use clap::Parser;
use cudarc::cublas;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::result::mem_get_info;
use cudarc::driver::{sys, CudaDevice, CudaSlice};
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu;
use nvml_wrapper::Nvml;
use rand::rngs::SmallRng;
use rand::RngCore;
use rand::SeedableRng;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc::{
    unbounded_channel, UnboundedReceiver as Receiver, UnboundedSender as Sender,
};

const SIZE: usize = 8192; // Ensure SIZE % 16 == 0 for Tensor Core optimization
const MEM_TO_USE_PCT: f64 = 0.9; // Use 90% of GPU memory
const MIN_DURATION_SECS: u64 = 10;

const GPU_THROTTLING_REASON: &str =
    "GPU is throttled. Check the throttling reasons and temperatures";
const GPU_FLOPS_REASON: &str =
    "GPU is not performing as expected. Check the flops values and temperatures";

type AllocBufferTuple = (CudaSlice<f32>, CudaSlice<f32>, Vec<CudaSlice<f32>>);

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Duration in seconds to burn the GPUs
    #[clap(default_value = "60")]
    duration_secs: u64,
    /// Path to NVIDIA Management Library (libnvidia-ml.so)
    #[clap(long, default_value = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")]
    nvml_lib_path: String,
    /// Tolerate software throttling if the TFLOPS are in the acceptable range
    #[clap(long, default_value = "false")]
    tolerate_software_throttling: bool,
    /// TFLOPS tolerance (%) compared to best GPU
    /// If the TFLOPS are within `tflops_tolerance`% of the best performing GPU, test will pass
    #[clap(long, default_value = "10")]
    tflops_tolerance: f64,
}

#[derive(Debug, Clone)]
struct BurnResult {
    gpu_idx: usize,
    flops_max: usize,
    flops_min: usize,
    flops_sum: usize,
    n_iters: usize,
    temp_max: usize,
    temp_sum: usize,
    temp_min: usize,
    throttling_hw: usize,
    throttling_thermal_sw: usize,
    throttling_thermal_hw: usize,
}

impl BurnResult {
    fn new(gpu_idx: usize) -> Self {
        Self {
            gpu_idx,
            flops_min: usize::MAX,
            temp_min: usize::MAX,
            ..Default::default()
        }
    }

    fn flops_avg(&self) -> f64 {
        if self.n_iters == 0 {
            0.0
        } else {
            self.flops_sum as f64 / self.n_iters as f64
        }
    }

    fn temp_avg(&self) -> f64 {
        if self.n_iters == 0 {
            0.0
        } else {
            self.temp_sum as f64 / self.n_iters as f64
        }
    }

    fn is_throttled(&self) -> bool {
        self.throttling_hw > 0 || self.throttling_thermal_sw > 0 || self.throttling_thermal_hw > 0
    }
}

impl Default for BurnResult {
    fn default() -> Self {
        Self {
            gpu_idx: 0,
            flops_max: 0,
            flops_min: usize::MAX,
            flops_sum: 0,
            n_iters: 0,
            temp_max: 0,
            temp_sum: 0,
            temp_min: usize::MAX,
            throttling_hw: 0,
            throttling_thermal_sw: 0,
            throttling_thermal_hw: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    duration_secs: u64,
    nvml_lib_path: String,
    tflops_tolerance: f64,
    tolerate_software_throttling: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.duration_secs < MIN_DURATION_SECS {
        eprintln!("Duration must be at least {} seconds", MIN_DURATION_SECS);
        std::process::exit(1);
    }
    if args.tflops_tolerance < 0.0 || args.tflops_tolerance > 100.0 {
        eprintln!("TFLOPS tolerance must be between 0 and 100");
        std::process::exit(1);
    }

    let config = Config {
        duration_secs: args.duration_secs,
        nvml_lib_path: args.nvml_lib_path.clone(),
        tflops_tolerance: args.tflops_tolerance,
        tolerate_software_throttling: args.tolerate_software_throttling,
    };

    match run(config).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
}

fn uuid_to_string(uuid: sys::CUuuid) -> String {
    let bytes = uuid.bytes;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

async fn run(config: Config) -> anyhow::Result<()> {
    let mut gpus = detect_gpus()?;
    if gpus.is_empty() {
        return Err(anyhow::anyhow!("No GPUs detected"));
    }
    // sort gpus per ordinal
    gpus.sort_by_key(|gpu| gpu.ordinal());
    for gpu in gpus.clone() {
        println!(
            "Detected GPU #{}: {:?} ({})",
            gpu.ordinal(),
            gpu.name()?,
            uuid_to_string(gpu.uuid()?)
        );
    }

    // create 2 matrix with random values
    println!("Creating random matrices");
    // use SmallRng to create random values, we don't need cryptographic security but we need speed
    let mut small_rng = SmallRng::from_entropy();
    let mut a = vec![0.0f32; SIZE * SIZE];
    let mut b = vec![0.0f32; SIZE * SIZE];
    for i in 0..SIZE * SIZE {
        a[i] = small_rng.next_u32() as f32;
        b[i] = small_rng.next_u32() as f32;
    }
    println!("Matrices created");

    let (tx, rx) = unbounded_channel::<(usize, usize)>();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    tokio::spawn(shutdown_signal(stop.clone()));
    let mut handles = Vec::new();
    for gpu in gpus.clone() {
        let tx = tx.clone();
        let stop = stop.clone();
        let gpu = gpu.clone();
        let a = a.clone();
        let b = b.clone();
        let t = tokio::spawn(async move {
            burn_gpu(gpu.ordinal(), a, b, tx, stop)
                .await
                .unwrap_or_else(|_| panic!("Unable to burn GPU #{}", gpu.ordinal()));
        });
        handles.push(t);
    }
    // report progress
    let stop_clone = stop.clone();
    let gpus_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let gpus_healthy_clone = gpus_healthy.clone();
    let nvml = Nvml::builder().lib_path(config.nvml_lib_path.as_ref()).init().expect("Unable to initialize NVML. Check if the NVIDIA driver is installed and the NVIDIA Management Library is available (libnvidia-ml.so).");
    let config_clone = config.clone();
    let t = tokio::spawn(async move {
        report_progress(
            config_clone,
            gpus.len(),
            nvml,
            rx,
            stop_clone,
            gpus_healthy_clone,
        )
        .await;
    });
    handles.push(t);
    // burn the GPU for given duration
    tokio::time::sleep(std::time::Duration::from_secs(config.duration_secs)).await;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    drop(tx);
    for handle in handles {
        handle.await.expect("Thread panicked");
    }
    if gpus_healthy.load(std::sync::atomic::Ordering::Relaxed) {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Some GPUs are not healthy"))
    }
}

fn poll_temperatures(nvml: &Nvml, gpu_count: usize) -> anyhow::Result<Vec<usize>> {
    let mut temps = vec![0usize; gpu_count];
    for (i, temp) in temps.iter_mut().enumerate().take(gpu_count) {
        let gpu = nvml.device_by_index(i as u32)?;
        *temp = gpu.temperature(Gpu)? as usize;
    }
    Ok(temps)
}

fn poll_throttling(nvml: &Nvml, gpu_count: usize) -> anyhow::Result<Vec<ThrottleReasons>> {
    let mut throttling = vec![];
    for i in 0..gpu_count {
        let gpu = nvml.device_by_index(i as u32)?;
        throttling.push(gpu.current_throttle_reasons()?);
    }
    Ok(throttling)
}

async fn report_progress(
    config: Config,
    gpu_count: usize,
    nvml: Nvml,
    mut rx: Receiver<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    gpus_healthy: Arc<std::sync::atomic::AtomicBool>,
) {
    // Use a fixed interval for reporting
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));

    let mut burn_results = (0..gpu_count).map(BurnResult::new).collect::<Vec<_>>();

    let mut tick = 0;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        interval.tick().await;
        let mut nops = vec![0usize; gpu_count];
        // Drain the channel to get the latest updates
        while let Ok(ops) = rx.try_recv() {
            nops[ops.0] += ops.1; // Accumulate operations
        }
        for i in 0..gpu_count {
            let flops = nops[i] * SIZE * SIZE * SIZE * 2;
            print!("{} ({} Gflops/s)", nops[i], flops / 1_000_000_000);
            if i < gpu_count - 1 {
                print!(" - ");
            } else {
                print!(" | ");
            }

            if tick > 4 {
                // Skip the first 4 ticks to avoid caches effects
                burn_results[i].flops_max = burn_results[i].flops_max.max(flops);
                burn_results[i].flops_min = burn_results[i].flops_min.min(flops);
                burn_results[i].flops_sum += flops;
                burn_results[i].n_iters += 1;
            }
        }
        // Report GPU temperatures
        let temps = poll_temperatures(&nvml, gpu_count).expect("Unable to poll temperatures");
        print!("Temperatures: ");
        for i in 0..gpu_count {
            print!("{}°C", temps[i]);
            if i < gpu_count - 1 {
                print!(" - ");
            } else {
                print!(" | ");
            }
            if tick > 4 {
                burn_results[i].temp_max = burn_results[i].temp_max.max(temps[i]);
                burn_results[i].temp_min = burn_results[i].temp_min.min(temps[i]);
                burn_results[i].temp_sum += temps[i];
            }
        }
        // Report throttling
        let throttling = poll_throttling(&nvml, gpu_count).expect("Unable to poll throttling");
        print!("Throttling: ");
        for i in 0..gpu_count {
            match throttling[i] {
                ThrottleReasons::SW_THERMAL_SLOWDOWN => {
                    print!("Thermal SW");
                    burn_results[i].throttling_thermal_sw += 1;
                }
                ThrottleReasons::HW_THERMAL_SLOWDOWN => {
                    print!("Thermal HW");
                    burn_results[i].throttling_thermal_hw += 1;
                }
                ThrottleReasons::HW_SLOWDOWN => {
                    print!("HW slowdown");
                    burn_results[i].throttling_hw += 1;
                }
                _ => {
                    print!("None");
                }
            }
            if i < gpu_count - 1 {
                print!(" - ");
            } else {
                println!();
            }
        }
        tick += 1;
    }
    for r in burn_results.clone() {
        println!(
            "GPU #{}: {:6.0} Gflops/s (min: {:.2}, max: {:.2}, dev: {:.2})",
            r.gpu_idx,
            r.flops_avg() / 1_000_000_000.0,
            r.flops_min as f64 / 1_000_000_000.0,
            r.flops_max as f64 / 1_000_000_000.0,
            r.flops_avg() / 1_000_000_000.0
        );
        println!(
            "         Temperature: {:.2}°C (min: {:.2}, max: {:.2})",
            r.temp_avg(),
            r.temp_min as f64,
            r.temp_max as f64
        );
        println!(
            "         Throttling HW: {}, Thermal SW: {}, Thermal HW: {}",
            r.throttling_hw > 0,
            r.throttling_thermal_sw > 0,
            r.throttling_thermal_hw > 0
        );
    }

    let (healthy, reasons) = are_gpus_healthy(
        burn_results,
        config.tflops_tolerance,
        config.tolerate_software_throttling,
    );
    if healthy {
        println!("All GPUs seem healthy");
    } else {
        println!("Some GPUs are not healthy. Reasons:");
        for r in reasons {
            println!("  - {}", r);
        }
    }
    gpus_healthy.store(healthy, std::sync::atomic::Ordering::Relaxed);
    println!("Freeing GPUs...");
}

fn are_gpus_healthy(
    burn_results: Vec<BurnResult>,
    tflops_tolerance: f64,
    tolerate_software_throttling: bool,
) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    // acceptable_flops is tflops_tolerance% lower than best gpu avg flops
    let acceptable_flops: f64 = burn_results
        .iter()
        .map(|r| r.flops_avg())
        .fold(0., |max, avg| {
            max.max(avg * (100. - tflops_tolerance) / 100.)
        });
    for r in burn_results.iter() {
        let mut low_flops = false;
        if r.flops_avg() < acceptable_flops {
            reasons.push(format!("GPU {} - ", r.gpu_idx) + GPU_FLOPS_REASON);
            low_flops = true;
        }
        // if we have any throttling
        if r.is_throttled() {
            if !low_flops
                && tolerate_software_throttling
                && (r.throttling_thermal_hw == 0 && r.throttling_hw == 0)
            {
                continue;
            }
            reasons.push(format!("GPU {} - ", r.gpu_idx) + GPU_THROTTLING_REASON);
        }
    }
    (reasons.is_empty(), reasons)
}

async fn burn_gpu(
    gpu_idx: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    tx: Sender<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<usize> {
    let gpu = CudaDevice::new(gpu_idx)?;
    // compute the output matrix size
    let (free_mem, _) = get_gpu_memory(gpu.clone())?;
    let mem_to_use = (free_mem as f64 * MEM_TO_USE_PCT) as usize;
    println!(
        "GPU #{}: Using {} MB out of {} MB",
        gpu_idx,
        mem_to_use / 1024 / 1024,
        free_mem / 1024 / 1024
    );
    let iters =
        (mem_to_use - 2 * SIZE * SIZE * size_of::<f32>()) / (SIZE * SIZE * size_of::<f32>());
    let (a_gpu, b_gpu, mut out_slices_gpu) = alloc_buffers(gpu.clone(), a, b, iters)?;
    let handle = cublas::safe::CudaBlas::new(gpu)?;
    let mut i = 0;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        for out in out_slices_gpu.iter_mut() {
            compute(&handle, &a_gpu, &b_gpu, out)?;
            i += 1;
            _ = tx.send((gpu_idx, 1));
        }
    }
    drop(tx);
    Ok(i)
}

fn get_gpu_memory(gpu: Arc<CudaDevice>) -> anyhow::Result<(usize, usize)> {
    CudaDevice::new(gpu.ordinal())?;
    let mem_info = mem_get_info()?;
    Ok(mem_info)
}

fn alloc_buffers(
    gpu: Arc<CudaDevice>,
    a: Vec<f32>,
    b: Vec<f32>,
    num_out_slices: usize,
) -> anyhow::Result<AllocBufferTuple> {
    let a_gpu = gpu.htod_copy(a)?;
    let b_gpu = gpu.htod_copy(b)?;
    let mut out_slices = vec![];
    for _ in 0..num_out_slices {
        let out = gpu.alloc_zeros::<f32>(SIZE * SIZE)?;
        out_slices.push(out);
    }
    Ok((a_gpu, b_gpu, out_slices))
}

fn compute(
    handle: &cublas::safe::CudaBlas,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: SIZE as i32,
        n: SIZE as i32,
        k: SIZE as i32,
        alpha: 1.0,
        lda: SIZE as i32,
        ldb: SIZE as i32,
        beta: 0.0,
        ldc: SIZE as i32,
    };
    unsafe {
        handle.gemm(cfg, a, b, out)?;
    }
    Ok(())
}

fn detect_gpus() -> anyhow::Result<Vec<Arc<CudaDevice>>> {
    let num_gpus = CudaDevice::count()? as usize;
    let mut devices = Vec::new();
    for i in 0..num_gpus {
        let dev = CudaDevice::new(i)?;
        devices.push(dev);
    }
    Ok(devices)
}

async fn shutdown_signal(stop: Arc<std::sync::atomic::AtomicBool>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to hook signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
}