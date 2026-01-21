use clap::Parser;
use cudarc::cublaslt::result::{create_matmul_desc, create_matrix_layout, matmul};
use cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t;
use cudarc::cublaslt::sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
use cudarc::cublaslt::sys::{cublasComputeType_t, cudaDataType};
use cudarc::cublaslt::{result, CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::driver::result::mem_get_info;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{
    sys, CudaContext, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
};
use float8::F8E4M3;
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu;
use nvml_wrapper::Nvml;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc::{
    unbounded_channel, UnboundedReceiver as Receiver, UnboundedSender as Sender,
};
use tokio::task::JoinHandle;

const SIZE: usize = 8192; // Ensure SIZE % 16 == 0 for Tensor Core optimization
const MEM_TO_USE_PCT: usize = 90; // Use 90% of GPU memory
const MIN_DURATION_SECS: u64 = 10;

const GPU_THROTTLING_REASON: &str =
    "GPU is throttled. Check the throttling reasons and temperatures";
const GPU_FLOPS_REASON: &str =
    "GPU is not performing as expected. Check the flops values and temperatures";
const GPU_ZERO_FLOPS_REASON: &str =
    "GPU reported 0 FLOPS, meaning it did not do any work. Check the GPU state for any XID errors and reset the GPU if needed";

type AllocBufferTuple<T> = (
    CudaSlice<T>,
    CudaSlice<T>,
    Vec<CudaSlice<<T as GpuCompute>::Out>>,
);

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Duration in seconds to burn the GPUs
    #[clap(default_value = "60")]
    duration_secs: u64,
    /// Path to NVIDIA Management Library (libnvidia-ml.so)
    #[clap(long)]
    nvml_lib_path: Option<String>,
    /// Tolerate software throttling if the TFLOPS are in the acceptable range
    #[clap(long, default_value = "false")]
    tolerate_software_throttling: bool,
    /// TFLOPS tolerance (%) compared to best GPU
    /// If the TFLOPS are within `tflops_tolerance`% of the best performing GPU, test will pass
    #[clap(long, default_value = "10")]
    tflops_tolerance: f64,
    /// Use FP32 precision. If unset, will use FP32 if no GPUs support BF16 or FP8.
    #[clap(long)]
    use_fp32: bool,
    /// Use BF16 precision. GPU must support BF16 type. If unset, will use BF16 only if all GPUs support it.
    #[clap(long)]
    use_bf16: bool,
    /// Use FP8 precision. GPU must support FP8 type.
    #[clap(long)]
    use_fp8: bool,
    /// Write final results as pretty JSON to the given file path
    #[clap(long = "output-file")]
    output_file: Option<String>,
}

#[derive(Debug, Clone)]
struct BurnResult {
    gpu_idx: usize,
    flops_max: usize,
    flops_min: usize,
    flops_sum: usize,
    flops_mean: f64,
    flops_m2: f64, // For Welford's algorithm: sum of squared differences from mean
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

    fn flops_stddev(&self) -> f64 {
        if self.n_iters < 2 {
            0.0
        } else {
            (self.flops_m2 / (self.n_iters - 1) as f64).sqrt()
        }
    }

    /// Update running statistics + Welford's algorithm for variance
    fn update_flops(&mut self, flops: usize) {
        self.flops_max = self.flops_max.max(flops);
        self.flops_min = self.flops_min.min(flops);
        self.flops_sum += flops;
        self.n_iters += 1;

        let flops_f64 = flops as f64;
        let delta = flops_f64 - self.flops_mean;
        self.flops_mean += delta / self.n_iters as f64;
        let delta2 = flops_f64 - self.flops_mean;
        self.flops_m2 += delta * delta2;
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
            flops_mean: 0.0,
            flops_m2: 0.0,
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
    nvml_lib_path: Option<String>,
    tflops_tolerance: f64,
    tolerate_software_throttling: bool,
    use_bf16: bool,
    use_fp8: bool,
    use_fp32: bool,
    output_file: Option<String>,
}

trait VariablePrecisionFloat:
    Copy + Debug + Send + Sync + Unpin + DeviceRepr + ValidAsZeroBits + 'static
{
    fn from_f32(f: f32) -> Self;
}

impl VariablePrecisionFloat for f32 {
    fn from_f32(f: f32) -> Self {
        f
    }
}

impl VariablePrecisionFloat for half::bf16 {
    fn from_f32(f: f32) -> Self {
        half::bf16::from_f32(f)
    }
}

impl VariablePrecisionFloat for F8E4M3 {
    fn from_f32(f: f32) -> Self {
        F8E4M3::from_f32(f)
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if args.duration_secs < MIN_DURATION_SECS {
        eprintln!("Duration must be at least {MIN_DURATION_SECS} seconds");
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
        use_fp32: args.use_fp32,
        use_bf16: args.use_bf16,
        use_fp8: args.use_fp8,
        output_file: args.output_file.clone(),
    };

    match run(config).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {e}");
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
    let use_fp8 = if config.use_fp8 {
        // If explicitly requested, check if all GPUs support it
        let all_support_fp8 = gpus.iter().all(|gpu| supports_fp8(gpu).unwrap_or(false));
        if !all_support_fp8 {
            return Err(anyhow::anyhow!(
                "FP8 was explicitly requested but not all GPUs support it. Remove the --use-fp8 flag"
            ));
        }
        config.use_fp8
    } else {
        false
    };

    let use_bf16 = if config.use_bf16 {
        // If explicitly requested, check if all GPUs support it
        let all_support_bf16 = gpus.iter().all(|gpu| supports_bf16(gpu).unwrap_or(false));
        if config.use_bf16 && !all_support_bf16 {
            return Err(anyhow::anyhow!(
                "BF16 was explicitly requested but not all GPUs support it. Remove the --use-bf16 flag"
            ));
        }
        config.use_bf16
    } else {
        // Auto-detect: use BF16 only if all GPUs support it
        gpus.iter().all(|gpu| supports_bf16(gpu).unwrap_or(false)) && !use_fp8 && !config.use_fp32
    };

    let use_fp32 = config.use_fp32 || (!use_bf16 && !use_fp8);

    println!(
        "Using precision(s): {} {} {}",
        if use_fp32 { "FP32" } else { "" },
        if use_bf16 { "BF16" } else { "" },
        if use_fp8 { "FP8" } else { "" }
    );

    let (tx, rx) = unbounded_channel::<(usize, usize)>();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    tokio::spawn(shutdown_signal(stop.clone()));

    // report progress
    let stop_clone = stop.clone();
    let gpus_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let gpus_healthy_clone = gpus_healthy.clone();
    let mut nvml_builder = Nvml::builder();
    if let Some(ref path) = config.nvml_lib_path {
        nvml_builder.lib_path(path.as_ref());
    }
    let nvml = nvml_builder.init().expect("Unable to initialize NVML. Check if the NVIDIA driver is installed and the NVIDIA Management Library is available (libnvidia-ml.so). Use --nvml-lib-path to manually define a path.");
    let config_clone = config.clone();
    let mut handles = Vec::new();
    let gpu_len = gpus.len();
    let t = tokio::spawn(async move {
        report_progress(
            config_clone,
            gpu_len,
            nvml,
            rx,
            stop_clone,
            gpus_healthy_clone,
        )
        .await;
    });
    handles.push(t);

    // compute the memory to use for each GPU, depending on the number of precisions we compute simultaneously
    let num_precisions = usize::from(use_fp32) + usize::from(use_bf16) + usize::from(use_fp8);
    let mem_to_use_pct = MEM_TO_USE_PCT / num_precisions;
    let mem_to_use_mb = gpus
        .iter()
        .map(|gpu| {
            let (free_mem, _) = get_gpu_memory(gpu.clone())?;
            Ok((gpu.ordinal(), free_mem / 1024 / 1024 * mem_to_use_pct / 100))
        })
        .collect::<Result<HashMap<usize, usize>, anyhow::Error>>()?;
    let mem_to_use = Arc::new(mem_to_use_mb);

    if use_bf16 {
        handles.append(
            &mut run_with_precision::<half::bf16>(
                gpus.clone(),
                mem_to_use.clone(),
                tx.clone(),
                stop.clone(),
            )
            .await,
        );
    }
    if use_fp8 {
        handles.append(
            &mut run_with_precision::<F8E4M3>(
                gpus.clone(),
                mem_to_use.clone(),
                tx.clone(),
                stop.clone(),
            )
            .await,
        );
    }
    if use_fp32 {
        handles.append(
            &mut run_with_precision::<f32>(gpus, mem_to_use.clone(), tx.clone(), stop.clone())
                .await,
        );
    }

    // burn the GPU for given duration
    let stop_cloned = stop.clone();
    let wait = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
        let mut tick = 0;
        while !stop_cloned.load(std::sync::atomic::Ordering::Relaxed) && tick < config.duration_secs
        {
            interval.tick().await;
            tick += 1;
        }
        stop_cloned.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    handles.push(wait);
    // for handle in handles {
    //     handle.await.expect("Thread panicked");
    // }
    // join all handles simultaneously
    let _ = futures::future::join_all(handles).await;
    if gpus_healthy.load(std::sync::atomic::Ordering::Relaxed) {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Some GPUs are not healthy"))
    }
}

async fn run_with_precision<T: VariablePrecisionFloat + GpuCompute>(
    gpus: Vec<Arc<CudaContext>>,
    mem_to_use_mb: Arc<HashMap<usize, usize>>,
    tx: Sender<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> Vec<JoinHandle<()>>
where
    CudaBlasLT: Matmul<T>,
{
    // create 2 matrix with random values
    println!("Creating random matrices");
    // use SmallRng to create random values, we don't need cryptographic security but we need speed
    let mut small_rng = SmallRng::from_rng(&mut rand::rng());
    let mut a = vec![T::from_f32(0.0); SIZE * SIZE];
    let mut b = vec![T::from_f32(0.0); SIZE * SIZE];
    // fill matrices with random values and scale them to a small range so that they fit in the float8 range
    for i in 0..SIZE * SIZE {
        a[i] = T::from_f32(small_rng.next_u32() as f32);
        b[i] = T::from_f32(small_rng.next_u32() as f32);
    }
    println!("Matrices created");

    let mut handles = Vec::new();
    for gpu in gpus.clone() {
        let tx = tx.clone();
        let stop = stop.clone();
        let gpu = gpu.clone();
        let a = a.clone();
        let b = b.clone();
        let mem = mem_to_use_mb[&gpu.ordinal()];
        let t = tokio::spawn(async move {
            unsafe {
                burn_gpu::<T>(gpu.ordinal(), mem, a, b, tx, stop)
                    .await
                    .unwrap_or_else(|e| panic!("Unable to burn GPU #{}: {e}", gpu.ordinal()));
            }
        });
        handles.push(t);
    }
    handles
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

fn supports_bf16(gpu: &Arc<CudaContext>) -> anyhow::Result<bool> {
    Ok(
        gpu.attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
            >= 8
            && gpu.attribute(
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )? >= 0,
    )
}

fn supports_fp8(gpu: &Arc<CudaContext>) -> anyhow::Result<bool> {
    Ok(
        gpu.attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
            >= 9
            && gpu.attribute(
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )? >= 0,
    )
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
        // If no operations were received, continue to the next tick (means no work done yet, still loading data)
        if nops.iter().all(|&x| x == 0) {
            continue;
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
                // Skip the first 5 ticks to avoid caches effects
                burn_results[i].update_flops(flops);
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
    // Keep a copy for JSON serialization before moving the vector
    let results_for_json = burn_results.clone();
    for r in burn_results.clone() {
        println!(
            "GPU #{}: {:6.0} Gflops/s (min: {:.2}, max: {:.2}, dev: {:.2})",
            r.gpu_idx,
            r.flops_avg() / 1_000_000_000.0,
            r.flops_min as f64 / 1_000_000_000.0,
            r.flops_max as f64 / 1_000_000_000.0,
            r.flops_stddev() / 1_000_000_000.0
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
    // If requested, write a pretty JSON report to the specified path
    if let Some(path) = &config.output_file {
        #[derive(Serialize)]
        struct GpuSummary {
            gpu_idx: usize,
            flops_gflops_avg: f64,
            flops_gflops_min: f64,
            flops_gflops_max: f64,
            temperature_celsius_avg: f64,
            temperature_celsius_min: f64,
            temperature_celsius_max: f64,
            throttling_hw: bool,
            throttling_thermal_sw: bool,
            throttling_thermal_hw: bool,
        }
        #[derive(Serialize)]
        struct JsonReport {
            gpus: Vec<GpuSummary>,
            healthy: bool,
            reasons: Vec<String>,
        }
        let gpus = results_for_json
            .into_iter()
            .map(|r| GpuSummary {
                gpu_idx: r.gpu_idx,
                flops_gflops_avg: r.flops_avg() / 1_000_000_000.0,
                flops_gflops_min: r.flops_min as f64 / 1_000_000_000.0,
                flops_gflops_max: r.flops_max as f64 / 1_000_000_000.0,
                temperature_celsius_avg: r.temp_avg(),
                temperature_celsius_min: r.temp_min as f64,
                temperature_celsius_max: r.temp_max as f64,
                throttling_hw: r.throttling_hw > 0,
                throttling_thermal_sw: r.throttling_thermal_sw > 0,
                throttling_thermal_hw: r.throttling_thermal_hw > 0,
            })
            .collect::<Vec<_>>();
        let report = JsonReport {
            gpus,
            healthy,
            reasons: reasons.clone(),
        };
        if let Ok(file) = std::fs::File::create(path) {
            let _ = serde_json::to_writer_pretty(file, &report);
        } else {
            eprintln!("Unable to create JSON output file at {}", path);
        }
    }
    if healthy {
        println!("All GPUs seem healthy");
    } else {
        println!("Some GPUs are not healthy. Reasons:");
        for r in reasons {
            println!("  - {r}");
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
        // if we report 0 FLOPS, it means the GPU did not do any work due to an inconsistent state
        // see https://github.com/huggingface/gpu-fryer/issues/10
        if r.flops_sum == 0 {
            reasons.push(format!("GPU {}", r.gpu_idx) + GPU_ZERO_FLOPS_REASON);
        }
    }
    (reasons.is_empty(), reasons)
}

trait GpuCompute: VariablePrecisionFloat {
    type Out: DeviceRepr + ValidAsZeroBits + 'static;
    unsafe fn compute(
        handle: &CudaBlasLT,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        out: &mut CudaSlice<Self::Out>,
    ) -> anyhow::Result<()>
    where
        Self: Sized + VariablePrecisionFloat;
}

impl GpuCompute for F8E4M3 {
    type Out = half::bf16;
    unsafe fn compute(
        handle: &CudaBlasLT,
        a: &CudaSlice<F8E4M3>,
        b: &CudaSlice<F8E4M3>,
        out: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        compute_fp8(handle, a, b, out)
    }
}

impl GpuCompute for half::bf16 {
    type Out = half::bf16;
    unsafe fn compute(
        handle: &CudaBlasLT,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        out: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        compute(handle, a, b, out)
    }
}

impl GpuCompute for f32 {
    type Out = f32;
    unsafe fn compute(
        handle: &CudaBlasLT,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        compute(handle, a, b, out)
    }
}

async unsafe fn burn_gpu<T>(
    gpu_idx: usize,
    mem_to_use_mb: usize,
    a: Vec<T>,
    b: Vec<T>,
    tx: Sender<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<usize>
where
    CudaBlasLT: Matmul<T>,
    T: GpuCompute + VariablePrecisionFloat,
{
    let gpu = CudaContext::new(gpu_idx)?;
    // compute the output matrix size
    println!("GPU #{gpu_idx}: Using {mem_to_use_mb} MB");
    let iters = (mem_to_use_mb * 1024 * 1024 - 2 * SIZE * SIZE * get_memory_size::<T::Out>())
        / (SIZE * SIZE * get_memory_size::<T::Out>());
    let (a_gpu, b_gpu, mut out_slices_gpu) = alloc_buffers::<T>(gpu.clone(), a, b, iters)?;
    let handle = CudaBlasLT::new(gpu.new_stream()?)?;
    let mut i = 0;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        for out in out_slices_gpu.iter_mut() {
            T::compute(&handle, &a_gpu, &b_gpu, out)?;
            i += 1;
            _ = tx.send((gpu_idx, 1));
        }
    }
    drop(tx);
    Ok(i)
}

fn get_memory_size<T>() -> usize {
    size_of::<T>()
}

fn get_gpu_memory(gpu: Arc<CudaContext>) -> anyhow::Result<(usize, usize)> {
    CudaContext::new(gpu.ordinal())?;
    let mem_info = mem_get_info()?;
    Ok(mem_info)
}

fn alloc_buffers<T>(
    gpu: Arc<CudaContext>,
    a: Vec<T>,
    b: Vec<T>,
    num_out_slices: usize,
) -> anyhow::Result<AllocBufferTuple<T>>
where
    T: GpuCompute + VariablePrecisionFloat,
{
    let stream = gpu.default_stream();
    let a_gpu = stream.memcpy_stod(&a)?;
    let b_gpu = stream.memcpy_stod(&b)?;
    let mut out_slices = vec![];
    for _ in 0..num_out_slices {
        let out = stream.alloc_zeros::<T::Out>(SIZE * SIZE)?;
        out_slices.push(out);
    }
    Ok((a_gpu, b_gpu, out_slices))
}

/// Matrix multiplication for FP8 using CUBLAS LT as it is not supported in the standard CUBLAS API.
/// We use F8E4M3 as the input type and produce half::bf16 as the output type.
/// The function uses cudarc low-level bindings as it does not support Matmul with heterogeneous types (F8E4M3 matmul with bf16 output).
unsafe fn compute_fp8(
    handle: &CudaBlasLT,
    a: &CudaSlice<F8E4M3>,
    b: &CudaSlice<F8E4M3>,
    out: &mut CudaSlice<half::bf16>,
) -> anyhow::Result<()> {
    let stream = handle.stream().clone();
    let major = stream
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 }; // that should be exposed in cudarc API, but for now everything is private
    let mut buffer = stream.alloc_zeros::<u8>(workspace_size)?;

    let desc = create_matmul_desc(
        cublasComputeType_t::CUBLAS_COMPUTE_32F, // compute type
        cudaDataType::CUDA_R_32F,                // scale type
    )?;
    result::set_matmul_desc_attribute(
        desc,
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        (&1) as *const _ as *const _,
        size_of::<u32>(),
    )?;
    let layout_a = create_matrix_layout(
        cudaDataType::CUDA_R_8F_E4M3, // data type for A
        SIZE as u64,                  // rows
        SIZE as u64,                  // cols
        SIZE as i64,                  // leading dimension
    )?;
    let layout_b = create_matrix_layout(
        cudaDataType::CUDA_R_8F_E4M3, // data type for B
        SIZE as u64,                  // rows
        SIZE as u64,                  // cols
        SIZE as i64,                  // leading dimension
    )?;
    let layout_c = create_matrix_layout(
        cudaDataType::CUDA_R_16F, // data type for C
        SIZE as u64,              // rows
        SIZE as u64,              // cols
        SIZE as i64,              // leading dimension
    )?;
    let alpha = 1.0f32;
    let beta = 0.0f32;

    // get the heuristic for the best algorithm
    let matmul_pref_handle = result::create_matmul_pref()?;
    result::set_matmul_pref_attribute(
        matmul_pref_handle,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        (&buffer) as *const _ as *const _,
        size_of::<usize>(),
    )?;
    let heuristic = result::get_matmul_algo_heuristic(
        *handle.handle(),
        desc,
        layout_a,
        layout_b,
        layout_c,
        layout_c,
        matmul_pref_handle,
    )?;

    // run matmul kernel
    let (a, _) = a.device_ptr(handle.stream());
    let (b, _) = b.device_ptr(handle.stream());
    let (out, _) = out.device_ptr_mut(handle.stream());
    let (w, _) = buffer.device_ptr_mut(handle.stream());
    matmul(
        *handle.handle(),
        desc,
        &alpha as *const _ as *const _, // alpha
        &beta as *const _ as *const _,  // beta
        a as *const _,
        layout_a,
        b as *const _,
        layout_b,
        out as *const _,
        layout_c,
        out as *mut _,
        layout_c,
        (&heuristic.algo) as *const _,
        w as *mut _, // workspace
        workspace_size,
        handle.stream().cu_stream() as *mut _,
    )?;
    Ok(())
}

fn compute<T: VariablePrecisionFloat>(
    handle: &CudaBlasLT,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    out: &mut CudaSlice<T>,
) -> anyhow::Result<()>
where
    CudaBlasLT: Matmul<T>,
{
    let cfg = MatmulConfig {
        transa: false,
        transb: false,
        transc: false,
        m: SIZE as u64,
        n: SIZE as u64,
        k: SIZE as u64,
        alpha: 1.0,
        lda: SIZE as i64,
        ldb: SIZE as i64,
        beta: 0.0,
        ldc: SIZE as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    unsafe {
        handle.matmul(cfg, a, b, out, None, None)?;
    }
    Ok(())
}

fn detect_gpus() -> anyhow::Result<Vec<Arc<CudaContext>>> {
    let num_gpus = CudaContext::device_count()? as usize;
    let mut devices = Vec::new();
    for i in 0..num_gpus {
        let dev = CudaContext::new(i)?;
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
