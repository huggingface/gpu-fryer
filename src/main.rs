use cudarc::cublas;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaSlice};
use nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu;
use nvml_wrapper::Nvml;
use rand::rngs::SmallRng;
use rand::RngCore;
use rand::SeedableRng;
use std::sync::Arc;
use tokio::sync::mpsc::{
    unbounded_channel, UnboundedReceiver as Receiver, UnboundedSender as Sender,
};

const SIZE: usize = 8192; // 8192x8192 matrix, ensure that SIZE%16==0 to optimize Tensor Core usage

#[tokio::main]
async fn main() {
    run().await.expect("TODO: panic message");
    println!("End of program");
}

async fn run() -> anyhow::Result<()> {
    let mut gpus = detect_gpus()?;
    if gpus.is_empty() {
        return Err(anyhow::anyhow!("No GPUs detected"));
    }
    // sort gpus per ordinal
    gpus.sort_by_key(|gpu| gpu.ordinal());
    for gpu in gpus.clone() {
        println!(
            "Detected GPU #{}: {:?} ({:?})",
            gpu.ordinal(),
            gpu.name()?,
            gpu.uuid()
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
    let mut handles = Vec::new();
    for gpu in gpus.clone() {
        let tx = tx.clone();
        let stop = stop.clone();
        let gpu = gpu.clone();
        let a = a.clone();
        let b = b.clone();
        let t = tokio::spawn(async move {
            burn_gpu(gpu.clone(), a, b, tx, stop)
                .await
                .expect("TODO: panic message");
        });
        handles.push(t);
    }
    // report progress
    let stop_clone = stop.clone();
    let t = tokio::spawn(async move {
        report_progress(gpus.len(), rx, stop_clone).await;
    });
    handles.push(t);
    // burn the GPU for 10 seconds
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    drop(tx);
    for handle in handles {
        handle.await.expect("TODO: panic message");
    }
    Ok(())
}

fn poll_temperatures(nvml: &Nvml, gpu_count: usize) -> anyhow::Result<Vec<usize>> {
    let mut temps = vec![0usize; gpu_count];
    for i in 0..gpu_count {
        let gpu = nvml.device_by_index(i as u32)?;
        temps[i] = gpu.temperature(Gpu)? as usize;
    }
    Ok(temps)
}

async fn report_progress(
    gpu_count: usize,
    mut rx: Receiver<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) {
    let nvml = Nvml::builder().lib_path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1".as_ref()).init().expect("Unable to initialize NVML. Check if the NVIDIA driver is installed and the NVIDIA Management Library is available (libnvidia-ml.so).");

    // Use a fixed interval for reporting
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));

    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        interval.tick().await;
        let mut nops = vec![0usize; gpu_count];
        // Drain the channel to get the latest updates
        while let Ok(ops) = rx.try_recv() {
            nops[ops.0] += ops.1; // Accumulate operations
        }
        for i in 0..gpu_count {
            print!("{} ({} Gflops/s)", nops[i], nops[i]* SIZE * SIZE * SIZE); // / 1_000_000_000);
            if i < gpu_count - 1 {
                print!(" - ");
            } else {
                print!(" | ");
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
                println!();
            }
        }
    }
}

async fn burn_gpu(
    gpu: Arc<CudaDevice>,
    a: Vec<f32>,
    b: Vec<f32>,
    tx: Sender<(usize, usize)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<usize> {
    let ordinal = gpu.ordinal();
    let (a_gpu, b_gpu, mut out_gpu) = alloc_buffers(gpu.clone(), a, b)?;
    let handle = cublas::safe::CudaBlas::new(gpu.clone())?;
    let mut i = 0;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        compute(&handle, &a_gpu, &b_gpu, &mut out_gpu)?;
        i += 100;
        tx.send((ordinal, i))?;
    }
    drop(tx);
    Ok(i)
}

fn alloc_buffers(
    gpu: Arc<CudaDevice>,
    a: Vec<f32>,
    b: Vec<f32>,
) -> anyhow::Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
    let a_gpu = gpu.htod_copy(a)?;
    let b_gpu = gpu.htod_copy(b)?;
    let out = gpu.alloc_zeros::<f32>(SIZE * SIZE)?;
    Ok((a_gpu, b_gpu, out))
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
        for _ in 0..100 {
            let _res = handle.gemm(cfg, a, b, out)?;
        }
    }
    Ok(())
}

// fn init_compare_kernel(gpu:Arc<CudaDevice>,a:Vec<f32>,b:Vec<f32>)->anyhow::Result<()>{
//     load_kernel(gpu.clone())?;
//     let a_gpu=gpu.htod_copy(a)?;
//     let b_gpu=gpu.htod_copy(b)?;
//     let mut out = gpu.alloc_zeros::<f32>(SIZE*SIZE)?;
//     let kernel=gpu.get_func("compare","compare").expect("TODO: panic message");
//     let cfg=LaunchConfig::for_num_elems(100);
//     unsafe {
//         kernel.launch(cfg, (out,))?;
//     }
// }

// fn load_kernel(gpu: Arc<CudaDevice>) -> anyhow::Result<()> {
//     let ptx = cudarc::nvrtc::compile_ptx(
//         "
//         extern \"C\" __global__ void compare(float *C, int *faultyElems, size_t iters) {
//             size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
//             size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
//                 gridDim.x*blockDim.x + // W
//                 blockIdx.x*blockDim.x + threadIdx.x; // X
//
//             int myFaulty = 0;
//             for (size_t i = 1; i < iters; ++i)
//                 if (fabsf(C[myIndex] - C[myIndex + i*iterStep]) > EPSILON)
//                     myFaulty++;
//
//             atomicAdd(faultyElems, myFaulty);
//         }
//     ",
//     );
//     let kernel = gpu.load_ptx(ptx, "compare", &["compare"])?;
//     Ok(())
// }

fn detect_gpus() -> anyhow::Result<Vec<Arc<CudaDevice>>> {
    let dev_count = CudaDevice::count()? as usize;
    let mut devices = Vec::new();
    for i in 0..dev_count {
        let dev = CudaDevice::new(i)?;
        devices.push(dev);
    }
    Ok(devices)
}
