# GPU fryer 🍳

GPU fryer is a tool to stress test GPUs and detect any abnormal thermal throttling or performance degradation.
It is especially useful to test GPUs running ML inference or training workloads for which
performances are dictated by the slowest GPU in the system.

We use it at [Hugging Face](https://huggingface.co) 🤗 to monitor our HPC clusters and ensure that all GPUs are running at
peak performance.

![cooking.jpg](../assets/cooking.jpg)

## Usage

```bash
$ gpu-fryer 60  # Run the test for 60 seconds
```

GPU fryer relies on NVIDIA's CUDA toolkit to run the stress test, so make sure
that your PATH includes the CUDA libs.
NVML is used to monitor the GPU's temperature and performance, in case of non default
installations, you can use the `--nvml-lib-path` flag to specify the path to `libnvidia-ml.so`.

GPU fryer checks for homogeneous performance across all GPUs in the system (if multiple GPUs are present) and reports
any performance degradation or thermal throttling.
There is currently no absolute performance metric. For reference:

| GPU                   | TFLOPS |
|-----------------------|--------|
| NVIDIA H100 80GB HBM3 | ~51    |

## Installation

```bash
$ cargo install gpu-fryer
```

## How it works

GPU fryer creates two 8192x8192 matrix and performs a matrix multiplication using CUBLAS.
Test allocates 95% of the GPU memory to write results in a ring buffer fashion.  