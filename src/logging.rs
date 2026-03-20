use std::fmt::Display;
use std::sync::OnceLock;

use nvml_wrapper::bitmasks::device::ThrottleReasons;
use serde::Serialize;

use crate::{BadResultReasons, BurnResult, GpuProblem};

#[derive(Clone, Copy)]
enum OutputMode {
    Human,
    Json,
}

static OUTPUT_MODE: OnceLock<OutputMode> = OnceLock::new();

fn output_mode() -> OutputMode {
    *OUTPUT_MODE
        .get()
        .expect("Output mode not initialized, call init_output_mode first")
}

impl OutputMode {
    fn emit_log(&self, value: &(impl Serialize + Display)) {
        match self {
            Self::Human => println!("{value}"),
            Self::Json => {
                let json = serde_json::to_string(value).expect("Failed to serialize log output");
                println!("{json}");
            }
        }
    }
}

pub fn init_output_mode(json: bool) {
    let mode = if json {
        OutputMode::Json
    } else {
        OutputMode::Human
    };
    let _ = OUTPUT_MODE.set(mode);
}

#[derive(Serialize)]
struct ErrorLog {
    message: String,
}

impl Display for ErrorLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn log_error(msg: impl Into<String>) {
    let error_log = ErrorLog {
        message: msg.into(),
    };
    match output_mode() {
        OutputMode::Json => {
            let json = serde_json::to_string(&error_log).expect("Failed to serialize error log");
            eprintln!("{json}");
        }
        OutputMode::Human => eprintln!("Error: {error_log}"),
    }
}

pub fn log(out_struct: impl Serialize + Display) {
    output_mode().emit_log(&out_struct);
}

pub fn log_raw(msg: &str) {
    match output_mode() {
        OutputMode::Human => println!("{msg}"),
        // Keep raw messages out of JSON stdout stream while preserving visibility.
        OutputMode::Json => eprintln!("{msg}"),
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "detected_gpus")]
pub struct LogDetectedGPUs {
    pub detected_gpus: Vec<LogGPUInfo>,
}

#[derive(Serialize)]
pub struct LogGPUInfo {
    pub index: usize,
    pub name: String,
    pub uuid: String,
}

impl Display for LogDetectedGPUs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, gpu) in self.detected_gpus.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(
                f,
                "Detected GPU #{}: {:?} ({})",
                gpu.index, gpu.name, gpu.uuid
            )?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "feature_detection")]
pub struct LogFeatureDetection {
    pub bf16: bool,
    pub fp8: bool,
    pub fp32: bool,
    pub gpu_memory_used: Vec<LogGPUMemoryInfo>,
}

#[derive(Serialize)]
pub struct LogGPUMemoryInfo {
    pub gpu_idx: usize,
    pub memory_mb: usize,
}

impl Display for LogFeatureDetection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Using precision(s): {} {} {}",
            if self.fp32 { "FP32" } else { "" },
            if self.bf16 { "BF16" } else { "" },
            if self.fp8 { "FP8" } else { "" },
        )?;
        for gpu in &self.gpu_memory_used {
            writeln!(f)?;
            write!(f, "GPU #{}: Using {} MB", gpu.gpu_idx, gpu.memory_mb)?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "progress")]
pub struct LogProgress {
    pub gpus: Vec<GpuProgress>,
}

#[derive(Serialize)]
pub struct GpuProgress {
    pub gpu_idx: usize,
    pub nops: usize,
    pub flops: usize,
    pub temp_celsius: usize,
    pub throttle_reason: Option<&'static str>,
}

pub fn throttle_reason_for_log(reason: ThrottleReasons) -> Option<&'static str> {
    if reason == ThrottleReasons::SW_THERMAL_SLOWDOWN {
        Some("SW_THERMAL_SLOWDOWN")
    } else if reason == ThrottleReasons::HW_THERMAL_SLOWDOWN {
        Some("HW_THERMAL_SLOWDOWN")
    } else if reason == ThrottleReasons::HW_SLOWDOWN {
        Some("HW_SLOWDOWN")
    } else {
        None
    }
}

impl Display for LogProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.gpus.len() {
            write!(
                f,
                "{} ({} Gflops/s)",
                self.gpus[i].nops,
                self.gpus[i].flops / 1_000_000_000
            )?;
            if i < self.gpus.len() - 1 {
                write!(f, " - ")?;
            } else {
                write!(f, " | ")?;
            }
        }
        write!(f, "Temperatures: ")?;
        for i in 0..self.gpus.len() {
            write!(f, "{}°C", self.gpus[i].temp_celsius)?;
            if i < self.gpus.len() - 1 {
                write!(f, " - ")?;
            } else {
                write!(f, " | ")?;
            }
        }
        // Report throttling
        write!(f, "Throttling: ")?;
        for i in 0..self.gpus.len() {
            match self.gpus[i].throttle_reason {
                Some("SW_THERMAL_SLOWDOWN") => {
                    write!(f, "Thermal SW")?;
                }
                Some("HW_THERMAL_SLOWDOWN") => {
                    write!(f, "Thermal HW")?;
                }
                Some("HW_SLOWDOWN") => {
                    write!(f, "HW slowdown")?;
                }
                _ => {
                    write!(f, "None")?;
                }
            }
            if i < self.gpus.len() - 1 {
                write!(f, " - ")?;
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "results")]
pub struct LogResults {
    pub gpus: Vec<AggregatedResults>,
}

#[derive(Serialize)]
pub struct AggregatedResults {
    pub burn_result: BurnResult,
    pub flops_avg: f64,
}

impl Display for LogResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, r) in self.gpus.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            writeln!(
                f,
                "GPU #{}: {:6.0} Gflops/s (min: {:.2}, max: {:.2}, dev: {:.2})",
                r.burn_result.gpu_idx,
                r.flops_avg / 1_000_000_000.0,
                r.burn_result.flops_min as f64 / 1_000_000_000.0,
                r.burn_result.flops_max as f64 / 1_000_000_000.0,
                r.burn_result.flops_stddev() / 1_000_000_000.0
            )?;
            writeln!(
                f,
                "         Temperature: {:.2}°C (min: {:.2}, max: {:.2})",
                r.burn_result.temp_avg(),
                r.burn_result.temp_min as f64,
                r.burn_result.temp_max as f64
            )?;
            write!(
                f,
                "         Throttling HW: {}, Thermal SW: {}, Thermal HW: {}",
                r.burn_result.throttling_hw > 0,
                r.burn_result.throttling_thermal_sw > 0,
                r.burn_result.throttling_thermal_hw > 0
            )?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "problems")]
pub struct LogProblems {
    pub problems: Vec<GpuProblem>,
}

impl Display for LogProblems {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.problems.is_empty() {
            write!(f, "All GPUs seem healthy")?;
        } else {
            write!(f, "Some GPUs are not healthy. Reasons:")?;
            for p in &self.problems {
                writeln!(f)?;
                write!(f, "  - GPU {} - {}", p.gpu_idx, p.reason)?;
            }
        }
        Ok(())
    }
}

impl Display for BadResultReasons {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BadResultReasons::Throttling => write!(f, "GPU is throttled. Check the throttling reasons and temperatures"),
            BadResultReasons::LowFlops => write!(f, "GPU is not performing as expected. Check the flops values and temperatures"),
            BadResultReasons::ZeroFlops => write!(f, "GPU reported 0 FLOPS, meaning it did not do any work. Check the GPU state for any XID errors and reset the GPU if needed"),
        }
    }
}
