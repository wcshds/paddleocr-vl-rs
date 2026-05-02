use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::prelude::Backend;
use burn::tensor::DType;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use paddleocr_vl::{OcrTask, PaddleOcrVersion, PaddleOcrVl};

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif"];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Cpu,
    Cuda,
    Flex,
    Metal,
    Mlx,
    Ndarray,
    Vulkan,
    Wgpu,
}

// Backend/dtype support follows Burn 0.21.0-pre.4:
// - NdArray is f32-only in this example because Burn's NdArray backend does
//   not implement f16/bf16 float element types.
// - Flex uses a single backend type for all supported dtypes; precision is
//   selected by the model-loading `DType`.
// - Vulkan requires the Vulkan SDK on macOS when the Burn/CubeCL SPIR-V path is
//   enabled, so build availability depends on the local toolchain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DTypeArg {
    F32,
    F16,
    Bf16,
}

impl From<DTypeArg> for DType {
    fn from(value: DTypeArg) -> Self {
        match value {
            DTypeArg::F32 => DType::F32,
            DTypeArg::F16 => DType::F16,
            DTypeArg::Bf16 => DType::BF16,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run direct PaddleOCR-VL recognition on images")]
struct Cli {
    #[arg(long, value_parser = parse_version, default_value = "1.5")]
    version: PaddleOcrVersion,

    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    backend: BackendArg,

    #[arg(long)]
    device: Option<String>,

    #[arg(long, value_enum, default_value_t = DTypeArg::F32)]
    dtype: DTypeArg,

    #[arg(long)]
    model_dir: Option<PathBuf>,

    #[arg(long, value_parser = parse_task, default_value = "ocr")]
    task: OcrTask,

    #[arg(long)]
    max_new_tokens: Option<usize>,

    #[arg(required = true)]
    input: Vec<PathBuf>,
}

impl Cli {
    fn model_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(self.version.default_model_dir()))
    }
}

fn parse_version(value: &str) -> Result<PaddleOcrVersion, String> {
    value.parse()
}

fn parse_task(value: &str) -> Result<OcrTask, String> {
    value.parse()
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

fn collect_images(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(input)
                .map_err(|e| format!("failed to read directory {}: {e}", input.display()))?
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    (path.is_file() && is_image_file(&path)).then_some(path)
                })
                .collect();
            entries.sort();
            paths.extend(entries);
        } else if input.is_file() {
            if is_image_file(input) {
                paths.push(input.clone());
            } else {
                return Err(format!("{} is not a supported image file", input.display()));
            }
        } else {
            eprintln!(
                "Warning: {} is not a valid file or directory, skipping.",
                input.display()
            );
        }
    }
    Ok(paths)
}

fn spinner_style() -> Result<ProgressStyle, String> {
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .map_err(|e| e.to_string())
}

fn device_spec(cli: &Cli) -> Option<&str> {
    cli.device
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn validate_task_version(cli: &Cli) -> Result<(), String> {
    if cli.task.is_spotting() && cli.version != PaddleOcrVersion::V1_5 {
        Err("spotting is only supported by the PaddleOCR-VL 1.5 recipe".into())
    } else {
        Ok(())
    }
}

#[cfg(any(feature = "cpu", feature = "flex", feature = "ndarray"))]
fn ensure_default_device(cli: &Cli, backend: &str) -> Result<(), String> {
    match device_spec(cli) {
        None | Some("default") | Some("cpu") => Ok(()),
        Some(other) => Err(format!(
            "backend {backend} only supports the default CPU device, got '{other}'"
        )),
    }
}

#[cfg(feature = "cuda")]
fn parse_index_device(cli: &Cli, backend: &str) -> Result<usize, String> {
    let Some(spec) = device_spec(cli) else {
        return Ok(0);
    };
    let spec = spec
        .strip_prefix("cuda:")
        .or_else(|| spec.strip_prefix("gpu:"))
        .unwrap_or(spec);
    if spec == "default" {
        return Ok(0);
    }
    spec.parse::<usize>()
        .map_err(|_| format!("backend {backend} expects --device <index>, got '{spec}'"))
}

#[cfg(any(feature = "metal", feature = "vulkan", feature = "wgpu"))]
fn parse_wgpu_device<B: Backend>(cli: &Cli, backend: &str) -> Result<B::Device, String> {
    use burn::tensor::backend::{Device, DeviceId};

    fn device<B: Backend>(type_id: u16, index_id: u16) -> B::Device {
        <B::Device as Device>::from_id(DeviceId::new(type_id, index_id))
    }

    let Some(spec) = device_spec(cli) else {
        return Ok(B::Device::default());
    };
    let lower = spec.to_ascii_lowercase();
    if lower == "default" {
        return Ok(B::Device::default());
    }
    if lower == "cpu" {
        return Ok(device::<B>(3, 0));
    }
    if let Ok(index) = lower.parse::<usize>() {
        return Ok(device::<B>(0, index as u16));
    }
    let (kind, index) = lower
        .split_once(':')
        .ok_or_else(|| format!("unsupported --device '{spec}' for backend {backend}"))?;
    let index = index
        .parse::<usize>()
        .map_err(|_| format!("invalid device index in --device '{spec}'"))?;
    match kind {
        "discrete" | "discrete-gpu" | "discrete_gpu" => Ok(device::<B>(0, index as u16)),
        "integrated" | "integrated-gpu" | "integrated_gpu" => Ok(device::<B>(1, index as u16)),
        "virtual" | "virtual-gpu" | "virtual_gpu" => Ok(device::<B>(2, index as u16)),
        "existing" => Ok(device::<B>(5, index as u16)),
        _ => Err(format!(
            "unsupported --device '{spec}' for backend {backend}"
        )),
    }
}

#[cfg(feature = "mlx")]
fn parse_mlx_device(cli: &Cli) -> Result<burn_mlx::MlxDevice, String> {
    match device_spec(cli) {
        None | Some("default") | Some("gpu") => Ok(burn_mlx::MlxDevice::Gpu),
        Some("cpu") => Ok(burn_mlx::MlxDevice::Cpu),
        Some(other) => Err(format!(
            "backend mlx expects --device gpu, cpu, or default, got '{other}'"
        )),
    }
}

fn run<B: Backend>(device: B::Device, cli: &Cli) -> Result<(), String> {
    let model_dir = cli.model_dir();
    let image_paths = collect_images(&cli.input)?;
    if image_paths.is_empty() {
        return Err("no image files found in the given paths".into());
    }
    validate_task_version(cli)?;

    eprintln!(
        "[init] Loading PaddleOCR-VL {:?} from {} (backend: {:?}, dtype: {:?}, task: {:?})...",
        cli.version,
        model_dir.display(),
        cli.backend,
        cli.dtype,
        cli.task
    );
    let t = Instant::now();
    let mut builder = PaddleOcrVl::<B>::builder(device)
        .version(cli.version)
        .model_dir(model_dir)
        .dtype(cli.dtype.into());
    if let Some(max_new_tokens) = cli.max_new_tokens {
        builder = builder.max_new_tokens(max_new_tokens);
    }
    let ocr = builder.build()?;
    eprintln!("[init] Loaded in {:.1}s\n", t.elapsed().as_secs_f64());

    for (idx, path) in image_paths.iter().enumerate() {
        eprintln!(
            "━━━ [{}/{}] {} ━━━",
            idx + 1,
            image_paths.len(),
            path.display()
        );
        let t = Instant::now();
        let progress = ProgressBar::new_spinner();
        progress.set_style(spinner_style()?);
        progress.enable_steady_tick(Duration::from_millis(120));
        progress.set_message(format!("recognizing {}", path.display()));
        let result = ocr.recognize(path, cli.task);
        if result.is_err() {
            progress.abandon_with_message("failed");
        }
        let result = result?;
        progress.finish_with_message(format!(
            "done: generated {} tokens",
            result.generated_tokens.len()
        ));
        eprintln!(
            "  Grid: {}×{}, generated {} tokens in {:.2}s",
            result.image_grid_hw[0],
            result.image_grid_hw[1],
            result.generated_tokens.len(),
            t.elapsed().as_secs_f64()
        );
        println!("{}", result.text);
        eprintln!();
    }

    Ok(())
}

fn run_selected_backend(cli: &Cli) -> Result<(), String> {
    match cli.backend {
        BackendArg::Cpu => run_cpu(cli),
        BackendArg::Cuda => run_cuda(cli),
        BackendArg::Flex => run_flex(cli),
        BackendArg::Metal => run_metal(cli),
        BackendArg::Mlx => run_mlx(cli),
        BackendArg::Ndarray => run_ndarray(cli),
        BackendArg::Vulkan => run_vulkan(cli),
        BackendArg::Wgpu => run_wgpu(cli),
    }
}

#[cfg(feature = "cpu")]
fn run_cpu(cli: &Cli) -> Result<(), String> {
    ensure_default_device(cli, "cpu")?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::Cpu>(Default::default(), cli),
        DTypeArg::F16 => run::<burn::backend::Cpu<half::f16>>(Default::default(), cli),
        DTypeArg::Bf16 => run::<burn::backend::Cpu<half::bf16>>(Default::default(), cli),
    }
}

#[cfg(not(feature = "cpu"))]
fn run_cpu(_cli: &Cli) -> Result<(), String> {
    Err("backend cpu was requested, but feature `cpu` is not enabled".into())
}

#[cfg(feature = "cuda")]
fn run_cuda(cli: &Cli) -> Result<(), String> {
    let device = burn::backend::cuda::CudaDevice::new(parse_index_device(cli, "cuda")?);
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::Cuda>(device, cli),
        DTypeArg::F16 => run::<burn::backend::Cuda<half::f16>>(device, cli),
        DTypeArg::Bf16 => run::<burn::backend::Cuda<half::bf16>>(device, cli),
    }
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(_cli: &Cli) -> Result<(), String> {
    Err("backend cuda was requested, but feature `cuda` is not enabled".into())
}

#[cfg(feature = "flex")]
fn run_flex(cli: &Cli) -> Result<(), String> {
    ensure_default_device(cli, "flex")?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::Flex>(Default::default(), cli),
        DTypeArg::F16 => run::<burn::backend::Flex>(Default::default(), cli),
        DTypeArg::Bf16 => run::<burn::backend::Flex>(Default::default(), cli),
    }
}

#[cfg(not(feature = "flex"))]
fn run_flex(_cli: &Cli) -> Result<(), String> {
    Err("backend flex was requested, but feature `flex` is not enabled".into())
}

#[cfg(feature = "metal")]
fn run_metal(cli: &Cli) -> Result<(), String> {
    let device = parse_wgpu_device::<burn::backend::Metal>(cli, "metal")?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::Metal>(device, cli),
        DTypeArg::F16 => run::<burn::backend::Metal<half::f16>>(device, cli),
        DTypeArg::Bf16 => run::<burn::backend::Metal<half::bf16>>(device, cli),
    }
}

#[cfg(not(feature = "metal"))]
fn run_metal(_cli: &Cli) -> Result<(), String> {
    Err("backend metal was requested, but feature `metal` is not enabled".into())
}

#[cfg(feature = "mlx")]
fn run_mlx(cli: &Cli) -> Result<(), String> {
    let device = parse_mlx_device(cli)?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn_mlx::Mlx>(device, cli),
        DTypeArg::F16 => run::<burn_mlx::Mlx<half::f16>>(device, cli),
        DTypeArg::Bf16 => run::<burn_mlx::Mlx<half::bf16>>(device, cli),
    }
}

#[cfg(not(feature = "mlx"))]
fn run_mlx(_cli: &Cli) -> Result<(), String> {
    Err("backend mlx was requested, but feature `mlx` is not enabled".into())
}

#[cfg(feature = "ndarray")]
fn run_ndarray(cli: &Cli) -> Result<(), String> {
    ensure_default_device(cli, "ndarray")?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::NdArray>(Default::default(), cli),
        DTypeArg::F16 | DTypeArg::Bf16 => Err(format!(
            "backend ndarray does not support dtype {:?}",
            cli.dtype
        )),
    }
}

#[cfg(not(feature = "ndarray"))]
fn run_ndarray(_cli: &Cli) -> Result<(), String> {
    Err("backend ndarray was requested, but feature `ndarray` is not enabled".into())
}

#[cfg(feature = "vulkan")]
fn run_vulkan(cli: &Cli) -> Result<(), String> {
    let device = parse_wgpu_device::<burn::backend::Vulkan>(cli, "vulkan")?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::Vulkan>(device, cli),
        DTypeArg::F16 => run::<burn::backend::Vulkan<half::f16>>(device, cli),
        DTypeArg::Bf16 => run::<burn::backend::Vulkan<half::bf16>>(device, cli),
    }
}

#[cfg(not(feature = "vulkan"))]
fn run_vulkan(_cli: &Cli) -> Result<(), String> {
    Err("backend vulkan was requested, but feature `vulkan` is not enabled".into())
}

#[cfg(feature = "wgpu")]
fn run_wgpu(cli: &Cli) -> Result<(), String> {
    let device = parse_wgpu_device::<burn::backend::Wgpu>(cli, "wgpu")?;
    match cli.dtype {
        DTypeArg::F32 => run::<burn::backend::Wgpu>(device, cli),
        DTypeArg::F16 => run::<burn::backend::Wgpu<half::f16>>(device, cli),
        DTypeArg::Bf16 => run::<burn::backend::Wgpu<half::bf16>>(device, cli),
    }
}

#[cfg(not(feature = "wgpu"))]
fn run_wgpu(_cli: &Cli) -> Result<(), String> {
    Err("backend wgpu was requested, but feature `wgpu` is not enabled".into())
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run_selected_backend(&cli) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(version: PaddleOcrVersion, task: OcrTask) -> Cli {
        Cli {
            version,
            backend: BackendArg::Mlx,
            device: None,
            dtype: DTypeArg::Bf16,
            model_dir: None,
            task,
            max_new_tokens: None,
            input: vec![PathBuf::from("image.png")],
        }
    }

    #[test]
    fn spotting_requires_v1_5_recipe() {
        assert!(validate_task_version(&cli(PaddleOcrVersion::V1_5, OcrTask::Spotting)).is_ok());
        assert!(validate_task_version(&cli(PaddleOcrVersion::V1_0, OcrTask::Spotting)).is_err());
    }

    #[test]
    fn non_spotting_allows_v1_0() {
        assert!(validate_task_version(&cli(PaddleOcrVersion::V1_0, OcrTask::Ocr)).is_ok());
    }
}
