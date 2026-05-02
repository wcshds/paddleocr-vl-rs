use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::{prelude::Backend, tensor::DType};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use paddleocr_vl_pipeline::{PaddleOcrVlPipeline, PipelineProgress, PipelineVersion};

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

// Keep this support matrix explicit: NdArray is f32-only in Burn 0.21.0-pre.4,
// Flex uses one backend type with dtype-selected storage/arithmetic, and Vulkan
// builds require the Vulkan SDK on macOS.
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
#[command(about = "Run local PaddleOCR-VL page-level parsing")]
struct Cli {
    #[arg(long, value_parser = parse_version, default_value = "v1.5")]
    version: PipelineVersion,

    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    backend: BackendArg,

    #[arg(long)]
    device: Option<String>,

    #[arg(long, value_enum, default_value_t = DTypeArg::F32)]
    dtype: DTypeArg,

    #[arg(long)]
    layout_model: Option<PathBuf>,

    #[arg(long)]
    vl_model_dir: Option<PathBuf>,

    #[arg(long)]
    layout_threshold: Option<f32>,

    #[arg(long)]
    max_new_tokens: Option<usize>,

    #[arg(long, default_value_t = false)]
    use_chart_recognition: bool,

    #[arg(long)]
    use_seal_recognition: Option<bool>,

    #[arg(long, default_value_t = false)]
    use_ocr_for_image_block: bool,

    #[arg(long, value_delimiter = ',')]
    markdown_ignore_labels: Option<Vec<String>>,

    #[arg(long, default_value_t = false)]
    print: bool,

    #[arg(long)]
    save_json: Option<PathBuf>,

    #[arg(long)]
    save_markdown: Option<PathBuf>,

    #[arg(required = true)]
    input: Vec<PathBuf>,
}

fn parse_version(value: &str) -> Result<PipelineVersion, String> {
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
            return Err(format!(
                "{} is not a valid file or directory",
                input.display()
            ));
        }
    }
    Ok(paths)
}

fn progress_style() -> Result<ProgressStyle, String> {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
    )
    .map(|style| style.progress_chars("=>-"))
    .map_err(|e| e.to_string())
}

fn device_spec(cli: &Cli) -> Option<&str> {
    cli.device
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
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

fn run<B: Backend>(device: B::Device, cli: &Cli) -> Result<(), String>
where
    B::Device: Clone,
{
    let image_paths = collect_images(&cli.input)?;
    if image_paths.is_empty() {
        return Err("no image files found in the given paths".into());
    }

    eprintln!(
        "[init] Loading PaddleOCR-VL pipeline {:?} (backend: {:?}, dtype: {:?})...",
        cli.version, cli.backend, cli.dtype
    );
    let t = Instant::now();
    let mut builder = PaddleOcrVlPipeline::<B>::builder(device)
        .version(cli.version)
        .dtype(cli.dtype.into())
        .use_chart_recognition(cli.use_chart_recognition)
        .use_ocr_for_image_block(cli.use_ocr_for_image_block);

    if let Some(path) = cli.layout_model.clone() {
        builder = builder.layout_model_path(path);
    }
    if let Some(dir) = cli.vl_model_dir.clone() {
        builder = builder.vl_model_dir(dir);
    }
    if let Some(threshold) = cli.layout_threshold {
        builder = builder.layout_threshold(threshold);
    }
    if let Some(max_new_tokens) = cli.max_new_tokens {
        builder = builder.max_new_tokens(max_new_tokens);
    }
    if let Some(enabled) = cli.use_seal_recognition {
        builder = builder.use_seal_recognition(enabled);
    }
    if let Some(labels) = cli.markdown_ignore_labels.clone() {
        builder = builder.markdown_ignore_labels(labels);
    }

    let pipeline = builder.build()?;
    eprintln!("[init] Loaded in {:.1}s\n", t.elapsed().as_secs_f64());

    for (idx, path) in image_paths.iter().enumerate() {
        eprintln!(
            "━━━ [{}/{}] {} ━━━",
            idx + 1,
            image_paths.len(),
            path.display()
        );
        let t = Instant::now();
        let progress = ProgressBar::new(0);
        progress.set_style(progress_style()?);
        let result = pipeline.predict_image_path_with_progress(path.clone(), |event| match event {
            PipelineProgress::LayoutDetected { total_blocks } => {
                progress.set_length(total_blocks as u64);
                if total_blocks == 0 {
                    progress.set_message("no layout blocks");
                }
            }
            PipelineProgress::BlockStarted {
                current,
                total,
                label,
                will_recognize,
            } => {
                let action = if will_recognize {
                    "recognizing"
                } else {
                    "skipping recognition"
                };
                progress.set_message(format!("{action} {current}/{total} {label}"));
            }
            PipelineProgress::BlockFinished {
                current,
                total,
                label,
                recognized,
            } => {
                progress.set_position(current as u64);
                let status = if recognized { "done" } else { "skipped" };
                progress.set_message(format!("{status} {current}/{total} {label}"));
            }
        });
        if result.is_err() {
            progress.abandon_with_message("failed");
        }
        let result = result?;
        progress.finish_with_message(format!("done {} block(s)", result.blocks.len()));
        if cli.print {
            result.print()?;
        }
        if let Some(save_json) = &cli.save_json {
            result.save_to_json(save_json)?;
        }
        if let Some(save_markdown) = &cli.save_markdown {
            result.save_to_markdown(save_markdown)?;
        }
        eprintln!(
            "Processed {} in {:.2}s\n",
            path.display(),
            t.elapsed().as_secs_f64()
        );
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
