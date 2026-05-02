# paddleocr-vl-pipeline

Local page-level PaddleOCR-VL parsing in Rust. This crate combines `pp-doclayout` layout detection with `paddleocr-vl` element recognition and returns JSON/Markdown-friendly page results. Direct text spotting is not part of this pipeline; use `paddleocr-vl` directly for the `spotting` task.

## API Usage

```rust
use burn::backend::Flex;
use paddleocr_vl_pipeline::{PaddleOcrVlPipeline, PipelineVersion};

fn main() -> Result<(), String> {
    let device = Default::default();
    let pipeline = PaddleOcrVlPipeline::<Flex>::builder(device)
        .version(PipelineVersion::V1_5)
        .layout_model_path("./model1.5/PP-DocLayoutV3/model.safetensors")
        .vl_model_dir("./model1.5/PaddleOCR-VL")
        .build()?;

    let output = pipeline.predict("image.png")?;
    for page in output {
        page.print()?;
        page.save_to_json("output")?;
        page.save_to_markdown("output")?;
    }
    Ok(())
}
```

Useful builder options:

- `version(...)`: `PipelineVersion::V1` uses PaddleOCR-VL-1.0 +
  PP-DocLayoutV2; `PipelineVersion::V1_5` uses PaddleOCR-VL-1.5 +
  PP-DocLayoutV3.
- `layout_model_path(...)`: layout checkpoint path.
- `vl_model_dir(...)`: PaddleOCR-VL model directory.
- `dtype(...)`: load layout and recognizer weights as `DType::F32`,
  `DType::F16`, or `DType::BF16` when supported by the selected backend.
- `layout_threshold(...)`: detection threshold.
- `max_new_tokens(...)`: override recognition generation length.
- `use_chart_recognition(...)`, `use_seal_recognition(...)`,
  `use_ocr_for_image_block(...)`: enable optional block recognition behavior.
- `markdown_ignore_labels(...)`: labels skipped only when rendering Markdown.

Result helpers:

- `PipelineResult::print()`
- `PipelineResult::save_to_json(path)`
- `PipelineResult::to_markdown()`
- `PipelineResult::save_to_markdown(path)`

## CLI Example

Run the example with:

```bash
cargo run -p paddleocr-vl-pipeline --example cli -- image.png
```

Options:

- `--version <v1|v1.5>`: pipeline preset. Default: `v1.5`.
- `--backend <cpu|cuda|flex|metal|mlx|ndarray|vulkan|wgpu>`: Burn backend.
  Default: `flex`.
- `--device <SPEC>`: backend device selector. Defaults to the backend's default
  device. Examples: `0` or `cuda:0` for CUDA, `gpu`/`cpu` for MLX, and
  `default`, `cpu`, `discrete:0`, `integrated:0`, `virtual:0`, or bare `0` for
  WGPU-style backends (`metal`, `vulkan`, `wgpu`).
- `--dtype <f32|f16|bf16>`: model weight dtype. Default: `f32`.
- `--layout-model <FILE>`: layout safetensors checkpoint path.
- `--vl-model-dir <DIR>`: PaddleOCR-VL model directory.
- `--layout-threshold <FLOAT>`: detection score threshold.
- `--max-new-tokens <N>`: recognition generation limit.
- `--use-chart-recognition`: run chart recognition for chart blocks.
- `--use-seal-recognition <true|false>`: override seal recognition behavior.
- `--use-ocr-for-image-block`: run OCR on image blocks.
- `--markdown-ignore-labels <A,B,C>`: comma-separated labels to skip.
- `--print`: print JSON results to stdout.
- `--save-json <PATH_OR_DIR>`: save JSON output. A directory writes one file per
  input stem.
- `--save-markdown <PATH_OR_DIR>`: save Markdown output. A directory writes one
  file per input stem.
- `<input>...`: image files or directories.

Backend notes:

- `ndarray` supports only `f32` in this example.
- `flex` uses one backend type; precision is selected by `--dtype`.
- `vulkan` requires a local Vulkan SDK on macOS.
