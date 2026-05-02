# PaddleOCR-VL-rs

Rust port of PaddleOCR-VL element recognition, PP-DocLayout layout detection, and a local page-level parsing pipeline built on Burn.

## Disclaimer

Most of this repository’s source code was produced with substantial assistance from AI coding tools. Treat it accordingly for review, security, and production use.

## Crates

- `crates/paddleocr-vl`: direct PaddleOCR-VL element recognition for OCR, table, chart, formula, seal, and PaddleOCR-VL-1.5 spotting tasks.
- `crates/pp-doclayout`: PP-DocLayoutV2/V3 layout detection.
- `crates/paddleocr-vl-pipeline`: page-level parsing that combines layout detection and element recognition.
- `crates/burn-mlx`: local MLX backend used when the `mlx` feature is enabled.

Default features enable `flex` and `cpu`, which are the most portable backends. GPU backends such as `metal`, `wgpu`, `cuda`, `vulkan`, and `mlx` can be enabled explicitly.

## Download Models

Install the downloader dependency:

```bash
python -m pip install huggingface_hub
```

Download all default models into the workspace root:

```bash
python scripts/download_models.py
```

The script creates the layout expected by the Rust defaults:

```text
model1.0/
  model.safetensors
  tokenizer.json
  preprocessor_config.json
  PP-DocLayoutV2/model.safetensors

model1.5/
  model.safetensors
  tokenizer.json
  preprocessor_config.json
  PP-DocLayoutV3/model.safetensors
```

Download only one version:

```bash
python scripts/download_models.py --version 1.0
python scripts/download_models.py --version 1.5
```

Use `--output <DIR>` to place `model1.0` and `model1.5` somewhere else. If you do that, pass `--model-dir`, `--model`, `--vl-model-dir`, or `--layout-model` to the CLI examples.

## CLI Examples

Page-level parsing:

```bash
# Test on MacBook Pro M4 Pro.
cargo run --release --package paddleocr-vl-pipeline --example cli --features mlx -- --version 1.5 --backend mlx --dtype bf16 --save-json ./output --save-markdown ./output ./samples
```

Layout detection:

```bash
cargo run --release --package pp-doclayout --example cli --features wgpu,metal,mlx -- --version v2 --backend flex --dtype f32 ./samples/0506.png
```

Direct PaddleOCR-VL element recognition & text spotting:

```bash
cargo run --release --package paddleocr-vl  --example cli --features wgpu,metal,mlx -- --version 1.5 --backend flex --dtype f32 --task seal ./samples/seal.png
cargo run --release --package paddleocr-vl  --example cli --features wgpu,metal,mlx -- --version 1.5 --backend mlx --dtype f32 --task spotting ./samples/0506.png
```

Common CLI options:

- `--backend <cpu|cuda|flex|metal|mlx|ndarray|vulkan|wgpu>`: backend to use.
  Default: `flex`.
- `--device <SPEC>`: backend device. Examples: `cuda:1` for Cuda, `gpu` or `cpu` for MLX, and `discrete:0`, `integrated:0`, `cpu`, or `default` for WGPU-style backends.
- `--dtype <f32|f16|bf16>`: model weight dtype. Default: `f32`. Low precision is a speed/memory option and is not guaranteed to match f32 token-for-token; precision-sensitive preprocessing, RoPE, and greedy logits are kept in f32 internally where supported.

Enable a backend feature when running a non-default backend:

```bash
cargo run -p paddleocr-vl --example cli --no-default-features --features wgpu -- \
  --backend wgpu image.png

cargo run -p paddleocr-vl --example cli --no-default-features --features cuda -- \
  --backend cuda --device cuda:1 image.png
```

For realistic performance measurements, use release builds:

```bash
cargo run --release -p paddleocr-vl --example cli -- image.png
```

Debug builds are much slower, especially during autoregressive token generation.

## API Usage

Direct recognition:

```rust
use burn::backend::Flex;
use paddleocr_vl::{PaddleOcrVersion, PaddleOcrVl};

fn main() -> Result<(), String> {
    let device = Default::default();
    let recognizer = PaddleOcrVl::<Flex>::builder(device)
        .version(PaddleOcrVersion::V1_5)
        .model_dir("./model1.5")
        .build()?;

    let result = recognizer.ocr("image.png")?;
    println!("{}", result.as_text());
    Ok(())
}
```

Page-level parsing:

```rust
use burn::backend::Flex;
use paddleocr_vl_pipeline::{PaddleOcrVlPipeline, PipelineVersion};

fn main() -> Result<(), String> {
    let device = Default::default();
    let pipeline = PaddleOcrVlPipeline::<Flex>::builder(device)
        .version(PipelineVersion::V1_5)
        .build()?;

    for result in pipeline.predict("image.png")? {
        result.print()?;
        result.save_to_json("output")?;
        result.save_to_markdown("output")?;
    }
    Ok(())
}
```

Each crate has a focused README with more API and CLI details.
