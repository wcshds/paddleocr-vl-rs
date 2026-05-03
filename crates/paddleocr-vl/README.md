# paddleocr-vl

Direct PaddleOCR-VL element recognition in Rust. This crate is for running the
vision-language recognizer on one image or crop at a time, including OCR, table,
chart, formula, seal, and PaddleOCR-VL-1.5 spotting tasks. Page-level document
parsing lives in `paddleocr-vl-pipeline`.

## API Usage

```rust
use burn::backend::Flex;
use paddleocr_vl::{PaddleOcrVersion, PaddleOcrVl};

fn main() -> Result<(), String> {
    let device = Default::default();
    let recognizer = PaddleOcrVl::<Flex>::builder(device)
        .version(PaddleOcrVersion::V1_5)
        .model_dir("./model1.5/PaddleOCR-VL")
        .build()?;

    let result = recognizer.ocr("image.png")?;
    println!("{}", result.as_text());

    let spotting = recognizer.spotting("image.png")?;
    spotting.print();
    Ok(())
}
```

Useful builder options:

- `version(...)`: `PaddleOcrVersion::V1_0` or `PaddleOcrVersion::V1_5`.
- `model_dir(...)`: directory containing `model.safetensors`, `tokenizer.json`,
  and processor/template config files.
- `dtype(...)`: load weights as `DType::F32`, `DType::F16`, or `DType::BF16`
  when supported by the selected backend.
- `max_new_tokens(...)`: override the generation limit.

Recognition methods:

- `recognize_path(path, task)` / `recognize_image(image, task)` for explicit
  `OcrTask` dispatch.
- Convenience methods: `ocr`, `table`, `chart`, `formula`, `seal`, `spotting`.

## CLI Example

Run the example with:

```bash
cargo run -p paddleocr-vl --example cli -- image.png
```

Common options:

- `--version <1.0|1.5>`: PaddleOCR-VL checkpoint version. Default: `1.0`.
- `--backend <cpu|cuda|flex|tch|metal|mlx|vulkan|wgpu>`: Burn backend.
  Default: `flex`.
- `--device <SPEC>`: backend device selector. Defaults to the backend's default
  device. Examples: `0` or `cuda:0` for CUDA/tch, `gpu`/`cpu` for MLX,
  `mps`/`metal` for tch on Apple Silicon, and
  `default`, `cpu`, `discrete:0`, `integrated:0`, `virtual:0`, or bare `0` for
  WGPU-style backends (`metal`, `vulkan`, `wgpu`).
- `--dtype <f32|f16|bf16>`: model weight dtype. Default: `f32`.
  Low precision is a speed/memory option and is not guaranteed to match f32
  token-for-token. Precision-sensitive preprocessing, RoPE, and greedy logits
  are kept in f32 internally where supported. Spotting follows the
  PaddleOCR-VL-1.5 recipe.
- `--model-dir <DIR>`: model directory. If omitted, the version default path is
  used.
- `--task <ocr|table|chart|formula|seal|spotting>`: recognition task. Default:
  `ocr`.
- `--max-new-tokens <N>`: override generation length.
- `<input>...`: image files or directories. Directories are scanned for common
  image extensions.

Backend notes:

- `flex` uses one backend type; precision is selected by `--dtype`.
