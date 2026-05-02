# pp-doclayout

PP-DocLayout detection in Rust. This crate loads PP-DocLayoutV2 or PP-DocLayoutV3 checkpoints and returns structured layout blocks with labels, scores, bounding boxes, and optional polygons.

## API Usage

```rust
use burn::backend::Flex;
use pp_doclayout::{DocLayout, DocLayoutVersion};

fn main() -> Result<(), String> {
    let device = Default::default();
    let detector = DocLayout::<Flex>::try_load_from_safetensors(
        DocLayoutVersion::V3,
        "./model1.5/PP-DocLayoutV3/model.safetensors",
        device,
        0.3,
    )?;

    let image = image::open("image.png").map_err(|e| e.to_string())?;
    let result = detector.detect(&image);

    for block in result.blocks {
        println!(
            "{} {:.3} bbox={:?} polygon={:?}",
            block.label, block.score, block.bbox, block.polygon
        );
    }
    Ok(())
}
```

Public types:

- `DocLayoutVersion`: `V2` or `V3`.
- `DocLayout::try_load_from_safetensors(...)`: fallible model loader for public API use.
- `DocLayout::load_from_safetensors(...)`: convenience loader that panics on loading failure.
- `DocLayout::detect(&DynamicImage) -> LayoutResult`.
- `LayoutResult`: image `width`, `height`, and detected `blocks`.
- `LayoutBlock`: `id`, `label_id`, `label`, `score`, `bbox`, and optional `polygon`.

## CLI Example

Run the example with:

```bash
cargo run -p pp-doclayout --example cli -- image.png
```

Options:

- `--version <v2|v3>`: layout model version. Default: `v2`.
- `--backend <cpu|cuda|flex|metal|mlx|ndarray|vulkan|wgpu>`: Burn backend.
  Default: `flex`.
- `--device <SPEC>`: backend device selector. Defaults to the backend's default device. Examples: `0` or `cuda:0` for CUDA, `gpu`/`cpu` for MLX, and `default`, `cpu`, `discrete:0`, `integrated:0`, `virtual:0`, or bare `0` for WGPU-style backends (`metal`, `vulkan`, `wgpu`).
- `--dtype <f32|f16|bf16>`: model weight dtype. Default: `f32`.
- `--model <FILE>`: safetensors checkpoint path. If omitted, the version default path is used.
- `--threshold <FLOAT>`: detection score threshold. Defaults are `0.5` for V2 and `0.3` for V3.
- `<input>...`: image files or directories. Directories are scanned for common image extensions.

Backend notes:

- `ndarray` supports only `f32` in this example.
- `flex` uses one backend type; precision is selected by `--dtype`.
