// ========================================================================
// PPDocLayout Image Preprocessing
// ========================================================================
//
// Preprocesses input images to match the model's expected input format.
//
// Processing steps:
//   1. Resize the image to 800×800 (torchvision bicubic interpolation)
//   2. Convert to float32 and rescale to [0, 1] (rescale_factor = 1/255)
//   3. Convert to channel-first format: [3, 800, 800]
//   4. Add a batch dimension: [1, 3, 800, 800]
//
// Note: PP-DocLayoutV2 preprocessing does **not** apply mean/std normalization.
// mean=[0, 0, 0], std=[1, 1, 1] means no additional normalization is done.
// This differs from PaddleOCR-VL preprocessing (which uses mean=std=0.5).
//
// Aligned with HuggingFace PPDocLayoutV2/PPDocLayoutV3ImageProcessor:
//   - do_resize=True, size=(800, 800), resample=BICUBIC, antialias=False
//   - do_rescale=True, rescale_factor=1/255
//   - do_normalize=True with mean=[0, 0, 0], std=[1, 1, 1] (a no-op)
// ========================================================================

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use image::{DynamicImage, RgbImage};

/// PP-DocLayoutV2 preprocessing configuration.
pub struct DocLayoutPreprocessConfig {
    /// Target image size (height, width).
    pub target_size: (usize, usize),
    /// Pixel rescale factor.
    pub rescale_factor: f32,
}

impl Default for DocLayoutPreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (800, 800),
            rescale_factor: 1.0 / 255.0,
        }
    }
}

/// Preprocess an image for model inference.
///
/// - `img`: input RGB image
/// - `config`: preprocessing configuration
/// - `device`: compute device
///
/// Returns:
/// - `pixel_values`: `[1, 3, target_h, target_w]` float tensor
/// - `original_size`: original image (height, width)
pub fn preprocess_image<B: Backend>(
    img: &DynamicImage,
    config: &DocLayoutPreprocessConfig,
    device: &B::Device,
    target_dtype: DType,
) -> (Tensor<B, 4>, (usize, usize)) {
    let (target_h, target_w) = config.target_size;

    // Convert to RGB8
    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = rgb.dimensions();
    let original_size = (orig_h as usize, orig_w as usize);

    // Match HuggingFace's TorchvisionBackend resize path.
    //
    // PP-DocLayoutV3 is handled by torchvision v2, which keeps PIL inputs as
    // uint8 tensors and calls resize with BICUBIC + antialias=false. On CPU,
    // PyTorch uses a specialized uint8 path with quantized int16 weights and a
    // rounded uint8 intermediate after the horizontal pass. Replicating that
    // path avoids small but visible differences at high-contrast edges.
    let resized = resize_bicubic_torchvision(&rgb, target_w, target_h);

    // Build tensor [H, W, 3] in the dtype selected for the model weights.
    // Dynamic-dtype backends such as Flex have `B::FloatElem = f32`, so using
    // the runtime `target_dtype` is required to keep inputs and weights aligned.
    let pixels: &[u8] = resized.as_raw();
    let pixel_data = pixels.iter().map(|&b| b as f32).collect::<Vec<f32>>();
    let img_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(pixel_data, [target_h * target_w * 3]).convert_dtype(target_dtype),
        (device, target_dtype),
    )
    .reshape([target_h, target_w, 3]);

    // Rescale: [0, 255] → [0, 1]
    let img_tensor = img_tensor
        .mul_scalar(config.rescale_factor)
        .cast(target_dtype);

    // Channel-first: [H, W, 3] → [3, H, W]
    let img_tensor = img_tensor.permute([2, 0, 1]);

    // Add batch dimension: [1, 3, H, W]
    let pixel_values = img_tensor.unsqueeze_dim::<4>(0);

    (pixel_values, original_size)
}

fn resize_bicubic_torchvision(src: &RgbImage, target_w: usize, target_h: usize) -> RgbImage {
    let (src_w, src_h) = src.dimensions();
    let src_w = src_w as usize;
    let src_h = src_h as usize;
    let src_data = src.as_raw();

    let x_weights = compute_uint8_bicubic_weights(src_w, target_w);
    let y_weights = compute_uint8_bicubic_weights(src_h, target_h);

    // PyTorch's uint8 resize is separable and stores the horizontal pass back
    // into uint8 before the vertical pass. Keeping that intermediate rounding
    // is required for byte-exact parity with torchvision v2 on CPU.
    let mut tmp = vec![0u8; src_h * target_w * 3];
    let x_rounding = 1i32 << (x_weights.precision - 1);
    for y in 0..src_h {
        for out_x in 0..target_w {
            let x_min = x_weights.index_min[out_x];
            let x_size = x_weights.index_size[out_x];
            for channel in 0..3 {
                let mut acc = x_rounding;
                for k in 0..x_size {
                    let src_idx = (y * src_w + x_min + k) * 3 + channel;
                    acc += src_data[src_idx] as i32 * x_weights.weights[out_x][k] as i32;
                }
                let tmp_idx = (y * target_w + out_x) * 3 + channel;
                tmp[tmp_idx] = clamp_u8(acc >> x_weights.precision);
            }
        }
    }

    let mut dst = vec![0u8; target_w * target_h * 3];
    let y_rounding = 1i32 << (y_weights.precision - 1);
    for out_y in 0..target_h {
        let y_min = y_weights.index_min[out_y];
        let y_size = y_weights.index_size[out_y];
        for out_x in 0..target_w {
            for channel in 0..3 {
                let mut acc = y_rounding;
                for k in 0..y_size {
                    let tmp_idx = ((y_min + k) * target_w + out_x) * 3 + channel;
                    acc += tmp[tmp_idx] as i32 * y_weights.weights[out_y][k] as i32;
                }
                let dst_idx = (out_y * target_w + out_x) * 3 + channel;
                dst[dst_idx] = clamp_u8(acc >> y_weights.precision);
            }
        }
    }

    RgbImage::from_raw(target_w as u32, target_h as u32, dst)
        .expect("resized RGB buffer should match target dimensions")
}

struct Uint8BicubicWeights {
    index_min: Vec<usize>,
    index_size: Vec<usize>,
    weights: Vec<[i16; 4]>,
    precision: u32,
}

fn compute_uint8_bicubic_weights(input_size: usize, output_size: usize) -> Uint8BicubicWeights {
    const INTERP_SIZE: usize = 4;

    let scale = input_size as f64 / output_size as f64;
    let mut index_min = Vec::with_capacity(output_size);
    let mut index_size = Vec::with_capacity(output_size);
    let mut float_weights = Vec::with_capacity(output_size);
    let mut weight_max = 0.0f64;

    for out_index in 0..output_size {
        let real_input_index = scale * (out_index as f64 + 0.5) - 0.5;
        let mut input_index = real_input_index.floor() as isize;
        input_index = input_index.min(input_size.saturating_sub(1) as isize);
        let lambda = (real_input_index - input_index as f64).clamp(0.0, 1.0);

        let support = (INTERP_SIZE / 2) as isize;
        let unbound_min = input_index - support + 1;
        let unbound_max = input_index + support + 1;
        let min_index = unbound_min.max(0) as usize;
        let size = (unbound_max.min(input_size as isize) - min_index as isize)
            .clamp(0, INTERP_SIZE as isize) as usize;

        let mut weights = [0.0f64; INTERP_SIZE];
        let mut weight_index = 0usize;
        for k in 0..INTERP_SIZE {
            let sample_index = unbound_min + k as isize;
            let weight = cubic_weight((k as isize + 1 - support) as f64 - lambda);
            if sample_index <= 0 {
                weight_index = 0;
            } else if sample_index >= input_size.saturating_sub(1) as isize {
                weight_index = size.saturating_sub(1);
            }
            weights[weight_index] += weight;
            weight_max = weight_max.max(weights[weight_index]);
            weight_index += 1;
        }

        index_min.push(min_index);
        index_size.push(size);
        float_weights.push(weights);
    }

    let mut precision = 0u32;
    while precision < 22 {
        let next_value = (0.5 + weight_max * (1u32 << (precision + 1)) as f64) as i32;
        if next_value >= (1 << 15) {
            break;
        }
        precision += 1;
    }

    let scale = (1u32 << precision) as f64;
    let weights = float_weights
        .into_iter()
        .map(|weights| {
            let mut quantized = [0i16; INTERP_SIZE];
            for (idx, weight) in weights.into_iter().enumerate() {
                let scaled = weight * scale;
                let rounded = if scaled < 0.0 {
                    (-0.5 + scaled) as i32
                } else {
                    (0.5 + scaled) as i32
                };
                quantized[idx] = rounded as i16;
            }
            quantized
        })
        .collect();

    Uint8BicubicWeights {
        index_min,
        index_size,
        weights,
        precision,
    }
}

fn cubic_weight(x: f64) -> f64 {
    // PyTorch uses a = -0.75 for bicubic when antialias=false.
    let a = -0.75f64;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}
