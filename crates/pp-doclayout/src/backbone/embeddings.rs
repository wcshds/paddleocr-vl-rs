// ========================================================================
// HGNetV2 Embeddings (Stem)
// ========================================================================
//
// The HGNetV2 Stem module is responsible for initial feature extraction and
// downsampling of raw images. It is the entry point of the network, converting
// images of shape [B, 3, H, W] into feature maps of shape [B, 48, H/4, W/4].
//
// Computation flow:
//
// ```text
// pixel_values [B, 3, H, W]
//       │
//       ▼
//  stem1: Conv(3→32, k=3, s=2) + FBN + ReLU  → [B, 32, H/2, W/2]
//       │
//       ├─ pad(0,1,0,1) ────┐
//       │                   │
//       ▼                   ▼
//  stem2a: Conv(32→16,   MaxPool(k=2, s=1,
//    k=2, s=1) + FBN     ceil_mode=True)
//    + ReLU               → [B, 32, H/2, W/2]
//       │
//       ▼ pad(0,1,0,1)
//  stem2b: Conv(16→32,
//    k=2, s=1) + FBN
//    + ReLU
//       │                   │
//       └───── cat ─────────┘
//              │
//              ▼
//         [B, 64, H/2, W/2]
//              │
//              ▼
//  stem3: Conv(64→32, k=3, s=2) + FBN + ReLU  → [B, 32, H/4, W/4]
//              │
//              ▼
//  stem4: Conv(32→48, k=1, s=1) + FBN + ReLU  → [B, 48, H/4, W/4]
// ```
//
// Notes:
// - stem2a and stem2b use k=2 convolutions with padding=(k-1)/2=0,
//   so manual pad(0,1,0,1) (right+1, bottom+1) is needed to preserve spatial size.
// - MaxPool uses ceil_mode=True, which requires manual handling in burn.
// - All BatchNorm layers are FrozenBatchNorm2d (inference mode).
// ========================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::Backend,
    tensor::{Tensor, activation},
};

use crate::frozen_batch_norm::{FrozenBatchNorm2d, FrozenBatchNorm2dConfig};

// ========================================================================
// ConvBnAct: Conv2d + FrozenBatchNorm + optional activation
// ========================================================================
//
// The most basic convolutional unit in HGNetV2.
// Composition: Conv2d(bias=False) → FrozenBatchNorm2d → Activation
//
// The activation function is determined by model configuration (default: ReLU).
// In the PP-DocLayoutV2 backbone, use_learnable_affine_block=false, so
// the LAB (Learnable Affine Block) is not included.
// ========================================================================

#[derive(Module, Debug)]
pub struct ConvBnAct<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: FrozenBatchNorm2d<B>,
    pub use_activation: bool,
}

impl<B: Backend> ConvBnAct<B> {
    /// Forward pass: Conv → BN → (ReLU)
    ///
    /// - Input:  `[B, in_ch, H, W]`
    /// - Output: `[B, out_ch, H', W']` (size depends on stride and padding)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        if self.use_activation {
            activation::relu(x)
        } else {
            x
        }
    }
}

/// ConvBnAct configuration.
///
/// - `in_channels`: number of input channels
/// - `out_channels`: number of output channels
/// - `kernel_size`: convolution kernel size
/// - `stride`: stride
/// - `groups`: number of groups for grouped convolution (groups=in_ch for depthwise)
/// - `use_activation`: whether to apply ReLU activation
/// - `padding`: convolution padding (None defaults to (kernel_size-1)/2)
/// - `bn_eps`: FrozenBatchNorm2d epsilon
#[derive(Config, Debug)]
pub struct ConvBnActConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    #[config(default = 1)]
    pub stride: usize,
    #[config(default = 1)]
    pub groups: usize,
    #[config(default = true)]
    pub use_activation: bool,
    pub padding: Option<usize>,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl ConvBnActConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBnAct<B> {
        let pad = self.padding.unwrap_or((self.kernel_size - 1) / 2);
        let conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_groups(self.groups)
        .with_bias(false)
        .init(device);
        let bn = FrozenBatchNorm2dConfig::new(self.out_channels)
            .with_epsilon(self.bn_eps)
            .init(device);

        ConvBnAct {
            conv,
            bn,
            use_activation: self.use_activation,
        }
    }
}

// ========================================================================
// ConvLayerLight: Lightweight convolution (depthwise separable convolution)
// ========================================================================
//
// Used in HGNetV2 Stage3 and Stage4 (stage_light_block=true).
// Consists of two parts:
//   1. 1×1 Pointwise Conv: changes channel count (no activation)
//   2. k×k Depthwise Conv: spatial convolution (groups=out_ch)
//
// Compared to standard convolution, depthwise separable convolution
// significantly reduces parameters and computation:
//   Standard:   O(in × out × k²)
//   Separable:  O(in × out × 1²) + O(out × 1 × k²) = O(in × out + out × k²)
// ========================================================================

#[derive(Module, Debug)]
pub struct ConvLayerLight<B: Backend> {
    /// 1×1 pointwise convolution (no activation)
    pub conv1: ConvBnAct<B>,
    /// k×k depthwise convolution (groups=out_channels, no activation)
    pub conv2: ConvBnAct<B>,
}

impl<B: Backend> ConvLayerLight<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        self.conv2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ConvLayerLightConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 3)]
    pub kernel_size: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl ConvLayerLightConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvLayerLight<B> {
        ConvLayerLight {
            // conv1: 1×1 pointwise, no activation
            conv1: ConvBnActConfig::new(self.in_channels, self.out_channels, 1)
                .with_use_activation(false)
                .with_bn_eps(self.bn_eps)
                .init(device),
            // conv2: k×k depthwise, with ReLU (Python default activation="relu")
            conv2: ConvBnActConfig::new(self.out_channels, self.out_channels, self.kernel_size)
                .with_groups(self.out_channels)
                .with_use_activation(true)
                .with_bn_eps(self.bn_eps)
                .init(device),
        }
    }
}

// ========================================================================
// HGNetV2Embeddings (Stem)
// ========================================================================

#[derive(Module, Debug)]
pub struct HGNetV2Embeddings<B: Backend> {
    pub stem1: ConvBnAct<B>,
    pub stem2a: ConvBnAct<B>,
    pub stem2b: ConvBnAct<B>,
    pub stem3: ConvBnAct<B>,
    pub stem4: ConvBnAct<B>,
    pub pool: MaxPool2d,
}

impl<B: Backend> HGNetV2Embeddings<B> {
    /// Stem forward pass.
    ///
    /// - Input:  `[B, 3, H, W]` (raw image)
    /// - Output: `[B, 48, H/4, W/4]`
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 4> {
        // stem1: [B, 3, H, W] → [B, 32, H/2, W/2]
        let embedding = self.stem1.forward(pixel_values);

        // Manual pad(0,1,0,1): +1 on the right and bottom.
        // Needed so that k=2, s=1 convolutions and MaxPool preserve spatial size.
        let padded = pad_right_bottom(embedding.clone());

        // stem2a: [B, 32, H/2+1, W/2+1] → [B, 16, H/2, W/2]
        let emb_2a = self.stem2a.forward(padded.clone());
        // pad + stem2b: [B, 16, H/2, W/2] → pad → [B, 32, H/2, W/2]
        let emb_2a = self.stem2b.forward(pad_right_bottom(emb_2a));

        // MaxPool on padded stem1 output: [B, 32, H/2+1, W/2+1] → [B, 32, H/2, W/2]
        let pooled = self.pool.forward(padded);

        // Channel-wise concat: [B, 64, H/2, W/2]
        let embedding = Tensor::cat(vec![pooled, emb_2a], 1);

        // stem3: [B, 64, H/2, W/2] → [B, 32, H/4, W/4]
        let embedding = self.stem3.forward(embedding);
        // stem4: [B, 32, H/4, W/4] → [B, 48, H/4, W/4]
        self.stem4.forward(embedding)
    }
}

/// HGNetV2 Stem configuration.
///
/// Default configuration (arch="L"):
///   stem_channels = [3, 32, 48]
///   stem_strides  = [2, 1, 1, 2, 1]
#[derive(Config, Debug)]
pub struct HGNetV2EmbeddingsConfig {
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl HGNetV2EmbeddingsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HGNetV2Embeddings<B> {
        // stem1: Conv(3→32, k=3, s=2) + BN + ReLU
        let stem1 = ConvBnActConfig::new(3, 32, 3)
            .with_stride(2)
            .with_bn_eps(self.bn_eps)
            .init(device);
        // stem2a: Conv(32→16, k=2, s=1) + BN + ReLU (padding=0, manual pad)
        let stem2a = ConvBnActConfig::new(32, 16, 2)
            .with_padding(Some(0))
            .with_bn_eps(self.bn_eps)
            .init(device);
        // stem2b: Conv(16→32, k=2, s=1) + BN + ReLU (padding=0, manual pad)
        let stem2b = ConvBnActConfig::new(16, 32, 2)
            .with_padding(Some(0))
            .with_bn_eps(self.bn_eps)
            .init(device);
        // stem3: Conv(64→32, k=3, s=2) + BN + ReLU
        let stem3 = ConvBnActConfig::new(64, 32, 3)
            .with_stride(2)
            .with_bn_eps(self.bn_eps)
            .init(device);
        // stem4: Conv(32→48, k=1, s=1) + BN + ReLU
        let stem4 = ConvBnActConfig::new(32, 48, 1)
            .with_bn_eps(self.bn_eps)
            .init(device);

        // MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        // burn's MaxPool2d does not support ceil_mode, so manual padding is used instead.
        let pool = MaxPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init();

        HGNetV2Embeddings {
            stem1,
            stem2a,
            stem2b,
            stem3,
            stem4,
            pool,
        }
    }
}

/// Pad a 4D tensor with F.pad(x, (0, 1, 0, 1)) — add one row of zeros on
/// the right and one on the bottom.
///
/// PyTorch F.pad parameter order is (left, right, top, bottom).
/// Input: `[B, C, H, W]` → Output: `[B, C, H+1, W+1]`
fn pad_right_bottom<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let device = x.device();
    let dtype = x.dtype();

    // Create a zero tensor [B, C, H+1, W+1] and assign the original data into it
    let padded = Tensor::<B, 4>::zeros([b, c, h + 1, w + 1], &device).cast(dtype);
    padded.slice_assign([0..b, 0..c, 0..h, 0..w], x)
}
