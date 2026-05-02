// ========================================================================
// HGNetV2 Stage and BasicLayer (HG-Block)
// ========================================================================
//
// The core building blocks of HGNetV2.  Each Stage contains:
//   1. An optional Downsample (depthwise Conv, stride=2)
//   2. Several HGBlocks (BasicLayer)
//
// The HGBlock design is inspired by HarDNet.  Its core idea is to aggregate
// (concatenate) the outputs of all intermediate layers, then compress the
// channel count with a 1×1 Conv:
//
// ```text
// input ──┬─ layer_0 ─┬─ layer_1 ─┬─ ... ─┬─ layer_5 ─┐
//         │           │           │       │           │
//         └───────────┴───────────┴───────┴───────────┘
//                              │
//                          cat(dim=1)  → [B, in + 6*mid, H, W]
//                              │
//                          squeeze_conv(1×1) → [B, out/2, H, W]
//                              │
//                          excite_conv(1×1)  → [B, out, H, W]
//                              │
//                     (+ identity if residual)
// ```
//
// Stage configurations for arch="L":
//
// | Stage | in_ch | mid_ch | out_ch | blocks | downsample | light | kernel |
// |-------|-------|--------|--------|--------|------------|-------|--------|
// | 1     | 48    | 48     | 128    | 1      | false      | false | 3      |
// | 2     | 128   | 96     | 512    | 1      | true       | false | 3      |
// | 3     | 512   | 192    | 1024   | 3      | true       | true  | 5      |
// | 4     | 1024  | 384    | 2048   | 1      | true       | true  | 5      |
//
// When light=true, ConvLayerLight (depthwise separable convolution) is used
// to reduce computation at higher feature-map levels.
// ========================================================================

use burn::{config::Config, module::Module, prelude::Backend, tensor::Tensor};

use super::embeddings::{ConvBnAct, ConvBnActConfig, ConvLayerLight, ConvLayerLightConfig};

// ========================================================================
// ConvBlock: unified wrapper for standard or lightweight convolution
// ========================================================================

/// Selects between standard convolution and depthwise separable convolution
/// based on the light_block setting.
#[derive(Module, Debug)]
pub enum ConvBlock<B: Backend> {
    Standard(ConvBnAct<B>),
    Light(ConvLayerLight<B>),
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            ConvBlock::Standard(conv) => conv.forward(x),
            ConvBlock::Light(conv) => conv.forward(x),
        }
    }
}

// ========================================================================
// HGNetV2BasicLayer (HG-Block)
// ========================================================================

/// The basic building block of HGNetV2 (HG-Block).
///
/// Contains a chain of convolutional layers whose intermediate outputs are
/// concatenated, then compressed through an aggregation layer.
/// Supports a residual connection (for all blocks except the first in each Stage).
#[derive(Module, Debug)]
pub struct HGNetV2BasicLayer<B: Backend> {
    pub layers: Vec<ConvBlock<B>>,
    /// Aggregation layer: 1×1 squeeze conv + 1×1 excite conv.
    /// Compresses (in_ch + layer_num * mid_ch) channels down to out_ch.
    pub agg_squeeze: ConvBnAct<B>,
    pub agg_excite: ConvBnAct<B>,
    pub residual: bool,
}

impl<B: Backend> HGNetV2BasicLayer<B> {
    /// HGBlock forward pass.
    ///
    /// - Input:  `[B, in_ch, H, W]`
    /// - Output: `[B, out_ch, H, W]` (spatial dimensions unchanged)
    pub fn forward(&self, hidden_state: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = hidden_state.clone();

        // Collect all intermediate outputs (including the original input)
        let mut outputs = vec![hidden_state.clone()];
        let mut current = hidden_state;
        for layer in &self.layers {
            current = layer.forward(current);
            outputs.push(current.clone());
        }

        // Concatenate all intermediate outputs: [B, in_ch + layer_num * mid_ch, H, W]
        let aggregated = Tensor::cat(outputs, 1);

        // Compress through the aggregation layer: [B, out_ch, H, W]
        let out = self.agg_squeeze.forward(aggregated);
        let out = self.agg_excite.forward(out);

        // Residual connection (only when residual=true, i.e. not the first block)
        if self.residual { out + identity } else { out }
    }
}

/// HGBlock configuration.
///
/// - `in_ch`: input channel count
/// - `mid_ch`: intermediate layer channel count
/// - `out_ch`: output channel count
/// - `layer_num`: number of convolutional layers (default 6)
/// - `kernel_size`: convolution kernel size
/// - `residual`: whether to use a residual connection
/// - `light_block`: whether to use lightweight (depthwise separable) convolutions
/// - `bn_eps`: FrozenBatchNorm2d epsilon
#[derive(Config, Debug)]
pub struct HGNetV2BasicLayerConfig {
    pub in_ch: usize,
    pub mid_ch: usize,
    pub out_ch: usize,
    #[config(default = 6)]
    pub layer_num: usize,
    pub kernel_size: usize,
    pub residual: bool,
    pub light_block: bool,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl HGNetV2BasicLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HGNetV2BasicLayer<B> {
        let mut layers = Vec::with_capacity(self.layer_num);
        for i in 0..self.layer_num {
            let ch_in = if i == 0 { self.in_ch } else { self.mid_ch };
            if self.light_block {
                layers.push(ConvBlock::Light(
                    ConvLayerLightConfig::new(ch_in, self.mid_ch)
                        .with_kernel_size(self.kernel_size)
                        .with_bn_eps(self.bn_eps)
                        .init(device),
                ));
            } else {
                layers.push(ConvBlock::Standard(
                    ConvBnActConfig::new(ch_in, self.mid_ch, self.kernel_size)
                        .with_bn_eps(self.bn_eps)
                        .init(device),
                ));
            }
        }

        // Aggregation layer input channels = original input + all intermediate outputs
        let total_ch = self.in_ch + self.layer_num * self.mid_ch;
        let agg_squeeze = ConvBnActConfig::new(total_ch, self.out_ch / 2, 1)
            .with_bn_eps(self.bn_eps)
            .init(device);
        let agg_excite = ConvBnActConfig::new(self.out_ch / 2, self.out_ch, 1)
            .with_bn_eps(self.bn_eps)
            .init(device);

        HGNetV2BasicLayer {
            layers,
            agg_squeeze,
            agg_excite,
            residual: self.residual,
        }
    }
}

// ========================================================================
// HGNetV2Stage
// ========================================================================

/// One HGNetV2 Stage.
///
/// Contains an optional downsample (depthwise Conv, stride=2) followed by
/// several HGBlocks.  When downsample is None, spatial dimensions are preserved.
#[derive(Module, Debug)]
pub struct HGNetV2Stage<B: Backend> {
    pub downsample: Option<ConvBnAct<B>>,
    pub blocks: Vec<HGNetV2BasicLayer<B>>,
}

impl<B: Backend> HGNetV2Stage<B> {
    /// Stage forward pass.
    ///
    /// - Input:  `[B, in_ch, H, W]`
    /// - Output: `[B, out_ch, H/2, W/2]` (when downsample=true)
    ///   or `[B, out_ch, H, W]`   (when downsample=false)
    pub fn forward(&self, hidden_state: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = match &self.downsample {
            Some(conv) => conv.forward(hidden_state),
            None => hidden_state,
        };
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

/// Stage configuration parameters.
#[derive(Config, Debug)]
pub struct HGNetV2StageConfig {
    pub in_ch: usize,
    pub mid_ch: usize,
    pub out_ch: usize,
    pub num_blocks: usize,
    pub do_downsample: bool,
    pub light_block: bool,
    pub kernel_size: usize,
    #[config(default = 6)]
    pub layer_num: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl HGNetV2StageConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HGNetV2Stage<B> {
        // Downsample: depthwise Conv(k=3, s=2, groups=in_ch), no activation
        let downsample = if self.do_downsample {
            Some(
                ConvBnActConfig::new(self.in_ch, self.in_ch, 3)
                    .with_stride(2)
                    .with_groups(self.in_ch) // depthwise
                    .with_use_activation(false) // no activation
                    .with_bn_eps(self.bn_eps)
                    .init(device),
            )
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(self.num_blocks);
        for i in 0..self.num_blocks {
            let block_in_ch = if i == 0 { self.in_ch } else { self.out_ch };
            blocks.push(
                HGNetV2BasicLayerConfig::new(
                    block_in_ch,
                    self.mid_ch,
                    self.out_ch,
                    self.kernel_size,
                    i != 0, // first block has no residual connection
                    self.light_block,
                )
                .with_layer_num(self.layer_num)
                .with_bn_eps(self.bn_eps)
                .init(device),
            );
        }

        HGNetV2Stage { downsample, blocks }
    }
}
