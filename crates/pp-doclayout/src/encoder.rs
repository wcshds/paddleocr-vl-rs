// ========================================================================
// PPDocLayoutV2 HybridEncoder
// ========================================================================
//
// The hybrid encoder consists of three stages:
//
// 1. **AIFI** (Attention-based Intra-scale Feature Interaction):
//    Applies a Transformer Encoder only to the highest-level feature map
//    (stride=32, 25×25), using 2D sinusoidal position embeddings to capture
//    global relationships on the coarsest feature map.
//
// 2. **FPN** (Feature Pyramid Network, top-down):
//    Starting from the highest level, progressively upsample + concat +
//    CSPRepLayer, fusing high-level semantic information into lower-level
//    features.
//
// 3. **PAN** (Path Aggregation Network, bottom-up):
//    Starting from the lowest level, progressively downsample + concat +
//    CSPRepLayer, fusing multi-scale features again.
//
// ```text
// Backbone outputs (after 1×1 projection):
//   S2: [B, 256, 100, 100]  (stride=8)
//   S3: [B, 256, 50, 50]    (stride=16)
//   S4: [B, 256, 25, 25]    (stride=32)
//
// ── AIFI ──
//   S4 → Transformer Encoder → S4'
//
// ── FPN (top-down) ──
//   S4' → lateral_conv → upsample(2x) → cat(S3) → CSPRep → F1
//   F1  → lateral_conv → upsample(2x) → cat(S2) → CSPRep → F0
//   Result: [F0, F1, S4'] (low → high resolution)
//
// ── PAN (bottom-up) ──
//   F0 → downsample_conv(s=2) → cat(F1) → CSPRep → P1
//   P1 → downsample_conv(s=2) → cat(S4') → CSPRep → P2
//   Result: [F0, P1, P2]  (final three-scale feature maps)
// ```
//
// Core sub-components:
// - **RepVGGBlock**: parallel 3×3 Conv + 1×1 Conv → SiLU
// - **CSPRepLayer**: Cross Stage Partial + RepVGG for efficient feature fusion
// - **SinePositionEmbedding**: 2D sinusoidal position encoding
// - **EncoderLayer**: Self-Attention + FFN (post-norm)
//
// References:
//   - RT-DETR: https://arxiv.org/abs/2304.08069
//   - RepVGG: https://arxiv.org/abs/2101.03697
// ========================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2dConfig, InterpolateMode},
    },
    prelude::Backend,
    tensor::{Int, Tensor, activation, module::attention},
};

use crate::frozen_batch_norm::{FrozenBatchNorm2d, FrozenBatchNorm2dConfig};

// ========================================================================
// ConvNormLayer: Conv2d + FrozenBN + optional SiLU
// ========================================================================
//
// Used for various convolutional layers within the HybridEncoder (lateral,
// downsample, CSPRep, etc.).  Similar to the backbone's ConvBnAct but uses
// SiLU activation instead of ReLU.
// ========================================================================

#[derive(Module, Debug)]
pub struct ConvNormLayer<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: FrozenBatchNorm2d<B>,
    pub use_activation: bool,
}

impl<B: Backend> ConvNormLayer<B> {
    /// Conv → BN → (SiLU)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        if self.use_activation {
            activation::silu(x)
        } else {
            x
        }
    }
}

#[derive(Config, Debug)]
pub struct ConvNormLayerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 3)]
    pub kernel_size: usize,
    #[config(default = 1)]
    pub stride: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
    #[config(default = true)]
    pub use_activation: bool,
    pub padding: Option<usize>,
}

impl ConvNormLayerConfig {
    /// Build a `ConvNormLayer`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvNormLayer<B> {
        let pad = self.padding.unwrap_or((self.kernel_size - 1) / 2);
        let conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_bias(false)
        .init(device);
        let norm = FrozenBatchNorm2dConfig::new(self.out_channels)
            .with_epsilon(self.bn_eps)
            .init(device);

        ConvNormLayer {
            conv,
            norm,
            use_activation: self.use_activation,
        }
    }
}

// ========================================================================
// RepVGGBlock
// ========================================================================
//
// RepVGG-style convolution block.  At inference the output is the sum of
// two parallel paths followed by activation:
//   output = SiLU(conv3x3(x) + conv1x1(x))
//
// The 3×3 path captures local spatial features; the 1×1 path provides
// channel-level feature mixing.  The two-branch design is equivalent to a
// multi-branch structure during training and can be fused into a single
// convolution at inference.
//
// In PP-DocLayoutV2, hidden_expansion=1.0, so hidden_ch = out_ch = 256.
// ========================================================================

#[derive(Module, Debug)]
pub struct RepVGGBlock<B: Backend> {
    /// 3×3 convolution path: Conv(ch→ch, k=3, s=1, p=1) + BN (no activation)
    pub conv1: ConvNormLayer<B>,
    /// 1×1 convolution path: Conv(ch→ch, k=1, s=1, p=0) + BN (no activation)
    pub conv2: ConvNormLayer<B>,
}

impl<B: Backend> RepVGGBlock<B> {
    /// RepVGG forward: SiLU(conv3x3(x) + conv1x1(x))
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv1.forward(x.clone()) + self.conv2.forward(x);
        activation::silu(y)
    }
}

#[derive(Config, Debug)]
pub struct RepVGGBlockConfig {
    pub channels: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl RepVGGBlockConfig {
    /// Build a `RepVGGBlock`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RepVGGBlock<B> {
        RepVGGBlock {
            conv1: ConvNormLayerConfig::new(self.channels, self.channels)
                .with_kernel_size(3)
                .with_stride(1)
                .with_padding(Some(1))
                .with_use_activation(false)
                .with_bn_eps(self.bn_eps)
                .init(device),
            conv2: ConvNormLayerConfig::new(self.channels, self.channels)
                .with_kernel_size(1)
                .with_stride(1)
                .with_padding(Some(0))
                .with_use_activation(false)
                .with_bn_eps(self.bn_eps)
                .init(device),
        }
    }
}

// ========================================================================
// CSPRepLayer (Cross Stage Partial + RepVGG)
// ========================================================================
//
// CSP structure splits the input into two paths:
//   path1: conv1(1×1) → 3 RepVGGBlocks in series
//   path2: conv2(1×1) pass-through
//
// The two paths are then added (since hidden_ch == out_ch, conv3 is Identity).
//
// Input channels are 256*2=512 (after concat), output channels are 256.
// ========================================================================

#[derive(Module, Debug)]
pub struct CSPRepLayer<B: Backend> {
    /// Path-1 entry: 1×1 Conv 512→256
    pub conv1: ConvNormLayer<B>,
    /// Path-2 pass-through: 1×1 Conv 512→256
    pub conv2: ConvNormLayer<B>,
    /// Path-1's 3 RepVGGBlocks
    pub bottlenecks: Vec<RepVGGBlock<B>>,
}

impl<B: Backend> CSPRepLayer<B> {
    /// CSPRepLayer forward pass.
    ///
    /// - Input:  `[B, 512, H, W]` (concatenated feature map)
    /// - Output: `[B, 256, H, W]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Path 1: 1×1 Conv → 3 RepVGG blocks
        let mut h1 = self.conv1.forward(x.clone());
        for block in &self.bottlenecks {
            h1 = block.forward(h1);
        }
        // Path 2: 1×1 Conv (pass-through)
        let h2 = self.conv2.forward(x);
        // Sum (since hidden_ch == out_ch, no additional conv3 is needed)
        h1 + h2
    }
}

#[derive(Config, Debug)]
pub struct CSPRepLayerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 3)]
    pub num_blocks: usize,
    #[config(default = 1.0)]
    pub expansion: f64,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl CSPRepLayerConfig {
    /// Build a CSPRepLayer.
    ///
    /// - `in_channels`: input channel count (typically encoder_hidden_dim*2 = 512)
    /// - `out_channels`: output channel count (typically encoder_hidden_dim = 256)
    /// - `expansion`: hidden-channel expansion ratio (default 1.0)
    pub fn init<B: Backend>(&self, device: &B::Device) -> CSPRepLayer<B> {
        let hidden_ch = (self.out_channels as f64 * self.expansion) as usize;

        let conv1 = ConvNormLayerConfig::new(self.in_channels, hidden_ch)
            .with_kernel_size(1)
            .with_stride(1)
            .with_use_activation(true)
            .with_bn_eps(self.bn_eps)
            .init(device);
        let conv2 = ConvNormLayerConfig::new(self.in_channels, hidden_ch)
            .with_kernel_size(1)
            .with_stride(1)
            .with_use_activation(true)
            .with_bn_eps(self.bn_eps)
            .init(device);

        let bottlenecks = (0..self.num_blocks)
            .map(|_| {
                RepVGGBlockConfig::new(hidden_ch)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();

        // Since hidden_ch == encoder_hidden_dim (256 == 256), conv3 is Identity
        CSPRepLayer {
            conv1,
            conv2,
            bottlenecks,
        }
    }
}

// ========================================================================
// 2D Sine Position Embedding
// ========================================================================
//
// 2D sinusoidal position encoding used by AIFI.
//
// For embed_dim=256:
//   pos_dim = 256 / 4 = 64
//   omega[i] = 1 / (temperature ^ (i / 64))   i ∈ [0, 64)
//
// For a position (h, w) the encoding is:
//   [sin(h·ω₀), ..., sin(h·ω₆₃), cos(h·ω₀), ..., cos(h·ω₆₃),
//    sin(w·ω₀), ..., sin(w·ω₆₃), cos(w·ω₀), ..., cos(w·ω₆₃)]
//
// Final shape: [1, H*W, 256]
// ========================================================================

pub fn sine_position_embedding_2d<B: Backend>(
    height: usize,
    width: usize,
    embed_dim: usize,
    temperature: f64,
    device: &B::Device,
) -> Tensor<B, 3> {
    assert!(
        embed_dim.is_multiple_of(4),
        "embed_dim must be a multiple of 4"
    );

    let pos_dim = embed_dim / 4;

    // omega[i] = 1 / (temperature ^ (i / pos_dim))
    let omega = Tensor::<B, 1, Int>::arange(0..pos_dim as i64, device)
        .float()
        .div_scalar(pos_dim as f32)
        .mul_scalar(-(temperature as f32).ln())
        .exp();

    // Build grid_h [H*W, 1] and grid_w [H*W, 1] entirely on device.
    // grid_h: [0,0,..,0, 1,1,..,1, ..., H-1,...,H-1]  (each repeated W times)
    // grid_w: [0,1,..,W-1, 0,1,..,W-1, ...]           (repeated H times)
    let grid_h = Tensor::<B, 1, Int>::arange(0..height as i64, device)
        .float()
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, width)
        .reshape([height * width, 1]);
    let grid_w = Tensor::<B, 1, Int>::arange(0..width as i64, device)
        .float()
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, height)
        .reshape([height * width, 1]);

    let omega_row = omega.unsqueeze_dim::<2>(0); // [1, pos_dim]

    // Outer product: [H*W, pos_dim]
    let out_h = grid_h.matmul(omega_row.clone());
    let out_w = grid_w.matmul(omega_row);

    // Concatenate: [sin(h*ω), cos(h*ω), sin(w*ω), cos(w*ω)] → [H*W, 256]
    let pos = Tensor::cat(
        vec![
            out_h.clone().sin(),
            out_h.cos(),
            out_w.clone().sin(),
            out_w.cos(),
        ],
        1,
    );

    // Add batch dimension: [1, H*W, 256]
    pos.unsqueeze_dim::<3>(0)
}

// ========================================================================
// AIFI Encoder Layer (Self-Attention + FFN, post-norm)
// ========================================================================
//
// RT-DETR's AIFI uses a post-norm Transformer Encoder Layer:
//   1. Self-Attention: Q=K=(hidden+pos_embed), V=hidden
//   2. residual + LayerNorm
//   3. FFN: Linear(256→1024) → GELU → Linear(1024→256)
//   4. residual + LayerNorm
//
// Note: position embeddings are added to Q/K but not to V.
// Scaled dot-product attention (is_causal=false for bidirectional attention).
// ========================================================================

#[derive(Module, Debug)]
pub struct AIFIEncoderLayer<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub self_attn_layer_norm: LayerNorm<B>,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub final_layer_norm: LayerNorm<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> AIFIEncoderLayer<B> {
    /// AIFI Encoder Layer forward pass (post-norm).
    ///
    /// - `hidden_states`: `[B, H*W, 256]`
    /// - `pos_embed`: `[1, H*W, 256]` (2D sinusoidal position encoding)
    ///
    /// Returns: `[B, H*W, 256]`
    pub fn forward(&self, hidden_states: Tensor<B, 3>, pos_embed: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();

        // Add position embeddings to Q/K but not to V
        let qk_input = hidden_states.clone() + pos_embed.clone();

        // Self-Attention
        let q = self
            .q_proj
            .forward(qk_input.clone())
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2); // [B, heads, seq, head_dim]
        let k = self
            .k_proj
            .forward(qk_input)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let attn_output = attention(q, k, v, None, None, Default::default());
        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ]);
        let attn_output = self.o_proj.forward(attn_output);

        // Post-norm: residual + LayerNorm
        let hidden_states = self
            .self_attn_layer_norm
            .forward(hidden_states + attn_output);

        // FFN: Linear → GELU → Linear
        let residual = hidden_states.clone();
        let hidden_states = self.fc1.forward(hidden_states);
        let hidden_states = activation::gelu(hidden_states);
        let hidden_states = self.fc2.forward(hidden_states);

        // Post-norm: residual + LayerNorm
        self.final_layer_norm.forward(residual + hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct AIFIEncoderLayerConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub num_heads: usize,
    #[config(default = 1024)]
    pub ffn_dim: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl AIFIEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AIFIEncoderLayer<B> {
        let head_dim = self.d_model / self.num_heads;

        AIFIEncoderLayer {
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            o_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            self_attn_layer_norm: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            fc1: LinearConfig::new(self.d_model, self.ffn_dim).init(device),
            fc2: LinearConfig::new(self.ffn_dim, self.d_model).init(device),
            final_layer_norm: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

// ========================================================================
// AIFI Layer
// ========================================================================

#[derive(Module, Debug)]
pub struct AIFILayer<B: Backend> {
    pub layers: Vec<AIFIEncoderLayer<B>>,
    pub encoder_hidden_dim: usize,
}

impl<B: Backend> AIFILayer<B> {
    /// AIFI forward pass.
    ///
    /// - Input:  `[B, 256, H, W]` (one feature map)
    /// - Output: `[B, 256, H, W]` (Transformer-processed feature map)
    pub fn forward(&self, hidden_states: Tensor<B, 4>, temperature: f64) -> Tensor<B, 4> {
        let [batch_size, _c, height, width] = hidden_states.dims();

        // Flatten spatial dimensions: [B, 256, H, W] → [B, H*W, 256]
        let mut x = hidden_states
            .clone()
            .reshape([batch_size, self.encoder_hidden_dim, height * width])
            .swap_dims(1, 2);

        // Generate 2D sinusoidal position encoding: [1, H*W, 256]
        let pos_embed = sine_position_embedding_2d(
            height,
            width,
            self.encoder_hidden_dim,
            temperature,
            &x.device(),
        )
        .cast(x.dtype());

        for layer in &self.layers {
            x = layer.forward(x, &pos_embed);
        }

        // Restore spatial dimensions: [B, H*W, 256] → [B, 256, H, W]
        x.swap_dims(1, 2)
            .reshape([batch_size, self.encoder_hidden_dim, height, width])
    }
}

#[derive(Config, Debug)]
pub struct AIFILayerConfig {
    #[config(default = 256)]
    pub encoder_hidden_dim: usize,
    #[config(default = 8)]
    pub num_heads: usize,
    #[config(default = 1024)]
    pub ffn_dim: usize,
    #[config(default = 1)]
    pub num_layers: usize,
    #[config(default = 10000.0)]
    pub temperature: f64,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl AIFILayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AIFILayer<B> {
        let layers = (0..self.num_layers)
            .map(|_| {
                AIFIEncoderLayerConfig::new()
                    .with_d_model(self.encoder_hidden_dim)
                    .with_num_heads(self.num_heads)
                    .with_ffn_dim(self.ffn_dim)
                    .with_layer_norm_eps(self.layer_norm_eps)
                    .init(device)
            })
            .collect();

        AIFILayer {
            layers,
            encoder_hidden_dim: self.encoder_hidden_dim,
        }
    }
}

// ========================================================================
// Nearest Upsample (2x)
// ========================================================================

/// Nearest-neighbor 2× upsampling.
///
/// Replicates each pixel into a 2×2 block:
/// `[B, C, H, W]` → `[B, C, 2H, 2W]`
fn upsample_nearest_2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    upsample_nearest_2x_impl(x)
}

fn upsample_nearest_2x_impl<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    // [B, C, H, 1, W, 1] → repeat → [B, C, H, 2, W, 2] → reshape
    x.reshape([b, c, h, 1, w, 1])
        .repeat(&[1, 1, 1, 2, 1, 2])
        .reshape([b, c, h * 2, w * 2])
}

// ========================================================================
// PPDocLayoutV2HybridEncoder
// ========================================================================

#[derive(Module, Debug)]
pub struct HybridEncoder<B: Backend> {
    /// AIFI layer (applies Transformer to the highest-level feature map)
    pub aifi: Vec<AIFILayer<B>>,

    // ── FPN (top-down) ──
    /// Lateral 1×1 Conv + BN + SiLU (2 instances)
    pub lateral_convs: Vec<ConvNormLayer<B>>,
    /// FPN fusion CSPRepLayers (2 instances)
    pub fpn_blocks: Vec<CSPRepLayer<B>>,

    // ── PAN (bottom-up) ──
    /// Downsample 3×3 Conv(s=2) + BN + SiLU (2 instances)
    pub downsample_convs: Vec<ConvNormLayer<B>>,
    /// PAN fusion CSPRepLayers (2 instances)
    pub pan_blocks: Vec<CSPRepLayer<B>>,

    pub positional_encoding_temperature: f64,
}

impl<B: Backend> HybridEncoder<B> {
    /// HybridEncoder forward pass.
    ///
    /// - `feature_maps`: three projected feature maps `[B, 256, H_i, W_i]`
    ///   - feature_maps[0]: stride=8  (100×100)
    ///   - feature_maps[1]: stride=16 (50×50)
    ///   - feature_maps[2]: stride=32 (25×25)
    ///
    /// Returns: three fused feature maps `Vec<Tensor<B, 4>>`
    pub fn forward(&self, feature_maps: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        assert_eq!(
            feature_maps.len(),
            3,
            "HybridEncoder requires 3 feature maps"
        );

        let mut fmaps = feature_maps;

        // ── AIFI: apply Transformer to the highest level (index=2) ──
        fmaps[2] = self.aifi[0].forward(fmaps[2].clone(), self.positional_encoding_temperature);

        // ── FPN (top-down) ──
        // fpn_feature_maps starts from the highest level
        let mut fpn_maps = vec![fmaps[2].clone()];

        for idx in 0..2 {
            let backbone_fmap = fmaps[1 - idx].clone(); // first fmaps[1], then fmaps[0]

            // Lateral conv: apply 1×1 Conv to the previous FPN output
            let mut top_fpn = fpn_maps.last().unwrap().clone();
            top_fpn = self.lateral_convs[idx].forward(top_fpn);
            // Replace the last element (Python: fpn_feature_maps[-1] = ...)
            let last_idx = fpn_maps.len() - 1;
            fpn_maps[last_idx] = top_fpn.clone();

            // Upsample 2× + concat + CSPRep
            let upsampled = upsample_nearest_2x(top_fpn);
            let fused = Tensor::cat(vec![upsampled, backbone_fmap], 1); // [B, 512, H, W]
            let new_fpn = self.fpn_blocks[idx].forward(fused);
            fpn_maps.push(new_fpn);
        }

        // Reverse: [S4', F1, F0] → [F0, F1, S4']
        fpn_maps.reverse();

        // ── PAN (bottom-up) ──
        let mut pan_maps = vec![fpn_maps[0].clone()];

        for idx in 0..2 {
            let fpn_fmap = fpn_maps[idx + 1].clone();

            // Downsample conv (stride=2) + concat + CSPRep
            let top_pan = pan_maps.last().unwrap().clone();
            let downsampled = self.downsample_convs[idx].forward(top_pan);
            let fused = Tensor::cat(vec![downsampled, fpn_fmap], 1); // [B, 512, H, W]
            let new_pan = self.pan_blocks[idx].forward(fused);
            pan_maps.push(new_pan);
        }

        pan_maps
    }
}

#[derive(Config, Debug)]
pub struct HybridEncoderConfig {
    #[config(default = 256)]
    pub encoder_hidden_dim: usize,
    #[config(default = 1)]
    pub encoder_layers: usize,
    #[config(default = 1024)]
    pub encoder_ffn_dim: usize,
    #[config(default = 8)]
    pub encoder_attention_heads: usize,
    #[config(default = 1.0)]
    pub hidden_expansion: f64,
    #[config(default = 10000.0)]
    pub positional_encoding_temperature: f64,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl HybridEncoderConfig {
    /// Build the HybridEncoder.
    ///
    /// Parameters are sourced from PPDocLayoutV2Config:
    /// - encoder_hidden_dim: 256
    /// - encoder_layers: 1 (AIFI)
    /// - encoder_ffn_dim: 1024
    /// - encoder_attention_heads: 8
    /// - hidden_expansion: 1.0
    /// - positional_encoding_temperature: 10000.0
    pub fn init<B: Backend>(&self, device: &B::Device) -> HybridEncoder<B> {
        // AIFI: 1 AIFILayer (only applied at encode_proj_layers=[2], i.e. the coarsest level)
        let aifi = vec![
            AIFILayerConfig::new()
                .with_encoder_hidden_dim(self.encoder_hidden_dim)
                .with_num_heads(self.encoder_attention_heads)
                .with_ffn_dim(self.encoder_ffn_dim)
                .with_num_layers(self.encoder_layers)
                .with_temperature(self.positional_encoding_temperature)
                .with_layer_norm_eps(self.layer_norm_eps)
                .init(device),
        ];

        // FPN: 2 lateral_convs + 2 CSPRepLayers
        let num_fpn_stages = 2; // len(encoder_in_channels) - 1
        let lateral_convs = (0..num_fpn_stages)
            .map(|_| {
                ConvNormLayerConfig::new(self.encoder_hidden_dim, self.encoder_hidden_dim)
                    .with_kernel_size(1)
                    .with_stride(1)
                    .with_use_activation(true)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();
        let fpn_blocks = (0..num_fpn_stages)
            .map(|_| {
                CSPRepLayerConfig::new(self.encoder_hidden_dim * 2, self.encoder_hidden_dim)
                    .with_expansion(self.hidden_expansion)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();

        // PAN: 2 downsample_convs + 2 CSPRepLayers
        let num_pan_stages = 2;
        let downsample_convs = (0..num_pan_stages)
            .map(|_| {
                ConvNormLayerConfig::new(self.encoder_hidden_dim, self.encoder_hidden_dim)
                    .with_kernel_size(3)
                    .with_stride(2)
                    .with_use_activation(true)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();
        let pan_blocks = (0..num_pan_stages)
            .map(|_| {
                CSPRepLayerConfig::new(self.encoder_hidden_dim * 2, self.encoder_hidden_dim)
                    .with_expansion(self.hidden_expansion)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();

        HybridEncoder {
            aifi,
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
            positional_encoding_temperature: self.positional_encoding_temperature,
        }
    }
}

// ========================================================================
// PPDocLayoutV3 HybridEncoder Mask Branch
// ========================================================================
//
// PP-DocLayoutV3 keeps the V2 AIFI/FPN/PAN feature fusion path and adds a
// lightweight mask feature branch. The mask branch produces 32 prototype
// feature maps at stride 4, which are combined with query-specific mask
// embeddings in the decoder.
// ========================================================================

fn bilinear_resize<B: Backend>(x: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4> {
    Interpolate2dConfig::new()
        .with_output_size(Some(output_size))
        .with_mode(InterpolateMode::Linear)
        .with_align_corners(false)
        .init()
        .forward(x)
}

fn bilinear_upsample_2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_b, _c, h, w] = x.dims();
    bilinear_resize(x, [h * 2, w * 2])
}

#[derive(Module, Debug)]
pub struct ScaleHead<B: Backend> {
    /// Convolutional layers in the scale head. The Python module stores
    /// upsampling layers in the same list; in Burn we keep only weighted
    /// layers and drive interpolation from `upsample_after`.
    pub layers: Vec<ConvNormLayer<B>>,
    pub upsample_after: Vec<bool>,
}

impl<B: Backend> ScaleHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if self.upsample_after[idx] {
                x = bilinear_upsample_2x(x);
            }
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct ScaleHeadConfig {
    pub in_channels: usize,
    pub feature_channels: usize,
    pub fpn_stride: usize,
    pub base_stride: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl ScaleHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ScaleHead<B> {
        let mut head_length = 0usize;
        let mut stride = self.fpn_stride;
        while stride > self.base_stride {
            head_length += 1;
            stride /= 2;
        }
        head_length = head_length.max(1);

        let mut layers = Vec::with_capacity(head_length);
        let mut upsample_after = Vec::with_capacity(head_length);
        for idx in 0..head_length {
            let in_channels = if idx == 0 {
                self.in_channels
            } else {
                self.feature_channels
            };
            layers.push(
                ConvNormLayerConfig::new(in_channels, self.feature_channels)
                    .with_kernel_size(3)
                    .with_stride(1)
                    .with_padding(Some(1))
                    .with_use_activation(true)
                    .with_bn_eps(self.bn_eps)
                    .init(device),
            );
            upsample_after.push(self.fpn_stride != self.base_stride);
        }

        ScaleHead {
            layers,
            upsample_after,
        }
    }
}

#[derive(Module, Debug)]
pub struct MaskFeatureFpn<B: Backend> {
    pub scale_heads: Vec<ScaleHead<B>>,
    pub output_conv: ConvNormLayer<B>,
    pub reorder_index: Vec<usize>,
}

impl<B: Backend> MaskFeatureFpn<B> {
    pub fn forward(&self, inputs: &[Tensor<B, 4>]) -> Tensor<B, 4> {
        assert_eq!(inputs.len(), 3, "MaskFeatureFpn requires 3 feature maps");

        let first_idx = self.reorder_index[0];
        let mut output = self.scale_heads[0].forward(inputs[first_idx].clone());
        let [_b, _c, out_h, out_w] = output.dims();

        for idx in 1..self.scale_heads.len() {
            let input_idx = self.reorder_index[idx];
            let scaled = self.scale_heads[idx].forward(inputs[input_idx].clone());
            output = output + bilinear_resize(scaled, [out_h, out_w]);
        }

        self.output_conv.forward(output)
    }
}

#[derive(Config, Debug)]
pub struct MaskFeatureFpnConfig {
    pub in_channels: [usize; 3],
    pub fpn_strides: [usize; 3],
    #[config(default = 64)]
    pub feature_channels: usize,
    #[config(default = 64)]
    pub out_channels: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl MaskFeatureFpnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MaskFeatureFpn<B> {
        let mut reorder_index = vec![0usize, 1, 2];
        reorder_index.sort_by_key(|&idx| self.fpn_strides[idx]);
        let base_stride = self.fpn_strides[reorder_index[0]];

        let scale_heads = reorder_index
            .iter()
            .map(|&idx| {
                ScaleHeadConfig::new(
                    self.in_channels[idx],
                    self.feature_channels,
                    self.fpn_strides[idx],
                    base_stride,
                )
                .with_bn_eps(self.bn_eps)
                .init(device)
            })
            .collect();

        let output_conv = ConvNormLayerConfig::new(self.feature_channels, self.out_channels)
            .with_kernel_size(3)
            .with_stride(1)
            .with_padding(Some(1))
            .with_use_activation(true)
            .with_bn_eps(self.bn_eps)
            .init(device);

        MaskFeatureFpn {
            scale_heads,
            output_conv,
            reorder_index,
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderMaskOutput<B: Backend> {
    pub base_conv: ConvNormLayer<B>,
    pub conv: Conv2d<B>,
}

impl<B: Backend> EncoderMaskOutput<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.base_conv.forward(x);
        self.conv.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct EncoderMaskOutputConfig {
    pub in_channels: usize,
    pub num_prototypes: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl EncoderMaskOutputConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderMaskOutput<B> {
        EncoderMaskOutput {
            base_conv: ConvNormLayerConfig::new(self.in_channels, self.in_channels)
                .with_kernel_size(3)
                .with_stride(1)
                .with_padding(Some(1))
                .with_use_activation(true)
                .with_bn_eps(self.bn_eps)
                .init(device),
            conv: Conv2dConfig::new([self.in_channels, self.num_prototypes], [1, 1])
                .with_bias(true)
                .init(device),
        }
    }
}

pub struct HybridEncoderV3Output<B: Backend> {
    pub feature_maps: Vec<Tensor<B, 4>>,
    pub mask_feat: Tensor<B, 4>,
}

#[derive(Module, Debug)]
pub struct HybridEncoderV3<B: Backend> {
    pub aifi: Vec<AIFILayer<B>>,
    pub lateral_convs: Vec<ConvNormLayer<B>>,
    pub fpn_blocks: Vec<CSPRepLayer<B>>,
    pub downsample_convs: Vec<ConvNormLayer<B>>,
    pub pan_blocks: Vec<CSPRepLayer<B>>,
    pub mask_feature_head: MaskFeatureFpn<B>,
    pub encoder_mask_lateral: ConvNormLayer<B>,
    pub encoder_mask_output: EncoderMaskOutput<B>,
    pub positional_encoding_temperature: f64,
}

impl<B: Backend> HybridEncoderV3<B> {
    /// Forward pass for PP-DocLayoutV3.
    ///
    /// `feature_maps` are the projected stride 8/16/32 maps, while `x4_feat`
    /// is the stride-4 stage1 backbone map used to recover fine mask detail.
    pub fn forward(
        &self,
        feature_maps: Vec<Tensor<B, 4>>,
        x4_feat: Tensor<B, 4>,
    ) -> HybridEncoderV3Output<B> {
        assert_eq!(
            feature_maps.len(),
            3,
            "HybridEncoderV3 requires 3 feature maps"
        );

        let mut fmaps = feature_maps;
        fmaps[2] = self.aifi[0].forward(fmaps[2].clone(), self.positional_encoding_temperature);

        let mut fpn_maps = vec![fmaps[2].clone()];
        for idx in 0..2 {
            let backbone_fmap = fmaps[1 - idx].clone();
            let mut top_fpn = fpn_maps.last().unwrap().clone();
            top_fpn = self.lateral_convs[idx].forward(top_fpn);
            let last_idx = fpn_maps.len() - 1;
            fpn_maps[last_idx] = top_fpn.clone();

            let upsampled = upsample_nearest_2x(top_fpn);
            let fused = Tensor::cat(vec![upsampled, backbone_fmap], 1);
            fpn_maps.push(self.fpn_blocks[idx].forward(fused));
        }
        fpn_maps.reverse();

        let mut pan_maps = vec![fpn_maps[0].clone()];
        for idx in 0..2 {
            let fpn_fmap = fpn_maps[idx + 1].clone();
            let top_pan = pan_maps.last().unwrap().clone();
            let downsampled = self.downsample_convs[idx].forward(top_pan);
            let fused = Tensor::cat(vec![downsampled, fpn_fmap], 1);
            pan_maps.push(self.pan_blocks[idx].forward(fused));
        }

        let mut mask_feat = self.mask_feature_head.forward(&pan_maps);
        mask_feat = bilinear_upsample_2x(mask_feat);
        mask_feat = mask_feat + self.encoder_mask_lateral.forward(x4_feat);
        mask_feat = self.encoder_mask_output.forward(mask_feat);

        HybridEncoderV3Output {
            feature_maps: pan_maps,
            mask_feat,
        }
    }
}

#[derive(Config, Debug)]
pub struct HybridEncoderV3Config {
    #[config(default = 256)]
    pub encoder_hidden_dim: usize,
    #[config(default = 1)]
    pub encoder_layers: usize,
    #[config(default = 1024)]
    pub encoder_ffn_dim: usize,
    #[config(default = 8)]
    pub encoder_attention_heads: usize,
    #[config(default = 1.0)]
    pub hidden_expansion: f64,
    #[config(default = 10000.0)]
    pub positional_encoding_temperature: f64,
    #[config(default = 64)]
    pub mask_feature_channels_in: usize,
    #[config(default = 64)]
    pub mask_feature_channels_out: usize,
    #[config(default = 128)]
    pub x4_feat_dim: usize,
    #[config(default = 32)]
    pub num_prototypes: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl HybridEncoderV3Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HybridEncoderV3<B> {
        let aifi = vec![
            AIFILayerConfig::new()
                .with_encoder_hidden_dim(self.encoder_hidden_dim)
                .with_num_heads(self.encoder_attention_heads)
                .with_ffn_dim(self.encoder_ffn_dim)
                .with_num_layers(self.encoder_layers)
                .with_temperature(self.positional_encoding_temperature)
                .with_layer_norm_eps(self.layer_norm_eps)
                .init(device),
        ];

        let lateral_convs = (0..2)
            .map(|_| {
                ConvNormLayerConfig::new(self.encoder_hidden_dim, self.encoder_hidden_dim)
                    .with_kernel_size(1)
                    .with_stride(1)
                    .with_use_activation(true)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();
        let fpn_blocks = (0..2)
            .map(|_| {
                CSPRepLayerConfig::new(self.encoder_hidden_dim * 2, self.encoder_hidden_dim)
                    .with_expansion(self.hidden_expansion)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();
        let downsample_convs = (0..2)
            .map(|_| {
                ConvNormLayerConfig::new(self.encoder_hidden_dim, self.encoder_hidden_dim)
                    .with_kernel_size(3)
                    .with_stride(2)
                    .with_use_activation(true)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();
        let pan_blocks = (0..2)
            .map(|_| {
                CSPRepLayerConfig::new(self.encoder_hidden_dim * 2, self.encoder_hidden_dim)
                    .with_expansion(self.hidden_expansion)
                    .with_bn_eps(self.bn_eps)
                    .init(device)
            })
            .collect();

        HybridEncoderV3 {
            aifi,
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
            mask_feature_head: MaskFeatureFpnConfig::new([self.encoder_hidden_dim; 3], [8, 16, 32])
                .with_feature_channels(self.mask_feature_channels_in)
                .with_out_channels(self.mask_feature_channels_out)
                .with_bn_eps(self.bn_eps)
                .init(device),
            encoder_mask_lateral: ConvNormLayerConfig::new(
                self.x4_feat_dim,
                self.mask_feature_channels_out,
            )
            .with_kernel_size(3)
            .with_stride(1)
            .with_padding(Some(1))
            .with_use_activation(true)
            .with_bn_eps(self.bn_eps)
            .init(device),
            encoder_mask_output: EncoderMaskOutputConfig::new(
                self.mask_feature_channels_out,
                self.num_prototypes,
            )
            .with_bn_eps(self.bn_eps)
            .init(device),
            positional_encoding_temperature: self.positional_encoding_temperature,
        }
    }
}
