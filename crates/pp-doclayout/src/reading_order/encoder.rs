// ========================================================================
// PPDocLayoutV2 Reading Order — Transformer Encoder
// ========================================================================
//
// The core component of the reading-order module: a 6-layer Transformer
// Encoder with 2D spatial attention bias.
//
// Architecture (each ReadingOrderLayer):
//
// ```text
// hidden_states ──┐
//                  ▼
//   ┌─────────────────────────────┐
//   │ SelfAttention               │
//   │   Q/K/V linear → 8 heads    │
//   │   scores = Q@K^T / √head_d  │
//   │   scores += rel_2d_pos      │  ← unscaled 2D spatial attention bias
//   │   attn = CogView(scores)    │  ← PB-Relax stabilization
//   │   output = attn @ V         │
//   └─────────────────────────────┘
//   │
//   ▼ dense → dropout → LayerNorm(residual + output)
//   │
//   ▼ Intermediate: Linear(512→2048) → GELU
//   │
//   ▼ Output: Linear(2048→512) → dropout → LayerNorm(residual + output)
// ```
//
// 2D spatial attention bias (PositionRelationEmbedding):
//
// Computes an attention bias based on the relative spatial relationship
// between each pair of bounding boxes.
//
// 1. bbox → center_width_height format [cx, cy, w, h]
// 2. box_relative_encoding:
//    - Coordinate diff: log(|src_xy - tgt_xy| / src_wh + 1)
//    - Size ratio:      log(src_wh / tgt_wh)
//    → yields [B, L, L, 4] relative encoding
// 3. RoPE-style sin/cos embedding:
//    (rel * scale).unsqueeze(-1) * inv_freq → sin/cos → flatten
//    → yields [B, L, L, embed_dim*4=64]
// 4. Conv2d(64→8) projects to [B, 8, L, L] attention bias
//
// Note: rel_2d_pos is added **unscaled** directly to the attention scores.
// This differs from LayoutLMv3 (which divides by √head_dim).
//
// CogView PB-Relax stabilization:
//   scores' = (scores / α − max(scores / α)) × α
//   attn_weights = softmax(scores')
// where α=32.  This prevents overflow/NaN from large attention scores.
//
// References:
//   - LayoutLMv3: https://arxiv.org/abs/2204.08387
//   - CogView: https://arxiv.org/abs/2105.13290
// ========================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{Tensor, TensorData, activation, s},
};

// ========================================================================
// PositionRelationEmbedding
// ========================================================================
//
// Generates a 2D attention bias from the relative spatial relationship
// between bounding boxes.
//
// Computation flow:
//   1. bbox [B, L, 4] (cx, cy, w, h)
//   2. box_relative_encoding → [B, L, L, 4]
//   3. get_position_embedding (RoPE-style sin/cos) → [B, L, L, 64]
//   4. permute → [B, 64, L, L]
//   5. Conv2d(64, 8, 1) → [B, 8, L, L] (= num_attention_heads)
// ========================================================================

#[derive(Module, Debug)]
pub struct PositionRelationEmbedding<B: Backend> {
    /// Conv2d: embed_dim*4 → num_attention_heads, kernel=1
    pub pos_proj: Conv2d<B>,
    /// RoPE inverse frequencies: [half_dim] = [8]
    pub inv_freq: Tensor<B, 1>,
    pub embed_dim: usize,
    pub scale: f64,
}

impl<B: Backend> PositionRelationEmbedding<B> {
    /// Compute relative spatial encoding between bounding boxes.
    ///
    /// For each (source, target) pair, computes a 4-dimensional encoding:
    ///   - [0:2] = log(|src_center - tgt_center| / src_size + 1)  (coordinate diff)
    ///   - [2:4] = log(src_size / tgt_size)                        (size ratio)
    ///
    /// - `boxes`: `[B, L, 4]`, format [cx, cy, w, h]
    /// - Returns: `[B, L, L, 4]`
    fn box_relative_encoding(&self, boxes: &Tensor<B, 3>) -> Tensor<B, 4> {
        let [_batch, _seq_len, _] = boxes.dims();
        let eps = 1e-5f32;

        // source: [B, L, 1, 4], target: [B, 1, L, 4]
        let source = boxes.clone().unsqueeze_dim::<4>(2);
        let target = boxes.clone().unsqueeze_dim::<4>(1);

        // Separate coordinates (cx, cy) and dimensions (w, h)
        let src_coord = source.clone().slice(s![.., .., 0..1, 0..2]);
        let src_dim = source.slice(s![.., .., 0..1, 2..4]);
        let tgt_coord = target.clone().slice(s![.., 0..1, .., 0..2]);
        let tgt_dim = target.slice(s![.., 0..1, .., 2..4]);

        // Coordinate diff: log(|src_xy - tgt_xy| / (src_wh + eps) + 1.0)
        let coord_diff = (src_coord - tgt_coord).abs();
        let relative_coordinates = (coord_diff / src_dim.clone().add_scalar(eps) + 1.0f32).log();

        // Size ratio: log((src_wh + eps) / (tgt_wh + eps))
        let relative_dim = (src_dim.add_scalar(eps) / tgt_dim.add_scalar(eps)).log();

        // Concatenate: [B, L, L, 4]
        Tensor::cat(vec![relative_coordinates, relative_dim], 3)
    }

    /// RoPE-style position embedding.
    ///
    /// - `x`: `[B, L, L, 4]` relative encoding
    /// - Returns: `[B, L, L, embed_dim*4]` (4 components × embed_dim sin/cos dimensions)
    fn get_position_embedding(&self, x: &Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, l1, l2, four] = x.dims();

        // (x * scale).unsqueeze(-1): [B, L, L, 4, 1]
        let scaled = x.clone().mul_scalar(self.scale as f32);
        let scaled = scaled.unsqueeze_dim::<5>(4); // [B, L, L, 4, 1]

        // * inv_freq: [B, L, L, 4, half_dim]
        let inv_freq =
            self.inv_freq
                .clone()
                .cast(x.dtype())
                .reshape([1, 1, 1, 1, self.embed_dim / 2]);
        let angles = scaled * inv_freq; // [B, L, L, 4, half_dim]

        // sin/cos → cat → [B, L, L, 4, embed_dim]
        let sin_part = angles.clone().sin();
        let cos_part = angles.cos();
        let embedding = Tensor::cat(vec![sin_part, cos_part], 4); // [B, L, L, 4, embed_dim]

        // Flatten last two dims: [B, L, L, 4*embed_dim]
        embedding.reshape([batch, l1, l2, four * self.embed_dim])
    }

    /// PositionRelationEmbedding forward pass.
    ///
    /// - `boxes`: `[B, L, 4]`, format [cx, cy, w, h]
    /// - Returns: `[B, num_attention_heads, L, L]` 2D spatial attention bias
    pub fn forward(&self, boxes: &Tensor<B, 3>) -> Tensor<B, 4> {
        // 1. Compute relative encoding: [B, L, L, 4]
        let relative_encoding = self.box_relative_encoding(boxes);

        // 2. RoPE-style embedding: [B, L, L, 64]
        let position_embedding = self.get_position_embedding(&relative_encoding);

        // 3. Permute: [B, 64, L, L]
        let position_embedding = position_embedding
            .swap_dims(2, 3) // [B, L, 64, L]
            .swap_dims(1, 2); // [B, 64, L, L]

        // 4. Conv2d(64, 8, 1): [B, 8, L, L]
        self.pos_proj.forward(position_embedding)
    }
}

#[derive(Config, Debug)]
pub struct PositionRelationEmbeddingConfig {
    #[config(default = 16)]
    pub embed_dim: usize,
    #[config(default = 8)]
    pub num_attention_heads: usize,
    #[config(default = 10000.0)]
    pub theta: f64,
    #[config(default = 100.0)]
    pub scale: f64,
}

impl PositionRelationEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionRelationEmbedding<B> {
        let half_dim = self.embed_dim / 2;

        // inv_freq = 1 / (theta ^ (arange(0, embed_dim, 2) / half_dim))
        let inv_freq_data: Vec<f32> = (0..self.embed_dim)
            .step_by(2)
            .map(|i| 1.0 / (self.theta as f32).powf(i as f32 / half_dim as f32))
            .collect();
        let inv_freq = Tensor::<B, 1>::from_data(
            TensorData::new(inv_freq_data, [half_dim]).convert::<B::FloatElem>(),
            device,
        );

        // Conv2d(embed_dim * 4, num_attention_heads, kernel_size=1)
        let pos_proj = Conv2dConfig::new([self.embed_dim * 4, self.num_attention_heads], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
            .init(device);

        PositionRelationEmbedding {
            pos_proj,
            inv_freq,
            embed_dim: self.embed_dim,
            scale: self.scale,
        }
    }
}

// ========================================================================
// ReadingOrder Self-Attention
// ========================================================================
//
// Multi-head self-attention with CogView PB-Relax stabilization.
// Because we need to add unscaled rel_2d_pos bias, burn's built-in
// attention() cannot be used.
// ========================================================================

#[derive(Module, Debug)]
pub struct ReadingOrderSelfAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub num_attention_heads: usize,
    pub attention_head_size: usize,
}

impl<B: Backend> ReadingOrderSelfAttention<B> {
    /// CogView PB-Relax softmax stabilization.
    ///
    /// Reference: https://arxiv.org/abs/2105.13290 Section 2.4
    ///
    /// ```text
    /// scores' = scores / α
    /// scores' = (scores' − max(scores')) × α
    /// attn = softmax(scores')
    /// ```
    ///
    /// - `attention_scores`: `[B, heads, L, L]`
    /// - Returns: `[B, heads, L, L]` attention weights
    fn cogview_attention(&self, attention_scores: Tensor<B, 4>) -> Tensor<B, 4> {
        let alpha = 32.0f32;

        // scores / alpha
        let scaled = attention_scores.div_scalar(alpha);
        // max per row (last dim): [B, heads, L, 1]
        let max_val = scaled.clone().max_dim(3);
        // (scaled − max) × alpha
        let stabilized = (scaled - max_val).mul_scalar(alpha);

        activation::softmax(stabilized, 3)
    }

    /// Self-Attention forward pass.
    ///
    /// - `hidden_states`: `[B, L, hidden_size]`
    /// - `attention_mask`: `[B, 1, 1, L]` or `None` (additive mask)
    /// - `rel_2d_pos`: `[B, heads, L, L]` or `None` (unscaled 2D bias)
    /// - Returns: `[B, L, hidden_size]`
    pub fn forward(
        &self,
        hidden_states: &Tensor<B, 3>,
        attention_mask: Option<&Tensor<B, 4>>,
        rel_2d_pos: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();

        // Q/K/V: [B, L, hidden] → [B, heads, L, head_dim]
        let q = self
            .query
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ])
            .swap_dims(1, 2);
        let k = self
            .key
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ])
            .swap_dims(1, 2);
        let v = self
            .value
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ])
            .swap_dims(1, 2);

        // CogView style: scale Q by 1/√d, then multiply by K^T
        let scale = (self.attention_head_size as f32).sqrt();
        let q_scaled = q.div_scalar(scale);
        let mut attention_scores = q_scaled.matmul(k.swap_dims(2, 3)); // [B, heads, L, L]

        // Add 2D spatial attention bias (unscaled)
        if let Some(rel_2d) = rel_2d_pos {
            attention_scores = attention_scores + rel_2d.clone();
        }

        // Add attention mask (already in additive format; padding positions are large negative)
        if let Some(mask) = attention_mask {
            attention_scores = attention_scores + mask.clone();
        }

        // CogView PB-Relax softmax
        let attention_probs = self.cogview_attention(attention_scores);

        // attn @ V: [B, heads, L, head_dim]
        let context = attention_probs.matmul(v);

        // Reshape: [B, L, hidden_size]
        context
            .swap_dims(1, 2) // [B, L, heads, head_dim]
            .reshape([
                batch_size,
                seq_len,
                self.num_attention_heads * self.attention_head_size,
            ])
    }
}

#[derive(Config, Debug)]
pub struct ReadingOrderSelfAttentionConfig {
    #[config(default = 512)]
    pub hidden_size: usize,
    #[config(default = 8)]
    pub num_attention_heads: usize,
}

impl ReadingOrderSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ReadingOrderSelfAttention<B> {
        let attention_head_size = self.hidden_size / self.num_attention_heads;
        ReadingOrderSelfAttention {
            query: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            key: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            value: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            num_attention_heads: self.num_attention_heads,
            attention_head_size,
        }
    }
}

// ========================================================================
// ReadingOrderLayer
// ========================================================================
//
// A complete Transformer encoder layer:
//   SelfAttention → dense+dropout+LN(residual) → Linear+GELU → dense+dropout+LN(residual)
//
// Corresponds to the Python-side:
//   PPDocLayoutV2ReadingOrderAttention → PPDocLayoutV2ReadingOrderIntermediate → PPDocLayoutV2ReadingOrderOutput
// ========================================================================

#[derive(Module, Debug)]
pub struct ReadingOrderLayer<B: Backend> {
    // Self-Attention + Output
    pub self_attention: ReadingOrderSelfAttention<B>,
    /// SelfOutput: dense(hidden → hidden)
    pub attn_dense: Linear<B>,
    pub attn_norm: LayerNorm<B>,

    // Intermediate: Linear(hidden → intermediate) + GELU
    pub intermediate_dense: Linear<B>,

    // Output: dense(intermediate → hidden) + LayerNorm
    pub output_dense: Linear<B>,
    pub output_norm: LayerNorm<B>,
}

impl<B: Backend> ReadingOrderLayer<B> {
    /// ReadingOrderLayer forward pass.
    ///
    /// - `hidden_states`: `[B, L, 512]`
    /// - `attention_mask`: `[B, 1, 1, L]` additive mask
    /// - `rel_2d_pos`: `[B, 8, L, L]` 2D spatial attention bias
    /// - Returns: `[B, L, 512]`
    pub fn forward(
        &self,
        hidden_states: &Tensor<B, 3>,
        attention_mask: Option<&Tensor<B, 4>>,
        rel_2d_pos: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        // 1. Self-Attention
        let attn_output = self
            .self_attention
            .forward(hidden_states, attention_mask, rel_2d_pos);

        // 2. SelfOutput: dense → dropout (no-op at inference) → LayerNorm(residual + x)
        let attn_output = self.attn_dense.forward(attn_output);
        let hidden_states = self.attn_norm.forward(attn_output + hidden_states.clone());

        // 3. Intermediate: Linear → GELU
        let intermediate = activation::gelu(self.intermediate_dense.forward(hidden_states.clone()));

        // 4. Output: Linear → dropout (no-op at inference) → LayerNorm(residual + x)
        let output = self.output_dense.forward(intermediate);
        self.output_norm.forward(output + hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct ReadingOrderLayerConfig {
    #[config(default = 512)]
    pub hidden_size: usize,
    #[config(default = 8)]
    pub num_attention_heads: usize,
    #[config(default = 2048)]
    pub intermediate_size: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl ReadingOrderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ReadingOrderLayer<B> {
        ReadingOrderLayer {
            self_attention: ReadingOrderSelfAttentionConfig::new()
                .with_hidden_size(self.hidden_size)
                .with_num_attention_heads(self.num_attention_heads)
                .init(device),
            attn_dense: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            attn_norm: LayerNormConfig::new(self.hidden_size)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            intermediate_dense: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .init(device),
            output_dense: LinearConfig::new(self.intermediate_size, self.hidden_size).init(device),
            output_norm: LayerNormConfig::new(self.hidden_size)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
        }
    }
}

// ========================================================================
// ReadingOrderEncoder
// ========================================================================
//
// 6 ReadingOrderLayers + PositionRelationEmbedding (2D spatial attention bias)
//
// Forward flow:
//   1. bbox → center_width_height → PositionRelationEmbedding → rel_2d_pos
//   2. 6 ReadingOrderLayers(hidden_states, attention_mask, rel_2d_pos)
//   3. Return the last layer's hidden_states
// ========================================================================

#[derive(Module, Debug)]
pub struct ReadingOrderEncoder<B: Backend> {
    pub layers: Vec<ReadingOrderLayer<B>>,
    pub rel_bias_module: PositionRelationEmbedding<B>,
    pub relation_box_dim_min: f32,
}

impl<B: Backend> ReadingOrderEncoder<B> {
    /// Compute the 2D spatial attention bias.
    ///
    /// Converts bboxes from [x1, y1, x2, y2] to [cx, cy, w, h] before computing.
    ///
    /// - `bbox`: `[B, L, 4]` float tensor ([x1, y1, x2, y2] in [0, 1000] range)
    /// - Returns: `[B, num_heads, L, L]`
    fn cal_2d_pos_emb(&self, bbox: &Tensor<B, 3>) -> Tensor<B, 4> {
        let [_batch, _seq_len, _four] = bbox.dims();

        let x_min = bbox.clone().slice(s![.., .., 0..1]);
        let y_min = bbox.clone().slice(s![.., .., 1..2]);
        let x_max = bbox.clone().slice(s![.., .., 2..3]);
        let y_max = bbox.clone().slice(s![.., .., 3..4]);

        // Match Python's 1e-3 clamp by default, but allow the loader to raise
        // it for f16. In true f16 execution, padded or degenerate boxes can
        // make `1000 / 1e-3` overflow before the subsequent logarithm.
        let width = (x_max.clone() - x_min.clone()).clamp_min(self.relation_box_dim_min);
        let height = (y_max.clone() - y_min.clone()).clamp_min(self.relation_box_dim_min);
        let center_x = (x_min + x_max).mul_scalar(0.5f32);
        let center_y = (y_min + y_max).mul_scalar(0.5f32);

        // [cx, cy, w, h]: [B, L, 4]
        let center_wh_bbox = Tensor::cat(vec![center_x, center_y, width, height], 2);

        self.rel_bias_module.forward(&center_wh_bbox)
    }

    /// ReadingOrderEncoder forward pass.
    ///
    /// - `hidden_states`: `[B, L, 512]`
    /// - `bbox`: `[B, L, 4]` float tensor ([x1, y1, x2, y2])
    /// - `attention_mask`: `[B, 1, 1, L]` additive mask
    /// - Returns: `[B, L, 512]`
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        bbox: &Tensor<B, 3>,
        attention_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let rel_2d_pos = self.cal_2d_pos_emb(bbox);

        let mut h = hidden_states;
        for layer in &self.layers {
            h = layer.forward(&h, attention_mask, Some(&rel_2d_pos));
        }
        h
    }
}

#[derive(Config, Debug)]
pub struct ReadingOrderEncoderConfig {
    #[config(default = 512)]
    pub hidden_size: usize,
    #[config(default = 8)]
    pub num_attention_heads: usize,
    #[config(default = 2048)]
    pub intermediate_size: usize,
    #[config(default = 6)]
    pub num_hidden_layers: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    #[config(default = 16)]
    pub relation_bias_embed_dim: usize,
    #[config(default = 10000.0)]
    pub relation_bias_theta: f64,
    #[config(default = 100.0)]
    pub relation_bias_scale: f64,
    #[config(default = 1e-3)]
    pub relation_box_dim_min: f64,
}

impl ReadingOrderEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ReadingOrderEncoder<B> {
        let layers = (0..self.num_hidden_layers)
            .map(|_| {
                ReadingOrderLayerConfig::new()
                    .with_hidden_size(self.hidden_size)
                    .with_num_attention_heads(self.num_attention_heads)
                    .with_intermediate_size(self.intermediate_size)
                    .with_layer_norm_eps(self.layer_norm_eps)
                    .init(device)
            })
            .collect();

        let rel_bias_module = PositionRelationEmbeddingConfig::new()
            .with_embed_dim(self.relation_bias_embed_dim)
            .with_num_attention_heads(self.num_attention_heads)
            .with_theta(self.relation_bias_theta)
            .with_scale(self.relation_bias_scale)
            .init(device);

        ReadingOrderEncoder {
            layers,
            rel_bias_module,
            relation_box_dim_min: self.relation_box_dim_min as f32,
        }
    }
}
