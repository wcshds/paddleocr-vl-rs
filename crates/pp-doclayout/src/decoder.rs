// ========================================================================
// PPDocLayoutV2 RT-DETR Decoder
// ========================================================================
//
// An RT-DETR–style Transformer Decoder comprising:
//
// 1. **Anchor generation**: produces 13 125 candidate anchors across three
//    feature-map levels.
// 2. **Encoder head**: runs classification + bbox regression on the memory
//    features and selects the Top-300 queries.
// 3. **Decoder**: 6 DecoderLayers, each containing:
//    - Self-Attention (Q/K receive the position embedding; V does not)
//    - Multiscale deformable cross-attention (the core operation)
//    - FFN (Linear → ReLU → Linear)
//    - Per-layer iterative bbox refinement
//
// Key idea behind multiscale deformable attention:
//   Instead of attending over all 13 125 tokens globally, each query
//   samples only 4 points near its reference point at every level,
//   yielding 3 × 4 = 12 sampling locations total.  Features at those
//   locations are gathered via bilinear grid_sample and then weighted-
//   summed.  This reduces complexity from O(Q × 13 125) to O(Q × 12).
//
// ```text
// encoder_features [B, 13125, 256]
//         │
//         ▼
//   generate_anchors → [1, 13125, 4]
//   enc_output → enc_score_head + enc_bbox_head
//   Top-300 → reference_points [B, 300, 4]
//             target [B, 300, 256]
//         │
//         ▼
//   ┌──────────────────────────────────────┐
//   │  DecoderLayer ×6                     │
//   │  ┌────────────────────────────────┐  │
//   │  │ Self-Attn (Q/K + pos_embed)   │  │
//   │  │ + LayerNorm                    │  │
//   │  │ Deformable Cross-Attn         │  │
//   │  │ + LayerNorm                    │  │
//   │  │ FFN + LayerNorm                │  │
//   │  └────────────────────────────────┘  │
//   │  bbox_embed[i] → refine ref_points  │
//   │  class_embed[i] → logits            │
//   └──────────────────────────────────────┘
// ```
//
// References:
//   - RT-DETR: https://arxiv.org/abs/2304.08069
//   - Deformable DETR: https://arxiv.org/abs/2010.04159
// ========================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{
        DType, Tensor, TensorData, activation,
        module::attention,
        ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
        s,
    },
};

use crate::reading_order::global_pointer::GlobalPointer;

// ========================================================================
// MLP Prediction Head
// ========================================================================
//
// Used for bbox regression (enc_bbox_head and bbox_embed) as well as
// query_pos_head.
//
// Example – bbox_embed: 256 → 256 → 256 → 4 (3 layers, ReLU after the
//   first two).
// Example – query_pos_head: 4 → 512 → 256 (2 layers, ReLU after the
//   first).
// ========================================================================

#[derive(Module, Debug)]
pub struct MLPPredictionHead<B: Backend> {
    pub layers: Vec<Linear<B>>,
    pub num_layers: usize,
}

impl<B: Backend> MLPPredictionHead<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.num_layers - 1 {
                x = activation::relu(x);
            }
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct MLPPredictionHeadConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    #[config(default = 3)]
    pub num_layers: usize,
}

impl MLPPredictionHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLPPredictionHead<B> {
        let mut dims = vec![self.hidden_dim; self.num_layers - 1];
        dims.push(self.output_dim);
        let mut in_dims = vec![self.input_dim];
        in_dims.extend_from_slice(&dims[..self.num_layers - 1]);

        let layers = in_dims
            .iter()
            .zip(dims.iter())
            .map(|(&in_d, &out_d)| LinearConfig::new(in_d, out_d).init(device))
            .collect();

        MLPPredictionHead {
            layers,
            num_layers: self.num_layers,
        }
    }
}

// ========================================================================
// Multiscale Deformable Attention
// ========================================================================
//
// Core inference pipeline:
//
// 1. Generate sampling_offsets and attention_weights from
//    hidden_states + pos_embed.
// 2. Compute per-level sampling locations from reference_points + offsets.
// 3. Gather features from each level's feature map via grid_sample
//    (bilinear interpolation).
// 4. Weighted-sum the gathered features to produce the output.
//
// Key dimensions:
//   n_heads = 8, head_dim = 32, n_levels = 3, n_points = 4
//   sampling_offsets : [B, Q, 8, 3, 4, 2]  – per-head, per-level, per-point (x, y) offset
//   attention_weights: [B, Q, 8, 12]        – softmax over the 3 × 4 = 12 points
// ========================================================================

#[derive(Module, Debug)]
pub struct MultiscaleDeformableAttention<B: Backend> {
    pub sampling_offsets: Linear<B>,
    pub attention_weights: Linear<B>,
    pub value_proj: Linear<B>,
    pub output_proj: Linear<B>,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_levels: usize,
    pub n_points: usize,
}

impl<B: Backend> MultiscaleDeformableAttention<B> {
    /// Multiscale deformable attention forward pass.
    ///
    /// - `hidden_states`: `[B, num_queries, d_model]` – query vectors.
    /// - `pos_embed`: `[B, num_queries, d_model]` – position embeddings.
    /// - `encoder_hidden_states`: `[B, total_tokens, d_model]` – encoder features.
    /// - `reference_points`: `[B, num_queries, n_levels, 4]` – (cx, cy, w, h).
    /// - `spatial_shapes`: per-level (H, W) list.
    /// - `_level_start_index`: per-level start offsets in the flattened sequence.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        pos_embed: &Tensor<B, 3>,
        encoder_hidden_states: &Tensor<B, 3>,
        reference_points: &Tensor<B, 4>,
        spatial_shapes: &[(usize, usize)],
        // Kept for API symmetry with the Python reference; the actual per-level offset is
        // computed locally via `offset_in_seq` instead.
        _level_start_index: &[usize],
    ) -> Tensor<B, 3> {
        let device = hidden_states.device();
        let [batch_size, num_queries, _] = hidden_states.dims();
        let head_dim = self.d_model / self.n_heads;

        // Q = hidden_states + pos_embed
        let query = hidden_states + pos_embed.clone();

        // Value projection: [B, total_tokens, d_model]
        let value = self.value_proj.forward(encoder_hidden_states.clone());

        // Sampling offsets: [B, Q, n_heads * n_levels * n_points * 2]
        let offsets = self.sampling_offsets.forward(query.clone());
        let offsets = offsets.reshape([
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        ]);

        // Attention weights: [B, Q, n_heads * n_levels * n_points] → softmax → reshape
        let attn_weights = self.attention_weights.forward(query);
        let attn_weights = activation::softmax(
            attn_weights.reshape([
                batch_size,
                num_queries,
                self.n_heads,
                self.n_levels * self.n_points,
            ]),
            3,
        );
        let attn_weights = attn_weights.reshape([
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
        ]);

        // Check whether the last dim of reference_points is 2 (center only) or 4 (center+size).
        let num_coordinates = reference_points.dims()[3];

        // Per-level grid_sample loop.
        let mut sampled_values = Vec::with_capacity(self.n_levels);

        let mut offset_in_seq = 0usize;
        for level_id in 0..self.n_levels {
            let (height, width) = spatial_shapes[level_id];
            let num_tokens = height * width;

            let value_l = value
                .clone()
                .slice(s![
                    0..batch_size,
                    offset_in_seq..(offset_in_seq + num_tokens),
                    ..
                ])
                .reshape([batch_size, num_tokens, self.n_heads, head_dim])
                .swap_dims(1, 2)
                .reshape([batch_size * self.n_heads, height, width, head_dim])
                .swap_dims(1, 3)
                .swap_dims(2, 3);

            let offsets_l = offsets
                .clone()
                .slice(s![.., .., .., level_id..(level_id + 1), .., ..,])
                .squeeze_dim::<5>(3);

            let sampling_locs = if num_coordinates == 2 {
                let ref_pts = reference_points
                    .clone()
                    .slice(s![.., .., level_id..(level_id + 1), 0..2])
                    .squeeze_dim::<3>(2);

                let normalizer_data = vec![width as f32, height as f32];
                let dtype = reference_points.dtype();
                let offset_normalizer = Tensor::<B, 1>::from_data(
                    TensorData::new(normalizer_data, [2]).convert_dtype(dtype),
                    (&device, dtype),
                );

                let ref_pts_expanded = ref_pts.unsqueeze_dim::<4>(2).unsqueeze_dim::<5>(3);
                ref_pts_expanded + offsets_l / offset_normalizer.reshape([1, 1, 1, 1, 2])
            } else {
                let ref_center = reference_points
                    .clone()
                    .slice(s![.., .., level_id..(level_id + 1), 0..2])
                    .squeeze_dim::<3>(2);
                let ref_size = reference_points
                    .clone()
                    .slice(s![.., .., level_id..(level_id + 1), 2..4])
                    .squeeze_dim::<3>(2);

                let center_exp = ref_center.unsqueeze_dim::<4>(2).unsqueeze_dim::<5>(3);
                let size_exp = ref_size.unsqueeze_dim::<4>(2).unsqueeze_dim::<5>(3);

                center_exp
                    + offsets_l
                        .div_scalar(self.n_points as f32)
                        .mul(size_exp)
                        .mul_scalar(0.5f32)
            };

            // Map sampling locations from [0, 1] to [-1, 1] for grid_sample.
            let sampling_grids = sampling_locs.mul_scalar(2.0f32).add_scalar(-1.0f32);

            // grid_sample expects: value [B*n_heads, head_dim, H, W], grid [B*n_heads, Q, n_points, 2]
            let grid = sampling_grids
                .swap_dims(1, 2) // [B, n_heads, Q, n_points, 2]
                .reshape([batch_size * self.n_heads, num_queries, self.n_points, 2]);

            let sampled = value_l.grid_sample_2d(
                grid,
                GridSampleOptions::new(InterpolateMode::Bilinear)
                    .with_padding_mode(GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
            // sampled: [B*n_heads, head_dim, Q, n_points]

            sampled_values.push(sampled);
            offset_in_seq += num_tokens;
        }

        // Stack all levels: [B*n_heads, head_dim, Q, n_levels, n_points].
        // We concatenate along dim 3 (before the n_points axis) so that the
        // [n_levels, n_points] layout matches the attention_weights ordering.
        // (Python equivalent: torch.stack(list, dim=-2) stacks at the second-to-last dim.)
        let stacked = Tensor::cat(
            sampled_values
                .into_iter()
                .map(|s| s.unsqueeze_dim::<5>(3))
                .collect::<Vec<_>>(),
            3,
        );
        // [B*n_heads, head_dim, Q, n_levels, n_points] → [B*n_heads, head_dim, Q, n_levels*n_points]
        let stacked = stacked.reshape([
            batch_size * self.n_heads,
            head_dim,
            num_queries,
            self.n_levels * self.n_points,
        ]);

        // attention_weights: [B, Q, n_heads, n_levels, n_points]
        // → [B, n_heads, Q, n_levels*n_points] → [B*n_heads, 1, Q, n_levels*n_points]
        let weights = attn_weights
            .swap_dims(1, 2) // [B, n_heads, Q, n_levels, n_points]
            .reshape([
                batch_size * self.n_heads,
                1,
                num_queries,
                self.n_levels * self.n_points,
            ]);

        // Weighted sum: [B*n_heads, head_dim, Q, 12] * [B*n_heads, 1, Q, 12] → sum → [B*n_heads, head_dim, Q]
        let output = (stacked * weights).sum_dim(3);
        // output: [B*n_heads, head_dim, Q] → [B, n_heads*head_dim, Q] → [B, Q, d_model]
        let output = output
            .reshape([batch_size, self.n_heads * head_dim, num_queries])
            .swap_dims(1, 2); // [B, Q, d_model]

        self.output_proj.forward(output)
    }
}

#[derive(Config, Debug)]
pub struct MultiscaleDeformableAttentionConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 3)]
    pub n_levels: usize,
    #[config(default = 4)]
    pub n_points: usize,
}

impl MultiscaleDeformableAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiscaleDeformableAttention<B> {
        MultiscaleDeformableAttention {
            sampling_offsets: LinearConfig::new(
                self.d_model,
                self.n_heads * self.n_levels * self.n_points * 2,
            )
            .init(device),
            attention_weights: LinearConfig::new(
                self.d_model,
                self.n_heads * self.n_levels * self.n_points,
            )
            .init(device),
            value_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            output_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_levels: self.n_levels,
            n_points: self.n_points,
        }
    }
}

// ========================================================================
// Self-Attention (for Decoder)
// ========================================================================
//
// The decoder's self-attention layer.  Q and K receive the position
// embedding; V does not.
// Scaled dot-product attention (is_causal = false for bidirectional).
// ========================================================================

#[derive(Module, Debug)]
pub struct DecoderSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> DecoderSelfAttention<B> {
    /// Self-Attention: Q=K=(hidden+pos), V=hidden
    pub fn forward(&self, hidden_states: Tensor<B, 3>, pos_embed: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [bs, seq_len, _] = hidden_states.dims();
        let qk_input = hidden_states.clone() + pos_embed.clone();

        let q = self
            .q_proj
            .forward(qk_input.clone())
            .reshape([bs, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(qk_input)
            .reshape([bs, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(hidden_states)
            .reshape([bs, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let attn = attention(q, k, v, None, None, Default::default());
        let attn = attn
            .swap_dims(1, 2)
            .reshape([bs, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn)
    }
}

#[derive(Config, Debug)]
pub struct DecoderSelfAttentionConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub n_heads: usize,
}

impl DecoderSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderSelfAttention<B> {
        let head_dim = self.d_model / self.n_heads;
        DecoderSelfAttention {
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            o_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            num_heads: self.n_heads,
            head_dim,
        }
    }
}

// ========================================================================
// Decoder Layer
// ========================================================================

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub self_attn: DecoderSelfAttention<B>,
    pub self_attn_layer_norm: LayerNorm<B>,
    pub cross_attn: MultiscaleDeformableAttention<B>,
    pub cross_attn_layer_norm: LayerNorm<B>,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> DecoderLayer<B> {
    /// DecoderLayer forward pass (post-norm residual).
    ///
    /// - `hidden_states`: `[B, 300, 256]`
    /// - `pos_embed`: `[B, 300, 256]` – projected from reference_points.
    /// - `encoder_hidden_states`: `[B, 13125, 256]`
    /// - `reference_points`: `[B, 300, 1, 4]` – expanded to `[B, 300, n_levels, 4]` internally.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        pos_embed: &Tensor<B, 3>,
        encoder_hidden_states: &Tensor<B, 3>,
        reference_points: &Tensor<B, 4>,
        spatial_shapes: &[(usize, usize)],
        level_start_index: &[usize],
    ) -> Tensor<B, 3> {
        // 1. Self-Attention + residual + LayerNorm
        let residual = hidden_states.clone();
        let x = self.self_attn.forward(hidden_states, pos_embed);
        let x = self.self_attn_layer_norm.forward(residual + x);

        // 2. Cross-Attention (Deformable) + residual + LayerNorm
        let residual = x.clone();
        let x = self.cross_attn.forward(
            x,
            pos_embed,
            encoder_hidden_states,
            reference_points,
            spatial_shapes,
            level_start_index,
        );
        let x = self.cross_attn_layer_norm.forward(residual + x);

        // 3. FFN (Linear→ReLU→Linear) + residual + LayerNorm
        let residual = x.clone();
        let x = activation::relu(self.fc1.forward(x));
        let x = self.fc2.forward(x);
        self.final_layer_norm.forward(residual + x)
    }
}

#[derive(Config, Debug)]
pub struct DecoderLayerConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 1024)]
    pub ffn_dim: usize,
    #[config(default = 3)]
    pub n_levels: usize,
    #[config(default = 4)]
    pub n_points: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl DecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderLayer<B> {
        DecoderLayer {
            self_attn: DecoderSelfAttentionConfig::new()
                .with_d_model(self.d_model)
                .with_n_heads(self.n_heads)
                .init(device),
            self_attn_layer_norm: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            cross_attn: MultiscaleDeformableAttentionConfig::new()
                .with_d_model(self.d_model)
                .with_n_heads(self.n_heads)
                .with_n_levels(self.n_levels)
                .with_n_points(self.n_points)
                .init(device),
            cross_attn_layer_norm: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            fc1: LinearConfig::new(self.d_model, self.ffn_dim).init(device),
            fc2: LinearConfig::new(self.ffn_dim, self.d_model).init(device),
            final_layer_norm: LayerNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
        }
    }
}

// ========================================================================
// PPDocLayoutV2Decoder
// ========================================================================

#[derive(Module, Debug)]
pub struct PPDocLayoutV2Decoder<B: Backend> {
    pub layers: Vec<DecoderLayer<B>>,
    /// Projects reference_points (4D) into query position embeddings (d_model).
    /// MLP: 4 → 2*d_model → d_model
    pub query_pos_head: MLPPredictionHead<B>,
    /// Per-layer classification head: Linear(d_model → num_labels).
    pub class_embed: Vec<Linear<B>>,
    /// Per-layer bbox regression head: MLP(d_model → d_model → d_model → 4).
    pub bbox_embed: Vec<MLPPredictionHead<B>>,
}

/// Output bundle returned by the decoder.
pub struct DecoderOutput<B: Backend> {
    /// Hidden states from the last decoder layer: `[B, 300, 256]`.
    pub last_hidden_state: Tensor<B, 3>,
    /// Final-layer reference points: `[B, 300, 4]`.
    pub last_reference_points: Tensor<B, 3>,
    /// Final-layer classification logits: `[B, 300, num_labels]`.
    pub last_logits: Tensor<B, 3>,
    /// Stacked intermediate reference points: `[B, 6, 300, 4]`.
    pub intermediate_reference_points: Tensor<B, 4>,
    /// Stacked intermediate classification logits: `[B, 6, 300, num_labels]`.
    pub intermediate_logits: Tensor<B, 4>,
}

impl<B: Backend> PPDocLayoutV2Decoder<B> {
    /// Decoder forward pass.
    ///
    /// - `target`: `[B, 300, 256]` – query embeddings (from encoder top-k or learnable weights).
    /// - `reference_points_unact`: `[B, 300, 4]` – pre-sigmoid reference points.
    /// - `encoder_hidden_states`: `[B, 13125, 256]` – flattened encoder features.
    /// - `spatial_shapes`: per-level (H, W).
    /// - `level_start_index`: per-level start offsets in the flattened sequence.
    pub fn forward(
        &self,
        target: Tensor<B, 3>,
        reference_points_unact: Tensor<B, 3>,
        encoder_hidden_states: &Tensor<B, 3>,
        spatial_shapes: &[(usize, usize)],
        level_start_index: &[usize],
    ) -> DecoderOutput<B> {
        // sigmoid(reference_points): [B, 300, 4]
        let mut reference_points = activation::sigmoid(reference_points_unact);
        let mut hidden_states = target;

        let mut all_ref_points = Vec::with_capacity(self.layers.len());
        let mut all_logits = Vec::with_capacity(self.layers.len());
        let mut last_logits = None;

        for (idx, layer) in self.layers.iter().enumerate() {
            let pos_embed = self.query_pos_head.forward(reference_points.clone());

            let n_levels = spatial_shapes.len();
            let ref_pts_input = reference_points
                .clone()
                .unsqueeze_dim::<4>(2)
                .repeat_dim(2, n_levels);

            hidden_states = layer.forward(
                hidden_states,
                &pos_embed,
                encoder_hidden_states,
                &ref_pts_input,
                spatial_shapes,
                level_start_index,
            );

            let predicted_corners = self.bbox_embed[idx].forward(hidden_states.clone());
            let new_ref =
                activation::sigmoid(predicted_corners + inverse_sigmoid(reference_points.clone()));
            reference_points = new_ref.clone();

            all_ref_points.push(new_ref.unsqueeze_dim::<4>(1));

            let logits = self.class_embed[idx].forward(hidden_states.clone());
            last_logits = Some(logits.clone());
            all_logits.push(logits.unsqueeze_dim::<4>(1));
        }

        // Stack across layers: [B, 6, 300, 4] and [B, 6, 300, num_labels].
        let last_reference_points = reference_points;
        let last_logits = last_logits.expect("decoder should have at least one layer");
        let intermediate_reference_points = Tensor::cat(all_ref_points, 1);
        let intermediate_logits = Tensor::cat(all_logits, 1);

        DecoderOutput {
            last_hidden_state: hidden_states,
            last_reference_points,
            last_logits,
            intermediate_reference_points,
            intermediate_logits,
        }
    }
}

#[derive(Config, Debug)]
pub struct PPDocLayoutV2DecoderConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 1024)]
    pub ffn_dim: usize,
    #[config(default = 6)]
    pub num_layers: usize,
    #[config(default = 3)]
    pub n_levels: usize,
    #[config(default = 4)]
    pub n_points: usize,
    #[config(default = 25)]
    pub num_labels: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl PPDocLayoutV2DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PPDocLayoutV2Decoder<B> {
        let layers = (0..self.num_layers)
            .map(|_| {
                DecoderLayerConfig::new()
                    .with_d_model(self.d_model)
                    .with_n_heads(self.n_heads)
                    .with_ffn_dim(self.ffn_dim)
                    .with_n_levels(self.n_levels)
                    .with_n_points(self.n_points)
                    .with_layer_norm_eps(self.layer_norm_eps)
                    .init(device)
            })
            .collect();

        let query_pos_head = MLPPredictionHeadConfig::new(4, 2 * self.d_model, self.d_model)
            .with_num_layers(2)
            .init(device);

        let class_embed = (0..self.num_layers)
            .map(|_| LinearConfig::new(self.d_model, self.num_labels).init(device))
            .collect();

        let bbox_embed = (0..self.num_layers)
            .map(|_| MLPPredictionHeadConfig::new(self.d_model, self.d_model, 4).init(device))
            .collect();

        PPDocLayoutV2Decoder {
            layers,
            query_pos_head,
            class_embed,
            bbox_embed,
        }
    }
}

// ========================================================================
// PPDocLayoutV3Decoder
// ========================================================================
//
// V3 uses the same RT-DETR decoder layers as V2, but its prediction heads are
// tied to the encoder heads and it adds query-specific mask/order heads. During
// inference we keep only the final layer outputs to avoid retaining six large
// `[B, Q, 200, 200]` mask tensors.
// ========================================================================

#[derive(Module, Debug)]
pub struct PPDocLayoutV3Decoder<B: Backend> {
    pub layers: Vec<DecoderLayer<B>>,
    pub query_pos_head: MLPPredictionHead<B>,
    pub num_queries: usize,
}

pub struct V3DecoderOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
    pub pred_boxes: Tensor<B, 3>,
    pub order_logits: Tensor<B, 3>,
    pub out_masks: Tensor<B, 4>,
}

impl<B: Backend> PPDocLayoutV3Decoder<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        target: Tensor<B, 3>,
        reference_points_unact: Tensor<B, 3>,
        encoder_hidden_states: &Tensor<B, 3>,
        spatial_shapes: &[(usize, usize)],
        level_start_index: &[usize],
        class_head: &Linear<B>,
        bbox_head: &MLPPredictionHead<B>,
        order_head: &[Linear<B>],
        global_pointer: &GlobalPointer<B>,
        mask_query_head: &MLPPredictionHead<B>,
        decoder_norm: &LayerNorm<B>,
        mask_feat: &Tensor<B, 4>,
    ) -> V3DecoderOutput<B> {
        let [batch_size, _num_queries, _] = target.dims();
        let [_mask_b, mask_channels, mask_h, mask_w] = mask_feat.dims();
        let mask_feat_flat =
            mask_feat
                .clone()
                .reshape([batch_size, mask_channels, mask_h * mask_w]);

        let mut reference_points = activation::sigmoid(reference_points_unact);
        let mut hidden_states = target;

        let mut last_logits = None;
        let mut last_pred_boxes = None;
        let mut last_order_logits = None;
        let mut last_masks = None;

        for (idx, layer) in self.layers.iter().enumerate() {
            let pos_embed = self.query_pos_head.forward(reference_points.clone());
            let n_levels = spatial_shapes.len();
            let ref_pts_input = reference_points
                .clone()
                .unsqueeze_dim::<4>(2)
                .repeat_dim(2, n_levels);

            hidden_states = layer.forward(
                hidden_states,
                &pos_embed,
                encoder_hidden_states,
                &ref_pts_input,
                spatial_shapes,
                level_start_index,
            );

            // V3 ties the decoder bbox/class heads to the encoder heads in the
            // HuggingFace model; the safetensors file stores only the encoder
            // copy, so the Rust model explicitly reuses those modules here.
            let predicted_corners = bbox_head.forward(hidden_states.clone());
            let new_reference_points =
                activation::sigmoid(predicted_corners + inverse_sigmoid(reference_points.clone()));
            reference_points = new_reference_points.clone();

            let out_query = decoder_norm.forward(hidden_states.clone());
            let logits = class_head.forward(out_query.clone());

            let mask_query_embed = mask_query_head.forward(out_query.clone());
            let out_mask = mask_query_embed.matmul(mask_feat_flat.clone()).reshape([
                batch_size,
                self.num_queries,
                mask_h,
                mask_w,
            ]);

            let valid_query = if out_query.dims()[1] > self.num_queries {
                let start = out_query.dims()[1] - self.num_queries;
                out_query.slice(s![.., start.., ..])
            } else {
                out_query
            };
            let order_query = order_head[idx].forward(valid_query);
            let order_logits = global_pointer.forward(order_query);

            last_logits = Some(logits);
            last_pred_boxes = Some(new_reference_points);
            last_order_logits = Some(order_logits);
            last_masks = Some(out_mask);
        }

        V3DecoderOutput {
            last_hidden_state: hidden_states,
            logits: last_logits.expect("decoder has no layers"),
            pred_boxes: last_pred_boxes.expect("decoder has no layers"),
            order_logits: last_order_logits.expect("decoder has no layers"),
            out_masks: last_masks.expect("decoder has no layers"),
        }
    }
}

#[derive(Config, Debug)]
pub struct PPDocLayoutV3DecoderConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 1024)]
    pub ffn_dim: usize,
    #[config(default = 6)]
    pub num_layers: usize,
    #[config(default = 3)]
    pub n_levels: usize,
    #[config(default = 4)]
    pub n_points: usize,
    #[config(default = 300)]
    pub num_queries: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl PPDocLayoutV3DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PPDocLayoutV3Decoder<B> {
        let layers = (0..self.num_layers)
            .map(|_| {
                DecoderLayerConfig::new()
                    .with_d_model(self.d_model)
                    .with_n_heads(self.n_heads)
                    .with_ffn_dim(self.ffn_dim)
                    .with_n_levels(self.n_levels)
                    .with_n_points(self.n_points)
                    .with_layer_norm_eps(self.layer_norm_eps)
                    .init(device)
            })
            .collect();

        let query_pos_head = MLPPredictionHeadConfig::new(4, 2 * self.d_model, self.d_model)
            .with_num_layers(2)
            .init(device);

        PPDocLayoutV3Decoder {
            layers,
            query_pos_head,
            num_queries: self.num_queries,
        }
    }
}

// ========================================================================
// Helper functions
// ========================================================================

/// inverse_sigmoid: log(clamp(x, eps) / clamp(1-x, eps))
///
/// The mathematical inverse of the sigmoid function.  Used during iterative
/// bbox refinement to convert reference points from sigmoid-space back into
/// logit-space.
pub fn inverse_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let eps = 1e-5f32;
    let x = x.clamp(eps, 1.0 - eps);
    let x1 = x.clone();
    let x2 = x.mul_scalar(-1.0f32).add_scalar(1.0f32); // 1 - x
    x1.log() - x2.log()
}

/// Generate multiscale anchors.
///
/// Produces uniformly distributed anchor boxes across the three feature
/// levels.  Each anchor is in (cx, cy, w, h) format, normalised to [0, 1].
///
/// - `spatial_shapes`: e.g. `[(100, 100), (50, 50), (25, 25)]`.
/// - `grid_size`: base anchor size (0.05).
///
/// Returns:
/// - `anchors`: `[1, 13125, 4]` in inverse-sigmoid (logit) space.
/// - `valid_mask`: `[1, 13125, 1]` – boolean mask for valid anchors.
pub fn generate_anchors<B: Backend>(
    spatial_shapes: &[(usize, usize)],
    grid_size: f32,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let target_dtype = Tensor::<B, 1>::zeros([1], device).dtype();
    generate_anchors_with_dtype::<B>(spatial_shapes, grid_size, device, target_dtype)
}

pub fn generate_anchors_with_dtype<B: Backend>(
    spatial_shapes: &[(usize, usize)],
    grid_size: f32,
    device: &B::Device,
    target_dtype: DType,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let mut all_anchors = Vec::new();

    for (level, &(height, width)) in spatial_shapes.iter().enumerate() {
        let level_size = grid_size * (2.0f32).powi(level as i32);
        let mut level_data = Vec::with_capacity(height * width * 4);

        for y in 0..height {
            for x in 0..width {
                let cx = (x as f32 + 0.5) / width as f32;
                let cy = (y as f32 + 0.5) / height as f32;
                level_data.push(cx);
                level_data.push(cy);
                level_data.push(level_size);
                level_data.push(level_size);
            }
        }

        all_anchors.extend(level_data);
    }

    let total_anchors: usize = spatial_shapes.iter().map(|(h, w)| h * w).sum();
    let anchors = Tensor::<B, 3>::from_data(
        TensorData::new(all_anchors, [1, total_anchors, 4]).convert_dtype(target_dtype),
        (device, target_dtype),
    );

    // valid_mask: all 4 coordinates must lie within (0.01, 0.99).
    let eps = 0.01f32;
    let gt_eps = anchors.clone().greater_elem(eps);
    let lt_1_eps = anchors.clone().lower_elem(1.0 - eps);
    let valid_4 = gt_eps.bool_and(lt_1_eps); // [1, N, 4]

    // An anchor is valid only if all 4 dimensions pass.
    let valid_mask = valid_4
        .clone()
        .slice(s![.., .., 0..1])
        .bool_and(valid_4.clone().slice(s![.., .., 1..2]))
        .bool_and(valid_4.clone().slice(s![.., .., 2..3]))
        .bool_and(valid_4.slice(s![.., .., 3..4]));

    // log(x / (1-x)): inverse sigmoid
    let anchors_logit = inverse_sigmoid(anchors).cast(target_dtype);

    // Set invalid positions to a large positive logit so they are never selected.
    //
    // This must stay finite for half precision backends. Using `f32::MAX` is
    // harmless in pure f32, but it becomes `inf` when materialized by f16/bf16
    // backends and can later spread through decoder arithmetic as NaN.
    let invalid_anchor_logit = 1.0e4f32;
    let valid_float = valid_mask.clone().float().cast(target_dtype);
    let invalid_float = valid_float
        .clone()
        .mul_scalar(-1.0f32)
        .add_scalar(1.0f32)
        .cast(target_dtype); // 1 - valid
    let anchors_logit = anchors_logit * valid_float
        + invalid_float
            .mul_scalar(invalid_anchor_logit)
            .cast(target_dtype);

    (anchors_logit, valid_mask.float().cast(target_dtype))
}
