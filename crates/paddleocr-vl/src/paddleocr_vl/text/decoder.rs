//! Core decoder components for the PaddleOCR text model (Ernie4.5)
//!
//! This module implements every sub-layer of the Transformer Decoder-only
//! architecture, corresponding to the Python-side `Ernie4_5Model` (wrapped at
//! inference time by `PaddleOCRTextModel`).
//!
//! ## Overall Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │  PaddleOcrTextDecoderLayer × num_hidden_layers (default 18)   │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │  input_layernorm (RMSNorm)                              │  │
//! │  │  ↓                                                      │  │
//! │  │  self_attn (Grouped Query Attention + M-RoPE)           │  │
//! │  │  ↓  + residual                                          │  │
//! │  │  post_attention_layernorm (RMSNorm)                     │  │
//! │  │  ↓                                                      │  │
//! │  │  mlp (SwiGLU: silu(gate_proj(x)) * up_proj(x) → down)  │  │
//! │  │  ↓  + residual                                          │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Design Points
//!
//! - **RMSNorm**: Compared to LayerNorm, RMSNorm skips mean-centering and
//!   normalizes only by the root-mean-square. This is more efficient at
//!   inference and is the mainstream choice in modern LLMs (LLaMA, Qwen, etc.).
//!
//! - **Grouped Query Attention (GQA)**: The number of query heads (16) exceeds
//!   the number of KV heads (2). Every 8 query heads share a single KV head
//!   pair, drastically reducing KV-cache memory overhead.
//!
//! - **Multimodal RoPE (M-RoPE)**: 3D rotary position embeddings that split
//!   `head_dim` into three segments `[16, 24, 24]` (temporal, height, width).
//!   Each segment uses position IDs from a different spatial axis, enabling the
//!   model to perceive the 2D spatial structure of image patches.
//!   For plain text, all three axes share identical position IDs, so M-RoPE
//!   degenerates to standard 1D RoPE.
//!
//! - **SwiGLU MLP**: `SiLU(gate(x)) ⊙ up(x) → down(·)`. Compared to standard
//!   GeLU-MLP, SwiGLU offers better training stability and expressiveness.

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{
        DType, Int, Tensor, activation, backend::Backend, module::attention,
        ops::AttentionModuleOptions, s,
    },
};

// ========================================================================
// Rotary Position Embedding output containers
// ========================================================================

/// Pre-computed cos/sin position-embedding tensors shared by all decoder layers.
pub struct TextPositionEmbeddings<B: Backend> {
    /// Shape: [batch_size, 1, seq_len, head_dim].
    /// dim=1 is kept at size 1 so it broadcasts across all attention heads.
    pub cos: Tensor<B, 4>,
    /// Shape: [batch_size, 1, seq_len, head_dim].
    pub sin: Tensor<B, 4>,
}

/// Per-layer KV cache used during autoregressive generation to avoid
/// redundant recomputation of keys and values for past tokens.
pub struct LayerKVCache<B: Backend> {
    /// [batch_size, num_kv_heads, cached_len, head_dim]
    pub key: Tensor<B, 4>,
    /// [batch_size, num_kv_heads, cached_len, head_dim]
    pub value: Tensor<B, 4>,
}

// ========================================================================
// Multimodal Rotary Position Embedding (M-RoPE)
// ========================================================================
//
// M-RoPE was introduced by Qwen2-VL as a 3D rotary position encoding scheme.
// Standard RoPE uses a single dimension of position IDs; M-RoPE uses three:
//   - Temporal: video-frame time position
//   - Height:   vertical position of image patches
//   - Width:    horizontal position of image patches
//
// head_dim is split into three segments (specified by mrope_section).
// Each segment computes rotation frequencies from a different axis's position IDs.
// For example, with mrope_section = [16, 24, 24]:
//   - head_dim[0..16]  + head_dim[64..80]:  uses temporal-axis positions
//   - head_dim[16..40] + head_dim[80..104]: uses height-axis positions
//   - head_dim[40..64] + head_dim[104..128]: uses width-axis positions
//
// For plain-text input, all three axes share identical position IDs
// ([0, 1, 2, …, seq_len-1]), so M-RoPE degenerates to standard 1D RoPE.
//
// Mathematical derivation:
//   inv_freq[i] = 1 / (θ ^ (2i / d)),  where θ = rope_theta, d = head_dim
//   freqs[pos, i] = pos × inv_freq[i]
//   RoPE(x, pos) = x * cos(freqs) + rotate_half(x) * sin(freqs)
//
// References:
//   - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
//   - Qwen2-VL: https://qwenlm.github.io/blog/qwen2-vl/
// ========================================================================

#[derive(Module, Debug)]
pub struct TextRotaryEmbedding<B: Backend> {
    /// Inverse-frequency vector. Shape: [head_dim / 2].
    /// inv_freq[i] = 1 / (theta ^ (2i / head_dim)).
    /// Not a learnable parameter — deterministically computed from rope_theta and head_dim.
    pub inv_freq: Tensor<B, 1>,
    /// M-RoPE segment sizes for each of the three axes (temporal, height, width).
    /// Their sum must equal head_dim / 2.
    pub mrope_section_t: usize,
    pub mrope_section_h: usize,
    pub mrope_section_w: usize,
    /// Attention head dimension.
    pub head_dim: usize,
}

impl<B: Backend> TextRotaryEmbedding<B> {
    /// Construct the rotary position embedding module.
    ///
    /// - `head_dim`: attention head dimension (default 128)
    /// - `rope_theta`: RoPE base frequency (default 500000.0)
    /// - `mrope_section`: M-RoPE three-axis segment widths [temporal, height, width]
    pub fn new(
        head_dim: usize,
        rope_theta: f64,
        mrope_section: [usize; 3],
        device: &B::Device,
    ) -> Self {
        // Compute inv_freq = exp(-2i / d * ln(theta)),
        // which is equivalent to 1 / (theta ^ (2i / d)) but more numerically stable
        // when computed in log-space.
        let inv_freq = Tensor::<B, 1, Int>::arange_step(0..head_dim as i64, 2, device)
            .float()
            .cast(DType::F32)
            .div_scalar(head_dim as f32)
            .mul_scalar(-(rope_theta as f32).ln())
            .exp();

        Self {
            inv_freq,
            mrope_section_t: mrope_section[0],
            mrope_section_h: mrope_section[1],
            mrope_section_w: mrope_section[2],
            head_dim,
        }
    }

    /// Compute M-RoPE cos/sin from 3D position IDs.
    ///
    /// - `position_ids`: Int tensor of shape `[3, batch_size, seq_len]`
    ///   - position_ids[0]: temporal-axis positions (0 for images; monotonically increasing for text)
    ///   - position_ids[1]: height-axis positions (patch row index for images)
    ///   - position_ids[2]: width-axis positions (patch column index for images)
    ///
    /// Returns `TextPositionEmbeddings` where cos/sin have shape
    /// `[batch_size, 1, seq_len, head_dim]`, ready to be element-wise multiplied
    /// with Q/K (broadcasting across the heads dimension).
    pub fn forward(&self, position_ids: &Tensor<B, 3, Int>) -> TextPositionEmbeddings<B> {
        let sections = [
            self.mrope_section_t,
            self.mrope_section_h,
            self.mrope_section_w,
        ];
        let half_dim = self.head_dim / 2;
        let mut offset = 0usize;
        let mut freqs_parts = Vec::with_capacity(3);

        for (axis, &section_size) in sections.iter().enumerate() {
            // Extract the inv_freq slice for this segment: [section_size].
            // e.g. axis=0 (temporal): inv_freq[0..16]
            //      axis=1 (height):   inv_freq[16..40]
            //      axis=2 (width):    inv_freq[40..64]
            let inv_freq_seg = self
                .inv_freq
                .clone()
                .slice(s![offset..(offset + section_size)]);

            // Extract position IDs for this axis: [3, bs, seq] → [1, bs, seq] → [bs, seq]
            let pos = position_ids
                .clone()
                .slice(s![axis..(axis + 1), .., ..])
                .squeeze_dim::<2>(0) // [batch_size, seq_len] (Int)
                .float()
                .cast(DType::F32); // keep rotary frequencies in f32 before applying to q/k

            // Outer product: pos[b, s] × inv_freq[d] → freqs[b, s, d]
            //   pos_expanded:      [bs, seq, 1]
            //   inv_freq_expanded: [1, 1, section_size]
            //   result:            [bs, seq, section_size] (broadcast multiplication)
            let pos_expanded = pos.unsqueeze_dim::<3>(2);
            let inv_freq_expanded = inv_freq_seg.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
            let freqs_seg = pos_expanded * inv_freq_expanded;

            freqs_parts.push(freqs_seg);
            offset += section_size;
        }

        debug_assert_eq!(
            offset, half_dim,
            "Sum of mrope_section segments ({offset}) must equal head_dim/2 ({half_dim})"
        );

        // Concatenate three frequency segments → [bs, seq, half_dim]
        let freqs_half = Tensor::cat(freqs_parts, 2);

        // Duplicate to cover the full head_dim → [bs, seq, head_dim].
        // This mirrors the Python-side `emb = torch.cat((freqs, freqs), dim=-1)`.
        // The rotate_half operation needs the first half paired with the second half.
        let emb = Tensor::cat(vec![freqs_half.clone(), freqs_half], 2);

        // Compute cos/sin and insert a heads dimension → [bs, 1, seq, head_dim].
        // dim=1 is size 1 so it broadcasts to all attention heads.
        TextPositionEmbeddings {
            cos: emb.clone().cos().unsqueeze_dim::<4>(1),
            sin: emb.sin().unsqueeze_dim::<4>(1),
        }
    }
}

// ========================================================================
// Helper functions
// ========================================================================

/// Rotate half of the dimensions (rotate_half).
///
/// Splits head_dim into the first half and the second half, then swaps them
/// with a sign flip on the second half:
///   rotate_half([x1, x2]) = [-x2, x1]
///
/// This is a core operation of RoPE, making the embedding equivalent to a
/// rotation in the complex plane.
///
/// Input/Output shape: `[batch_size, num_heads, seq_len, head_dim]`
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, _, head_dim] = x.dims();
    let half = head_dim / 2;

    let x1 = x.clone().slice(s![.., .., .., 0..half]); // first half
    let x2 = x.slice(s![.., .., .., half..head_dim]); // second half

    // [-x2, x1]
    Tensor::cat(vec![x2 * (-1), x1], 3)
}

/// Apply rotary position embeddings to query and key tensors.
///
/// RoPE formula:
///   q_embed = q * cos + rotate_half(q) * sin
///   k_embed = k * cos + rotate_half(k) * sin
///
/// The cos/sin tensors already incorporate M-RoPE axis selection.
///
/// - `q`: `[bs, num_heads, seq, head_dim]`
/// - `k`: `[bs, num_kv_heads, seq, head_dim]`
/// - `cos`, `sin`: `[bs, 1, seq, head_dim]` (broadcast over the heads dimension)
fn apply_rotary_pos_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let dtype = q.dtype();
    let cos = cos.clone().cast(dtype);
    let sin = sin.clone().cast(dtype);
    let q_embed = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
    let k_embed = k.clone() * cos + rotate_half(k) * sin;
    (q_embed, k_embed)
}

/// Repeat KV heads to match the number of query heads (Grouped Query Attention).
///
/// In GQA, num_attention_heads > num_kv_heads, so each group of query heads
/// shares the same KV head pair.  This function expands KV from
/// `[bs, kv_heads, seq, dim]` to `[bs, q_heads, seq, dim]`.
///
/// Example: num_heads=16, num_kv_heads=2, n_rep=8
///   [bs, 2, seq, dim] → [bs, 16, seq, dim]
///
/// When n_rep=1 (i.e. standard Multi-Head Attention), returns the tensor
/// as-is with zero overhead.
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [bs, num_kv_heads, seq_len, head_dim] = x.dims();

    // [bs, kv_heads, seq, dim]
    //   → unsqueeze → [bs, kv_heads, 1, seq, dim]
    //   → repeat    → [bs, kv_heads, n_rep, seq, dim]
    //   → reshape   → [bs, kv_heads * n_rep, seq, dim]
    x.unsqueeze_dim::<5>(2)
        .repeat(&[1, 1, n_rep, 1, 1])
        .reshape([bs, num_kv_heads * n_rep, seq_len, head_dim])
}

// ========================================================================
// SwiGLU MLP (Gated Linear Unit with SiLU activation)
// ========================================================================
//
// SwiGLU is the feed-forward network variant adopted by PaLM / LLaMA / Qwen:
//
//   output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
//
// where ⊙ denotes element-wise multiplication and SiLU(x) = x · sigmoid(x).
//
// Compared to the standard MLP (ReLU/GeLU + a single linear layer), SwiGLU
// uses two parallel linear projections (gate and up) followed by a gated
// multiplication, which improves model expressiveness and training stability.
//
// Dimension flow:
//   hidden_states: [bs, seq, hidden_size=1024]
//   gate_proj(x):  [bs, seq, intermediate_size=3072]
//   up_proj(x):    [bs, seq, intermediate_size=3072]
//   SiLU(gate) ⊙ up: [bs, seq, 3072]
//   down_proj(·):  [bs, seq, hidden_size=1024]
//
// Reference: Shazeer, "GLU Variants Improve Transformer" (2020)
// ========================================================================

#[derive(Module, Debug)]
pub struct PaddleOcrTextMLP<B: Backend> {
    /// Gate projection: hidden_size → intermediate_size (no bias)
    pub gate_proj: Linear<B>,
    /// Up projection: hidden_size → intermediate_size (no bias)
    pub up_proj: Linear<B>,
    /// Down projection: intermediate_size → hidden_size (no bias)
    pub down_proj: Linear<B>,
}

impl<B: Backend> PaddleOcrTextMLP<B> {
    /// SwiGLU forward pass.
    ///
    /// - Input:  `[batch_size, seq_len, hidden_size]`
    /// - Output: `[batch_size, seq_len, hidden_size]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // gate: SiLU(gate_proj(x)), shape [bs, seq, intermediate_size]
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        // up: up_proj(x), shape [bs, seq, intermediate_size]
        let up = self.up_proj.forward(x);
        // Gated down-projection: down_proj(gate ⊙ up), shape [bs, seq, hidden_size]
        self.down_proj.forward(gate * up)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrTextMLPConfig {
    #[config(default = 1024)]
    pub hidden_size: usize,
    #[config(default = 3072)]
    pub intermediate_size: usize,
    #[config(default = false)]
    pub use_bias: bool,
}

impl PaddleOcrTextMLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrTextMLP<B> {
        PaddleOcrTextMLP {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(self.use_bias)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(self.use_bias)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(self.use_bias)
                .init(device),
        }
    }
}

// ========================================================================
// Grouped Query Attention (GQA)
// ========================================================================
//
// GQA is a compromise between Multi-Head Attention (MHA) and Multi-Query
// Attention (MQA):
//
//   MHA: each attention head has its own Q, K, V  (num_kv_heads == num_heads)
//   MQA: all heads share a single K, V pair       (num_kv_heads == 1)
//   GQA: every G query heads share one K, V pair   (1 < num_kv_heads < num_heads)
//
// This model: num_heads=16, num_kv_heads=2, head_dim=128
//   → Q projection: [1024 → 16×128 = 2048]
//   → K projection: [1024 → 2×128  = 256]
//   → V projection: [1024 → 2×128  = 256]
//   → O projection: [2048 → 1024]
//   → every 8 query heads share 1 KV head pair
//
// Advantages of GQA:
//   1. KV-cache size is only 1/8 of MHA, relieving memory pressure during generation
//   2. Quality is close to MHA and far better than MQA
//
// Attention computation flow:
//   1. Q/K/V linear projections
//   2. Apply Rotary Position Embedding (RoPE) to Q and K
//   3. Expand KV heads: [bs, kv_heads, seq, dim] → [bs, num_heads, seq, dim]
//   4. Scaled Dot-Product Attention + causal mask
//   5. Output projection
//
// Reference: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
//            Models from Multi-Head Checkpoints" (2023)
// ========================================================================

#[derive(Module, Debug)]
pub struct PaddleOcrTextAttention<B: Backend> {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// num_heads / num_kv_heads — the KV-head repetition factor
    pub num_kv_groups: usize,
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

impl<B: Backend> PaddleOcrTextAttention<B> {
    /// Grouped Query Attention forward pass.
    ///
    /// - `hidden_states`: `[batch_size, seq_len, hidden_size]`
    /// - `position_embeddings`: pre-computed cos/sin of shape `[bs, 1, seq, head_dim]`
    ///
    /// Returns: `[batch_size, seq_len, hidden_size]`
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: &TextPositionEmbeddings<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();

        let query_states = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let key_states = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let value_states = self
            .v_proj
            .forward(hidden_states)
            .reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (query_states, key_states) = apply_rotary_pos_emb(
            query_states,
            key_states,
            &position_embeddings.cos,
            &position_embeddings.sin,
        );

        let key_states = repeat_kv(key_states, self.num_kv_groups);
        let value_states = repeat_kv(value_states, self.num_kv_groups);

        let attn_output = attention(
            query_states,
            key_states,
            value_states,
            None,
            None,
            AttentionModuleOptions {
                is_causal: true,
                ..Default::default()
            },
        );

        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        self.o_proj.forward(attn_output)
    }

    /// Forward pass with KV cache for efficient autoregressive generation.
    ///
    /// Prefill phase: `cache = None` — processes the full input sequence and
    ///   creates the initial cache.  A causal mask is applied internally.
    /// Decode phase:  `cache = Some(prev)` — processes a single new token.
    ///   The mask can be `None` because each new token attends to all past keys.
    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: &TextPositionEmbeddings<B>,
        cache: Option<LayerKVCache<B>>,
    ) -> (Tensor<B, 3>, LayerKVCache<B>) {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let is_prefill = cache.is_none();

        let query_states = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let key_states = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let value_states = self
            .v_proj
            .forward(hidden_states)
            .reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (query_states, key_states) = apply_rotary_pos_emb(
            query_states,
            key_states,
            &position_embeddings.cos,
            &position_embeddings.sin,
        );

        let (key_states, value_states) = match cache {
            Some(c) => (
                Tensor::cat(vec![c.key, key_states], 2),
                Tensor::cat(vec![c.value, value_states], 2),
            ),
            None => (key_states, value_states),
        };

        let new_cache = LayerKVCache {
            key: key_states.clone(),
            value: value_states.clone(),
        };

        let key_states = repeat_kv(key_states, self.num_kv_groups);
        let value_states = repeat_kv(value_states, self.num_kv_groups);

        let attn_output = attention(
            query_states,
            key_states,
            value_states,
            None,
            None,
            AttentionModuleOptions {
                is_causal: is_prefill,
                ..Default::default()
            },
        );

        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        (self.o_proj.forward(attn_output), new_cache)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrTextAttentionConfig {
    #[config(default = 1024)]
    pub hidden_size: usize,
    #[config(default = 16)]
    pub num_attention_heads: usize,
    /// Number of KV heads — the core GQA parameter.
    /// When smaller than num_attention_heads, GQA is activated.
    #[config(default = 2)]
    pub num_key_value_heads: usize,
    #[config(default = 128)]
    pub head_dim: usize,
    #[config(default = false)]
    pub use_bias: bool,
}

impl PaddleOcrTextAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrTextAttention<B> {
        let num_kv_groups = self.num_attention_heads / self.num_key_value_heads;

        PaddleOcrTextAttention {
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            num_kv_groups,
            q_proj: LinearConfig::new(self.hidden_size, self.num_attention_heads * self.head_dim)
                .with_bias(self.use_bias)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, self.num_key_value_heads * self.head_dim)
                .with_bias(self.use_bias)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, self.num_key_value_heads * self.head_dim)
                .with_bias(self.use_bias)
                .init(device),
            o_proj: LinearConfig::new(self.num_attention_heads * self.head_dim, self.hidden_size)
                .with_bias(self.use_bias)
                .init(device),
        }
    }
}

// ========================================================================
// Decoder Layer
// ========================================================================
//
// Computation flow per layer (Pre-Norm architecture):
//
//   residual = hidden_states
//   hidden_states = input_layernorm(hidden_states)     ← RMSNorm
//   hidden_states = self_attn(hidden_states, ...)      ← GQA + M-RoPE
//   hidden_states = residual + hidden_states           ← residual connection
//
//   residual = hidden_states
//   hidden_states = post_attention_layernorm(hidden_states)  ← RMSNorm
//   hidden_states = mlp(hidden_states)                      ← SwiGLU
//   hidden_states = residual + hidden_states                ← residual connection
//
// Pre-Norm (normalize before Attention/MLP) is more favorable for training
// deep models compared to Post-Norm.
// ========================================================================

#[derive(Module, Debug)]
pub struct PaddleOcrTextDecoderLayer<B: Backend> {
    pub self_attn: PaddleOcrTextAttention<B>,
    pub mlp: PaddleOcrTextMLP<B>,
    /// RMSNorm applied before the attention sub-layer
    pub input_layernorm: RmsNorm<B>,
    /// RMSNorm applied before the MLP sub-layer
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> PaddleOcrTextDecoderLayer<B> {
    /// Decoder layer forward pass.
    ///
    /// - `hidden_states`: `[batch_size, seq_len, hidden_size]`
    /// - `position_embeddings`: rotary position embeddings
    ///
    /// Returns: `[batch_size, seq_len, hidden_size]`
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: &TextPositionEmbeddings<B>,
    ) -> Tensor<B, 3> {
        // ---- Self-attention sub-layer (with residual connection) ----
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, position_embeddings);
        let hidden_states = residual + hidden_states;

        // ---- MLP sub-layer (with residual connection) ----
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }

    /// Decoder layer forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        position_embeddings: &TextPositionEmbeddings<B>,
        cache: Option<LayerKVCache<B>>,
    ) -> (Tensor<B, 3>, LayerKVCache<B>) {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let (hidden_states, new_cache) =
            self.self_attn
                .forward_with_cache(hidden_states, position_embeddings, cache);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        (residual + hidden_states, new_cache)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrTextDecoderLayerConfig {
    #[config(default = 1024)]
    pub hidden_size: usize,
    #[config(default = 3072)]
    pub intermediate_size: usize,
    #[config(default = 1e-05)]
    pub rms_norm_eps: f64,
    #[config(default = 16)]
    pub num_attention_heads: usize,
    #[config(default = 2)]
    pub num_key_value_heads: usize,
    #[config(default = 128)]
    pub head_dim: usize,
    #[config(default = false)]
    pub use_bias: bool,
}

impl PaddleOcrTextDecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrTextDecoderLayer<B> {
        PaddleOcrTextDecoderLayer {
            self_attn: PaddleOcrTextAttentionConfig::new()
                .with_hidden_size(self.hidden_size)
                .with_num_attention_heads(self.num_attention_heads)
                .with_num_key_value_heads(self.num_key_value_heads)
                .with_head_dim(self.head_dim)
                .with_use_bias(self.use_bias)
                .init(device),
            mlp: PaddleOcrTextMLPConfig::new()
                .with_hidden_size(self.hidden_size)
                .with_intermediate_size(self.intermediate_size)
                .with_use_bias(self.use_bias)
                .init(device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}
