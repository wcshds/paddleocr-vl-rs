//! PaddleOCR Text Model — Ernie4.5 Decoder-Only Transformer
//!
//! This module corresponds to the Python-side `Ernie4_5Model`, which is wrapped
//! at inference time by `PaddleOCRTextModel`.  It is a standard decoder-only
//! causal language model:
//!
//! ```text
//! input_ids / inputs_embeds
//!     │
//!     ▼
//! ┌──────────────────────┐
//! │  embed_tokens        │   Token embedding look-up: token_id → [hidden_size]
//! └──────┬───────────────┘
//!        │
//!        ▼
//! ┌──────────────────────┐
//! │  rotary_emb          │   Compute M-RoPE cos/sin (shared by all layers)
//! └──────┬───────────────┘
//!        │
//!        ▼
//! ┌──────────────────────┐
//! │  DecoderLayer × 18   │   Self-attention (GQA) + SwiGLU MLP + residual
//! └──────┬───────────────┘
//!        │
//!        ▼
//! ┌──────────────────────┐
//! │  norm (RMSNorm)      │   Final normalization
//! └──────┴───────────────┘
//!        │
//!        ▼
//!    hidden_states: [batch_size, seq_len, hidden_size]
//! ```
//!
//! ## Usage
//!
//! The text model accepts two kinds of input:
//! 1. **Plain text**: pass `input_ids` (a token-ID sequence); the internal
//!    Embedding layer performs the look-up.
//! 2. **Multimodal**: pass `inputs_embeds` (a vector sequence that already has
//!    image embeddings mixed in). The upstream VL model replaces the positions
//!    of image tokens with the Projector's output.
//!
//! `position_ids` is a 3D tensor `[3, batch_size, seq_len]` computed by the
//! upstream model's `get_rope_index()` method, carrying multimodal 3D position
//! information.
//!
//! ## Weight Loading
//!
//! In safetensors the text-model weights are prefixed with `model.`.
//! The following key mappings are applied during loading:
//! ```text
//! model.embed_tokens.weight              → embed_tokens.weight
//! model.layers.N.self_attn.*             → layers.N.self_attn.*
//! model.layers.N.mlp.*                   → layers.N.mlp.*
//! model.layers.N.*_layernorm.weight      → layers.N.*_layernorm.gamma
//! model.norm.weight                      → norm.gamma
//! ```
//! `rotary_emb.inv_freq` is not loaded (it is deterministically computed from
//! rope_theta and head_dim).

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::Int,
};

use crate::paddleocr_vl::text::decoder::{
    PaddleOcrTextDecoderLayer, PaddleOcrTextDecoderLayerConfig, TextRotaryEmbedding,
};

pub mod decoder;

// ========================================================================
// PaddleOCR Text Model (Ernie4_5Model)
// ========================================================================

#[derive(Module, Debug)]
pub struct PaddleOcrTextModel<B: Backend> {
    /// Token embedding layer: vocab_size × hidden_size.
    /// Maps token IDs to dense vector representations.
    pub embed_tokens: Embedding<B>,
    /// Stack of 18 decoder layers.
    pub layers: Vec<PaddleOcrTextDecoderLayer<B>>,
    /// Final RMSNorm normalization.
    pub norm: RmsNorm<B>,
    /// M-RoPE rotary position embedding (computes cos/sin shared by all layers).
    pub rotary_emb: TextRotaryEmbedding<B>,
}

impl<B: Backend> PaddleOcrTextModel<B> {
    /// Text model forward pass.
    ///
    /// # Arguments
    ///
    /// - `input_ids`: `Option<Tensor<B, 2, Int>>`, shape `[batch_size, seq_len]`.
    ///   Token-ID sequence for the plain-text mode.  Mutually exclusive with
    ///   `inputs_embeds`.
    ///
    /// - `inputs_embeds`: `Option<Tensor<B, 3>>`, shape `[batch_size, seq_len, hidden_size]`.
    ///   Pre-computed embeddings (in multimodal mode, image tokens have already
    ///   been replaced with projected visual features).
    ///
    /// - `position_ids`: `Tensor<B, 3, Int>`, shape `[3, batch_size, seq_len]`.
    ///   3D position indices required by M-RoPE:
    ///   - `[0, :, :]` = temporal-axis positions
    ///   - `[1, :, :]` = height-axis positions
    ///   - `[2, :, :]` = width-axis positions
    ///     For plain text all three axes are identical: `[0, 1, 2, ..., seq_len-1]`.
    ///
    /// # Returns
    ///
    /// `Tensor<B, 3>`, shape `[batch_size, seq_len, hidden_size]`.
    /// Hidden states from the last layer after RMSNorm.  The upstream model
    /// can then apply `lm_head` for next-token prediction.
    pub fn forward(
        &self,
        input_ids: Option<Tensor<B, 2, Int>>,
        inputs_embeds: Option<Tensor<B, 3>>,
        position_ids: Tensor<B, 3, Int>,
    ) -> Tensor<B, 3> {
        // ---- 1. Obtain input embeddings ----
        let hidden_states = match (input_ids, inputs_embeds) {
            (Some(ids), None) => self.embed_tokens.forward(ids),
            (None, Some(embeds)) => embeds,
            _ => panic!("Exactly one of input_ids or inputs_embeds must be provided"),
        };

        // ---- 2. Compute rotary position embeddings (once, shared by all layers) ----
        // cos, sin: [batch_size, 1, seq_len, head_dim]
        let position_embeddings = self.rotary_emb.forward(&position_ids);

        // ---- 3. Layer-by-layer forward pass ----
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, &position_embeddings);
        }

        // ---- 4. Final normalization ----
        self.norm.forward(hidden_states)
    }
}

// ========================================================================
// Configuration
// ========================================================================
//
// Default values correspond to PaddleOCR-VL's config.json:
//   vocab_size: 103424
//   hidden_size: 1024
//   intermediate_size: 3072
//   num_hidden_layers: 18
//   num_attention_heads: 16
//   num_key_value_heads: 2
//   head_dim: 128
//   rms_norm_eps: 1e-05
//   rope_theta: 500000.0
//   use_bias: false
//   mrope_section: [16, 24, 24]
// ========================================================================

#[derive(Config, Debug)]
pub struct PaddleOcrTextModelConfig {
    /// Vocabulary size
    #[config(default = 103424)]
    pub vocab_size: usize,
    /// Hidden-layer dimension
    #[config(default = 1024)]
    pub hidden_size: usize,
    /// MLP intermediate dimension
    #[config(default = 3072)]
    pub intermediate_size: usize,
    /// Number of decoder layers
    #[config(default = 18)]
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    #[config(default = 16)]
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (GQA)
    #[config(default = 2)]
    pub num_key_value_heads: usize,
    /// Dimension of each attention head
    #[config(default = 128)]
    pub head_dim: usize,
    /// RMSNorm epsilon
    #[config(default = 1e-05)]
    pub rms_norm_eps: f64,
    /// RoPE base frequency θ
    #[config(default = 500000.0)]
    pub rope_theta: f64,
    /// Whether linear layers include bias terms
    #[config(default = false)]
    pub use_bias: bool,
    /// M-RoPE temporal-axis segment width
    #[config(default = 16)]
    pub mrope_section_t: usize,
    /// M-RoPE height-axis segment width
    #[config(default = 24)]
    pub mrope_section_h: usize,
    /// M-RoPE width-axis segment width
    #[config(default = 24)]
    pub mrope_section_w: usize,
}

impl PaddleOcrTextModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrTextModel<B> {
        PaddleOcrTextModel {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers: (0..self.num_hidden_layers)
                .map(|_| {
                    PaddleOcrTextDecoderLayerConfig::new()
                        .with_hidden_size(self.hidden_size)
                        .with_intermediate_size(self.intermediate_size)
                        .with_rms_norm_eps(self.rms_norm_eps)
                        .with_num_attention_heads(self.num_attention_heads)
                        .with_num_key_value_heads(self.num_key_value_heads)
                        .with_head_dim(self.head_dim)
                        .with_use_bias(self.use_bias)
                        .init(device)
                })
                .collect(),
            norm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            rotary_emb: TextRotaryEmbedding::new(
                self.head_dim,
                self.rope_theta,
                [
                    self.mrope_section_t,
                    self.mrope_section_h,
                    self.mrope_section_w,
                ],
                device,
            ),
        }
    }
}
