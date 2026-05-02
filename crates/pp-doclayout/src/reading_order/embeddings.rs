// ========================================================================
// PPDocLayoutV2 Reading Order — TextEmbeddings
// ========================================================================
//
// The embedding layer for the reading-order module, based on LayoutLMv3's design.
// Converts text tokens, positional information, spatial coordinates, and layout
// category labels into a unified vector representation.
//
// Embedding composition:
//
// ```text
// embedding = word_embedding(input_ids)
//           + token_type_embedding(zeros)
//           + position_embedding(position_ids)
//           + spatial_proj(spatial_position_embedding(bbox))
// ```
//
// Spatial position embedding structure (LayoutLMv3-style concat):
//   spatial = cat([
//       x_emb(bbox[:,0]),     // left boundary  x1  → [B, L, 171]
//       y_emb(bbox[:,1]),     // top boundary   y1  → [B, L, 171]
//       x_emb(bbox[:,2]),     // right boundary x2  → [B, L, 171]
//       y_emb(bbox[:,3]),     // bottom boundary y2 → [B, L, 171]
//       h_emb(bbox[:,3]-bbox[:,1]),  // height       → [B, L, 170]
//       w_emb(bbox[:,2]-bbox[:,0]),  // width        → [B, L, 170]
//   ], dim=-1)               // total: 4×171 + 2×170 = 1024
//   spatial_proj(spatial)     // Linear(1024 → 512)
//
// Coordinate range is [0, 1000], so x/y embedding table size is 1024.
//
// The vocabulary is tiny (only 4 tokens):
//   0 = start, 1 = pad, 2 = end, 3 = pred
// ========================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Int, Tensor, s},
};

#[derive(Module, Debug)]
pub struct TextEmbeddings<B: Backend> {
    /// Word embeddings: vocab_size=4, hidden_size=512, padding_idx=1
    pub word_embeddings: Embedding<B>,
    /// Token-type embeddings: type_vocab_size=1, hidden_size=512
    pub token_type_embeddings: Embedding<B>,
    /// Position embeddings: max_position=514, hidden_size=512, padding_idx=1
    pub position_embeddings: Embedding<B>,

    // Spatial embeddings (LayoutLMv3 style)
    /// x-coordinate embedding (left/right): max_2d_pos=1024, dim=coordinate_size=171
    pub x_position_embeddings: Embedding<B>,
    /// y-coordinate embedding (top/bottom): max_2d_pos=1024, dim=coordinate_size=171
    pub y_position_embeddings: Embedding<B>,
    /// Height embedding: max_2d_pos=1024, dim=shape_size=170
    pub h_position_embeddings: Embedding<B>,
    /// Width embedding: max_2d_pos=1024, dim=shape_size=170
    pub w_position_embeddings: Embedding<B>,

    /// Spatial embedding projection: 4*171 + 2*170 = 1024 → hidden_size=512
    pub spatial_proj: Linear<B>,

    /// Final LayerNorm (called in ReadingOrder.forward)
    pub norm: LayerNorm<B>,

    pub padding_idx: usize,
}

impl<B: Backend> TextEmbeddings<B> {
    /// Compute spatial position embeddings.
    ///
    /// - `bbox`: `[B, L, 4]` Int tensor, coordinate range [0, 1000], format [x1, y1, x2, y2]
    /// - Returns: `[B, L, hidden_size]`
    pub fn calculate_spatial_position_embeddings(&self, bbox: &Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [_batch_size, _seq_len, _] = bbox.dims();

        // Extract each coordinate component: [B, L]
        let x1 = bbox.clone().slice(s![.., .., 0..1]).squeeze_dim::<2>(2);
        let y1 = bbox.clone().slice(s![.., .., 1..2]).squeeze_dim::<2>(2);
        let x2 = bbox.clone().slice(s![.., .., 2..3]).squeeze_dim::<2>(2);
        let y2 = bbox.clone().slice(s![.., .., 3..4]).squeeze_dim::<2>(2);

        // x/y coordinate embeddings: [B, L] → [B, L, coordinate_size=171]
        let left_emb = self.x_position_embeddings.forward(x1.clone());
        let upper_emb = self.y_position_embeddings.forward(y1.clone());
        let right_emb = self.x_position_embeddings.forward(x2.clone());
        let lower_emb = self.y_position_embeddings.forward(y2.clone());

        // Height and width: clamped to [0, 1023]
        let h = (y2 - y1).clamp(0, 1023);
        let w = (x2 - x1).clamp(0, 1023);
        let h_emb = self.h_position_embeddings.forward(h);
        let w_emb = self.w_position_embeddings.forward(w);

        // Concat: [B, L, 4*171 + 2*170 = 1024]
        let spatial = Tensor::cat(
            vec![left_emb, upper_emb, right_emb, lower_emb, h_emb, w_emb],
            2,
        );

        // Project to hidden_size: [B, L, 512]
        self.spatial_proj.forward(spatial)
    }

    /// Generate position_ids from input_ids.
    ///
    /// Non-padding positions are numbered sequentially; padding positions
    /// keep padding_idx.
    /// Example: input_ids    = [0, 3, 3, 2, 1, 1]  (start, pred, pred, end, pad, pad)
    ///          position_ids = [2, 3, 4, 5, 1, 1]  (padding_idx=1)
    pub fn create_position_ids_from_input_ids(
        &self,
        input_ids: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int> {
        let pad_id = self.padding_idx as i32;

        // mask: 1 where not pad, 0 where pad
        let mask = input_ids.clone().not_equal_elem(pad_id).int(); // [B, L]

        // cumsum along dim=1, then * mask to zero out padding positions
        let cumsum = mask.clone().cumsum(1); // [B, L]
        let incremental = cumsum * mask; // [B, L], zeros at pad positions

        // position_ids = incremental + padding_idx
        incremental.add_scalar(pad_id)
    }

    /// TextEmbeddings forward pass.
    ///
    /// - `input_ids`: `[B, L]` Int tensor (0=start, 1=pad, 2=end, 3=pred)
    /// - `bbox`: `[B, L, 4]` Int tensor, coordinate range [0, 1000]
    /// - Returns: `[B, L, hidden_size=512]` (without LayerNorm — called by the parent)
    pub fn forward(&self, input_ids: &Tensor<B, 2, Int>, bbox: &Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Position IDs
        let position_ids = self.create_position_ids_from_input_ids(input_ids);

        // Token-type IDs: all zeros
        let token_type_ids = Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device);

        // Sum all embeddings
        let word_emb = self.word_embeddings.forward(input_ids.clone());
        let token_type_emb = self.token_type_embeddings.forward(token_type_ids);
        let position_emb = self.position_embeddings.forward(position_ids);

        let mut embeddings = word_emb + token_type_emb + position_emb;

        // Add spatial position embeddings
        let spatial_emb = self.calculate_spatial_position_embeddings(bbox);
        embeddings = embeddings + spatial_emb;

        embeddings
    }
}

#[derive(Config, Debug)]
pub struct TextEmbeddingsConfig {
    #[config(default = 4)]
    pub vocab_size: usize,
    #[config(default = 512)]
    pub hidden_size: usize,
    #[config(default = 514)]
    pub max_position_embeddings: usize,
    #[config(default = 1024)]
    pub max_2d_position_embeddings: usize,
    #[config(default = 1)]
    pub type_vocab_size: usize,
    #[config(default = 171)]
    pub coordinate_size: usize,
    #[config(default = 170)]
    pub shape_size: usize,
    #[config(default = 1)]
    pub pad_token_id: usize,
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl TextEmbeddingsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextEmbeddings<B> {
        let spatial_embed_dim = 4 * self.coordinate_size + 2 * self.shape_size;

        TextEmbeddings {
            word_embeddings: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            token_type_embeddings: EmbeddingConfig::new(self.type_vocab_size, self.hidden_size)
                .init(device),
            position_embeddings: EmbeddingConfig::new(
                self.max_position_embeddings,
                self.hidden_size,
            )
            .init(device),
            x_position_embeddings: EmbeddingConfig::new(
                self.max_2d_position_embeddings,
                self.coordinate_size,
            )
            .init(device),
            y_position_embeddings: EmbeddingConfig::new(
                self.max_2d_position_embeddings,
                self.coordinate_size,
            )
            .init(device),
            h_position_embeddings: EmbeddingConfig::new(
                self.max_2d_position_embeddings,
                self.shape_size,
            )
            .init(device),
            w_position_embeddings: EmbeddingConfig::new(
                self.max_2d_position_embeddings,
                self.shape_size,
            )
            .init(device),
            spatial_proj: LinearConfig::new(spatial_embed_dim, self.hidden_size).init(device),
            norm: LayerNormConfig::new(self.hidden_size)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            padding_idx: self.pad_token_id,
        }
    }
}
