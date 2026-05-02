// ========================================================================
// PPDocLayoutV2 ReadingOrder Module
// ========================================================================
//
// Reading-order prediction module based on the LayoutLMv3 Transformer
// Encoder architecture.  Receives detected bounding boxes and class labels,
// and predicts the reading order among layout elements.
//
// Overall architecture:
//
// ```text
// boxes [B, 300, 4] (sorted by confidence, [0, 1000] scale)
// labels [B, 300]    (remapped class indices)
// mask [B, 300]      (valid element indicator)
//       │
//       ▼
// ┌───────────────────────────────────────────────┐
// │  Build input_ids:                             │
// │    [start, pred, pred, ..., end, pad, pad]    │
// │  Build pad_boxes:                             │
// │    [zeros, box1, box2, ..., zeros]            │
// └───────────────────────────────────────────────┘
//       │
//       ▼
// ┌───────────────────────────────────────────────┐
// │  TextEmbeddings:                              │
// │    word + position + token_type + spatial      │
// └───────────────────────────────────────────────┘
//       │
//       ▼
// ┌───────────────────────────────────────────────┐
// │  + label_embeddings → label_proj              │
// │  → LayerNorm → (dropout skipped at inference) │
// └───────────────────────────────────────────────┘
//       │
//       ▼
// ┌───────────────────────────────────────────────┐
// │  ReadingOrderEncoder (6 layers)               │
// │  + PositionRelationEmbedding (2D bias)        │
// └───────────────────────────────────────────────┘
//       │
//       ▼
// ┌───────────────────────────────────────────────┐
// │  Extract middle tokens: [1 : 1+seq_len]       │
// │  → GlobalPointer → order_logits [B, L, L]     │
// └───────────────────────────────────────────────┘
// ```
//
// input_ids format (example with 5 valid elements):
//   [0, 3, 3, 3, 3, 3, 2, 1, 1, ...]
//    ^  ^-----------^  ^  ^--------^
//  start   pred×5    end  pad tokens
//
// pad_boxes format:
//   [zeros, box1, box2, ..., box5, zeros, zeros, ...]
//    ^start               ^end    ^padding
// ========================================================================

pub mod embeddings;
pub mod encoder;
pub mod global_pointer;

use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, s},
};

use self::{
    embeddings::{TextEmbeddings, TextEmbeddingsConfig},
    encoder::{ReadingOrderEncoder, ReadingOrderEncoderConfig},
    global_pointer::{GlobalPointer, GlobalPointerConfig},
};

#[derive(Module, Debug)]
pub struct PPDocLayoutV2ReadingOrder<B: Backend> {
    pub embeddings: TextEmbeddings<B>,
    /// Layout-class embeddings: num_classes=20, hidden_size=512
    pub label_embeddings: Embedding<B>,
    /// Class-embedding projection: hidden_size → hidden_size
    pub label_features_projection: Linear<B>,
    pub encoder: ReadingOrderEncoder<B>,
    pub relative_head: GlobalPointer<B>,

    pub start_token_id: i32,
    pub pad_token_id: i32,
    pub end_token_id: i32,
    pub pred_token_id: i32,
}

// ========================================================================
// ReadingOrderConfig
// ========================================================================
//
// The reading-order module predicts the reading sequence among detected
// layout elements in a document.
// Architecture: TextEmbeddings → 6-layer Transformer Encoder → GlobalPointer
//
// TextEmbeddings contains:
//   - word_embeddings (vocab=4: start/pad/end/pred)
//   - position_embeddings (max=514)
//   - spatial embeddings (x/y/w/h coordinate embeddings → projected to hidden_size)
//   - label_embeddings (layout-class embeddings such as title, paragraph, etc.)
//
// The Transformer Encoder uses:
//   - CogView PB-Relax softmax stabilization
//   - PositionRelationEmbedding: 2D attention bias based on relative box positions
//
// GlobalPointer:
//   - Projects hidden_states into Q/K and computes pairwise reading-order scores
// ========================================================================

#[derive(Config, Debug)]
pub struct ReadingOrderConfig {
    /// Transformer hidden dimension
    #[config(default = 512)]
    pub hidden_size: usize,

    /// Number of attention heads
    #[config(default = 8)]
    pub num_attention_heads: usize,

    /// Attention dropout probability
    #[config(default = 0.1)]
    pub attention_probs_dropout_prob: f64,

    /// Whether to use 1D relative position bias
    #[config(default = false)]
    pub has_relative_attention_bias: bool,

    /// Whether to use 2D spatial attention bias (based on box relative positions)
    #[config(default = true)]
    pub has_spatial_attention_bias: bool,

    /// LayerNorm epsilon
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,

    /// Hidden dropout probability
    #[config(default = 0.1)]
    pub hidden_dropout_prob: f64,

    /// FFN intermediate dimension
    #[config(default = 2048)]
    pub intermediate_size: usize,

    /// Number of Transformer encoder layers
    #[config(default = 6)]
    pub num_hidden_layers: usize,

    /// Number of buckets for 1D relative position
    #[config(default = 32)]
    pub rel_pos_bins: usize,

    /// Maximum distance for 1D relative position
    #[config(default = 128)]
    pub max_rel_pos: usize,

    /// Number of buckets for 2D relative position
    #[config(default = 64)]
    pub rel_2d_pos_bins: usize,

    /// Maximum distance for 2D relative position
    #[config(default = 256)]
    pub max_rel_2d_pos: usize,

    /// Maximum sequence length for position embeddings
    #[config(default = 514)]
    pub max_position_embeddings: usize,

    /// Maximum value for spatial coordinate embeddings (x/y/w/h max 1024)
    #[config(default = 1024)]
    pub max_2d_position_embeddings: usize,

    /// Number of token-type embedding types
    #[config(default = 1)]
    pub type_vocab_size: usize,

    /// Vocabulary size (start=0, pad=1, end=2, pred=3)
    #[config(default = 4)]
    pub vocab_size: usize,

    /// Weight initialization range
    #[config(default = 0.01)]
    pub initializer_range: f64,

    /// Start token ID
    #[config(default = 0)]
    pub start_token_id: usize,

    /// Padding token ID
    #[config(default = 1)]
    pub pad_token_id: usize,

    /// End token ID
    #[config(default = 2)]
    pub end_token_id: usize,

    /// Prediction token ID
    #[config(default = 3)]
    pub pred_token_id: usize,

    /// x/y coordinate embedding dimension (4 coordinates × 171 dims each)
    #[config(default = 171)]
    pub coordinate_size: usize,

    /// w/h shape embedding dimension (2 shapes × 170 dims each)
    #[config(default = 170)]
    pub shape_size: usize,

    /// Number of layout classes
    #[config(default = 20)]
    pub num_classes: usize,

    /// PositionRelationEmbedding embedding dimension
    #[config(default = 16)]
    pub relation_bias_embed_dim: usize,

    /// PositionRelationEmbedding RoPE base frequency
    #[config(default = 10000.0)]
    pub relation_bias_theta: f64,

    /// PositionRelationEmbedding position scaling factor
    #[config(default = 100.0)]
    pub relation_bias_scale: f64,

    /// Minimum width/height used when building pairwise box relation features.
    ///
    /// The Python reference uses 1e-3. f16 loaders raise this at runtime to
    /// avoid overflow on padded boxes while keeping f32/bf16 numerics aligned
    /// with the reference implementation.
    #[config(default = 1e-3)]
    pub relation_box_dim_min: f64,

    /// GlobalPointer head dimension
    #[config(default = 64)]
    pub global_pointer_head_size: usize,

    /// GlobalPointer dropout value
    #[config(default = 0.0)]
    pub gp_dropout_value: f64,
}

impl ReadingOrderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PPDocLayoutV2ReadingOrder<B> {
        let embeddings = TextEmbeddingsConfig::new()
            .with_vocab_size(self.vocab_size)
            .with_hidden_size(self.hidden_size)
            .with_max_position_embeddings(self.max_position_embeddings)
            .with_max_2d_position_embeddings(self.max_2d_position_embeddings)
            .with_type_vocab_size(self.type_vocab_size)
            .with_coordinate_size(self.coordinate_size)
            .with_shape_size(self.shape_size)
            .with_pad_token_id(self.pad_token_id)
            .with_layer_norm_eps(self.layer_norm_eps)
            .init(device);

        let label_embeddings =
            EmbeddingConfig::new(self.num_classes, self.hidden_size).init(device);
        let label_features_projection =
            LinearConfig::new(self.hidden_size, self.hidden_size).init(device);

        let encoder = ReadingOrderEncoderConfig::new()
            .with_hidden_size(self.hidden_size)
            .with_num_attention_heads(self.num_attention_heads)
            .with_intermediate_size(self.intermediate_size)
            .with_num_hidden_layers(self.num_hidden_layers)
            .with_layer_norm_eps(self.layer_norm_eps)
            .with_relation_bias_embed_dim(self.relation_bias_embed_dim)
            .with_relation_bias_theta(self.relation_bias_theta)
            .with_relation_bias_scale(self.relation_bias_scale)
            .with_relation_box_dim_min(self.relation_box_dim_min)
            .init(device);

        let relative_head = GlobalPointerConfig::new()
            .with_hidden_size(self.hidden_size)
            .with_head_size(self.global_pointer_head_size)
            .init(device);

        PPDocLayoutV2ReadingOrder {
            embeddings,
            label_embeddings,
            label_features_projection,
            encoder,
            relative_head,
            start_token_id: self.start_token_id as i32,
            pad_token_id: self.pad_token_id as i32,
            end_token_id: self.end_token_id as i32,
            pred_token_id: self.pred_token_id as i32,
        }
    }
}

impl<B: Backend> PPDocLayoutV2ReadingOrder<B> {
    /// ReadingOrder forward pass.
    ///
    /// - `boxes`: `[B, seq_len, 4]` detected bboxes ([0, 1000] range,
    ///   valid elements first, padding at the end)
    /// - `labels`: `[B, seq_len]` Int tensor, remapped class indices
    /// - `mask`: `[B, seq_len]` Int tensor (1=valid, 0=padding)
    ///
    /// Returns: `[B, seq_len, seq_len]` reading-order logits
    ///   (trimmed to only include seq_len positions)
    pub fn forward(
        &self,
        boxes: &Tensor<B, 3>,
        labels: &Tensor<B, 2, Int>,
        mask: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = boxes.dims();
        let device = boxes.device();

        // Count valid elements per batch
        let num_pred = mask.clone().sum_dim(1); // [B, 1]

        // ---- Build input_ids [B, seq_len + 2] ----
        // Initialize as all-pad
        let mut input_ids_data = vec![self.pad_token_id; batch_size * (seq_len + 2)];
        let num_pred_vec: Vec<i32> = num_pred
            .clone()
            .reshape([batch_size])
            .to_data()
            .to_vec::<i32>()
            .unwrap();

        for b in 0..batch_size {
            let np = num_pred_vec[b] as usize;
            let offset = b * (seq_len + 2);
            // Start token at position 0
            input_ids_data[offset] = self.start_token_id;
            // Pred tokens at positions [1, np]
            for i in 1..=np {
                input_ids_data[offset + i] = self.pred_token_id;
            }
            // End token at position np + 1
            input_ids_data[offset + np + 1] = self.end_token_id;
        }
        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_ids_data, [batch_size, seq_len + 2]),
            &device,
        );

        // ---- Build pad_boxes [B, seq_len + 2, 4] ----
        let pad_box = Tensor::<B, 3>::zeros([batch_size, 1, 4], &device).cast(boxes.dtype());
        let pad_boxes = Tensor::cat(vec![pad_box.clone(), boxes.clone(), pad_box], 1); // [B, seq_len + 2, 4]

        // ---- TextEmbeddings ----
        let bbox_embedding = self
            .embeddings
            .forward(&input_ids, &pad_boxes.clone().int()); // [B, seq_len + 2, 512]

        // ---- Label Embeddings ----
        let label_embs = self.label_embeddings.forward(labels.clone()); // [B, seq_len, 512]
        let label_proj = self.label_features_projection.forward(label_embs); // [B, seq_len, 512]
        let label_pad = Tensor::<B, 3>::zeros([batch_size, 1, label_proj.dims()[2]], &device)
            .cast(label_proj.dtype());
        let label_proj = Tensor::cat(vec![label_pad.clone(), label_proj, label_pad], 1); // [B, seq_len + 2, 512]

        // ---- Merge embeddings + LayerNorm ----
        let final_embeddings = bbox_embedding + label_proj;
        let final_embeddings = self.embeddings.norm.forward(final_embeddings);

        // ---- Attention Mask ----
        // attention_mask[b, j] = 1 if j < num_pred[b] + 2, else 0
        // Then convert to additive mask: valid → 0.0, padding → -1e4
        let mut attn_mask_data = vec![0.0f32; batch_size * (seq_len + 2)];
        for b in 0..batch_size {
            let np = num_pred_vec[b] as usize;
            let valid_len = np + 2; // start + preds + end
            let offset = b * (seq_len + 2);
            for j in valid_len..(seq_len + 2) {
                attn_mask_data[offset + j] = -1e4;
            }
        }
        let attention_mask = Tensor::<B, 2>::from_data(
            TensorData::new(attn_mask_data, [batch_size, seq_len + 2]).convert_dtype(boxes.dtype()),
            (&device, boxes.dtype()),
        );
        // Reshape to [B, 1, 1, seq_len + 2] for broadcasting
        let attention_mask = attention_mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);

        // ---- Encoder ----
        let encoder_output =
            self.encoder
                .forward(final_embeddings, &pad_boxes, Some(&attention_mask)); // [B, seq_len + 2, 512]

        // ---- Extract middle tokens (skip start, take seq_len) ----
        let token = encoder_output.slice(s![.., 1..(1 + seq_len), ..]);

        // ---- GlobalPointer ----
        self.relative_head.forward(token)
    }
}
