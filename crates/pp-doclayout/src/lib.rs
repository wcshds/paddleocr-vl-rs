// ========================================================================
// PP-DocLayoutV2 Top-Level Model
// ========================================================================
//
// PPDocLayoutV2ForObjectDetection is the complete document layout analysis model.
//
// Forward pass flow:
//
// ```text
// pixel_values [B, 3, 800, 800]
//       │
//       ▼
// ┌────────────────────────────────────────────────────┐
// │ HGNetV2-L Backbone                                │
// │   → stage2 [B, 512, 100, 100]                     │
// │   → stage3 [B, 1024, 50, 50]                      │
// │   → stage4 [B, 2048, 25, 25]                      │
// └────────────────────────────────────────────────────┘
//       │
//       ▼ encoder_input_proj: 3x Conv(→256) + FrozenBN
//
// ┌────────────────────────────────────────────────────┐
// │ HybridEncoder (AIFI + FPN + PAN)                  │
// │   → 3x [B, 256, H_i, W_i] feature maps           │
// └────────────────────────────────────────────────────┘
//       │
//       ▼ decoder_input_proj: 3x Conv(→256) + FrozenBN
//       ▼ flatten + concat → [B, 13125, 256]
//
// ┌────────────────────────────────────────────────────┐
// │ Encoder Head                                      │
// │   enc_output → enc_score_head + enc_bbox_head     │
// │   Top-300 → target [B, 300, 256]                  │
// │             ref_points [B, 300, 4]                │
// └────────────────────────────────────────────────────┘
//       │
//       ▼
// ┌────────────────────────────────────────────────────┐
// │ RT-DETR Decoder ×6                                │
// │   → intermediate_logits [B, 6, 300, 20]           │
// │   → intermediate_ref_points [B, 6, 300, 4]        │
// └────────────────────────────────────────────────────┘
//       │
//       ▼ take last layer → threshold filtering + sorting
//
// ┌────────────────────────────────────────────────────┐
// │ ReadingOrder                                      │
// │   → order_logits [B, 300, 300]                    │
// └────────────────────────────────────────────────────┘
// ```
//
// Weight loading key mapping (safetensors → burn):
//   model.backbone.model.* → backbone.*
//   model.encoder_input_proj.* → encoder_input_proj.*
//   model.encoder.* → encoder.*
//   model.decoder_input_proj.* → decoder_input_proj.*
//   model.decoder.* → decoder.*
//   model.enc_output.* → enc_output_linear.* / enc_output_norm.*
//   model.enc_score_head.* → enc_score_head.*
//   model.enc_bbox_head.* → enc_bbox_head.*
//   reading_order.* → reading_order.*
// ========================================================================

pub mod api;
pub mod backbone;
pub mod decoder;
pub mod encoder;
pub mod frozen_batch_norm;
pub mod load_adapter;
pub mod postprocessing;
pub mod preprocessing;
pub mod reading_order;
pub mod v3;

pub use api::{DocLayout, DocLayoutVersion, LayoutBlock, LayoutResult, label_name};
pub use v3::{PPDocLayoutV3Config, PPDocLayoutV3ForObjectDetection, V3ModelOutput};

use std::path::Path;

use burn::{
    config::Config,
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{DType, Int, Tensor, TensorData, activation, s},
};

use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::load_adapter::PyTorchToBurnDTypeAdapter;

fn tensor_to_vec_f32<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
    tensor
        .clone()
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
}

fn relation_box_dim_min_for_dtype(target_dtype: DType, configured_min: f32) -> f32 {
    match target_dtype {
        // The Python reference uses 1e-3. In real f16 execution, however,
        // zero-padded boxes can make the pairwise relation term compute
        // `1000 / 1e-3`, which overflows f16 before the logarithm. Keep the
        // configured value for f32/bf16, and raise only f16 to a finite-safe
        // lower bound.
        DType::F16 => configured_min.max(0.1),
        _ => configured_min,
    }
}

use self::{
    backbone::{HGNetV2Backbone, HGNetV2BackboneConfig},
    decoder::{
        MLPPredictionHead, MLPPredictionHeadConfig, PPDocLayoutV2Decoder,
        PPDocLayoutV2DecoderConfig, generate_anchors_with_dtype,
    },
    encoder::{HybridEncoder, HybridEncoderConfig},
    frozen_batch_norm::{FrozenBatchNorm2d, FrozenBatchNorm2dConfig},
    reading_order::{PPDocLayoutV2ReadingOrder, ReadingOrderConfig},
};

// ========================================================================
// Encoder Input Projection: Conv2d(in_ch, 256, 1) + FrozenBN
// ========================================================================

#[derive(Module, Debug)]
pub struct InputProjection<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: FrozenBatchNorm2d<B>,
}

impl<B: Backend> InputProjection<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        self.norm.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct InputProjectionConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl InputProjectionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InputProjection<B> {
        InputProjection {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_bias(false)
                .init(device),
            norm: FrozenBatchNorm2dConfig::new(self.out_channels)
                .with_epsilon(self.bn_eps)
                .init(device),
        }
    }
}

// ========================================================================
// PPDocLayoutV2ForObjectDetection
// ========================================================================

/// Forward pass output of the model
pub struct ModelOutput<B: Backend> {
    /// Classification logits: `[B, 300, num_labels]`
    pub logits: Tensor<B, 3>,
    /// Predicted bbox (cx, cy, w, h) normalized: `[B, 300, 4]`
    pub pred_boxes: Tensor<B, 3>,
    /// Reading order logits: `[B, 300, 300]`
    pub order_logits: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct PPDocLayoutV2ForObjectDetection<B: Backend> {
    // ---- Backbone ----
    pub backbone: HGNetV2Backbone<B>,

    // ---- Encoder Input Projection (backbone → encoder) ----
    pub encoder_input_proj: Vec<InputProjection<B>>,

    // ---- HybridEncoder ----
    pub encoder: HybridEncoder<B>,

    // ---- Decoder Input Projection (encoder → decoder) ----
    pub decoder_input_proj: Vec<InputProjection<B>>,

    // ---- Encoder Head (Top-K query selection) ----
    pub enc_output_linear: Linear<B>,
    pub enc_output_norm: LayerNorm<B>,
    pub enc_score_head: Linear<B>,
    pub enc_bbox_head: MLPPredictionHead<B>,

    // ---- RT-DETR Decoder ----
    pub decoder: PPDocLayoutV2Decoder<B>,

    // ---- ReadingOrder ----
    pub reading_order: PPDocLayoutV2ReadingOrder<B>,

    // ---- Config values ----
    pub num_queries: usize,
    pub num_labels: usize,
    pub class_thresholds: Vec<f32>,
    pub class_order: Vec<usize>,
}

impl<B: Backend> PPDocLayoutV2ForObjectDetection<B> {
    /// Forward pass
    ///
    /// - `pixel_values`: `[B, 3, 800, 800]`
    /// - Returns: `ModelOutput`
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> ModelOutput<B> {
        let device = pixel_values.device();
        let [batch_size, _, _, _] = pixel_values.dims();

        // ---- 1. Backbone ----
        let backbone_out = self.backbone.forward(pixel_values);
        let feat_maps = vec![
            backbone_out.stage2,
            backbone_out.stage3,
            backbone_out.stage4,
        ];

        // ---- 2. Encoder Input Projection ----
        let proj_feats: Vec<Tensor<B, 4>> = feat_maps
            .into_iter()
            .zip(self.encoder_input_proj.iter())
            .map(|(feat, proj)| proj.forward(feat))
            .collect();

        // ---- 3. HybridEncoder ----
        let encoded = self.encoder.forward(proj_feats);

        // ---- 4. Decoder Input Projection + Flatten ----
        let mut source_flatten = Vec::new();
        let mut spatial_shapes = Vec::new();
        let mut level_start_index = Vec::new();
        let mut offset = 0usize;

        let d_model = 256;
        for (level, feat) in encoded.iter().enumerate() {
            let proj = self.decoder_input_proj[level].forward(feat.clone());
            let [_b, _c, h, w] = proj.dims();
            spatial_shapes.push((h, w));
            level_start_index.push(offset);
            offset += h * w;

            // [B, 256, H, W] → [B, H*W, 256]
            let flat = proj.reshape([batch_size, d_model, h * w]).swap_dims(1, 2);
            source_flatten.push(flat);
        }

        let source_flatten = Tensor::cat(source_flatten, 1); // [B, total, 256]

        // ---- 5. Generate Anchors ----
        let (anchors, valid_mask) = generate_anchors_with_dtype::<B>(
            &spatial_shapes,
            0.05,
            &device,
            source_flatten.dtype(),
        );

        // Mask features with valid_mask
        let memory = source_flatten.clone() * valid_mask; // [B, total, 256]

        // ---- 6. Encoder Head: Top-K query selection ----
        let output_memory = self.enc_output_linear.forward(memory.clone());
        let output_memory = self.enc_output_norm.forward(output_memory);

        let enc_scores = self.enc_score_head.forward(output_memory.clone()); // [B, total, num_labels]
        let enc_bboxes = self.enc_bbox_head.forward(output_memory.clone()); // [B, total, 4]
        let enc_bboxes = enc_bboxes + anchors; // in inverse sigmoid space

        // Top-K: select the top 300 queries with the highest scores
        let max_scores = enc_scores.clone().max_dim(2).squeeze_dim::<2>(2);

        let [_, total_tokens] = max_scores.dims();
        let max_scores_data = tensor_to_vec_f32(&max_scores);

        let mut topk_idx_data = Vec::with_capacity(batch_size * self.num_queries);
        for b in 0..batch_size {
            let offset = b * total_tokens;
            let mut indices: Vec<usize> = (0..total_tokens).collect();
            indices.sort_by(|&a, &b_idx| {
                max_scores_data[offset + b_idx]
                    .partial_cmp(&max_scores_data[offset + a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            topk_idx_data.extend(indices.iter().take(self.num_queries).map(|&i| i as i32));
        }

        let topk_indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(topk_idx_data, [batch_size, self.num_queries]),
            &device,
        );

        // Gather reference_points_unact and target
        let topk_3d = topk_indices.clone().unsqueeze_dim::<3>(2).repeat_dim(2, 4);
        let reference_points_unact = enc_bboxes.gather(1, topk_3d);

        let topk_3d_d = topk_indices.unsqueeze_dim::<3>(2).repeat_dim(2, 256);
        let target = output_memory.gather(1, topk_3d_d);

        // ---- 7. RT-DETR Decoder ----
        let decoder_output = self.decoder.forward(
            target,
            reference_points_unact,
            &source_flatten,
            &spatial_shapes,
            &level_start_index,
        );

        let logits = decoder_output.last_logits;
        let raw_bboxes = decoder_output.last_reference_points;

        // ---- 8. Threshold filtering + ReadingOrder ----
        let centers = raw_bboxes.clone().slice(s![.., .., 0..2]);
        let sizes = raw_bboxes.clone().slice(s![.., .., 2..4]);
        let bboxes_1000 = Tensor::cat(
            vec![
                (centers.clone() - sizes.clone().mul_scalar(0.5f32)),
                (centers + sizes.mul_scalar(0.5f32)),
            ],
            2,
        )
        .mul_scalar(1000.0f32)
        .clamp(0.0, 1000.0);

        let max_logits = logits.clone().max_dim(2).squeeze_dim::<2>(2);
        let class_ids = logits.clone().argmax(2).squeeze_dim::<2>(2);
        let max_probs = activation::sigmoid(max_logits);

        let class_ids_data: Vec<i32> = class_ids.to_data().to_vec::<i32>().unwrap();
        let probs_data = tensor_to_vec_f32(&max_probs);
        let mut mask_data = vec![0i32; batch_size * self.num_queries];
        for b in 0..batch_size {
            for q in 0..self.num_queries {
                let idx = b * self.num_queries + q;
                let cid = class_ids_data[idx] as usize;
                let threshold = if cid < self.class_thresholds.len() {
                    self.class_thresholds[cid]
                } else {
                    0.5
                };
                if probs_data[idx] >= threshold {
                    mask_data[idx] = 1;
                }
            }
        }

        let mut sort_indices_data = Vec::with_capacity(batch_size * self.num_queries);
        for b in 0..batch_size {
            let off = b * self.num_queries;
            let mut indices: Vec<usize> = (0..self.num_queries).collect();
            indices.sort_by(|&a, &b_idx| {
                let ma = mask_data[off + a];
                let mb = mask_data[off + b_idx];
                mb.cmp(&ma).then_with(|| a.cmp(&b_idx))
            });
            sort_indices_data.extend(indices.iter().map(|&i| i as i32));
        }

        let sort_indices_3d = Tensor::<B, 3, Int>::from_data(
            TensorData::new(sort_indices_data.clone(), [batch_size, self.num_queries, 1]),
            &device,
        );

        let sort_3d_4 = sort_indices_3d.clone().repeat_dim(2, 4);
        let sorted_boxes = bboxes_1000.gather(1, sort_3d_4.clone());
        let sorted_pred_boxes = raw_bboxes.gather(1, sort_3d_4);
        let sort_3d_nl = sort_indices_3d.repeat_dim(2, self.num_labels);
        let sorted_logits = logits.gather(1, sort_3d_nl);

        // Reorder class_ids and mask using the CPU-side sort_indices_data directly,
        // avoiding a wasteful device round-trip (upload then immediately download).
        let sorted_class_ids_data: Vec<i32> = {
            let mut result = vec![0i32; batch_size * self.num_queries];
            for b in 0..batch_size {
                for q in 0..self.num_queries {
                    let src = sort_indices_data[b * self.num_queries + q] as usize;
                    result[b * self.num_queries + q] = class_ids_data[b * self.num_queries + src];
                }
            }
            result
        };
        let sorted_mask_data: Vec<i32> = {
            let mut result = vec![0i32; batch_size * self.num_queries];
            for b in 0..batch_size {
                for q in 0..self.num_queries {
                    let src = sort_indices_data[b * self.num_queries + q] as usize;
                    result[b * self.num_queries + q] = mask_data[b * self.num_queries + src];
                }
            }
            result
        };

        let sorted_mask_3d_data: Vec<f32> = sorted_mask_data
            .iter()
            .flat_map(|&m| std::iter::repeat_n(m as f32, 4))
            .collect();
        let sorted_mask_3d = Tensor::<B, 3>::from_data(
            TensorData::new(sorted_mask_3d_data, [batch_size, self.num_queries, 4])
                .convert_dtype(sorted_boxes.dtype()),
            (&device, sorted_boxes.dtype()),
        );
        let pad_boxes = sorted_boxes * sorted_mask_3d;

        let mapped_data: Vec<i32> = sorted_class_ids_data
            .iter()
            .zip(sorted_mask_data.iter())
            .map(|(&cid, &m)| {
                if m == 0 {
                    return 0;
                }
                let idx = cid as usize;
                if idx < self.class_order.len() {
                    self.class_order[idx] as i32
                } else {
                    0
                }
            })
            .collect();
        let mapped_labels = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mapped_data, [batch_size, self.num_queries]),
            &device,
        );

        let sorted_mask_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(sorted_mask_data, [batch_size, self.num_queries]),
            &device,
        );

        // ---- ReadingOrder ----
        let order_logits =
            self.reading_order
                .forward(&pad_boxes, &mapped_labels, &sorted_mask_tensor);
        let order_logits = order_logits.slice(s![.., 0..self.num_queries, 0..self.num_queries,]);

        ModelOutput {
            logits: sorted_logits,
            pred_boxes: sorted_pred_boxes,
            order_logits,
        }
    }
}

// ========================================================================
// PP-DocLayoutV2 Main Model Configuration
// ========================================================================
//
// Overall architecture:
//
// ```text
//  pixel_values [B,3,800,800]
//       │
//       ▼
//  ┌─────────────────────────────────────────────┐
//  │  HGNetV2-L Backbone                         │
//  │  Stage2 → [B, 512, H/8, W/8]               │
//  │  Stage3 → [B, 1024, H/16, W/16]            │
//  │  Stage4 → [B, 2048, H/32, W/32]            │
//  └─────────────────────────────────────────────┘
//       │
//       ▼
//  ┌─────────────────────────────────────────────┐
//  │  HybridEncoder                              │
//  │  1×1 Conv (→256ch) → AIFI → FPN → PAN      │
//  └─────────────────────────────────────────────┘
//       │
//       ▼
//  ┌─────────────────────────────────────────────┐
//  │  Encoder Head: topk 300 queries             │
//  └─────────────────────────────────────────────┘
//       │
//       ▼
//  ┌─────────────────────────────────────────────┐
//  │  RT-DETR Decoder ×6 layers                  │
//  │  Self-Attn → Deformable Cross-Attn → FFN    │
//  └─────────────────────────────────────────────┘
//       │
//       ▼
//  class_embed + bbox_embed → ReadingOrder
// ```
// ========================================================================

#[derive(Config, Debug)]
pub struct PPDocLayoutV2Config {
    // ---- Initialization ----
    /// Standard deviation for weight initialization
    #[config(default = 0.01)]
    pub initializer_range: f64,

    /// LayerNorm epsilon
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,

    /// BatchNorm epsilon
    #[config(default = 1e-5)]
    pub batch_norm_eps: f64,

    // ---- Backbone (HGNetV2-L) ----
    /// Whether to freeze backbone BatchNorm (always true during inference)
    #[config(default = true)]
    pub freeze_backbone_batch_norms: bool,

    // ---- HybridEncoder ----
    /// Encoder hidden dimension (all feature maps are projected to this dimension)
    #[config(default = 256)]
    pub encoder_hidden_dim: usize,

    /// Output channel counts of each backbone stage [stage2, stage3, stage4]
    pub encoder_in_channels: Option<[usize; 3]>,

    /// Downsampling strides for each feature map level
    pub feat_strides: Option<[usize; 3]>,

    /// Number of Transformer encoder layers in AIFI
    #[config(default = 1)]
    pub encoder_layers: usize,

    /// Intermediate dimension of the FFN in AIFI
    #[config(default = 1024)]
    pub encoder_ffn_dim: usize,

    /// Number of self-attention heads in AIFI
    #[config(default = 8)]
    pub encoder_attention_heads: usize,

    /// Dropout probability
    #[config(default = 0.0)]
    pub dropout: f64,

    /// Activation dropout inside FFN
    #[config(default = 0.0)]
    pub activation_dropout: f64,

    /// Level indices where AIFI is applied (relative to backbone output; [2] means the coarsest level)
    pub encode_proj_layers: Option<Vec<usize>>,

    /// Temperature parameter for sinusoidal positional encoding
    #[config(default = 10000.0)]
    pub positional_encoding_temperature: f64,

    /// Hidden-layer expansion ratio in RepVggBlock
    #[config(default = 1.0)]
    pub hidden_expansion: f64,

    // ---- RT-DETR Decoder ----
    /// Model dimension of the Decoder
    #[config(default = 256)]
    pub d_model: usize,

    /// Number of queries (maximum number of detected objects per image)
    #[config(default = 300)]
    pub num_queries: usize,

    /// Channel counts after Decoder input projection
    pub decoder_in_channels: Option<[usize; 3]>,

    /// Decoder FFN intermediate dimension
    #[config(default = 1024)]
    pub decoder_ffn_dim: usize,

    /// Number of multi-scale feature levels
    #[config(default = 3)]
    pub num_feature_levels: usize,

    /// Number of sampling points for deformable attention
    #[config(default = 4)]
    pub decoder_n_points: usize,

    /// Number of Decoder layers
    #[config(default = 6)]
    pub decoder_layers: usize,

    /// Number of Decoder self-attention heads
    #[config(default = 8)]
    pub decoder_attention_heads: usize,

    /// Decoder attention dropout
    #[config(default = 0.0)]
    pub attention_dropout: f64,

    /// Number of layout classes (num_labels), corresponding to id2label entries
    #[config(default = 25)]
    pub num_labels: usize,

    // ---- Post-processing ----
    /// Per-class confidence thresholds (length = num_labels)
    ///
    /// Corresponds to class_thresholds in the HuggingFace config.json.
    /// See `PPDocLayoutV2Config::default_class_thresholds()` for defaults.
    pub class_thresholds: Option<Vec<f32>>,

    /// Mapping from class IDs to reading-order groups (length = num_labels)
    ///
    /// Maps detection class IDs to the category groups used by the ReadingOrder module.
    /// See `PPDocLayoutV2Config::default_class_order()` for defaults.
    pub class_order: Option<Vec<usize>>,

    // ---- Reading Order ----
    /// Reading order module configuration
    pub reading_order_config: Option<ReadingOrderConfig>,
}

impl PPDocLayoutV2Config {
    pub fn default_class_thresholds() -> Vec<f32> {
        vec![
            0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.5,
            0.4, 0.5, 0.5, 0.45, 0.5, 0.4, 0.4, 0.5,
        ]
    }

    pub fn default_class_order() -> Vec<usize> {
        vec![
            4, 2, 14, 1, 5, 7, 8, 6, 11, 11, 9, 13, 10, 10, 1, 2, 3, 0, 2, 2, 12, 1, 2, 15, 6,
        ]
    }

    pub fn default_encode_proj_layers() -> Vec<usize> {
        vec![2]
    }

    pub fn get_class_thresholds(&self) -> Vec<f32> {
        self.class_thresholds
            .clone()
            .unwrap_or_else(Self::default_class_thresholds)
    }

    pub fn get_class_order(&self) -> Vec<usize> {
        self.class_order
            .clone()
            .unwrap_or_else(Self::default_class_order)
    }

    pub fn get_encode_proj_layers(&self) -> Vec<usize> {
        self.encode_proj_layers
            .clone()
            .unwrap_or_else(Self::default_encode_proj_layers)
    }

    pub fn get_encoder_in_channels(&self) -> [usize; 3] {
        self.encoder_in_channels.unwrap_or([512, 1024, 2048])
    }

    pub fn get_feat_strides(&self) -> [usize; 3] {
        self.feat_strides.unwrap_or([8, 16, 32])
    }

    pub fn get_decoder_in_channels(&self) -> [usize; 3] {
        self.decoder_in_channels.unwrap_or([256, 256, 256])
    }

    /// Initialize the model (with random weights)
    pub fn init<B: Backend>(&self, device: &B::Device) -> PPDocLayoutV2ForObjectDetection<B> {
        let bn_eps = self.batch_norm_eps;
        let ln_eps = self.layer_norm_eps;
        let ro_config = self
            .reading_order_config
            .clone()
            .unwrap_or_else(ReadingOrderConfig::new);

        let backbone = HGNetV2BackboneConfig::new()
            .with_bn_eps(bn_eps)
            .init(device);

        let encoder_in_channels = self.get_encoder_in_channels();
        let encoder_input_proj: Vec<InputProjection<B>> = encoder_in_channels
            .iter()
            .map(|&in_ch| {
                InputProjectionConfig::new(in_ch, self.encoder_hidden_dim)
                    .with_bn_eps(bn_eps)
                    .init(device)
            })
            .collect();

        let encoder = HybridEncoderConfig::new()
            .with_encoder_hidden_dim(self.encoder_hidden_dim)
            .with_encoder_layers(self.encoder_layers)
            .with_encoder_ffn_dim(self.encoder_ffn_dim)
            .with_encoder_attention_heads(self.encoder_attention_heads)
            .with_hidden_expansion(self.hidden_expansion)
            .with_positional_encoding_temperature(self.positional_encoding_temperature)
            .with_layer_norm_eps(ln_eps)
            .with_bn_eps(bn_eps)
            .init(device);

        let decoder_in_channels = self.get_decoder_in_channels();
        let decoder_input_proj: Vec<InputProjection<B>> = decoder_in_channels
            .iter()
            .map(|&in_ch| {
                InputProjectionConfig::new(in_ch, self.d_model)
                    .with_bn_eps(bn_eps)
                    .init(device)
            })
            .collect();

        let enc_output_linear = LinearConfig::new(self.d_model, self.d_model).init(device);
        let enc_output_norm = LayerNormConfig::new(self.d_model)
            .with_epsilon(ln_eps)
            .init(device);
        let enc_score_head = LinearConfig::new(self.d_model, self.num_labels).init(device);
        let enc_bbox_head =
            MLPPredictionHeadConfig::new(self.d_model, self.d_model, 4).init(device);

        let decoder = PPDocLayoutV2DecoderConfig::new()
            .with_d_model(self.d_model)
            .with_n_heads(self.decoder_attention_heads)
            .with_ffn_dim(self.decoder_ffn_dim)
            .with_num_layers(self.decoder_layers)
            .with_n_levels(self.num_feature_levels)
            .with_n_points(self.decoder_n_points)
            .with_num_labels(self.num_labels)
            .with_layer_norm_eps(ln_eps)
            .init(device);

        let reading_order = ro_config.init(device);

        PPDocLayoutV2ForObjectDetection {
            backbone,
            encoder_input_proj,
            encoder,
            decoder_input_proj,
            enc_output_linear,
            enc_output_norm,
            enc_score_head,
            enc_bbox_head,
            decoder,
            reading_order,
            num_queries: self.num_queries,
            num_labels: self.num_labels,
            class_thresholds: self.get_class_thresholds(),
            class_order: self.get_class_order(),
        }
    }

    /// Initialize the model and load weights from a safetensors file.
    ///
    /// This convenience wrapper panics when loading fails. Public APIs should
    /// prefer [`Self::try_init_from_safetensors`] so checkpoint errors can be
    /// reported without aborting the process.
    pub fn init_from_safetensors<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> PPDocLayoutV2ForObjectDetection<B> {
        self.try_init_from_safetensors::<B>(path, device)
            .expect("failed to load PP-DocLayoutV2 weights")
    }

    pub fn init_from_safetensors_with_dtype<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
        target_dtype: DType,
    ) -> PPDocLayoutV2ForObjectDetection<B> {
        self.try_init_from_safetensors_with_dtype::<B>(path, device, target_dtype)
            .expect("failed to load PP-DocLayoutV2 weights")
    }

    /// Fallible checkpoint loader for PP-DocLayoutV2.
    ///
    /// Partial loading is intentionally allowed because the HuggingFace
    /// checkpoint can contain auxiliary keys outside the inference module graph.
    /// Shape mismatches and unreadable files are still returned as hard errors.
    pub fn try_init_from_safetensors<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<PPDocLayoutV2ForObjectDetection<B>, String> {
        self.try_init_from_safetensors_with_dtype::<B>(path, device, DType::F32)
    }

    pub fn try_init_from_safetensors_with_dtype<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
        target_dtype: DType,
    ) -> Result<PPDocLayoutV2ForObjectDetection<B>, String> {
        let path = path.as_ref();
        let mut model = self.init::<B>(device);
        model.reading_order.encoder.relation_box_dim_min = relation_box_dim_min_for_dtype(
            target_dtype,
            model.reading_order.encoder.relation_box_dim_min,
        );

        let mut st = SafetensorsStore::from_file(path.to_path_buf())
            // ========== 1. Top-level prefix ==========
            .with_key_remapping(r"^model\.backbone\.model\.", "backbone.")
            .with_key_remapping(r"^model\.", "")
            // ========== 2. Backbone structural mapping ==========
            .with_key_remapping(r"^backbone\.embedder\.", "backbone.embeddings.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.0\.", "backbone.stage1.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.1\.", "backbone.stage2.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.2\.", "backbone.stage3.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.3\.", "backbone.stage4.")
            // Backbone ConvBlock enum: light blocks (conv1/conv2) → Light variant (must precede standard)
            .with_key_remapping(
                r"\.layers\.(\d+)\.(conv[12])\.convolution\.",
                ".layers.$1.Light.$2.conv.",
            )
            .with_key_remapping(
                r"\.layers\.(\d+)\.(conv[12])\.normalization\.",
                ".layers.$1.Light.$2.bn.",
            )
            // Backbone ConvBlock enum: standard blocks → Standard variant
            .with_key_remapping(
                r"\.layers\.(\d+)\.convolution\.",
                ".layers.$1.Standard.conv.",
            )
            .with_key_remapping(
                r"\.layers\.(\d+)\.normalization\.",
                ".layers.$1.Standard.bn.",
            )
            // Backbone aggregation → agg_squeeze / agg_excite
            .with_key_remapping(r"\.aggregation\.0\.", ".agg_squeeze.")
            .with_key_remapping(r"\.aggregation\.1\.", ".agg_excite.")
            // Backbone generic: convolution→conv, normalization→bn (downsample, stems, agg internals)
            .with_key_remapping(r"\.convolution\.", ".conv.")
            .with_key_remapping(r"\.normalization\.", ".bn.")
            // ========== 3. Input Projections (Sequential→named) ==========
            .with_key_remapping(
                r"encoder_input_proj\.(\d+)\.0\.",
                "encoder_input_proj.$1.conv.",
            )
            .with_key_remapping(
                r"encoder_input_proj\.(\d+)\.1\.",
                "encoder_input_proj.$1.norm.",
            )
            .with_key_remapping(
                r"decoder_input_proj\.(\d+)\.0\.",
                "decoder_input_proj.$1.conv.",
            )
            .with_key_remapping(
                r"decoder_input_proj\.(\d+)\.1\.",
                "decoder_input_proj.$1.norm.",
            )
            // ========== 4. Encoder Output ==========
            .with_key_remapping(r"^enc_output\.0\.", "enc_output_linear.")
            .with_key_remapping(r"^enc_output\.1\.", "enc_output_norm.")
            // ========== 5. AIFI Encoder: encoder.encoder → encoder.aifi ==========
            .with_key_remapping(r"encoder\.encoder\.", "encoder.aifi.")
            // AIFI: self_attn.out_proj → o_proj (must precede the generic self_attn rule)
            .with_key_remapping(
                r"encoder\.aifi\.(\d+)\.layers\.(\d+)\.self_attn\.out_proj\.",
                "encoder.aifi.$1.layers.$2.o_proj.",
            )
            // AIFI: strip the self_attn. prefix (q_proj, k_proj, v_proj hang directly on the layer)
            .with_key_remapping(
                r"encoder\.aifi\.(\d+)\.layers\.(\d+)\.self_attn\.",
                "encoder.aifi.$1.layers.$2.",
            )
            // ========== 6. Decoder ==========
            // encoder_attn_layer_norm → cross_attn_layer_norm (must precede encoder_attn)
            .with_key_remapping(
                r"decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.",
                "decoder.layers.$1.cross_attn_layer_norm.",
            )
            // encoder_attn → cross_attn
            .with_key_remapping(
                r"decoder\.layers\.(\d+)\.encoder_attn\.",
                "decoder.layers.$1.cross_attn.",
            )
            // self_attn.out_proj → self_attn.o_proj
            .with_key_remapping(
                r"decoder\.layers\.(\d+)\.self_attn\.out_proj\.",
                "decoder.layers.$1.self_attn.o_proj.",
            )
            // ========== 7. Reading Order ==========
            .with_key_remapping(
                r"reading_order\.encoder\.layer\.(\d+)\.attention\.self\.",
                "reading_order.encoder.layers.$1.self_attention.",
            )
            .with_key_remapping(
                r"reading_order\.encoder\.layer\.(\d+)\.attention\.output\.dense\.",
                "reading_order.encoder.layers.$1.attn_dense.",
            )
            .with_key_remapping(
                r"reading_order\.encoder\.layer\.(\d+)\.attention\.output\.norm\.",
                "reading_order.encoder.layers.$1.attn_norm.",
            )
            .with_key_remapping(
                r"reading_order\.encoder\.layer\.(\d+)\.intermediate\.dense\.",
                "reading_order.encoder.layers.$1.intermediate_dense.",
            )
            .with_key_remapping(
                r"reading_order\.encoder\.layer\.(\d+)\.output\.dense\.",
                "reading_order.encoder.layers.$1.output_dense.",
            )
            .with_key_remapping(
                r"reading_order\.encoder\.layer\.(\d+)\.output\.norm\.",
                "reading_order.encoder.layers.$1.output_norm.",
            )
            // ========== 8. Adapter + partial loading ==========
            // PyTorchToBurnAdapter automatically remaps LayerNorm weight→gamma, bias→beta,
            // so no manual norm rename rules are needed.
            // FrozenBatchNorm2d weight/bias/running_mean/running_var names match PyTorch,
            // so no extra mapping is required either.
            .with_from_adapter(PyTorchToBurnDTypeAdapter::new(target_dtype))
            .allow_partial(true);

        model
            .load_from(&mut st)
            .map_err(|e| format!("failed to load {}: {e}", path.display()))?;

        Ok(model)
    }
}
