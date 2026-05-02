// ========================================================================
// PP-DocLayoutV3 Top-Level Model
// ========================================================================
//
// V3 keeps the V2 HGNetV2 + HybridEncoder + RT-DETR foundation, but adds a
// mask prototype branch and predicts reading order directly from decoder
// queries via GlobalPointer. The decoder class/bbox heads are tied to the
// encoder heads in the HuggingFace checkpoint, so this implementation reuses
// those modules explicitly instead of allocating separate, unloaded weights.
// ========================================================================

use std::path::Path;

use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{DType, Int, Tensor, TensorData},
};
use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::{
    InputProjection, InputProjectionConfig,
    backbone::{HGNetV2Backbone, HGNetV2BackboneConfig},
    decoder::{
        MLPPredictionHead, MLPPredictionHeadConfig, PPDocLayoutV3Decoder,
        PPDocLayoutV3DecoderConfig, generate_anchors_with_dtype, inverse_sigmoid,
    },
    encoder::{HybridEncoderV3, HybridEncoderV3Config},
    load_adapter::PyTorchToBurnDTypeAdapter,
    reading_order::global_pointer::{GlobalPointer, GlobalPointerConfig},
};

fn tensor_to_vec_f32<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
    tensor
        .clone()
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
}

pub struct V3ModelOutput<B: Backend> {
    /// Classification logits: `[B, 300, num_labels]`.
    pub logits: Tensor<B, 3>,
    /// Predicted bbox (cx, cy, w, h) normalized: `[B, 300, 4]`.
    pub pred_boxes: Tensor<B, 3>,
    /// Reading order logits: `[B, 300, 300]`.
    pub order_logits: Tensor<B, 3>,
    /// Query masks: `[B, 300, 200, 200]`.
    pub out_masks: Tensor<B, 4>,
}

#[derive(Module, Debug)]
pub struct PPDocLayoutV3ForObjectDetection<B: Backend> {
    pub backbone: HGNetV2Backbone<B>,
    pub encoder_input_proj: Vec<InputProjection<B>>,
    pub encoder: HybridEncoderV3<B>,
    pub decoder_input_proj: Vec<InputProjection<B>>,

    pub enc_output_linear: Linear<B>,
    pub enc_output_norm: LayerNorm<B>,
    pub enc_score_head: Linear<B>,
    pub enc_bbox_head: MLPPredictionHead<B>,

    pub decoder: PPDocLayoutV3Decoder<B>,
    pub decoder_order_head: Vec<Linear<B>>,
    pub decoder_global_pointer: GlobalPointer<B>,
    pub decoder_norm: LayerNorm<B>,
    pub mask_query_head: MLPPredictionHead<B>,

    pub num_queries: usize,
    pub num_labels: usize,
    pub d_model: usize,
    pub num_prototypes: usize,
    pub mask_enhanced: bool,
}

impl<B: Backend> PPDocLayoutV3ForObjectDetection<B> {
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> V3ModelOutput<B> {
        let device = pixel_values.device();
        let [batch_size, _, _, _] = pixel_values.dims();

        let backbone_out = self.backbone.forward(pixel_values);
        let x4_feat = backbone_out.stage1;
        let feat_maps = vec![
            backbone_out.stage2,
            backbone_out.stage3,
            backbone_out.stage4,
        ];

        let proj_feats: Vec<Tensor<B, 4>> = feat_maps
            .into_iter()
            .zip(self.encoder_input_proj.iter())
            .map(|(feat, proj)| proj.forward(feat))
            .collect();

        let encoder_output = self.encoder.forward(proj_feats, x4_feat);
        let mask_feat = encoder_output.mask_feat;

        let mut source_flatten = Vec::new();
        let mut spatial_shapes = Vec::new();
        let mut level_start_index = Vec::new();
        let mut offset = 0usize;

        for (level, feat) in encoder_output.feature_maps.iter().enumerate() {
            let proj = self.decoder_input_proj[level].forward(feat.clone());
            let [_b, _c, h, w] = proj.dims();
            spatial_shapes.push((h, w));
            level_start_index.push(offset);
            offset += h * w;

            let flat = proj
                .reshape([batch_size, self.d_model, h * w])
                .swap_dims(1, 2);
            source_flatten.push(flat);
        }
        let source_flatten = Tensor::cat(source_flatten, 1);

        let (anchors, valid_mask) = generate_anchors_with_dtype::<B>(
            &spatial_shapes,
            0.05,
            &device,
            source_flatten.dtype(),
        );
        let memory = source_flatten.clone() * valid_mask;

        let output_memory = self.enc_output_linear.forward(memory);
        let output_memory = self.enc_output_norm.forward(output_memory);

        let enc_scores = self.enc_score_head.forward(output_memory.clone());
        let enc_bboxes = self.enc_bbox_head.forward(output_memory.clone()) + anchors;

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

        let topk_3d = topk_indices.clone().unsqueeze_dim::<3>(2).repeat_dim(2, 4);
        let mut reference_points_unact = enc_bboxes.gather(1, topk_3d);

        let topk_3d_d = topk_indices.unsqueeze_dim::<3>(2).repeat_dim(2, 256);
        let target = output_memory.gather(1, topk_3d_d);

        if self.mask_enhanced {
            let out_query = self.decoder_norm.forward(target.clone());
            let mask_query_embed = self.mask_query_head.forward(out_query);
            let [_b, mask_channels, mask_h, mask_w] = mask_feat.dims();
            let mask_feat_flat =
                mask_feat
                    .clone()
                    .reshape([batch_size, mask_channels, mask_h * mask_w]);
            let enc_out_masks = mask_query_embed.matmul(mask_feat_flat).reshape([
                batch_size,
                self.num_queries,
                mask_h,
                mask_w,
            ]);
            reference_points_unact =
                inverse_sigmoid(mask_to_box_coordinate::<B>(&enc_out_masks, &device));
        }

        let decoder_output = self.decoder.forward(
            target,
            reference_points_unact,
            &source_flatten,
            &spatial_shapes,
            &level_start_index,
            &self.enc_score_head,
            &self.enc_bbox_head,
            &self.decoder_order_head,
            &self.decoder_global_pointer,
            &self.mask_query_head,
            &self.decoder_norm,
            &mask_feat,
        );

        V3ModelOutput {
            logits: decoder_output.logits,
            pred_boxes: decoder_output.pred_boxes,
            order_logits: decoder_output.order_logits,
            out_masks: decoder_output.out_masks,
        }
    }
}

fn mask_to_box_coordinate<B: Backend>(masks: &Tensor<B, 4>, device: &B::Device) -> Tensor<B, 3> {
    let [batch_size, num_queries, height, width] = masks.dims();
    let masks_data = tensor_to_vec_f32(masks);
    let mut boxes = Vec::with_capacity(batch_size * num_queries * 4);

    for b in 0..batch_size {
        for q in 0..num_queries {
            let base = (b * num_queries + q) * height * width;
            let mut min_x = width;
            let mut min_y = height;
            let mut max_x = 0usize;
            let mut max_y = 0usize;
            let mut any = false;

            for y in 0..height {
                for x in 0..width {
                    if masks_data[base + y * width + x] > 0.0 {
                        any = true;
                        min_x = min_x.min(x);
                        min_y = min_y.min(y);
                        max_x = max_x.max(x);
                        max_y = max_y.max(y);
                    }
                }
            }

            if any {
                let x1 = min_x as f32 / width as f32;
                let y1 = min_y as f32 / height as f32;
                let x2 = (max_x + 1) as f32 / width as f32;
                let y2 = (max_y + 1) as f32 / height as f32;
                boxes.extend_from_slice(&[(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1]);
            } else {
                boxes.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            }
        }
    }

    Tensor::<B, 3>::from_data(
        TensorData::new(boxes, [batch_size, num_queries, 4]).convert_dtype(masks.dtype()),
        (device, masks.dtype()),
    )
}

#[derive(Config, Debug)]
pub struct PPDocLayoutV3Config {
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
    #[config(default = 1e-5)]
    pub batch_norm_eps: f64,
    #[config(default = 256)]
    pub encoder_hidden_dim: usize,
    pub encoder_in_channels: Option<[usize; 3]>,
    #[config(default = 1)]
    pub encoder_layers: usize,
    #[config(default = 1024)]
    pub encoder_ffn_dim: usize,
    #[config(default = 8)]
    pub encoder_attention_heads: usize,
    #[config(default = 10000.0)]
    pub positional_encoding_temperature: f64,
    #[config(default = 1.0)]
    pub hidden_expansion: f64,
    pub decoder_in_channels: Option<[usize; 3]>,
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 300)]
    pub num_queries: usize,
    #[config(default = 1024)]
    pub decoder_ffn_dim: usize,
    #[config(default = 3)]
    pub num_feature_levels: usize,
    #[config(default = 4)]
    pub decoder_n_points: usize,
    #[config(default = 6)]
    pub decoder_layers: usize,
    #[config(default = 8)]
    pub decoder_attention_heads: usize,
    #[config(default = 25)]
    pub num_labels: usize,
    #[config(default = 64)]
    pub global_pointer_head_size: usize,
    #[config(default = 32)]
    pub num_prototypes: usize,
    #[config(default = 64)]
    pub mask_feature_channels_in: usize,
    #[config(default = 64)]
    pub mask_feature_channels_out: usize,
    #[config(default = 128)]
    pub x4_feat_dim: usize,
    #[config(default = true)]
    pub mask_enhanced: bool,
}

impl PPDocLayoutV3Config {
    pub fn get_encoder_in_channels(&self) -> [usize; 3] {
        self.encoder_in_channels.unwrap_or([512, 1024, 2048])
    }

    pub fn get_decoder_in_channels(&self) -> [usize; 3] {
        self.decoder_in_channels.unwrap_or([256, 256, 256])
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> PPDocLayoutV3ForObjectDetection<B> {
        let bn_eps = self.batch_norm_eps;
        let ln_eps = self.layer_norm_eps;

        let backbone = HGNetV2BackboneConfig::new()
            .with_bn_eps(bn_eps)
            .init(device);

        let encoder_input_proj = self
            .get_encoder_in_channels()
            .iter()
            .map(|&in_ch| {
                InputProjectionConfig::new(in_ch, self.encoder_hidden_dim)
                    .with_bn_eps(bn_eps)
                    .init(device)
            })
            .collect();

        let encoder = HybridEncoderV3Config::new()
            .with_encoder_hidden_dim(self.encoder_hidden_dim)
            .with_encoder_layers(self.encoder_layers)
            .with_encoder_ffn_dim(self.encoder_ffn_dim)
            .with_encoder_attention_heads(self.encoder_attention_heads)
            .with_hidden_expansion(self.hidden_expansion)
            .with_positional_encoding_temperature(self.positional_encoding_temperature)
            .with_mask_feature_channels_in(self.mask_feature_channels_in)
            .with_mask_feature_channels_out(self.mask_feature_channels_out)
            .with_x4_feat_dim(self.x4_feat_dim)
            .with_num_prototypes(self.num_prototypes)
            .with_layer_norm_eps(ln_eps)
            .with_bn_eps(bn_eps)
            .init(device);

        let decoder_input_proj = self
            .get_decoder_in_channels()
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

        let decoder = PPDocLayoutV3DecoderConfig::new()
            .with_d_model(self.d_model)
            .with_n_heads(self.decoder_attention_heads)
            .with_ffn_dim(self.decoder_ffn_dim)
            .with_num_layers(self.decoder_layers)
            .with_n_levels(self.num_feature_levels)
            .with_n_points(self.decoder_n_points)
            .with_num_queries(self.num_queries)
            .with_layer_norm_eps(ln_eps)
            .init(device);

        let decoder_order_head = (0..self.decoder_layers)
            .map(|_| LinearConfig::new(self.d_model, self.d_model).init(device))
            .collect();
        let decoder_global_pointer = GlobalPointerConfig::new()
            .with_hidden_size(self.d_model)
            .with_head_size(self.global_pointer_head_size)
            .init(device);
        let decoder_norm = LayerNormConfig::new(self.d_model)
            .with_epsilon(ln_eps)
            .init(device);
        let mask_query_head =
            MLPPredictionHeadConfig::new(self.d_model, self.d_model, self.num_prototypes)
                .init(device);

        PPDocLayoutV3ForObjectDetection {
            backbone,
            encoder_input_proj,
            encoder,
            decoder_input_proj,
            enc_output_linear,
            enc_output_norm,
            enc_score_head,
            enc_bbox_head,
            decoder,
            decoder_order_head,
            decoder_global_pointer,
            decoder_norm,
            mask_query_head,
            num_queries: self.num_queries,
            num_labels: self.num_labels,
            d_model: self.d_model,
            num_prototypes: self.num_prototypes,
            mask_enhanced: self.mask_enhanced,
        }
    }

    /// Initialize the model and load weights from a safetensors file.
    ///
    /// This convenience wrapper panics when loading fails. Public APIs should
    /// prefer [`Self::try_init_from_safetensors`] so checkpoint errors can be
    /// returned to callers.
    pub fn init_from_safetensors<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> PPDocLayoutV3ForObjectDetection<B> {
        self.try_init_from_safetensors::<B>(path, device)
            .expect("failed to load PP-DocLayoutV3 weights")
    }

    pub fn init_from_safetensors_with_dtype<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
        target_dtype: DType,
    ) -> PPDocLayoutV3ForObjectDetection<B> {
        self.try_init_from_safetensors_with_dtype::<B>(path, device, target_dtype)
            .expect("failed to load PP-DocLayoutV3 weights")
    }

    /// Fallible checkpoint loader for PP-DocLayoutV3.
    ///
    /// Partial loading is intentional because the official checkpoint contains
    /// auxiliary tensors that are not part of the inference-only Burn module
    /// graph. Incompatible tensor shapes and unreadable files are still
    /// reported as errors.
    pub fn try_init_from_safetensors<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<PPDocLayoutV3ForObjectDetection<B>, String> {
        self.try_init_from_safetensors_with_dtype::<B>(path, device, DType::F32)
    }

    pub fn try_init_from_safetensors_with_dtype<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
        target_dtype: DType,
    ) -> Result<PPDocLayoutV3ForObjectDetection<B>, String> {
        let path = path.as_ref();
        let mut model = self.init::<B>(device);

        let mut st = SafetensorsStore::from_file(path.to_path_buf())
            .with_key_remapping(r"^model\.backbone\.model\.", "backbone.")
            .with_key_remapping(r"^model\.", "")
            .with_key_remapping(r"^backbone\.embedder\.", "backbone.embeddings.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.0\.", "backbone.stage1.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.1\.", "backbone.stage2.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.2\.", "backbone.stage3.")
            .with_key_remapping(r"^backbone\.encoder\.stages\.3\.", "backbone.stage4.")
            // Mask FPN scale heads store weighted convolutions inside a
            // Python ModuleList that also contains upsampling layers. Map them
            // before the generic HGNetV2 `.layers.*` rules, otherwise these
            // keys are mistaken for backbone ConvBlock variants.
            .with_key_remapping(
                r"encoder\.mask_feature_head\.scale_heads\.2\.layers\.2\.",
                "encoder.mask_feature_head.scale_heads.2.layers.1.",
            )
            .with_key_remapping(
                r"encoder\.mask_feature_head\.scale_heads\.(\d+)\.layers\.(\d+)\.convolution\.",
                "encoder.mask_feature_head.scale_heads.$1.layers.$2.conv.",
            )
            .with_key_remapping(
                r"encoder\.mask_feature_head\.scale_heads\.(\d+)\.layers\.(\d+)\.normalization\.",
                "encoder.mask_feature_head.scale_heads.$1.layers.$2.norm.",
            )
            .with_key_remapping(
                r"\.layers\.(\d+)\.(conv[12])\.convolution\.",
                ".layers.$1.Light.$2.conv.",
            )
            .with_key_remapping(
                r"\.layers\.(\d+)\.(conv[12])\.normalization\.",
                ".layers.$1.Light.$2.bn.",
            )
            .with_key_remapping(
                r"\.layers\.(\d+)\.convolution\.",
                ".layers.$1.Standard.conv.",
            )
            .with_key_remapping(
                r"\.layers\.(\d+)\.normalization\.",
                ".layers.$1.Standard.bn.",
            )
            .with_key_remapping(r"\.aggregation\.0\.", ".agg_squeeze.")
            .with_key_remapping(r"\.aggregation\.1\.", ".agg_excite.")
            .with_key_remapping(
                r"encoder\.mask_feature_head\.(.*)\.normalization\.",
                "encoder.mask_feature_head.$1.norm.",
            )
            .with_key_remapping(
                r"encoder\.encoder_mask_lateral\.normalization\.",
                "encoder.encoder_mask_lateral.norm.",
            )
            .with_key_remapping(
                r"encoder\.encoder_mask_output\.base_conv\.normalization\.",
                "encoder.encoder_mask_output.base_conv.norm.",
            )
            .with_key_remapping(r"\.convolution\.", ".conv.")
            .with_key_remapping(r"\.normalization\.", ".bn.")
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
            .with_key_remapping(r"^enc_output\.0\.", "enc_output_linear.")
            .with_key_remapping(r"^enc_output\.1\.", "enc_output_norm.")
            .with_key_remapping(r"encoder\.encoder\.", "encoder.aifi.")
            .with_key_remapping(
                r"encoder\.aifi\.(\d+)\.layers\.(\d+)\.self_attn\.out_proj\.",
                "encoder.aifi.$1.layers.$2.o_proj.",
            )
            .with_key_remapping(
                r"encoder\.aifi\.(\d+)\.layers\.(\d+)\.self_attn\.",
                "encoder.aifi.$1.layers.$2.",
            )
            .with_key_remapping(
                r"decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.",
                "decoder.layers.$1.cross_attn_layer_norm.",
            )
            .with_key_remapping(
                r"decoder\.layers\.(\d+)\.encoder_attn\.",
                "decoder.layers.$1.cross_attn.",
            )
            .with_key_remapping(
                r"decoder\.layers\.(\d+)\.self_attn\.out_proj\.",
                "decoder.layers.$1.self_attn.o_proj.",
            )
            .with_from_adapter(PyTorchToBurnDTypeAdapter::new(target_dtype))
            .allow_partial(true);

        model
            .load_from(&mut st)
            .map_err(|e| format!("failed to load {}: {e}", path.display()))?;

        Ok(model)
    }
}
