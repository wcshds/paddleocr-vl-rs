use std::path::Path;

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{DType, Int, Tensor, TensorData, s},
};
use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::{PaddleOcrVersion, get_rope_index, load_adapter::PyTorchToBurnDTypeAdapter};

use self::{
    projector::{PaddleOcrProjector, PaddleOcrProjectorConfig},
    text::decoder::{LayerKVCache, TextPositionEmbeddings},
    text::{PaddleOcrTextModel, PaddleOcrTextModelConfig},
    vision::{PaddleOcrVisionModel, PaddleOcrVisionModelConfig},
};

pub mod projector;
pub mod text;
pub mod vision;

// ========================================================================
// PaddleOCR-VL Multimodal Model
// ========================================================================
//
// An end-to-end model that integrates the Vision Encoder, Projector,
// Text Decoder, and LM Head.
//
// ```text
//  pixel_values ──► Vision Encoder ──► Projector ──┐
//                                                   ├──► inputs_embeds
//  input_ids ──► Embedding ────────────────────────┘
//                                                      │
//                                        ┌─────────────┘
//                                        ▼
//                              Text Decoder (×18 layers)
//                                        │
//                                        ▼
//                                     RMSNorm
//                                        │
//                                        ▼
//                                     LM Head ──► logits
// ```
//
// Usage:
//   let config = PaddleOcrVlModelConfig::new();
//   let model = config.init_from_safetensors::<B>("model.safetensors", &device);
//   let token_ids = model.generate(pixel_values, &[(grid_h, grid_w)], &input_ids, 1024);
// ========================================================================

#[derive(Module, Debug)]
pub struct PaddleOcrVlModel<B: Backend> {
    pub vision: PaddleOcrVisionModel<B>,
    pub projector: PaddleOcrProjector<B>,
    pub text_model: PaddleOcrTextModel<B>,
    pub lm_head: Linear<B>,
    image_token_id: usize,
    vision_start_token_id: usize,
    eos_token_id: usize,
    spatial_merge_size: usize,
    hidden_size: usize,
}

impl<B: Backend> PaddleOcrVlModel<B> {
    /// KV-cached autoregressive generation
    ///
    /// # Arguments
    ///
    /// - `pixel_values`: `[1, num_images, C, H, W]` preprocessed image tensor
    /// - `image_grid_hw`: (grid_h, grid_w) for each image
    /// - `input_ids`: token ID sequence with image placeholders already expanded
    /// - `max_new_tokens`: maximum number of new tokens to generate
    ///
    /// # Returns
    ///
    /// Generated token ID sequence (excluding EOS)
    pub fn generate(
        &self,
        pixel_values: Tensor<B, 5>,
        image_grid_hw: &[(usize, usize)],
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Vec<u32> {
        let device = pixel_values.device();
        let image_token_id = self.image_token_id as u32;
        let eos_id = self.eos_token_id as u32;

        // ---- Vision + Projector ----
        // PaddlePaddle keeps `visual` and `mlp_AR` in fp32 even when the text
        // decoder runs in bf16. Mirror that path for vision precision.
        let image_embeds = self
            .vision
            .forward(pixel_values.cast(DType::F32), image_grid_hw);
        let image_embeds = self.projector.forward(image_embeds, image_grid_hw);

        // ---- Multimodal inputs_embeds ----
        let prefill_len = input_ids.len();
        let ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let ids_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i64, [1, prefill_len]), &device);
        let mut inputs_embeds = self.text_model.embed_tokens.forward(ids_tensor);

        let first_img_pos = input_ids
            .iter()
            .position(|&id| id == image_token_id)
            .expect("no image tokens in input_ids");
        let n_img_tokens = input_ids.iter().filter(|&&id| id == image_token_id).count();
        let n_proj = image_embeds.dims()[0];
        assert_eq!(
            n_img_tokens, n_proj,
            "image token count ({n_img_tokens}) ≠ projected count ({n_proj})"
        );
        let inputs_dtype = inputs_embeds.dtype();
        inputs_embeds = inputs_embeds.slice_assign(
            [
                0..1,
                first_img_pos..(first_img_pos + n_img_tokens),
                0..self.hidden_size,
            ],
            image_embeds.cast(inputs_dtype).unsqueeze_dim::<3>(0),
        );

        // ---- Pre-compute RoPE + causal mask ----
        let image_grid_thw: Vec<(usize, usize, usize)> =
            image_grid_hw.iter().map(|&(h, w)| (1, h, w)).collect();
        let rope = get_rope_index(
            input_ids,
            &image_grid_thw,
            image_token_id,
            self.vision_start_token_id as u32,
            self.spatial_merge_size,
        );
        let rope_deltas = rope.rope_deltas;
        let max_total_len = prefill_len + max_new_tokens;

        let mut all_t = rope.temporal;
        let mut all_h = rope.height;
        let mut all_w = rope.width;
        for i in 0..max_new_tokens {
            let pos = prefill_len as i64 + i as i64 + rope_deltas;
            all_t.push(pos);
            all_h.push(pos);
            all_w.push(pos);
        }
        let all_pos_flat: Vec<i64> = [&all_t[..], &all_h[..], &all_w[..]].concat();
        let all_pos_ids = Tensor::<B, 3, Int>::from_data(
            TensorData::new(all_pos_flat, [3, 1, max_total_len]),
            &device,
        );
        let full_rope_embs = self.text_model.rotary_emb.forward(&all_pos_ids);

        // ---- Prefill ----
        let num_layers = self.text_model.layers.len();
        let mut kv_caches: Vec<Option<LayerKVCache<B>>> = (0..num_layers).map(|_| None).collect();

        let prefill_pos = TextPositionEmbeddings {
            cos: full_rope_embs
                .cos
                .clone()
                .slice(s![0..1, 0..1, 0..prefill_len, ..]),
            sin: full_rope_embs
                .sin
                .clone()
                .slice(s![0..1, 0..1, 0..prefill_len, ..]),
        };

        let mut hidden = inputs_embeds;
        for (i, layer) in self.text_model.layers.iter().enumerate() {
            let (h, cache) = layer.forward_with_cache(hidden, &prefill_pos, kv_caches[i].take());
            hidden = h;
            kv_caches[i] = Some(cache);
        }

        let last_h = hidden.slice(s![0..1, (prefill_len - 1)..prefill_len, ..]);
        let last_h = self.text_model.norm.forward(last_h);
        let logits = self.lm_head.forward(last_h).cast(DType::F32);
        let mut next_id = logits
            .argmax(2)
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap()[0] as u32;

        // ---- Decode loop ----
        let mut generated: Vec<u32> = Vec::new();
        let mut cur_len = prefill_len;

        for _step in 0..max_new_tokens {
            if next_id == eos_id {
                break;
            }
            generated.push(next_id);
            cur_len += 1;

            let new_id_tensor = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![next_id as i64], [1, 1]),
                &device,
            );
            let new_embed = self.text_model.embed_tokens.forward(new_id_tensor);

            let pos_idx = cur_len - 1;
            let new_pos = TextPositionEmbeddings {
                cos: full_rope_embs
                    .cos
                    .clone()
                    .slice(s![0..1, 0..1, pos_idx..(pos_idx + 1), ..]),
                sin: full_rope_embs
                    .sin
                    .clone()
                    .slice(s![0..1, 0..1, pos_idx..(pos_idx + 1), ..]),
            };

            let mut hidden = new_embed;
            for (i, layer) in self.text_model.layers.iter().enumerate() {
                let (h, cache) = layer.forward_with_cache(hidden, &new_pos, kv_caches[i].take());
                hidden = h;
                kv_caches[i] = Some(cache);
            }

            let normed = self.text_model.norm.forward(hidden);
            let logits = self.lm_head.forward(normed).cast(DType::F32);
            next_id = logits
                .argmax(2)
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .unwrap()[0] as u32;
        }

        generated
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrVlModelConfig {
    #[config(default = 1152)]
    pub vision_hidden_size: usize,
    #[config(default = 1024)]
    pub text_hidden_size: usize,
    #[config(default = 103424)]
    pub vocab_size: usize,
    #[config(default = 2)]
    pub spatial_merge_size: usize,
    #[config(default = 100295)]
    pub image_token_id: usize,
    #[config(default = 101305)]
    pub vision_start_token_id: usize,
    #[config(default = 2)]
    pub eos_token_id: usize,
}

impl PaddleOcrVlModelConfig {
    pub fn for_version(_version: PaddleOcrVersion) -> Self {
        // PaddleOCR-VL-1.0 and 1.5 use the same module graph and tensor shapes.
        // The differences are in checkpoint values, tokenizer/template, and
        // image-processor defaults handled outside the model module.
        Self::new()
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrVlModel<B> {
        PaddleOcrVlModel {
            vision: PaddleOcrVisionModelConfig::new().init(device),
            projector: PaddleOcrProjectorConfig::new()
                .with_vision_hidden_size(self.vision_hidden_size)
                .with_text_hidden_size(self.text_hidden_size)
                .with_spatial_merge_size(self.spatial_merge_size)
                .init(device),
            text_model: PaddleOcrTextModelConfig::new()
                .with_hidden_size(self.text_hidden_size)
                .with_vocab_size(self.vocab_size)
                .init(device),
            lm_head: LinearConfig::new(self.text_hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
            image_token_id: self.image_token_id,
            vision_start_token_id: self.vision_start_token_id,
            eos_token_id: self.eos_token_id,
            spatial_merge_size: self.spatial_merge_size,
            hidden_size: self.text_hidden_size,
        }
    }

    /// Initialize the model and load weights from a safetensors file (defaults to F32).
    ///
    /// Prefix mapping from safetensors keys to sub-module paths:
    /// - `visual.vision_model.*` → `vision.*`
    /// - `mlp_AR.*`             → `projector.*`
    /// - `model.*`              → `text_model.*`
    /// - `lm_head.*`            → `lm_head.*` (no remapping needed)
    ///
    /// This convenience wrapper panics if the checkpoint cannot be loaded. Use
    /// [`Self::try_init_from_safetensors`] when exposing loading failures through
    /// public APIs or CLIs.
    pub fn init_from_safetensors<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> PaddleOcrVlModel<B> {
        self.init_from_safetensors_with_dtype::<B>(path, device, DType::F32)
    }

    /// Initialize the model and load weights from a safetensors file,
    /// converting all weights to the specified `target_dtype`.
    ///
    /// Supported `target_dtype` values: `DType::F32`, `DType::F16`, `DType::BF16`.
    ///
    /// This convenience wrapper panics if the checkpoint cannot be loaded. Use
    /// [`Self::try_init_from_safetensors_with_dtype`] when the caller needs a
    /// recoverable error.
    pub fn init_from_safetensors_with_dtype<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
        target_dtype: DType,
    ) -> PaddleOcrVlModel<B> {
        self.try_init_from_safetensors_with_dtype::<B>(path, device, target_dtype)
            .expect("failed to load model weights")
    }

    /// Fallible variant of [`Self::init_from_safetensors`] using `DType::F32`.
    pub fn try_init_from_safetensors<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<PaddleOcrVlModel<B>, String> {
        self.try_init_from_safetensors_with_dtype::<B>(path, device, DType::F32)
    }

    /// Fallible checkpoint loader used by public APIs and CLIs.
    ///
    /// The PaddleOCR-VL checkpoints contain a few keys that are not represented
    /// as standalone Burn modules, so partial loading is intentionally allowed.
    /// This still reports hard loading failures such as unreadable files,
    /// incompatible tensor shapes, or backend dtype errors.
    pub fn try_init_from_safetensors_with_dtype<B: Backend>(
        &self,
        path: impl AsRef<Path>,
        device: &B::Device,
        target_dtype: DType,
    ) -> Result<PaddleOcrVlModel<B>, String> {
        let path = path.as_ref();
        let mut model = self.init::<B>(device);

        let mut st = SafetensorsStore::from_file(path.to_path_buf())
            // ── Prefix mapping (safetensors key → Burn Module path) ──
            .with_key_remapping(r"^visual\.vision_model\.", "vision.")
            .with_key_remapping(r"^mlp_AR\.", "projector.")
            .with_key_remapping(r"^model\.", "text_model.")
            // ── Vision LayerNorm: weight→gamma, bias→beta ──
            .with_key_remapping(r"\.(.*?layer_norm.*?)\.weight$", ".$1.gamma")
            .with_key_remapping(r"\.(.*?layer_norm.*?)\.bias$", ".$1.beta")
            // ── Projector LayerNorm ──
            .with_key_remapping(r"(\.?)(.*?pre_norm.*?)\.weight$", "$1$2.gamma")
            .with_key_remapping(r"(\.?)(.*?pre_norm.*?)\.bias$", "$1$2.beta")
            // ── Text model RmsNorm ──
            .with_key_remapping(r"\.(.*?layernorm)\.weight$", ".$1.gamma")
            .with_key_remapping(r"\.norm\.weight$", ".norm.gamma")
            .with_from_adapter(PyTorchToBurnDTypeAdapter::new(target_dtype))
            .allow_partial(true);

        model
            .load_from(&mut st)
            .map_err(|e| format!("failed to load {}: {e}", path.display()))?;

        Ok(model)
    }
}
