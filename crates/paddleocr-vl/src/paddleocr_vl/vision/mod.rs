use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    prelude::Backend,
};

use crate::paddleocr_vl::vision::{
    embeddings::{PaddleOcrVisionEmbeddings, PaddleOcrVisionEmbeddingsConfig},
    encoder::{PaddleOCRVisionEncoder, PaddleOCRVisionEncoderConfig},
};

pub mod embeddings;
pub mod encoder;

#[derive(Module, Debug)]
pub struct PaddleOcrVisionModel<B: Backend> {
    pub embeddings: PaddleOcrVisionEmbeddings<B>,
    pub encoder: PaddleOCRVisionEncoder<B>,
    pub post_layernorm: LayerNorm<B>,
}

impl<B: Backend> PaddleOcrVisionModel<B> {
    pub fn forward(
        &self,
        pixel_values: Tensor<B, 5>,
        image_grid_hw: &[(usize, usize)],
    ) -> Tensor<B, 2> {
        let hidden_states = self.embeddings.forward(pixel_values, image_grid_hw);
        let last_hidden_state = self.encoder.forward(hidden_states, image_grid_hw);

        self.post_layernorm.forward(last_hidden_state)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrVisionModelConfig {
    #[config(default = 3)]
    pub num_channels: usize,
    #[config(default = 1152)]
    pub hidden_size: usize,
    #[config(default = 384)]
    pub image_size: usize,
    #[config(default = 14)]
    pub patch_size: usize,
    #[config(default = 27)]
    pub num_hidden_layers: usize,
    #[config(default = 4304)]
    pub intermediate_size: usize,
    #[config(default = 1e-06)]
    pub layer_norm_eps: f64,
    #[config(default = 16)]
    pub num_attention_heads: usize,
    #[config(default = 0.0)]
    pub attn_dropout: f64,
}

impl PaddleOcrVisionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrVisionModel<B> {
        PaddleOcrVisionModel {
            embeddings: PaddleOcrVisionEmbeddingsConfig::new()
                .with_num_channels(self.num_channels)
                .with_hidden_size(self.hidden_size)
                .with_image_size(self.image_size)
                .with_patch_size(self.patch_size)
                .init(device),
            encoder: PaddleOCRVisionEncoderConfig::new()
                .with_num_hidden_layers(self.num_hidden_layers)
                .with_embed_dim(self.hidden_size)
                .with_intermediate_size(self.intermediate_size)
                .with_layer_norm_eps(self.layer_norm_eps)
                .with_num_attention_heads(self.num_attention_heads)
                .with_attn_dropout(self.attn_dropout)
                .init(device),
            post_layernorm: LayerNormConfig::new(self.hidden_size)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
        }
    }
}
