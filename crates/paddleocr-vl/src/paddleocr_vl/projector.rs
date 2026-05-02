use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend, s},
};

#[derive(Module, Debug)]
pub struct PaddleOcrProjector<B: Backend> {
    pub spatial_merge_size: usize,
    pub pre_norm: LayerNorm<B>,
    pub linear_1: Linear<B>,
    pub linear_2: Linear<B>,
}

impl<B: Backend> PaddleOcrProjector<B> {
    pub fn forward(
        &self,
        image_features: Tensor<B, 2>,
        image_grid_hw: &[(usize, usize)],
    ) -> Tensor<B, 2> {
        if image_grid_hw.is_empty() {
            panic!("image_grid_hw must not be empty")
        }

        let [token_count, hidden_size] = image_features.dims();
        let [_, vision_hidden_size] = image_features.dims();

        assert_eq!(
            hidden_size, vision_hidden_size,
            "projector expects vision hidden size to match the input feature width"
        );

        let mut start = 0usize;
        let mut processed_features = Vec::with_capacity(image_grid_hw.len());
        for &(h, w) in image_grid_hw {
            assert_eq!(
                h % self.spatial_merge_size,
                0,
                "image grid height must be divisible by spatial_merge_size"
            );
            assert_eq!(
                w % self.spatial_merge_size,
                0,
                "image grid width must be divisible by spatial_merge_size"
            );

            let end = start + h * w;
            let image_feature = image_features.clone().slice(s![start..end, ..]);
            let image_feature = self.pre_norm.forward(image_feature);

            let h_block = h / self.spatial_merge_size;
            let w_block = w / self.spatial_merge_size;
            let image_feature = image_feature.reshape([
                h_block,
                self.spatial_merge_size,
                w_block,
                self.spatial_merge_size,
                vision_hidden_size,
            ]);
            // shape: [h_block, w_block, spatial_merge_size(h), spatial_merge_size(w), vision_hidden_size]
            let image_feature = image_feature.swap_dims(1, 2);
            let image_feature = image_feature.reshape([
                h_block * w_block,
                self.spatial_merge_size * self.spatial_merge_size * vision_hidden_size,
            ]);

            let hidden_states = self.linear_1.forward(image_feature);
            let hidden_states = activation::gelu(hidden_states);
            let hidden_states = self.linear_2.forward(hidden_states);
            processed_features.push(hidden_states);

            start = end;
        }

        assert_eq!(
            start, token_count,
            "image_grid_hw must cover all image feature tokens exactly"
        );

        Tensor::cat(processed_features, 0)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrProjectorConfig {
    #[config(default = 1152)]
    pub vision_hidden_size: usize,
    #[config(default = 1024)]
    pub text_hidden_size: usize,
    #[config(default = 2)]
    pub spatial_merge_size: usize,
}

impl PaddleOcrProjectorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrProjector<B> {
        let merged_hidden_size =
            self.spatial_merge_size * self.spatial_merge_size * self.vision_hidden_size;

        PaddleOcrProjector {
            spatial_merge_size: self.spatial_merge_size,
            pre_norm: LayerNormConfig::new(self.vision_hidden_size)
                .with_epsilon(1e-05)
                .init(device),
            linear_1: LinearConfig::new(merged_hidden_size, merged_hidden_size)
                .with_bias(true)
                .init(device),
            linear_2: LinearConfig::new(merged_hidden_size, self.text_hidden_size)
                .with_bias(true)
                .init(device),
        }
    }
}
