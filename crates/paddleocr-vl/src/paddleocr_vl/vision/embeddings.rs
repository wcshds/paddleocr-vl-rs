use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Int, Tensor,
        backend::Backend,
        ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
    },
};

#[derive(Module, Debug)]
pub struct PaddleOcrVisionEmbeddings<B: Backend> {
    pub patch_embedding: Conv2d<B>,
    pub position_embedding: Embedding<B>,
    pub embed_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_patches: usize,
    pub num_positions: usize,
}

impl<B: Backend> PaddleOcrVisionEmbeddings<B> {
    fn interpolate_pos_encoding(
        &self,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let sqrt_num_positions = (self.num_positions as f64).sqrt() as usize;
        let patch_pos_embed = self.position_embedding.weight.val();
        let [_, dim] = patch_pos_embed.dims();

        let patch_pos_embed = patch_pos_embed
            .reshape([1, sqrt_num_positions, sqrt_num_positions, dim])
            .permute([0, 3, 1, 2]);

        let sampling_grid =
            make_align_corners_false_grid::<B>(height, width, device).cast(patch_pos_embed.dtype());
        let patch_pos_embed = patch_pos_embed.grid_sample_2d(
            sampling_grid,
            GridSampleOptions::new(InterpolateMode::Bilinear)
                .with_padding_mode(GridSamplePaddingMode::Border)
                .with_align_corners(false),
        );

        patch_pos_embed
            .permute([0, 2, 3, 1])
            .reshape([1, height * width, dim])
    }

    /// Compute patch + position embeddings for all images.
    ///
    /// - `pixel_values`: `[batch, seq, channels, patch_h, patch_w]`
    /// - `image_grid_hw`: per-image grid dimensions (height_patches, width_patches)
    pub fn forward(
        &self,
        pixel_values: Tensor<B, 5>,
        image_grid_hw: &[(usize, usize)],
    ) -> Tensor<B, 2> {
        let [batch_size, sequence_len, channels, height, width] = pixel_values.dims();
        let device = pixel_values.device();

        assert_eq!(
            height, self.patch_size,
            "expected per-token height to equal patch_size"
        );
        assert_eq!(
            width, self.patch_size,
            "expected per-token width to equal patch_size"
        );

        let pixel_values =
            pixel_values.reshape([batch_size * sequence_len, channels, height, width]);
        let patch_embeds = self.patch_embedding.forward(pixel_values);
        let embeddings = patch_embeds.reshape([batch_size * sequence_len, self.embed_dim]);

        if image_grid_hw.is_empty() {
            return Tensor::<B, 2>::zeros([0, self.embed_dim], &device)
                .cast(self.position_embedding.weight.val().dtype());
        }

        let mut start = 0usize;
        let mut output = Vec::with_capacity(image_grid_hw.len());

        for &(h, w) in image_grid_hw {
            let token_count = h * w;
            let end = start + token_count;

            let image_embeddings = embeddings.clone().slice([start..end, 0..self.embed_dim]);
            let position_embedding = self
                .interpolate_pos_encoding(h, w, &device)
                .squeeze_dim::<2>(0);

            output.push(image_embeddings + position_embedding);
            start = end;
        }

        assert_eq!(
            start,
            batch_size * sequence_len,
            "image_grid_hw must cover all vision tokens exactly"
        );

        Tensor::cat(output, 0)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrVisionEmbeddingsConfig {
    #[config(default = 3)]
    num_channels: usize,
    #[config(default = 1152)]
    pub hidden_size: usize,
    #[config(default = 384)]
    pub image_size: usize,
    #[config(default = 14)]
    pub patch_size: usize,
}

impl PaddleOcrVisionEmbeddingsConfig {
    /// Build a `PaddleOcrVisionEmbeddings` module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrVisionEmbeddings<B> {
        let patch_embedding = Conv2dConfig::new(
            [self.num_channels, self.hidden_size],
            [self.patch_size, self.patch_size],
        )
        .with_bias(true)
        .with_stride([self.patch_size, self.patch_size])
        .with_padding(PaddingConfig2d::Valid)
        .init(device);

        let num_patches = (self.image_size / self.patch_size).pow(2);
        let num_positions = num_patches;
        let position_embedding = EmbeddingConfig::new(num_positions, self.hidden_size).init(device);

        PaddleOcrVisionEmbeddings {
            patch_embedding,
            position_embedding,
            embed_dim: self.hidden_size,
            image_size: self.image_size,
            patch_size: self.patch_size,
            num_patches,
            num_positions,
        }
    }
}

fn make_align_corners_false_grid<B: Backend>(
    out_height: usize,
    out_width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // x_norm[i] = 2 * (i + 0.5) / W - 1
    let x_norm = Tensor::<B, 1, Int>::arange(0..out_width as i64, device)
        .float()
        .add_scalar(0.5)
        .mul_scalar(2.0 / out_width as f32)
        .add_scalar(-1.0);

    // y_norm[i] = 2 * (i + 0.5) / H - 1
    let y_norm = Tensor::<B, 1, Int>::arange(0..out_height as i64, device)
        .float()
        .add_scalar(0.5)
        .mul_scalar(2.0 / out_height as f32)
        .add_scalar(-1.0);

    // Broadcast to [H, W] grids and stack along last dim → [1, H, W, 2]
    let x_grid = x_norm
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, out_height)
        .unsqueeze_dim::<3>(2);
    let y_grid = y_norm
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, out_width)
        .unsqueeze_dim::<3>(2);

    Tensor::cat(vec![x_grid, y_grid], 2).unsqueeze_dim::<4>(0)
}
