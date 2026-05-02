// ========================================================================
// HGNetV2 Backbone
// ========================================================================
//
// HGNetV2 (High-performance GPU Net V2) is an efficient CNN backbone
// developed by Baidu, optimized for GPU inference performance.
// It serves as the feature extractor in PP-DocLayoutV2.
//
// Overall architecture (arch="L"):
//
// ```text
// pixel_values [B, 3, 800, 800]
//       │
//       ▼
// ┌──────────────────────────┐
// │  Stem (Embeddings)       │  → [B, 48, 200, 200]
// └──────────────────────────┘
//       │
//       ▼
// ┌──────────────────────────┐
// │  Stage1                  │  48 → 128, no downsampling
// │  1 block, k=3, std conv  │  → [B, 128, 200, 200]
// └──────────────────────────┘
//       │
//       ▼
// ┌──────────────────────────┐
// │  Stage2 ★output          │  128 → 512, stride=2 downsampling
// │  1 block, k=3, std conv  │  → [B, 512, 100, 100]
// └──────────────────────────┘
//       │
//       ▼
// ┌──────────────────────────┐
// │  Stage3 ★output          │  512 → 1024, stride=2 downsampling
// │  3 blocks, k=5, light    │  → [B, 1024, 50, 50]
// └──────────────────────────┘
//       │
//       ▼
// ┌──────────────────────────┐
// │  Stage4 ★output          │  1024 → 2048, stride=2 downsampling
// │  1 block, k=5, light     │  → [B, 2048, 25, 25]
// └──────────────────────────┘
// ```
//
// PP-DocLayoutV2 uses the outputs of Stage2, Stage3, and Stage4 as multi-scale
// feature maps, corresponding to out_features=["stage2", "stage3", "stage4"].
// ========================================================================

pub mod embeddings;
pub mod stage;

use burn::{config::Config, module::Module, prelude::Backend, tensor::Tensor};

use self::{
    embeddings::{HGNetV2Embeddings, HGNetV2EmbeddingsConfig},
    stage::{HGNetV2Stage, HGNetV2StageConfig},
};

/// Output of the HGNetV2 backbone network
pub struct BackboneOutput<B: Backend> {
    /// Stage1 output: `[B, 128, H/4, W/4]` (stride=4). PP-DocLayoutV3 uses
    /// this high-resolution map for its mask feature branch.
    pub stage1: Tensor<B, 4>,
    /// Stage2 output: `[B, 512, H/8, W/8]` (stride=8)
    pub stage2: Tensor<B, 4>,
    /// Stage3 output: `[B, 1024, H/16, W/16]` (stride=16)
    pub stage3: Tensor<B, 4>,
    /// Stage4 output: `[B, 2048, H/32, W/32]` (stride=32)
    pub stage4: Tensor<B, 4>,
}

#[derive(Module, Debug)]
pub struct HGNetV2Backbone<B: Backend> {
    pub embeddings: HGNetV2Embeddings<B>,
    pub stage1: HGNetV2Stage<B>,
    pub stage2: HGNetV2Stage<B>,
    pub stage3: HGNetV2Stage<B>,
    pub stage4: HGNetV2Stage<B>,
}

impl<B: Backend> HGNetV2Backbone<B> {
    /// Forward pass, returning feature maps used by both V2 and V3.
    ///
    /// - Input: `[B, 3, H, W]` (H=W=800)
    /// - Output: `BackboneOutput` containing feature maps from stage1/2/3/4
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> BackboneOutput<B> {
        // Stem: [B, 3, 800, 800] → [B, 48, 200, 200]
        let x = self.embeddings.forward(pixel_values);

        // Stage1: [B, 48, 200, 200] → [B, 128, 200, 200]
        let feat1 = self.stage1.forward(x);

        // Stage2: [B, 128, 200, 200] → [B, 512, 100, 100]
        let feat2 = self.stage2.forward(feat1.clone());

        // Stage3: [B, 512, 100, 100] → [B, 1024, 50, 50]
        let feat3 = self.stage3.forward(feat2.clone());

        // Stage4: [B, 1024, 50, 50] → [B, 2048, 25, 25]
        let feat4 = self.stage4.forward(feat3.clone());

        BackboneOutput {
            stage1: feat1,
            stage2: feat2,
            stage3: feat3,
            stage4: feat4,
        }
    }
}

/// HGNetV2-L backbone network configuration
///
/// Fixed configuration for arch="L"; all BatchNorm layers use FrozenBatchNorm2d.
#[derive(Config, Debug)]
pub struct HGNetV2BackboneConfig {
    #[config(default = 1e-5)]
    pub bn_eps: f64,
}

impl HGNetV2BackboneConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HGNetV2Backbone<B> {
        let embeddings = HGNetV2EmbeddingsConfig::new()
            .with_bn_eps(self.bn_eps)
            .init(device);

        // Stage1: 48→128, no downsampling, 1 block, k=3, standard conv, 6 layers
        let stage1 = HGNetV2StageConfig::new(48, 48, 128, 1, false, false, 3)
            .with_bn_eps(self.bn_eps)
            .init(device);

        // Stage2: 128→512, stride=2, 1 block, k=3, standard conv, 6 layers
        let stage2 = HGNetV2StageConfig::new(128, 96, 512, 1, true, false, 3)
            .with_bn_eps(self.bn_eps)
            .init(device);

        // Stage3: 512→1024, stride=2, 3 blocks, k=5, light conv, 6 layers
        let stage3 = HGNetV2StageConfig::new(512, 192, 1024, 3, true, true, 5)
            .with_bn_eps(self.bn_eps)
            .init(device);

        // Stage4: 1024→2048, stride=2, 1 block, k=5, light conv, 6 layers
        let stage4 = HGNetV2StageConfig::new(1024, 384, 2048, 1, true, true, 5)
            .with_bn_eps(self.bn_eps)
            .init(device);

        HGNetV2Backbone {
            embeddings,
            stage1,
            stage2,
            stage3,
            stage4,
        }
    }
}
