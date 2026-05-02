// ========================================================================
// Frozen Batch Normalization (FrozenBatchNorm2d)
// ========================================================================
//
// A frozen BatchNorm layer used at inference time.  In PP-DocLayoutV2 every
// BatchNorm2d is replaced with this frozen variant, ensuring that
// running_mean and running_var are never updated.
//
// Standard BatchNorm formula:
//   y = (x − running_mean) / √(running_var + ε) × weight + bias
//
// The frozen version decomposes this into pre-computed multiply and add:
//   scale       = weight / √(running_var + ε)
//   bias_offset = bias − running_mean × scale
//   y = x × scale + bias_offset
//
// This pre-computation is more efficient at inference because it avoids
// a division in every forward pass.
//
// Weight sources (safetensors keys):
//   - weight:       [num_features]  learnable scale factor γ
//   - bias:         [num_features]  learnable bias β
//   - running_mean: [num_features]  accumulated mean from training
//   - running_var:  [num_features]  accumulated variance from training
//
// Reference: torchvision.misc.ops.FrozenBatchNorm2d
// ========================================================================

use burn::{
    config::Config,
    module::{Module, Param},
    prelude::Backend,
    tensor::Tensor,
};

/// Frozen BatchNorm2d.
///
/// All parameters are fixed at inference time and do not participate in
/// gradient computation.
/// Input shape:  `[batch_size, channels, height, width]`
/// Output shape: identical to input
#[derive(Module, Debug)]
pub struct FrozenBatchNorm2d<B: Backend> {
    /// Learnable scale factor γ. Shape: [num_features]
    pub weight: Param<Tensor<B, 1>>,
    /// Learnable bias β. Shape: [num_features]
    pub bias: Param<Tensor<B, 1>>,
    /// Accumulated training mean. Shape: [num_features]
    pub running_mean: Param<Tensor<B, 1>>,
    /// Accumulated training variance. Shape: [num_features]
    pub running_var: Param<Tensor<B, 1>>,
    /// Small constant to prevent division by zero
    pub epsilon: f64,
}

impl<B: Backend> FrozenBatchNorm2d<B> {
    /// Forward pass.
    ///
    /// Computes: output = x * scale + bias_offset
    /// where scale       = weight / √(running_var + ε)
    ///       bias_offset = bias − running_mean × scale
    ///
    /// - Input:  `[batch_size, channels, height, width]`
    /// - Output: `[batch_size, channels, height, width]` (shape unchanged)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Reshape all parameters to [1, C, 1, 1] for broadcasting with the 4D tensor
        let weight = self.weight.val().reshape([1, -1, 1, 1]);
        let bias = self.bias.val().reshape([1, -1, 1, 1]);
        let running_var = self.running_var.val().reshape([1, -1, 1, 1]);
        let running_mean = self.running_mean.val().reshape([1, -1, 1, 1]);

        // scale = weight * rsqrt(running_var + epsilon)
        let scale = weight * (running_var.add_scalar(self.epsilon)).sqrt().recip();
        // bias_offset = bias - running_mean * scale
        let bias_offset = bias - running_mean * scale.clone();

        x * scale + bias_offset
    }
}

#[derive(Config, Debug)]
pub struct FrozenBatchNorm2dConfig {
    pub num_features: usize,
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl FrozenBatchNorm2dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FrozenBatchNorm2d<B> {
        FrozenBatchNorm2d {
            weight: Param::from_tensor(Tensor::ones([self.num_features], device)),
            bias: Param::from_tensor(Tensor::zeros([self.num_features], device)),
            running_mean: Param::from_tensor(Tensor::zeros([self.num_features], device)),
            running_var: Param::from_tensor(Tensor::ones([self.num_features], device)),
            epsilon: self.epsilon,
        }
    }
}
