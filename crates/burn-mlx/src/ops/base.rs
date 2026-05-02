//! Base tensor operations for MLX backend.

use crate::tensor::MlxTensor;

/// Base operations on f32 MLX tensors.
impl MlxTensor<f32> {
    /// Reshape this tensor.
    pub fn reshape_to(&self, shape: &[i32]) -> Self {
        let array = self.array.reshape(shape).expect("Failed to reshape array");
        MlxTensor::new(array, self.device)
    }

    /// Transpose this tensor (reverses all axes).
    pub fn transpose_all(&self) -> Self {
        let array = mlx_rs::ops::transpose(&self.array).expect("Failed to transpose array");
        MlxTensor::new(array, self.device)
    }

    /// Expand tensor to a new shape.
    pub fn broadcast_to(&self, shape: &[i32]) -> Self {
        let array =
            mlx_rs::ops::broadcast_to(&self.array, shape).expect("Failed to broadcast array");
        MlxTensor::new(array, self.device)
    }
}
