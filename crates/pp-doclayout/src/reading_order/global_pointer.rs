// ========================================================================
// PPDocLayoutV2 Reading Order — GlobalPointer
// ========================================================================
//
// GlobalPointer is the final prediction head for reading-order estimation.
// It projects the encoder's hidden states into Q/K vectors and computes
// pairwise reading-order scores between all elements.
//
// Computation flow:
//
// ```text
// hidden_states [B, L, 512]
//       │
//       ▼
// Linear(512 → head_size*2 = 128) → [B, L, 128]
//       │
//       ▼
// reshape → [B, L, 2, head_size=64]
// split → queries [B, L, 64], keys [B, L, 64]
//       │
//       ▼
// logits = queries @ keys^T / √head_size
//       │
//       ▼
// Lower-triangular mask (positions where i ≤ j are set to -1e4)
// → retains only "i → j" (i precedes j) scores
//       │
//       ▼
// logits [B, L, L]
// ```
//
// Design intent:
//   logits[b, i, j] > 0 means element i should appear before element j.
//   The lower-triangular mask ensures that only the upper-triangular part
//   (i < j) is considered, avoiding redundant computation and self-loops.
//
// At inference, sigmoid(logits) is used with a voting scheme to determine
// the final reading order.
// ========================================================================

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{Tensor, TensorData, s},
};

#[derive(Module, Debug)]
pub struct GlobalPointer<B: Backend> {
    /// Linear(hidden_size → head_size * 2)
    pub dense: Linear<B>,
    pub head_size: usize,
}

impl<B: Backend> GlobalPointer<B> {
    /// GlobalPointer forward pass.
    ///
    /// - `inputs`: `[B, L, hidden_size]` — encoder output (middle tokens already extracted)
    /// - Returns: `[B, L, L]` reading-order logits
    ///
    /// logits[b, i, j] represents the score that element i precedes element j.
    /// The lower triangle (including the diagonal) is masked to -1e4.
    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = inputs.dims();
        let device = inputs.device();
        let dtype = inputs.dtype();

        // Linear: [B, L, hidden] → [B, L, head_size*2]
        let projected = self.dense.forward(inputs);

        // Reshape → [B, L, 2, head_size]
        let projected = projected.reshape([batch_size, seq_len, 2, self.head_size]);

        // Split into queries and keys: [B, L, head_size]
        let queries = projected
            .clone()
            .slice(s![.., .., 0..1, ..])
            .squeeze_dim::<3>(2);
        let keys = projected.slice(s![.., .., 1..2, ..]).squeeze_dim::<3>(2);

        // logits = Q @ K^T / √head_size: [B, L, L]
        let scale = (self.head_size as f32).sqrt();
        let logits = queries.matmul(keys.swap_dims(1, 2)).div_scalar(scale);

        // Lower-triangular mask (including diagonal): positions i ≥ j → -1e4
        // tril mask: [L, L], mask[i][j] = 1 if i >= j
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }
        let mask = Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, [seq_len, seq_len]).convert_dtype(dtype),
            (&device, dtype),
        );

        // Masked positions (lower triangle) are set to -1e4
        let large_neg = mask.mul_scalar(-1e4f32); // lower triangle = -1e4, upper triangle = 0
        let mask_expanded = large_neg.unsqueeze_dim::<3>(0); // [1, L, L]

        logits + mask_expanded
    }
}

#[derive(Config, Debug)]
pub struct GlobalPointerConfig {
    #[config(default = 512)]
    pub hidden_size: usize,
    #[config(default = 64)]
    pub head_size: usize,
}

impl GlobalPointerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GlobalPointer<B> {
        GlobalPointer {
            dense: LinearConfig::new(self.hidden_size, self.head_size * 2).init(device),
            head_size: self.head_size,
        }
    }
}
