use burn::tensor::{Tensor, TensorData, activation};
use burn_mlx::Mlx;
use half::{bf16, f16};
use serial_test::serial;

fn assert_half_reduction_readback<B: burn::prelude::Backend>() {
    let device = B::Device::default();
    let logits = Tensor::<B, 3>::from_data(
        TensorData::new(vec![-8.0, 0.25, 3.0, -1.0, 2.0, -4.0], [1, 3, 2]),
        &device,
    );

    let max_logits = logits.max_dim(2).squeeze_dim::<2>(2);
    let probs = activation::sigmoid(max_logits);
    let values = probs
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("half precision readback");

    assert!(
        values.iter().all(|value| value.is_finite()),
        "half precision reduction/sigmoid readback should not produce NaN"
    );
    assert!(
        values.iter().all(|value| (0.0..=1.0).contains(value)),
        "sigmoid probabilities should stay in range"
    );
}

#[test]
#[serial]
fn f16_reduction_readback_is_finite() {
    assert_half_reduction_readback::<Mlx<f16>>();
}

#[test]
#[serial]
fn bf16_reduction_readback_is_finite() {
    assert_half_reduction_readback::<Mlx<bf16>>();
}
