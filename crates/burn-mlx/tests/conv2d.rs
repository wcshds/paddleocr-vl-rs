use burn::tensor::{Tensor, TensorData, module::conv2d, ops::ConvOptions};
use burn_mlx::Mlx;
use serial_test::serial;

type B = Mlx;

fn assert_close(actual: Vec<f32>, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch at {idx}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
#[serial]
fn conv2d_matches_nchw_cross_correlation() {
    let device = Default::default();
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [1, 1, 3, 3],
        ),
        &device,
    );
    let weight = Tensor::<B, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
        &device,
    );

    let y = conv2d(x, weight, None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
    assert_eq!(y.dims(), [1, 1, 2, 2]);
    assert_close(
        y.into_data().to_vec::<f32>().expect("conv output"),
        &[37.0, 47.0, 67.0, 77.0],
    );
}

#[test]
#[serial]
fn conv2d_preserves_channel_order() {
    let device = Default::default();
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                10.0, 20.0, 30.0, 40.0,
            ],
            [1, 2, 2, 2],
        ),
        &device,
    );
    let weight = Tensor::<B, 4>::from_data(TensorData::new(vec![2.0, 3.0], [1, 2, 1, 1]), &device);

    let y = conv2d(x, weight, None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
    assert_eq!(y.dims(), [1, 1, 2, 2]);
    assert_close(
        y.into_data().to_vec::<f32>().expect("conv output"),
        &[32.0, 64.0, 96.0, 128.0],
    );
}

#[test]
#[serial]
fn conv2d_handles_stride_and_padding() {
    let device = Default::default();
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
        &device,
    );
    let weight = Tensor::<B, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
        &device,
    );

    let y = conv2d(x, weight, None, ConvOptions::new([2, 2], [1, 1], [1, 1], 1));
    assert_eq!(y.dims(), [1, 1, 2, 2]);
    assert_close(
        y.into_data().to_vec::<f32>().expect("conv output"),
        &[4.0, 6.0, 6.0, 4.0],
    );
}

#[test]
#[serial]
fn conv2d_preserves_output_channel_order() {
    let device = Default::default();
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
        &device,
    );
    let weight = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            [2, 1, 2, 2],
        ),
        &device,
    );

    let y = conv2d(x, weight, None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
    assert_eq!(y.dims(), [1, 2, 1, 1]);
    assert_close(
        y.into_data().to_vec::<f32>().expect("conv output"),
        &[1.0, 4.0],
    );
}

#[test]
#[serial]
fn conv2d_reads_nchw_output_with_channels_and_spatial_axes() {
    let device = Default::default();
    let x = Tensor::<B, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
        &device,
    );
    let weight = Tensor::<B, 4>::from_data(TensorData::new(vec![1.0, 10.0], [2, 1, 1, 1]), &device);

    let y = conv2d(x, weight, None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
    assert_eq!(y.dims(), [1, 2, 2, 2]);
    assert_close(
        y.into_data().to_vec::<f32>().expect("conv output"),
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    );
}
