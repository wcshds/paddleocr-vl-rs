//! Module operations for MLX backend (neural network primitives).

use burn_tensor::ops::{
    AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
    DeformConvOptions, InterpolateMode, InterpolateOptions, MaxPool1dWithIndices,
    MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use mlx_rs::Array;
use mlx_rs::module::Module as _;
use mlx_rs::nn::{Upsample, UpsampleMode};
use mlx_rs::ops::indexing::take_axis;

use crate::backend::{Mlx, MlxTensorPrimitive};
use crate::element::FloatMlxElement;

fn contiguous(array: &Array) -> Array {
    let stream = mlx_rs::Stream::default();
    let mut out = mlx_sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    let status =
        unsafe { mlx_sys::mlx_contiguous(&mut out, array.as_ptr(), false, stream.as_ptr()) };
    if status == 0 {
        unsafe { Array::from_ptr(out) }
    } else {
        panic!("mlx_contiguous failed")
    }
}

fn require_pool1d_options(dilation: usize, ceil_mode: bool) {
    if dilation != 1 {
        panic!("MLX backend only supports max_pool1d dilation=1, got {dilation}")
    }
    if ceil_mode {
        panic!("MLX backend does not support max_pool1d ceil_mode=true")
    }
}

fn require_pool2d_options(dilation: [usize; 2], ceil_mode: bool) {
    if dilation != [1, 1] {
        panic!("MLX backend only supports max_pool2d dilation=[1, 1], got {dilation:?}")
    }
    if ceil_mode {
        panic!("MLX backend does not support max_pool2d ceil_mode=true")
    }
}

fn require_avg_pool_options(count_include_pad: bool, ceil_mode: bool) {
    if count_include_pad {
        panic!("MLX backend does not support avg_pool count_include_pad=true")
    }
    if ceil_mode {
        panic!("MLX backend does not support avg_pool ceil_mode=true")
    }
}

/// Helper function to compute pooling using as_strided approach.
/// This follows the pattern from mlx-rs nn/pooling.rs.
///
/// Input shape for 2D: [N, H, W, C] (NHWC format - MLX native)
/// Returns: pooled output with shape [N, out_H, out_W, C]
fn pool2d_strided<F>(x: &Array, kernel_size: [usize; 2], stride: [usize; 2], pooling_op: F) -> Array
where
    F: Fn(&Array, &[i32]) -> Result<Array, mlx_rs::error::Exception>,
{
    let shape = x.shape();
    let n = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    let kh = kernel_size[0] as i32;
    let kw = kernel_size[1] as i32;
    let sh = stride[0] as i64;
    let sw = stride[1] as i64;

    // Calculate output dimensions
    let out_h = (h as i32 - kh) / stride[0] as i32 + 1;
    let out_w = (w as i32 - kw) / stride[1] as i32 + 1;

    // Build final shape: [N, out_H, out_W, kH, kW, C]
    let final_shape = vec![n, out_h, out_w, kh, kw, c];

    // Compute strides for a contiguous [N, H, W, C] array. Callers should pass
    // arrays produced by MLX ops that materialize or pad into a contiguous
    // layout; using this helper on arbitrary strided views would be incorrect.
    let orig_strides: Vec<i64> = {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    };

    // Final strides: [N_stride, H_stride*sh, W_stride*sw, H_stride, W_stride, C_stride]
    let final_strides = vec![
        orig_strides[0],      // N stride
        orig_strides[1] * sh, // out_H stride (moves by stride[0] in H dimension)
        orig_strides[2] * sw, // out_W stride (moves by stride[1] in W dimension)
        orig_strides[1],      // kH stride (moves by 1 in H dimension)
        orig_strides[2],      // kW stride (moves by 1 in W dimension)
        orig_strides[3],      // C stride
    ];

    // Create strided view
    let strided =
        mlx_rs::ops::as_strided(x, &final_shape[..], &final_strides[..], None).expect("as_strided");

    // Apply pooling operation on kernel dimensions (axes -3 and -2, i.e., 3 and 4)
    // This reduces [N, out_H, out_W, kH, kW, C] -> [N, out_H, out_W, C]
    let axes = [-3, -2];
    pooling_op(&strided, &axes).expect("pooling reduction")
}

/// Helper function for 1D pooling using as_strided approach.
/// Input shape: [N, L, C] (NLC format - MLX native)
/// Returns: pooled output with shape [N, out_L, C]
fn pool1d_strided<F>(x: &Array, kernel_size: usize, stride: usize, pooling_op: F) -> Array
where
    F: Fn(&Array, &[i32]) -> Result<Array, mlx_rs::error::Exception>,
{
    let shape = x.shape();
    let n = shape[0];
    let l = shape[1];
    let c = shape[2];

    let k = kernel_size as i32;
    let s = stride as i64;

    // Calculate output dimension
    let out_l = (l as i32 - k) / stride as i32 + 1;

    // Build final shape: [N, out_L, K, C]
    let final_shape = vec![n, out_l, k, c];

    // Compute strides for a contiguous [N, L, C] array. See the 2D helper for
    // the same restriction.
    let orig_strides: Vec<i64> = {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    };

    // Final strides: [N_stride, L_stride*s, L_stride, C_stride]
    let final_strides = vec![
        orig_strides[0],     // N stride
        orig_strides[1] * s, // out_L stride
        orig_strides[1],     // K stride
        orig_strides[2],     // C stride
    ];

    // Create strided view
    let strided =
        mlx_rs::ops::as_strided(x, &final_shape[..], &final_strides[..], None).expect("as_strided");

    // Apply pooling operation on kernel dimension (axis -2, i.e., 2)
    let axes = [-2];
    pooling_op(&strided, &axes).expect("pooling reduction")
}

/// Helper for max_pool2d_with_indices.
/// Returns both max values and flat indices into the padded input.
/// Input shape: [N, H, W, C] (NHWC format)
/// Returns: (output [N, out_H, out_W, C], indices [N, out_H, out_W, C])
fn max_pool2d_with_indices_impl(
    x: &Array,
    kernel_size: [usize; 2],
    stride: [usize; 2],
) -> (Array, Array) {
    let shape = x.shape();
    let n = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    let kh = kernel_size[0] as i32;
    let kw = kernel_size[1] as i32;
    let sh = stride[0] as i64;
    let sw = stride[1] as i64;

    // Calculate output dimensions
    let out_h = (h as i32 - kh) / stride[0] as i32 + 1;
    let out_w = (w as i32 - kw) / stride[1] as i32 + 1;

    // Build final shape: [N, out_H, out_W, kH, kW, C]
    let final_shape = vec![n, out_h, out_w, kh, kw, c];

    // Compute strides for the original array [N, H, W, C]
    let orig_strides: Vec<i64> = {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    };

    // Final strides: [N_stride, H_stride*sh, W_stride*sw, H_stride, W_stride, C_stride]
    let final_strides = vec![
        orig_strides[0],
        orig_strides[1] * sh,
        orig_strides[2] * sw,
        orig_strides[1],
        orig_strides[2],
        orig_strides[3],
    ];

    // Create strided view: [N, out_H, out_W, kH, kW, C]
    let strided =
        mlx_rs::ops::as_strided(x, &final_shape[..], &final_strides[..], None).expect("as_strided");

    // Flatten kernel dimensions: [N, out_H, out_W, kH*kW, C]
    let flat_kernel = kh * kw;
    let reshaped = strided
        .reshape(&[n, out_h, out_w, flat_kernel, c])
        .expect("reshape");

    // Get max values: reduce on axis 3 (the flattened kernel axis)
    let output = reshaped.max_axis(3, None).expect("max_axis");

    // Get argmax indices within each kernel window (axis 3)
    let local_indices = mlx_rs::ops::indexing::argmax_axis(&reshaped, 3, None).expect("argmax");

    // Convert local indices (within kernel) to flat indices into padded NHWC input
    let out_h_size = out_h as usize;
    let out_w_size = out_w as usize;
    let n_size = n as usize;
    let c_size = c as usize;
    let h_size = h as usize;
    let w_size = w as usize;

    // Create index arrays for n, oh, ow, c dimensions
    let n_range: Vec<i32> = (0..n_size as i32).collect();
    let n_idx = Array::from_slice(&n_range, &[n_size as i32])
        .reshape(&[n, 1, 1, 1])
        .expect("reshape");

    let oh_range: Vec<i32> = (0..out_h_size as i32).collect();
    let oh_idx = Array::from_slice(&oh_range, &[out_h_size as i32])
        .reshape(&[1, out_h, 1, 1])
        .expect("reshape");

    let ow_range: Vec<i32> = (0..out_w_size as i32).collect();
    let ow_idx = Array::from_slice(&ow_range, &[out_w_size as i32])
        .reshape(&[1, 1, out_w, 1])
        .expect("reshape");

    let c_range: Vec<i32> = (0..c_size as i32).collect();
    let c_idx = Array::from_slice(&c_range, &[c_size as i32])
        .reshape(&[1, 1, 1, c])
        .expect("reshape");

    // Compute local_h and local_w from local_indices
    let kw_arr = Array::from_int(kw);
    let local_h = mlx_rs::ops::floor_divide(&local_indices, &kw_arr).expect("div");
    let local_w = mlx_rs::ops::remainder(&local_indices, &kw_arr).expect("rem");

    // Compute actual h and w positions in padded input
    let sh_arr = Array::from_int(stride[0] as i32);
    let sw_arr = Array::from_int(stride[1] as i32);

    let actual_h = mlx_rs::ops::add(
        &mlx_rs::ops::multiply(&oh_idx, &sh_arr).expect("mul"),
        &local_h,
    )
    .expect("add");

    let actual_w = mlx_rs::ops::add(
        &mlx_rs::ops::multiply(&ow_idx, &sw_arr).expect("mul"),
        &local_w,
    )
    .expect("add");

    // Compute flat index: n * (H * W * C) + h * (W * C) + w * C + c
    let hwc = Array::from_int((h_size * w_size * c_size) as i32);
    let wc = Array::from_int((w_size * c_size) as i32);
    let c_stride = Array::from_int(c_size as i32);

    let flat_indices = mlx_rs::ops::add(
        &mlx_rs::ops::add(
            &mlx_rs::ops::add(
                &mlx_rs::ops::multiply(&n_idx, &hwc).expect("mul"),
                &mlx_rs::ops::multiply(&actual_h, &wc).expect("mul"),
            )
            .expect("add"),
            &mlx_rs::ops::multiply(&actual_w, &c_stride).expect("mul"),
        )
        .expect("add"),
        &c_idx,
    )
    .expect("add");

    (output, flat_indices)
}

impl<F: FloatMlxElement> ModuleOps<Self> for Mlx<F> {
    fn conv1d(
        x: MlxTensorPrimitive,
        weight: MlxTensorPrimitive,
        bias: Option<MlxTensorPrimitive>,
        options: ConvOptions<1>,
    ) -> MlxTensorPrimitive {
        // MLX conv1d: expects [N, L, C_in], weight [C_out, K, C_in]
        // Burn uses [N, C_in, L], weight [C_out, C_in, K]

        // Transpose input from [N, C_in, L] to [N, L, C_in]
        let x_t = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 1]).expect("transpose");

        // Transpose weight from [C_out, C_in, K] to [C_out, K, C_in]
        let w_t = mlx_rs::ops::transpose_axes(&weight.array, &[0, 2, 1]).expect("transpose");

        let result = mlx_rs::ops::conv1d(
            &x_t,
            &w_t,
            options.stride[0] as i32,
            options.padding[0] as i32,
            options.dilation[0] as i32,
            options.groups as i32,
        )
        .expect("conv1d");

        // Transpose output back from [N, L_out, C_out] to [N, C_out, L_out]
        let mut output = mlx_rs::ops::transpose_axes(&result, &[0, 2, 1]).expect("transpose");

        // Add bias if provided
        if let Some(b) = bias {
            let b_shape = b.shape();
            let b_reshaped = b
                .array
                .reshape(&[1, b_shape[0] as i32, 1])
                .expect("reshape bias");
            output = mlx_rs::ops::add(&output, &b_reshaped).expect("add bias");
        }

        MlxTensorPrimitive::new(output)
    }

    fn conv2d(
        x: MlxTensorPrimitive,
        weight: MlxTensorPrimitive,
        bias: Option<MlxTensorPrimitive>,
        options: ConvOptions<2>,
    ) -> MlxTensorPrimitive {
        // MLX conv2d: expects [N, H, W, C_in], weight [C_out, Kh, Kw, C_in]
        // Burn uses [N, C_in, H, W], weight [C_out, C_in, Kh, Kw]

        // Transpose input from [N, C_in, H, W] to [N, H, W, C_in]
        let x_t = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        // Transpose weight from [C_out, C_in, Kh, Kw] to [C_out, Kh, Kw, C_in]
        let w_t = mlx_rs::ops::transpose_axes(&weight.array, &[0, 2, 3, 1]).expect("transpose");

        let stride = (options.stride[0] as i32, options.stride[1] as i32);
        let padding = (options.padding[0] as i32, options.padding[1] as i32);
        let dilation = (options.dilation[0] as i32, options.dilation[1] as i32);

        let result =
            mlx_rs::ops::conv2d(&x_t, &w_t, stride, padding, dilation, options.groups as i32)
                .expect("conv2d");

        // Transpose output back from [N, H_out, W_out, C_out] to [N, C_out, H_out, W_out].
        // MLX transpose returns a view; Burn's data extraction and subsequent
        // NCHW ops expect the logical order to be materialized. Making the
        // result contiguous here prevents NHWC backing storage from leaking
        // through when tensors are later read or reused.
        let mut output = mlx_rs::ops::transpose_axes(&result, &[0, 3, 1, 2]).expect("transpose");
        output = contiguous(&output);

        // Add bias if provided
        if let Some(b) = bias {
            let b_shape = b.shape();
            let b_reshaped = b
                .array
                .reshape(&[1, b_shape[0] as i32, 1, 1])
                .expect("reshape bias");
            output = mlx_rs::ops::add(&output, &b_reshaped).expect("add bias");
        }

        MlxTensorPrimitive::new(output)
    }

    fn conv3d(
        _x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvOptions<3>,
    ) -> MlxTensorPrimitive {
        panic!("conv3d is not yet supported by the MLX backend")
    }

    fn conv_transpose1d(
        _x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvTransposeOptions<1>,
    ) -> MlxTensorPrimitive {
        panic!("conv_transpose1d is not yet supported by the MLX backend")
    }

    fn conv_transpose2d(
        _x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvTransposeOptions<2>,
    ) -> MlxTensorPrimitive {
        panic!("conv_transpose2d is not yet supported by the MLX backend")
    }

    fn conv_transpose3d(
        _x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvTransposeOptions<3>,
    ) -> MlxTensorPrimitive {
        panic!("conv_transpose3d is not yet supported by the MLX backend")
    }

    fn deform_conv2d(
        _x: MlxTensorPrimitive,
        _offset: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _mask: Option<MlxTensorPrimitive>,
        _bias: Option<MlxTensorPrimitive>,
        _options: DeformConvOptions<2>,
    ) -> MlxTensorPrimitive {
        panic!("deform_conv2d is not yet supported by the MLX backend")
    }

    fn deform_conv2d_backward(
        _x: MlxTensorPrimitive,
        _offset: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _mask: Option<MlxTensorPrimitive>,
        _bias: Option<MlxTensorPrimitive>,
        _out_grad: MlxTensorPrimitive,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Mlx<F>> {
        panic!("deform_conv2d_backward is not yet supported by the MLX backend")
    }

    fn avg_pool1d(
        x: MlxTensorPrimitive,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> MlxTensorPrimitive {
        require_avg_pool_options(count_include_pad, ceil_mode);
        // Burn uses NCL format, MLX uses NLC format.
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 1]).expect("transpose");

        let x_padded = if padding > 0 {
            let pad = padding as i32;
            mlx_rs::ops::pad(&x_nhwc, &[(0, 0), (pad, pad), (0, 0)], None, None).expect("pad")
        } else {
            x_nhwc
        };

        let pooled = pool1d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.mean_axes(axes, None)
        });

        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 2, 1]).expect("transpose");
        MlxTensorPrimitive::new(output)
    }

    fn avg_pool2d(
        x: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> MlxTensorPrimitive {
        require_avg_pool_options(count_include_pad, ceil_mode);
        // Burn uses NCHW format, MLX uses NHWC format
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        let x_padded = if padding[0] > 0 || padding[1] > 0 {
            let pad_h = padding[0] as i32;
            let pad_w = padding[1] as i32;
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)],
                None,
                None,
            )
            .expect("pad")
        } else {
            x_nhwc
        };

        let pooled = pool2d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.mean_axes(axes, None)
        });

        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 3, 1, 2]).expect("transpose");
        MlxTensorPrimitive::new(output)
    }

    fn avg_pool2d_backward(
        x: MlxTensorPrimitive,
        grad: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> MlxTensorPrimitive {
        require_avg_pool_options(count_include_pad, ceil_mode);
        let input_shape = x.shape();
        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];

        let kh = kernel_size[0];
        let kw = kernel_size[1];
        let sh = stride[0];
        let sw = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];

        let h_padded = h + 2 * pad_h;
        let w_padded = w + 2 * pad_w;

        let out_h = (h_padded - kh) / sh + 1;
        let out_w = (w_padded - kw) / sw + 1;

        let pool_size = (kh * kw) as f32;

        let grad_nhwc = mlx_rs::ops::transpose_axes(&grad.array, &[0, 2, 3, 1]).expect("transpose");

        let scale = F::f64_scalar_array(1.0 / pool_size as f64);
        let grad_scaled = mlx_rs::ops::multiply(&grad_nhwc, &scale).expect("multiply");

        let grad_input_padded =
            F::zeros_array(&[n as i32, h_padded as i32, w_padded as i32, c as i32]);

        let mut all_indices: Vec<i32> = Vec::with_capacity(n * out_h * out_w * kh * kw * c);
        let mut update_indices: Vec<usize> = Vec::with_capacity(n * out_h * out_w * kh * kw * c);

        for ni in 0..n {
            for ohi in 0..out_h {
                for owi in 0..out_w {
                    let h_start = ohi * sh;
                    let w_start = owi * sw;
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let hi = h_start + khi;
                            let wi = w_start + kwi;
                            for ci in 0..c {
                                let flat_idx = (ni * h_padded * w_padded * c
                                    + hi * w_padded * c
                                    + wi * c
                                    + ci) as i32;
                                all_indices.push(flat_idx);
                                let grad_idx =
                                    ni * out_h * out_w * c + ohi * out_w * c + owi * c + ci;
                                update_indices.push(grad_idx);
                            }
                        }
                    }
                }
            }
        }

        let grad_flat = grad_scaled.flatten(None, None).expect("flatten");
        let update_idx_arr = Array::from_slice(
            &update_indices.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &[update_indices.len() as i32],
        );
        let updates = take_axis(&grad_flat, &update_idx_arr, 0).expect("take");

        let grad_input_flat = grad_input_padded.flatten(None, None).expect("flatten");
        let indices_arr = Array::from_slice(&all_indices, &[all_indices.len() as i32]);

        let result_flat =
            mlx_rs::ops::scatter_add(&grad_input_flat, &[&indices_arr], &updates, &[0])
                .expect("scatter_add");

        let result_nhwc = result_flat
            .reshape(&[n as i32, h_padded as i32, w_padded as i32, c as i32])
            .expect("reshape");

        let result_unpadded = if pad_h > 0 || pad_w > 0 {
            mlx_rs::ops::slice(
                &result_nhwc,
                &[0, pad_h as i32, pad_w as i32, 0],
                &[n as i32, (pad_h + h) as i32, (pad_w + w) as i32, c as i32],
                None,
            )
            .expect("slice")
        } else {
            result_nhwc
        };

        let output =
            mlx_rs::ops::transpose_axes(&result_unpadded, &[0, 3, 1, 2]).expect("transpose");
        MlxTensorPrimitive::new(output)
    }

    fn max_pool1d(
        x: MlxTensorPrimitive,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MlxTensorPrimitive {
        require_pool1d_options(dilation, ceil_mode);
        let x_nlc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 1]).expect("transpose");

        let x_padded = if padding > 0 {
            let pad = padding as i32;
            let neg_inf = F::scalar_array(F::neg_infinity());
            mlx_rs::ops::pad(&x_nlc, &[(0, 0), (pad, pad), (0, 0)], neg_inf, None).expect("pad")
        } else {
            x_nlc
        };

        let pooled = pool1d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.max_axes(axes, None)
        });

        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 2, 1]).expect("transpose");
        MlxTensorPrimitive::new(output)
    }

    fn max_pool2d(
        x: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MlxTensorPrimitive {
        require_pool2d_options(dilation, ceil_mode);
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        let x_padded = if padding[0] > 0 || padding[1] > 0 {
            let pad_h = padding[0] as i32;
            let pad_w = padding[1] as i32;
            let neg_inf = F::scalar_array(F::neg_infinity());
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)],
                neg_inf,
                None,
            )
            .expect("pad")
        } else {
            x_nhwc
        };

        let pooled = pool2d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.max_axes(axes, None)
        });

        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 3, 1, 2]).expect("transpose");
        MlxTensorPrimitive::new(output)
    }

    fn max_pool1d_with_indices(
        x: MlxTensorPrimitive,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        _dilation: usize,
        _ceil_mode: bool,
    ) -> MaxPool1dWithIndices<Mlx<F>> {
        let _ = (x, kernel_size, stride, padding);
        panic!("max_pool1d_with_indices is not yet supported by the MLX backend")
    }

    fn max_pool2d_with_indices(
        x: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Mlx<F>> {
        require_pool2d_options(dilation, ceil_mode);
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        let x_padded = if padding[0] > 0 || padding[1] > 0 {
            let pad_h = padding[0] as i32;
            let pad_w = padding[1] as i32;
            let neg_inf = F::scalar_array(F::neg_infinity());
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)],
                neg_inf,
                None,
            )
            .expect("pad")
        } else {
            x_nhwc
        };

        let (output_nhwc, indices_nhwc) =
            max_pool2d_with_indices_impl(&x_padded, kernel_size, stride);

        let output = mlx_rs::ops::transpose_axes(&output_nhwc, &[0, 3, 1, 2]).expect("transpose");
        let indices = mlx_rs::ops::transpose_axes(&indices_nhwc, &[0, 3, 1, 2]).expect("transpose");

        MaxPool2dWithIndices::new(
            MlxTensorPrimitive::new(output),
            MlxTensorPrimitive::new(indices),
        )
    }

    fn max_pool2d_with_indices_backward(
        x: MlxTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        output_grad: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
    ) -> MaxPool2dBackward<Mlx<F>> {
        let input_shape = x.shape();
        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];

        let pad_h = padding[0];
        let pad_w = padding[1];

        let h_padded = h + 2 * pad_h;
        let w_padded = w + 2 * pad_w;

        let total_size = n * h_padded * w_padded * c;
        let grad_input_flat = F::zeros_array(&[total_size as i32]);

        let grad_nhwc =
            mlx_rs::ops::transpose_axes(&output_grad.array, &[0, 2, 3, 1]).expect("transpose");
        let indices_nhwc =
            mlx_rs::ops::transpose_axes(&indices.array, &[0, 2, 3, 1]).expect("transpose");

        let grad_flat = grad_nhwc.flatten(None, None).expect("flatten");
        let indices_flat = indices_nhwc.flatten(None, None).expect("flatten");

        let result_flat =
            mlx_rs::ops::scatter_add(&grad_input_flat, &[&indices_flat], &grad_flat, &[0])
                .expect("scatter_add");

        let result_nhwc = result_flat
            .reshape(&[n as i32, h_padded as i32, w_padded as i32, c as i32])
            .expect("reshape");

        let result_unpadded = if pad_h > 0 || pad_w > 0 {
            mlx_rs::ops::slice(
                &result_nhwc,
                &[0, pad_h as i32, pad_w as i32, 0],
                &[n as i32, (pad_h + h) as i32, (pad_w + w) as i32, c as i32],
                None,
            )
            .expect("slice")
        } else {
            result_nhwc
        };

        let output =
            mlx_rs::ops::transpose_axes(&result_unpadded, &[0, 3, 1, 2]).expect("transpose");
        MaxPool2dBackward::new(MlxTensorPrimitive::new(output))
    }

    fn adaptive_avg_pool1d(x: MlxTensorPrimitive, output_size: usize) -> MlxTensorPrimitive {
        let input_size = x.shape()[2];
        let stride = input_size / output_size;
        let kernel_size = input_size - (output_size - 1) * stride;
        Self::avg_pool1d(x, kernel_size, stride, 0, true, false)
    }

    fn adaptive_avg_pool2d(x: MlxTensorPrimitive, output_size: [usize; 2]) -> MlxTensorPrimitive {
        let input_h = x.shape()[2];
        let input_w = x.shape()[3];

        let stride_h = input_h / output_size[0];
        let stride_w = input_w / output_size[1];

        let kernel_h = input_h - (output_size[0] - 1) * stride_h;
        let kernel_w = input_w - (output_size[1] - 1) * stride_w;

        Self::avg_pool2d(
            x,
            [kernel_h, kernel_w],
            [stride_h, stride_w],
            [0, 0],
            true,
            false,
        )
    }

    fn adaptive_avg_pool2d_backward(
        _x: MlxTensorPrimitive,
        _grad: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        panic!("adaptive_avg_pool2d_backward is not yet supported by the MLX backend")
    }

    fn interpolate(
        x: MlxTensorPrimitive,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> MlxTensorPrimitive {
        let shape = x.shape();
        // Burn uses NCHW format; MLX Upsample expects spatial dims at [1..ndim-1]
        // (NHWC format). Transpose NCHW → NHWC, upsample, then transpose back.
        let [h_in, w_in] = [shape[2], shape[3]];
        let scale_h = output_size[0] as f32 / h_in as f32;
        let scale_w = output_size[1] as f32 / w_in as f32;

        let mode = match options.mode {
            InterpolateMode::Nearest => UpsampleMode::Nearest,
            InterpolateMode::Bilinear => UpsampleMode::Linear {
                align_corners: options.align_corners,
            },
            InterpolateMode::Bicubic => UpsampleMode::Cubic {
                align_corners: options.align_corners,
            },
            InterpolateMode::Lanczos3 => {
                panic!("lanczos3 interpolation is not supported by the MLX backend")
            }
        };

        // NCHW → NHWC
        let x_nhwc =
            mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose NCHW→NHWC");
        let mut upsample = Upsample::new(vec![scale_h, scale_w], mode);
        let result_nhwc = upsample.forward(&x_nhwc).expect("interpolate");
        // NHWC → NCHW
        let result =
            mlx_rs::ops::transpose_axes(&result_nhwc, &[0, 3, 1, 2]).expect("transpose NHWC→NCHW");
        MlxTensorPrimitive::new(result)
    }

    fn interpolate_backward(
        _x: MlxTensorPrimitive,
        _grad: MlxTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> MlxTensorPrimitive {
        panic!("interpolate_backward is not yet supported by the MLX backend")
    }

    fn embedding(weights: MlxTensorPrimitive, indices: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = take_axis(&weights.array, &indices.array, 0).expect("embedding");
        MlxTensorPrimitive::new(array)
    }

    fn embedding_backward(
        _weights: MlxTensorPrimitive,
        _output_grad: MlxTensorPrimitive,
        _indices: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        panic!("embedding_backward is not yet supported by the MLX backend")
    }

    fn attention(
        query: MlxTensorPrimitive,
        key: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
        mask: Option<MlxTensorPrimitive>,
        _attn_bias: Option<MlxTensorPrimitive>,
        options: AttentionModuleOptions,
    ) -> MlxTensorPrimitive {
        // query/key/value: [batch, num_heads, seq_len, head_dim]
        let head_dim = *query.shape().last().unwrap_or(&1);
        let scale = options.scale.unwrap_or(1.0 / (head_dim as f64).sqrt()) as f32;

        let use_fast_path = options.softcap.is_none();

        if use_fast_path {
            use mlx_rs::fast::{ScaledDotProductAttentionMask, scaled_dot_product_attention};

            let output = if options.is_causal {
                scaled_dot_product_attention(
                    &query.array,
                    &key.array,
                    &value.array,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                )
                .expect("fast sdpa causal")
            } else if let Some(m) = mask {
                // burn mask: bool tensor where true = keep, false = mask out
                // MLX SDPA additive mask: 0 where keep, -inf where mask out
                let zero = Array::from_f32(0.0);
                let neg_inf = Array::from_f32(-1e9);
                let additive_mask = mlx_rs::ops::r#where(&m.array, &zero, &neg_inf)
                    .expect("convert bool mask to additive");
                scaled_dot_product_attention(
                    &query.array,
                    &key.array,
                    &value.array,
                    scale,
                    &additive_mask,
                )
                .expect("fast sdpa masked")
            } else {
                scaled_dot_product_attention(
                    &query.array,
                    &key.array,
                    &value.array,
                    scale,
                    None::<ScaledDotProductAttentionMask>,
                )
                .expect("fast sdpa")
            };
            return MlxTensorPrimitive::new(output);
        }

        // Fallback for softcap: manual implementation
        let key_t = {
            let ndim = key.shape().len();
            let mut axes: Vec<i32> = (0..ndim as i32).collect();
            let last = ndim - 1;
            axes.swap(last - 1, last);
            mlx_rs::ops::transpose_axes(&key.array, &axes).expect("transpose key")
        };
        let scores = query.array.matmul(&key_t).expect("matmul QK^T");
        let scale_arr = F::f64_scalar_array(scale as f64);
        let mut scores = mlx_rs::ops::multiply(&scores, &scale_arr).expect("scale");

        if let Some(softcap) = options.softcap {
            let cap_arr = F::f64_scalar_array(softcap);
            let inv_cap = F::f64_scalar_array(1.0 / softcap);
            scores = mlx_rs::ops::multiply(&scores, &inv_cap).expect("div softcap");
            scores = mlx_rs::ops::tanh(&scores).expect("tanh");
            scores = mlx_rs::ops::multiply(&scores, &cap_arr).expect("mul softcap");
        }

        if options.is_causal {
            let seq_q = query.shape()[query.shape().len() - 2];
            let seq_k = key.shape()[key.shape().len() - 2];
            let mut causal_vals = vec![false; seq_q * seq_k];
            for i in 0..seq_q {
                let off = seq_k.saturating_sub(seq_q);
                for j in 0..=i + off {
                    if j < seq_k {
                        causal_vals[i * seq_k + j] = true;
                    }
                }
            }
            let causal_arr = Array::from_slice(&causal_vals, &[seq_q as i32, seq_k as i32]);
            let neg_inf = F::f64_scalar_array(-1e9);
            let neg_inf_b = mlx_rs::ops::broadcast_to(&neg_inf, scores.shape()).expect("broadcast");
            let causal_b =
                mlx_rs::ops::broadcast_to(&causal_arr, scores.shape()).expect("broadcast");
            scores = mlx_rs::ops::r#where(&causal_b, &scores, &neg_inf_b).expect("causal mask");
        }

        if let Some(m) = mask {
            let neg_inf = F::f64_scalar_array(-1e9);
            let neg_inf_b = mlx_rs::ops::broadcast_to(&neg_inf, scores.shape()).expect("broadcast");
            scores = mlx_rs::ops::r#where(&m.array, &scores, &neg_inf_b).expect("apply mask");
        }

        let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, None).expect("softmax");
        let output = attn_weights.matmul(&value.array).expect("matmul attn*V");
        MlxTensorPrimitive::new(output)
    }

    fn rfft(
        _signal: MlxTensorPrimitive,
        _dim: usize,
        _n: Option<usize>,
    ) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        panic!("rfft is not yet supported by the MLX backend")
    }

    fn irfft(
        _spectrum_re: MlxTensorPrimitive,
        _spectrum_im: MlxTensorPrimitive,
        _dim: usize,
        _n: Option<usize>,
    ) -> MlxTensorPrimitive {
        panic!("irfft is not yet supported by the MLX backend")
    }
}
