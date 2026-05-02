//! Additional ops implementations for MLX backend.

use burn_tensor::{
    DType, FloatDType, Scalar, Shape, Slice, TensorData, TensorPrimitive,
    backend::ExecutionError,
    ops::{ActivationOps, FloatTensorOps, QTensorOps, TransactionOps},
    quantization::{QuantLevel, QuantScheme, QuantValue, QuantizedBytes},
};
use mlx_rs::Array;

use crate::backend::{Mlx, MlxQuantizedTensorPrimitive, MlxTensorPrimitive};
use crate::device::MlxDevice;
use crate::element::FloatMlxElement;

impl<F: FloatMlxElement> ActivationOps<Self> for Mlx<F> {
    fn relu(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = F::f64_scalar_array(0.0);
        let array = mlx_rs::ops::maximum(&tensor.array, &zero).expect("relu");
        MlxTensorPrimitive::new(array)
    }

    fn sigmoid(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sigmoid(&tensor.array).expect("sigmoid");
        MlxTensorPrimitive::new(array)
    }

    fn gelu(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::nn::gelu(&tensor.array).expect("gelu");
        MlxTensorPrimitive::new(array)
    }

    fn leaky_relu(tensor: MlxTensorPrimitive, negative_slope: Scalar) -> MlxTensorPrimitive {
        let slope_f32: f32 = negative_slope.elem();
        let array = mlx_rs::nn::leaky_relu(&tensor.array, slope_f32).expect("leaky_relu");
        MlxTensorPrimitive::new(array)
    }

    fn hard_sigmoid(tensor: MlxTensorPrimitive, alpha: Scalar, beta: Scalar) -> MlxTensorPrimitive {
        let alpha_arr = F::f64_scalar_array(alpha.elem::<f64>());
        let beta_arr = F::f64_scalar_array(beta.elem::<f64>());
        let scaled = mlx_rs::ops::multiply(&tensor.array, &alpha_arr).expect("multiply");
        let shifted = mlx_rs::ops::add(&scaled, &beta_arr).expect("add");
        let zero = F::f64_scalar_array(0.0);
        let one = F::f64_scalar_array(1.0);
        let array = mlx_rs::ops::clip(&shifted, (&zero, &one)).expect("clip");
        MlxTensorPrimitive::new(array)
    }

    fn log_sigmoid(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let sig = mlx_rs::ops::sigmoid(&tensor.array).expect("sigmoid");
        let array = mlx_rs::ops::log(&sig).expect("log");
        MlxTensorPrimitive::new(array)
    }

    fn prelu(tensor: MlxTensorPrimitive, alpha: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = F::f64_scalar_array(0.0);
        let pos = mlx_rs::ops::maximum(&tensor.array, &zero).expect("max");
        let neg = mlx_rs::ops::minimum(&tensor.array, &zero).expect("min");
        let scaled_neg = mlx_rs::ops::multiply(&alpha.array, &neg).expect("multiply");
        let array = mlx_rs::ops::add(&pos, &scaled_neg).expect("add");
        MlxTensorPrimitive::new(array)
    }

    fn gelu_backward(x: MlxTensorPrimitive, grad: MlxTensorPrimitive) -> MlxTensorPrimitive {
        // d/dx[gelu(x)] = 0.5 * (1 + erf(x/sqrt(2))) + x * pdf(x)
        // where pdf(x) = exp(-x^2/2) / sqrt(2*pi)
        let sqrt2 = F::f64_scalar_array(std::f64::consts::SQRT_2);
        let half = F::f64_scalar_array(0.5);
        let one = F::f64_scalar_array(1.0);
        let inv_sqrt_2pi = F::f64_scalar_array((2.0 * std::f64::consts::PI).sqrt().recip());

        let x_over_sqrt2 = mlx_rs::ops::divide(&x.array, &sqrt2).expect("div");
        let erf_part = mlx_rs::ops::erf(&x_over_sqrt2).expect("erf");
        let cdf = mlx_rs::ops::multiply(&mlx_rs::ops::add(&one, &erf_part).expect("add"), &half)
            .expect("mul");

        let neg_half = F::f64_scalar_array(-0.5);
        let x_sq = mlx_rs::ops::multiply(&x.array, &x.array).expect("mul");
        let exp_arg = mlx_rs::ops::multiply(&x_sq, &neg_half).expect("mul");
        let pdf = mlx_rs::ops::multiply(&mlx_rs::ops::exp(&exp_arg).expect("exp"), &inv_sqrt_2pi)
            .expect("mul");

        let x_pdf = mlx_rs::ops::multiply(&x.array, &pdf).expect("mul");
        let dgelu = mlx_rs::ops::add(&cdf, &x_pdf).expect("add");
        let array = mlx_rs::ops::multiply(&grad.array, &dgelu).expect("mul");
        MlxTensorPrimitive::new(array)
    }

    fn relu_backward(x: MlxTensorPrimitive, grad: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = F::f64_scalar_array(0.0);
        let mask = mlx_rs::ops::gt(&x.array, &zero).expect("greater");
        let mask_float = F::cast_array(&mask);
        let array = mlx_rs::ops::multiply(&grad.array, &mask_float).expect("multiply");
        MlxTensorPrimitive::new(array)
    }
}

fn scheme_to_mlx_params(scheme: &QuantScheme, num_elements: usize) -> (i32, i32) {
    let bits: i32 = match scheme.value {
        QuantValue::Q8S | QuantValue::Q8F => 8,
        QuantValue::Q4S | QuantValue::Q4F => 4,
        _ => panic!("Unsupported quantization value: {:?}", scheme.value),
    };
    let group_size: i32 = match scheme.level {
        QuantLevel::Block(bs) => bs.as_slice()[0] as i32,
        QuantLevel::Tensor => num_elements as i32,
    };
    (bits, group_size)
}

impl<F: FloatMlxElement> QTensorOps<Self> for Mlx<F> {
    fn q_from_data(data: TensorData, _device: &MlxDevice) -> MlxQuantizedTensorPrimitive {
        let scheme = match data.dtype {
            DType::QFloat(scheme) => scheme,
            other => panic!("q_from_data called with non-quantized dtype: {:?}", other),
        };

        let shape_vec: Vec<usize> = data.shape.to_vec();
        let num_elements: usize = shape_vec.iter().product();
        let (bits, group_size) = scheme_to_mlx_params(&scheme, num_elements);

        let qb = QuantizedBytes {
            bytes: data.bytes,
            scheme,
            num_elements,
        };
        let (i8_values, qparams) = qb.into_vec_i8();
        let scales = qparams.scales;

        let mut float_vals = vec![0.0f32; i8_values.len()];
        for (block_idx, chunk) in i8_values.chunks(group_size as usize).enumerate() {
            let scale = scales[block_idx];
            for (i, &val) in chunk.iter().enumerate() {
                float_vals[block_idx * group_size as usize + i] = val as f32 * scale;
            }
        }

        let shape_i32: Vec<i32> = shape_vec.iter().map(|&s| s as i32).collect();
        let float_array = Array::from_slice(&float_vals, &shape_i32);

        let (quantized, mlx_scales, mlx_biases) =
            mlx_rs::ops::quantize(&float_array, group_size, bits).expect("MLX quantize failed");

        MlxQuantizedTensorPrimitive {
            quantized,
            scales: mlx_scales,
            biases: mlx_biases,
            shape: shape_vec,
            group_size,
            bits,
            scheme,
        }
    }

    fn quantize(
        tensor: MlxTensorPrimitive,
        scheme: &QuantScheme,
        _qparams: burn_tensor::quantization::QuantizationParametersPrimitive<Self>,
    ) -> MlxQuantizedTensorPrimitive {
        let num_elements: usize = tensor.shape.iter().product();
        let (bits, group_size) = scheme_to_mlx_params(scheme, num_elements);
        let shape = tensor.shape.clone();

        let (quantized, scales, biases) =
            mlx_rs::ops::quantize(&tensor.array, group_size, bits).expect("MLX quantize failed");

        MlxQuantizedTensorPrimitive {
            quantized,
            scales,
            biases,
            shape,
            group_size,
            bits,
            scheme: *scheme,
        }
    }

    fn dequantize(tensor: MlxQuantizedTensorPrimitive, _dtype: FloatDType) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::dequantize(
            &tensor.quantized,
            &tensor.scales,
            &tensor.biases,
            tensor.group_size,
            tensor.bits,
        )
        .expect("MLX dequantize failed");
        let array = F::cast_array(&array);
        MlxTensorPrimitive::new(array)
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let default_float_dtype = FloatDType::F32;
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs_f), TensorPrimitive::QFloat(rhs_q)) => {
                let result = mlx_rs::ops::quantized_matmul(
                    &lhs_f.array,
                    &rhs_q.quantized,
                    &rhs_q.scales,
                    &rhs_q.biases,
                    false,
                    rhs_q.group_size,
                    rhs_q.bits,
                )
                .expect("MLX quantized_matmul failed");
                let result = F::cast_array(&result);
                TensorPrimitive::Float(MlxTensorPrimitive::new(result))
            }
            (TensorPrimitive::QFloat(lhs_q), TensorPrimitive::Float(rhs_f)) => {
                let lhs_f = Self::dequantize(lhs_q, default_float_dtype);
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(lhs_f, rhs_f))
            }
            (TensorPrimitive::QFloat(lhs_q), TensorPrimitive::QFloat(rhs_q)) => {
                let lhs_f = Self::dequantize(lhs_q, default_float_dtype);
                let rhs_f = Self::dequantize(rhs_q, default_float_dtype);
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(lhs_f, rhs_f))
            }
            (TensorPrimitive::Float(lhs_f), TensorPrimitive::Float(rhs_f)) => {
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(lhs_f, rhs_f))
            }
        }
    }

    fn q_device(_tensor: &MlxQuantizedTensorPrimitive) -> MlxDevice {
        MlxDevice::Gpu
    }

    fn q_to_device(
        tensor: MlxQuantizedTensorPrimitive,
        _device: &MlxDevice,
    ) -> MlxQuantizedTensorPrimitive {
        tensor
    }

    fn q_reshape(tensor: MlxQuantizedTensorPrimitive, shape: Shape) -> MlxQuantizedTensorPrimitive {
        let new_dims: Vec<usize> = shape.to_vec();

        let old = &tensor.shape;
        if old.len() >= 2
            && new_dims.len() >= 2
            && old[old.len() - 2] == new_dims[new_dims.len() - 2]
            && old[old.len() - 1] == new_dims[new_dims.len() - 1]
        {
            return MlxQuantizedTensorPrimitive {
                shape: new_dims,
                ..tensor
            };
        }

        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let reshaped = <Self as FloatTensorOps<Self>>::float_reshape(float_tensor, shape);
        Self::quantize_dynamic(reshaped, &scheme)
    }

    async fn q_into_data(
        tensor: MlxQuantizedTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        <Self as FloatTensorOps<Self>>::float_into_data(float_tensor).await
    }

    fn q_swap_dims(
        tensor: MlxQuantizedTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> MlxQuantizedTensorPrimitive {
        let ndim = tensor.shape.len();

        if ndim >= 2 && dim1 < ndim - 2 && dim2 < ndim - 2 {
            let mut new_shape = tensor.shape.clone();
            new_shape.swap(dim1, dim2);
            return MlxQuantizedTensorPrimitive {
                shape: new_shape,
                ..tensor
            };
        }

        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let swapped = <Self as FloatTensorOps<Self>>::float_swap_dims(float_tensor, dim1, dim2);
        Self::quantize_dynamic(swapped, &scheme)
    }

    fn q_permute(
        tensor: MlxQuantizedTensorPrimitive,
        axes: &[usize],
    ) -> MlxQuantizedTensorPrimitive {
        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let permuted = <Self as FloatTensorOps<Self>>::float_permute(float_tensor, axes);
        Self::quantize_dynamic(permuted, &scheme)
    }

    fn q_flip(tensor: MlxQuantizedTensorPrimitive, axes: &[usize]) -> MlxQuantizedTensorPrimitive {
        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let flipped = <Self as FloatTensorOps<Self>>::float_flip(float_tensor, axes);
        Self::quantize_dynamic(flipped, &scheme)
    }

    fn q_select(
        tensor: MlxQuantizedTensorPrimitive,
        dim: usize,
        indices: MlxTensorPrimitive,
    ) -> MlxQuantizedTensorPrimitive {
        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let selected = <Self as FloatTensorOps<Self>>::float_select(float_tensor, dim, indices);
        Self::quantize_dynamic(selected, &scheme)
    }

    fn q_slice(
        tensor: MlxQuantizedTensorPrimitive,
        slices: &[Slice],
    ) -> MlxQuantizedTensorPrimitive {
        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let sliced = <Self as FloatTensorOps<Self>>::float_slice(float_tensor, slices);
        Self::quantize_dynamic(sliced, &scheme)
    }

    fn q_expand(tensor: MlxQuantizedTensorPrimitive, shape: Shape) -> MlxQuantizedTensorPrimitive {
        let new_dims: Vec<usize> = shape.to_vec();
        let old = &tensor.shape;

        if old.len() >= 2
            && new_dims.len() >= 2
            && old[old.len() - 2] == new_dims[new_dims.len() - 2]
            && old[old.len() - 1] == new_dims[new_dims.len() - 1]
        {
            let extra = new_dims.len().saturating_sub(old.len());
            if new_dims[..extra].iter().all(|&d| d == 1) {
                return MlxQuantizedTensorPrimitive {
                    shape: new_dims,
                    ..tensor
                };
            }
        }

        let scheme = tensor.scheme;
        let float_tensor = Self::dequantize(tensor, FloatDType::F32);
        let expanded = <Self as FloatTensorOps<Self>>::float_expand(float_tensor, shape);
        Self::quantize_dynamic(expanded, &scheme)
    }
}

impl<F: FloatMlxElement> TransactionOps<Self> for Mlx<F> {}
