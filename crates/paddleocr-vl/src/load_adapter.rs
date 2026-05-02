use std::rc::Rc;

use burn::tensor::DType;
use burn_store::{ModuleAdapter, PyTorchToBurnAdapter, TensorSnapshot};

/// A configurable target-DType adapter that converts weights to the requested
/// dtype while keeping PaddleOCR-VL's vision encoder and projector in f32.
#[derive(Debug, Clone)]
pub struct PyTorchToBurnDTypeAdapter {
    pub target_dtype: DType,
    pub keep_vision_projector_f32: bool,
}

impl PyTorchToBurnDTypeAdapter {
    pub fn new(target_dtype: DType) -> Self {
        Self {
            target_dtype,
            keep_vision_projector_f32: true,
        }
    }

    fn target_dtype_for_snapshot(&self, snapshot: &TensorSnapshot) -> DType {
        if self.keep_vision_projector_f32
            && snapshot.path_stack.as_ref().is_some_and(|path| {
                path.iter().any(|name| {
                    matches!(
                        name.as_str(),
                        "vision" | "projector" | "visual" | "vision_model" | "mlp_AR"
                    )
                })
            })
        {
            DType::F32
        } else {
            self.target_dtype
        }
    }
}

impl ModuleAdapter for PyTorchToBurnDTypeAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let snapshot = PyTorchToBurnAdapter.adapt(snapshot);

        let target = self.target_dtype_for_snapshot(&snapshot);
        if snapshot.dtype == target {
            return snapshot;
        }

        let data_fn = snapshot.clone_data_fn();

        TensorSnapshot::from_closure(
            Rc::new(move || {
                let data = data_fn()?;
                Ok(data.convert_dtype(target))
            }),
            target,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        PyTorchToBurnAdapter.get_alternative_param_name(param_name, container_type)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
