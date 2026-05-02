use std::rc::Rc;

use burn::tensor::DType;
use burn_store::{ModuleAdapter, PyTorchToBurnAdapter, TensorSnapshot};

#[derive(Debug, Clone)]
pub struct PyTorchToBurnDTypeAdapter {
    target_dtype: DType,
}

impl PyTorchToBurnDTypeAdapter {
    pub fn new(target_dtype: DType) -> Self {
        Self { target_dtype }
    }
}

impl ModuleAdapter for PyTorchToBurnDTypeAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let snapshot = PyTorchToBurnAdapter.adapt(snapshot);

        if snapshot.dtype == self.target_dtype {
            return snapshot;
        }

        let target = self.target_dtype;
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

#[derive(Debug, Clone, Default)]
pub struct PyTorchToBurnF32Adapter;

impl ModuleAdapter for PyTorchToBurnF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        PyTorchToBurnDTypeAdapter::new(DType::F32).adapt(snapshot)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        PyTorchToBurnAdapter.get_alternative_param_name(param_name, container_type)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
