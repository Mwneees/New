//! Implements a `wasi-nn` [`BackendInner`] using OpenVINO.

use super::{
    read, BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, Id,
};
use crate::wit::{self, ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use openvino::{DeviceType, ElementType, InferenceError, SetupError, Shape, Tensor as OvTensor};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct OpenvinoBackend(Option<openvino::Core>);
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 2 {
            return Err(BackendError::InvalidNumberOfBuilders(2, builders.len()).into());
        }
        // Construct the context if none is present; this is done lazily (i.e.
        // upon actually loading a model) because it may fail to find and load
        // the OpenVINO libraries. The laziness limits the extent of the error
        // only to wasi-nn users, not all WASI users.
        if self.0.is_none() {
            self.0.replace(openvino::Core::new()?);
        }
        // Read the guest array.
        let xml = builders[0];
        let weights = builders[1];

        //Construct new tensor with data.
        let dims: [i64; 2] = [1, weights.len() as i64];
        let shape = Shape::new(&dims)?;
        let mut weights_tensor = OvTensor::new(ElementType::F32, &shape)?;
        let buffer = weights_tensor.buffer_mut()?;
        for (index, bytes) in weights.iter().enumerate() {
            buffer[index] = *bytes;
        }

        // Construct OpenVINO graph structures: `model` contains the graph
        // structure, `compiled_model` can perform inference.
        let core = self
            .0
            .as_mut()
            .expect("openvino::Core was previously constructed");
        let model = core.read_model_from_buffer(&xml, Some(&weights_tensor))?;
        let compiled_model = core.compile_model(&model, map_execution_target_to_string(target))?;
        let box_: Box<dyn BackendGraph> =
            Box::new(OpenvinoGraph(Arc::new(Mutex::new(compiled_model))));
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for OpenvinoBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.xml"))?;
        let weights = read(&path.join("model.bin"))?;
        self.load(&[&model, &weights], target)
    }
}

struct OpenvinoGraph(Arc<Mutex<openvino::CompiledModel>>);

unsafe impl Send for OpenvinoGraph {}
unsafe impl Sync for OpenvinoGraph {}

impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let mut compiled_model = self.0.lock().unwrap();
        let infer_request = compiled_model.create_infer_request()?;
        let box_: Box<dyn BackendExecutionContext> =
            Box::new(OpenvinoExecutionContext(infer_request));
        Ok(box_.into())
    }
}

struct OpenvinoExecutionContext(openvino::InferRequest);

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        let input_name = match id {
            Id::Index(i) => self.0.get_input_name(i as usize)?,
            Id::Name(name) => name,
        };

        // Construct the blob structure. TODO: there must be some good way to
        // discover the layout here; `desc` should not have to default to NHWC.
        let precision = map_tensor_type_to_precision(tensor.ty);
        let dimensions = tensor
            .dimensions
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>();
        let shape = Shape::new(&dimensions)?;
        let mut new_tensor = OvTensor::new(precision, &shape)?;
        let buffer = new_tensor.buffer_mut()?;
        for (index, bytes) in tensor.data.iter().enumerate() {
            buffer[index] = *bytes;
        }

        // Assign the tensor to the request.
        self.0
            .set_input_tensor_by_index(index as usize, &new_tensor)?;
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.0.infer()?;
        Ok(())
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        let output_name = match id {
            Id::Index(i) => self.0.get_output_name(i as usize)?,
            Id::Name(name) => name,
        };
        let dimensions = vec![]; // TODO: get actual shape
        let ty = wit::TensorType::Fp32; // TODO: get actual type.
        let blob = self.1.get_blob(&output_name)?;
        let data = blob.buffer()?.to_vec();
        Ok(Tensor {
            dimensions,
            ty,
            data,
        })
    }
}

impl From<InferenceError> for BackendError {
    fn from(e: InferenceError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

impl From<SetupError> for BackendError {
    fn from(e: SetupError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_execution_target_to_string(target: ExecutionTarget) -> DeviceType<'static> {
    match target {
        ExecutionTarget::Cpu => DeviceType::CPU,
        ExecutionTarget::Gpu => DeviceType::GPU,
        ExecutionTarget::Tpu => unimplemented!("OpenVINO does not support TPU execution targets"),
    }
}

/// Return OpenVINO's precision type for the `TensorType` enum provided by
/// wasi-nn.
fn map_tensor_type_to_precision(tensor_type: TensorType) -> openvino::ElementType {
    match tensor_type {
        TensorType::Fp16 => ElementType::F16,
        TensorType::Fp32 => ElementType::F32,
        TensorType::Fp64 => ElementType::F64,
        TensorType::U8 => ElementType::U8,
        TensorType::I32 => ElementType::I32,
        TensorType::I64 => ElementType::I64,
        TensorType::Bf16 => todo!("not yet supported in `openvino` bindings"),
    }
}
