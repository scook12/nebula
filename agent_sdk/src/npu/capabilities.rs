//! NPU Capabilities definitions

use crate::npu::{ComputeUnit, DataType, MemoryType, ModelFormat, PerformanceSpecs};
use serde::{Deserialize, Serialize};

/// NPU device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpuCapabilities {
    pub compute: ComputeCapability,
    pub memory: MemoryCapability,
    pub model_support: ModelSupport,
    pub performance: PerformanceSpecs,
}

impl Default for NpuCapabilities {
    fn default() -> Self {
        Self {
            compute: ComputeCapability::default(),
            memory: MemoryCapability::default(),
            model_support: ModelSupport::default(),
            performance: PerformanceSpecs {
                peak_tops: 1.0,
                sustained_tops: 0.8,
                memory_bandwidth_gbps: 10.0,
                power_consumption_watts: 10.0,
                frequency_mhz: 1000,
            },
        }
    }
}

/// Compute capabilities of the NPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Available compute units
    pub compute_units: Vec<ComputeUnit>,
    /// Number of cores per compute unit type
    pub core_counts: std::collections::HashMap<ComputeUnit, u32>,
    /// Supported data types
    pub supported_data_types: Vec<DataType>,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Maximum tensor dimensions
    pub max_tensor_dims: u32,
    /// Supports concurrent inference
    pub concurrent_inference: bool,
    /// Supports mixed precision
    pub mixed_precision: bool,
}

impl Default for ComputeCapability {
    fn default() -> Self {
        let mut core_counts = std::collections::HashMap::new();
        core_counts.insert(ComputeUnit::TensorCore, 8);
        core_counts.insert(ComputeUnit::VectorCore, 4);

        Self {
            compute_units: vec![ComputeUnit::TensorCore, ComputeUnit::VectorCore],
            core_counts,
            supported_data_types: vec![DataType::Float32, DataType::Float16, DataType::Int8],
            max_batch_size: 32,
            max_tensor_dims: 8,
            concurrent_inference: true,
            mixed_precision: true,
        }
    }
}

/// Memory capabilities of the NPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCapability {
    /// Total device memory in bytes
    pub total_memory_bytes: u64,
    /// Memory types supported
    pub supported_memory_types: Vec<MemoryType>,
    /// Maximum allocation size
    pub max_allocation_bytes: u64,
    /// Memory alignment requirements
    pub alignment_bytes: u64,
    /// Supports memory pooling
    pub memory_pooling: bool,
    /// Supports unified memory
    pub unified_memory: bool,
}

impl Default for MemoryCapability {
    fn default() -> Self {
        Self {
            total_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            supported_memory_types: vec![MemoryType::Unified, MemoryType::Dedicated],
            max_allocation_bytes: 1024 * 1024 * 1024, // 1GB
            alignment_bytes: 256,
            memory_pooling: true,
            unified_memory: true,
        }
    }
}

/// Model format and feature support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSupport {
    /// Supported model formats
    pub supported_formats: Vec<ModelFormat>,
    /// Dynamic model loading
    pub dynamic_loading: bool,
    /// Model quantization support
    pub quantization: Vec<DataType>,
    /// Dynamic shapes support
    pub dynamic_shapes: bool,
    /// Graph optimization
    pub graph_optimization: bool,
    /// Custom operators
    pub custom_operators: bool,
}

impl Default for ModelSupport {
    fn default() -> Self {
        Self {
            supported_formats: vec![ModelFormat::Onnx, ModelFormat::TensorFlow],
            dynamic_loading: true,
            quantization: vec![DataType::Int8, DataType::Float16],
            dynamic_shapes: false,
            graph_optimization: true,
            custom_operators: false,
        }
    }
}

/// Query capabilities helper functions
impl NpuCapabilities {
    /// Check if a data type is supported
    pub fn supports_data_type(&self, data_type: &DataType) -> bool {
        self.compute.supported_data_types.contains(data_type)
    }

    /// Check if a model format is supported
    pub fn supports_model_format(&self, format: &ModelFormat) -> bool {
        self.model_support.supported_formats.contains(format)
    }

    /// Check if a compute unit is available
    pub fn has_compute_unit(&self, unit: &ComputeUnit) -> bool {
        self.compute.compute_units.contains(unit)
    }

    /// Get the number of cores for a compute unit
    pub fn get_core_count(&self, unit: &ComputeUnit) -> u32 {
        self.compute.core_counts.get(unit).copied().unwrap_or(0)
    }

    /// Check if concurrent inference is supported
    pub fn supports_concurrent_inference(&self) -> bool {
        self.compute.concurrent_inference
    }

    /// Get maximum supported batch size
    pub fn max_batch_size(&self) -> u32 {
        self.compute.max_batch_size
    }

    /// Check available memory
    pub fn available_memory(&self) -> u64 {
        self.memory.total_memory_bytes
    }

    /// Check if memory type is supported
    pub fn supports_memory_type(&self, memory_type: &MemoryType) -> bool {
        self.memory.supported_memory_types.contains(memory_type)
    }
}
