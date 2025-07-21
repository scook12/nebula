//! Core types for the NPU Hardware Abstraction Layer

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[cfg(feature = "npu")]
use uuid::Uuid;

use crate::types::{AgentId, TaskId};
use serde::{Deserialize, Serialize};

/// Unique identifier for NPU devices
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NpuDeviceId(String);

impl NpuDeviceId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[cfg(feature = "npu")]
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NpuDeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// NPU device types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NpuDeviceType {
    /// Apple Neural Engine (M1/M2/M3 chips)
    AppleNeuralEngine,
    /// Intel NPU (integrated)
    IntelNpu,
    /// NVIDIA GPU (used as NPU)
    NvidiaGpu,
    /// AMD GPU (used as NPU)
    AmdGpu,
    /// Qualcomm Hexagon DSP
    QualcommHexagon,
    /// Google Edge TPU
    GoogleEdgeTpu,
    /// Generic CPU fallback
    CpuFallback,
    /// Mock NPU for testing
    Mock,
    /// Unknown/custom NPU
    Unknown(String),
}

/// NPU device vendor
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NpuVendor {
    Apple,
    Intel,
    Nvidia,
    Amd,
    Qualcomm,
    Google,
    Unknown(String),
}

/// NPU power state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerState {
    /// Device is active and available for compute
    Active,
    /// Device is idle but ready
    Idle,
    /// Device is in power saving mode
    PowerSave,
    /// Device is suspended
    Suspended,
    /// Device is offline/unavailable
    Offline,
}

/// Memory types supported by NPU
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Unified memory (shared with CPU)
    Unified,
    /// Dedicated NPU memory
    Dedicated,
    /// High bandwidth memory
    Hbm,
    /// System RAM
    SystemRam,
}

/// Memory region information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub memory_type: MemoryType,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub bandwidth_gbps: f64,
}

/// Supported data types for NPU operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int16,
    Int32,
    UInt8,
    UInt16,
    UInt32,
    Bool,
}

/// NPU compute unit type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeUnit {
    /// Tensor/matrix processing units
    TensorCore,
    /// Vector processing units
    VectorCore,
    /// Scalar processing units
    ScalarCore,
    /// Custom accelerator units
    CustomAccelerator,
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSpecs {
    /// Peak TOPS (Tera Operations Per Second)
    pub peak_tops: f64,
    /// Sustained TOPS under thermal constraints
    pub sustained_tops: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Power consumption in watts
    pub power_consumption_watts: f64,
    /// Operating frequency in MHz
    pub frequency_mhz: u32,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// Task execution state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is queued and waiting
    Queued,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed(String),
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

/// Resource allocation for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// NPU device assigned to this task
    pub device_id: NpuDeviceId,
    /// Compute units allocated
    pub compute_units: Vec<ComputeUnit>,
    /// Memory allocated in bytes
    pub memory_bytes: u64,
    /// Power budget in watts
    pub power_budget_watts: f32,
    /// Maximum execution time
    pub timeout: Duration,
}

/// Model format support
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Onnx,
    TensorFlow,
    PyTorch,
    CoreMl,
    TfLite,
    OpenVino,
    Custom(String),
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub format: ModelFormat,
    pub input_shapes: Vec<Vec<u64>>,
    pub output_shapes: Vec<Vec<u64>>,
    pub input_types: Vec<DataType>,
    pub output_types: Vec<DataType>,
    pub parameter_count: u64,
    pub model_size_bytes: u64,
}

/// Inference input data
#[derive(Debug, Clone)]
pub struct InferenceInput {
    pub data: Vec<u8>,
    pub shape: Vec<u64>,
    pub data_type: DataType,
}

/// Inference output data
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub data: Vec<u8>,
    pub shape: Vec<u64>,
    pub data_type: DataType,
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Model to use for inference
    pub model_path: String,
    /// Input tensors
    pub inputs: Vec<InferenceInput>,
    /// Timeout for inference
    pub timeout: Duration,
    /// Priority of the request
    pub priority: TaskPriority,
    /// Agent requesting the inference
    pub agent_id: Option<AgentId>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Inference response
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Output tensors
    pub outputs: Vec<InferenceOutput>,
    /// Execution time
    pub execution_time: Duration,
    /// Device used for inference
    pub device_id: NpuDeviceId,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// NPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpuUsageStats {
    /// Total number of devices
    pub total_devices: usize,
    /// Number of active devices
    pub active_devices: usize,
    /// Total compute utilization (0.0 to 1.0)
    pub compute_utilization: f64,
    /// Total memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Average power consumption in watts
    pub power_consumption_watts: f64,
    /// Tasks completed in the last minute
    pub tasks_completed_last_minute: u64,
    /// Average task execution time
    pub average_task_time: Duration,
    /// Number of queued tasks
    pub queued_tasks: usize,
}

/// Device health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHealth {
    /// Overall health status
    pub is_healthy: bool,
    /// Temperature in Celsius
    pub temperature_celsius: f32,
    /// Power consumption in watts
    pub power_consumption_watts: f32,
    /// Memory errors detected
    pub memory_errors: u32,
    /// Compute errors detected
    pub compute_errors: u32,
    /// Last health check timestamp
    pub last_check: SystemTime,
    /// Detailed status message
    pub status_message: String,
}

/// Error types specific to NPU operations
#[cfg(feature = "npu")]
#[derive(thiserror::Error, Debug)]
pub enum NpuError {
    #[error("Device not found: {0}")]
    DeviceNotFound(NpuDeviceId),

    #[error("Device unavailable: {0}")]
    DeviceUnavailable(String),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Timeout waiting for device")]
    Timeout,

    #[error("Hardware error: {0}")]
    HardwareError(String),

    #[error("Driver error: {0}")]
    DriverError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Inference task that can be submitted to the scheduler
#[derive(Debug, Clone)]
pub struct InferenceTask {
    /// Unique task identifier
    pub id: TaskId,
    /// Inference request
    pub request: InferenceRequest,
    /// Task priority
    pub priority: TaskPriority,
    /// Resource allocation requirements
    pub resource_requirements: ResourceAllocation,
    /// Scheduling hints
    pub scheduling_hints: SchedulingHints,
}

/// Scheduling hints for task placement
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulingHints {
    /// Preferred device types
    pub preferred_devices: Vec<NpuDeviceType>,
    /// Avoid specific devices
    pub avoid_devices: Vec<NpuDeviceId>,
    /// Required memory type
    pub required_memory_type: Option<MemoryType>,
    /// Minimum required performance
    pub min_tops: Option<f64>,
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,
}
