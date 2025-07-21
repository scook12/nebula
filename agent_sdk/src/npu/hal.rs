//! Hardware Abstraction Layer traits and interfaces

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::npu::{
    InferenceRequest, InferenceResponse, ModelInfo, NpuDevice, NpuDeviceId, NpuScheduler,
    NpuUsageStats,
};

/// Main HAL interface for NPU hardware management
#[async_trait]
pub trait NpuHal: Send + Sync {
    /// Discover and initialize all available NPU devices
    async fn discover_devices(&self) -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>>;

    /// Create a scheduler for managing tasks across devices
    async fn create_scheduler(
        &self,
        devices: Arc<RwLock<Vec<Arc<dyn NpuDevice + Send + Sync>>>>,
    ) -> Result<Arc<dyn NpuScheduler + Send + Sync>>;

    /// Get HAL capabilities and version info
    fn get_hal_info(&self) -> HalInfo;

    /// Shutdown the HAL and cleanup resources
    async fn shutdown(&self) -> Result<()>;
}

/// Low-level driver interface for specific NPU devices
#[async_trait]
pub trait NpuDriver: Send + Sync {
    /// Initialize the driver
    async fn init(&mut self) -> Result<()>;

    /// Load a model into device memory
    async fn load_model(&self, model_path: &str) -> Result<ModelHandle>;

    /// Unload a model from device memory
    async fn unload_model(&self, handle: ModelHandle) -> Result<()>;

    /// Execute inference with the given model and inputs
    async fn run_inference(
        &self,
        handle: ModelHandle,
        request: InferenceRequest,
    ) -> Result<InferenceResponse>;

    /// Get current device status and health
    async fn get_device_status(&self) -> Result<crate::npu::DeviceHealth>;

    /// Set device power state
    async fn set_power_state(&self, state: crate::npu::PowerState) -> Result<()>;

    /// Get device memory information
    async fn get_memory_info(&self) -> Result<Vec<crate::npu::MemoryRegion>>;

    /// Allocate device memory
    async fn allocate_memory(&self, size_bytes: u64) -> Result<MemoryHandle>;

    /// Free device memory
    async fn free_memory(&self, handle: MemoryHandle) -> Result<()>;

    /// Reset the device (emergency recovery)
    async fn reset_device(&self) -> Result<()>;
}

/// HAL implementation information
#[derive(Debug, Clone)]
pub struct HalInfo {
    pub name: String,
    pub version: String,
    pub supported_devices: Vec<crate::npu::NpuDeviceType>,
    pub features: Vec<HalFeature>,
}

/// HAL feature capabilities
#[derive(Debug, Clone, PartialEq)]
pub enum HalFeature {
    /// Dynamic model loading/unloading
    DynamicModels,
    /// Multi-model execution
    MultiModel,
    /// Batch inference
    BatchInference,
    /// Streaming inference
    StreamingInference,
    /// Model quantization
    Quantization,
    /// Dynamic frequency scaling
    DynamicFrequency,
    /// Power management
    PowerManagement,
    /// Memory management
    MemoryManagement,
    /// Hardware monitoring
    HardwareMonitoring,
    /// Error recovery
    ErrorRecovery,
}

/// Opaque handle to a loaded model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelHandle(u64);

impl ModelHandle {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

/// Opaque handle to allocated device memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryHandle(u64);

impl MemoryHandle {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

/// Factory trait for creating HAL implementations
pub trait HalFactory: Send + Sync {
    /// Create a new HAL instance for the specified device type
    fn create_hal(
        &self,
        device_type: crate::npu::NpuDeviceType,
    ) -> Result<Arc<dyn NpuHal + Send + Sync>>;

    /// Get supported device types
    fn supported_devices(&self) -> Vec<crate::npu::NpuDeviceType>;
}

/// Registry for HAL factories
pub struct HalRegistry {
    factories: std::collections::HashMap<crate::npu::NpuDeviceType, Box<dyn HalFactory>>,
}

impl HalRegistry {
    pub fn new() -> Self {
        Self {
            factories: std::collections::HashMap::new(),
        }
    }

    /// Register a HAL factory for a device type
    pub fn register_factory(
        &mut self,
        device_type: crate::npu::NpuDeviceType,
        factory: Box<dyn HalFactory>,
    ) {
        self.factories.insert(device_type, factory);
    }

    /// Create a HAL instance for the specified device type
    pub fn create_hal(
        &self,
        device_type: &crate::npu::NpuDeviceType,
    ) -> Result<Arc<dyn NpuHal + Send + Sync>> {
        match self.factories.get(device_type) {
            Some(factory) => factory.create_hal(device_type.clone()),
            None => anyhow::bail!("No factory registered for device type: {:?}", device_type),
        }
    }

    /// Get all supported device types
    pub fn supported_devices(&self) -> Vec<crate::npu::NpuDeviceType> {
        self.factories.keys().cloned().collect()
    }
}

impl Default for HalRegistry {
    fn default() -> Self {
        Self::new()
    }
}
