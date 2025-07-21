//! Mock implementation of NPU Hardware Abstraction Layer

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::npu::hal::{HalFeature, HalInfo, MemoryHandle, ModelHandle};
use crate::npu::scheduler::MockScheduler;
use crate::npu::{
    DeviceHealth, InferenceOutput, InferenceRequest, InferenceResponse, MemoryRegion,
    NpuCapabilities, NpuDevice, NpuDeviceId, NpuDeviceInfo, NpuDeviceType, NpuHal, NpuScheduler,
    NpuUsageStats, NpuVendor, PowerState,
};

/// Mock HAL implementation
pub struct MockNpuHal {
    devices: Vec<Arc<dyn NpuDevice + Send + Sync>>,
}

impl MockNpuHal {
    pub async fn new() -> Result<Self> {
        Ok(Self { devices: vec![] })
    }
}

#[async_trait]
impl NpuHal for MockNpuHal {
    async fn discover_devices(&self) -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>> {
        let mut devices: Vec<Arc<dyn NpuDevice + Send + Sync>> = Vec::new();
        // Add mock devices for testing
        devices.push(Arc::new(MockNpuDevice::new().await?) as Arc<dyn NpuDevice + Send + Sync>);
        Ok(devices)
    }

    async fn create_scheduler(
        &self,
        _devices: Arc<RwLock<Vec<Arc<dyn NpuDevice + Send + Sync>>>>,
    ) -> Result<Arc<dyn NpuScheduler + Send + Sync>> {
        Ok(Arc::new(MockScheduler::default()))
    }

    fn get_hal_info(&self) -> crate::npu::HalInfo {
        crate::npu::HalInfo {
            name: "Mock HAL".to_string(),
            version: "1.0.0".to_string(),
            supported_devices: vec![NpuDeviceType::Mock],
            features: vec![
                crate::npu::HalFeature::DynamicModels,
                crate::npu::HalFeature::PowerManagement,
            ],
        }
    }

    async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down Mock HAL");
        Ok(())
    }
}

/// Mock NPU Device
pub struct MockNpuDevice {
    info: NpuDeviceInfo,
    capabilities: Arc<NpuCapabilities>,
}

impl MockNpuDevice {
    pub async fn new() -> Result<Self> {
        let info = NpuDeviceInfo::new(
            NpuDeviceId::new("mock-device"),
            "Mock NPU Device".to_string(),
            NpuDeviceType::Mock,
            NpuVendor::Unknown("MockVendor".to_string()),
        );

        let capabilities = Arc::new(NpuCapabilities::default());

        Ok(Self { info, capabilities })
    }
}

#[async_trait]
impl NpuDevice for MockNpuDevice {
    fn id(&self) -> NpuDeviceId {
        self.info.id.clone()
    }

    fn info(&self) -> NpuDeviceInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Arc<NpuCapabilities> {
        self.capabilities.clone()
    }

    async fn init(&self) -> Result<()> {
        log::info!("Initializing Mock NPU Device");
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down Mock NPU Device");
        Ok(())
    }

    async fn execute_inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        log::info!("Executing inference on Mock NPU Device: {:?}", request);
        // Mock output
        let outputs = vec![InferenceOutput {
            data: request.inputs[0].data.clone(),
            shape: request.inputs[0].shape.clone(),
            data_type: request.inputs[0].data_type.clone(),
        }];
        Ok(InferenceResponse {
            outputs,
            execution_time: std::time::Duration::from_millis(10),
            device_id: self.id(),
            metadata: HashMap::new(),
        })
    }

    async fn load_model(&self, _model_path: &str) -> Result<ModelHandle> {
        Ok(ModelHandle::new(0))
    }

    async fn unload_model(&self, _handle: ModelHandle) -> Result<()> {
        Ok(())
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn get_health(&self) -> Result<DeviceHealth> {
        Ok(DeviceHealth {
            is_healthy: true,
            temperature_celsius: 35.0,
            power_consumption_watts: 10.0,
            memory_errors: 0,
            compute_errors: 0,
            last_check: std::time::SystemTime::now(),
            status_message: "All systems nominal".to_string(),
        })
    }

    async fn get_power_state(&self) -> Result<PowerState> {
        Ok(PowerState::Active)
    }

    async fn set_power_state(&self, _state: PowerState) -> Result<()> {
        Ok(())
    }

    async fn get_memory_info(&self) -> Result<Vec<MemoryRegion>> {
        Ok(vec![MemoryRegion {
            memory_type: crate::npu::MemoryType::Unified,
            total_bytes: 4 * 1024 * 1024 * 1024,
            available_bytes: 3 * 1024 * 1024 * 1024,
            bandwidth_gbps: 10.0,
        }])
    }

    async fn allocate_memory(&self, _size_bytes: u64) -> Result<MemoryHandle> {
        Ok(MemoryHandle::new(0))
    }

    async fn free_memory(&self, _handle: MemoryHandle) -> Result<()> {
        Ok(())
    }

    async fn get_utilization(&self) -> f64 {
        0.1
    }

    async fn get_temperature(&self) -> f32 {
        35.0
    }

    async fn reset(&self) -> Result<()> {
        Ok(())
    }
}
