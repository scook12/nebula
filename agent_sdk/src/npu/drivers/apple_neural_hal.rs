//! Apple Neural Engine Hardware Abstraction Layer Implementation

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::npu::scheduler::MockScheduler;
use crate::npu::{HalFeature, HalInfo, NpuDevice, NpuHal, NpuScheduler};

use super::apple_neural_device::AppleNeuralDevice;

/// Apple Neural Engine HAL implementation
pub struct AppleNeuralEngineHal;

impl AppleNeuralEngineHal {
    pub async fn new() -> Result<Self> {
        Ok(AppleNeuralEngineHal)
    }
}

#[async_trait]
impl NpuHal for AppleNeuralEngineHal {
    async fn discover_devices(&self) -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>> {
        let mut devices: Vec<Arc<dyn NpuDevice + Send + Sync>> = Vec::new();

        // Create Apple Neural Engine device
        let apple_device = AppleNeuralDevice::new().await?;
        devices.push(Arc::new(apple_device) as Arc<dyn NpuDevice + Send + Sync>);

        log::info!("Discovered {} Apple Neural Engine device(s)", devices.len());
        Ok(devices)
    }

    async fn create_scheduler(
        &self,
        _devices: Arc<RwLock<Vec<Arc<dyn NpuDevice + Send + Sync>>>>,
    ) -> Result<Arc<dyn NpuScheduler + Send + Sync>> {
        // For now, use the mock scheduler
        // In a full implementation, we'd create an Apple-specific scheduler
        Ok(Arc::new(MockScheduler::default()))
    }

    fn get_hal_info(&self) -> HalInfo {
        HalInfo {
            name: "Apple Neural Engine HAL".to_string(),
            version: "1.0.0".to_string(),
            supported_devices: vec![crate::npu::NpuDeviceType::AppleNeuralEngine],
            features: vec![
                HalFeature::DynamicModels,
                HalFeature::MultiModel,
                HalFeature::PowerManagement,
                HalFeature::MemoryManagement,
                HalFeature::HardwareMonitoring,
            ],
        }
    }

    async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down Apple Neural Engine HAL");
        Ok(())
    }
}
