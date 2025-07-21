//! NPU Hardware Abstraction Layer
//!
//! This module provides a unified interface for NPU (Neural Processing Unit) devices,
//! supporting both real hardware and mock implementations for development and testing.

pub mod capabilities;
pub mod device;
pub mod drivers;
pub mod hal;
pub mod mock;
pub mod scheduler;
pub mod types;

// Re-export commonly used types and traits
pub use capabilities::{ComputeCapability, MemoryCapability, NpuCapabilities};
pub use device::{NpuDevice, NpuDeviceInfo};
pub use hal::{HalFeature, HalInfo, MemoryHandle, ModelHandle, NpuDriver, NpuHal};
pub use scheduler::NpuScheduler;
pub use types::*;

use crate::types::TaskId;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Global NPU manager that coordinates all NPU devices and scheduling
pub struct NpuManager {
    hal: Arc<dyn NpuHal + Send + Sync>,
    devices: Arc<RwLock<Vec<Arc<dyn NpuDevice + Send + Sync>>>>,
    scheduler: Arc<dyn NpuScheduler + Send + Sync>,
}

impl NpuManager {
    /// Create a new NPU manager with the specified HAL implementation
    pub async fn new(hal: Arc<dyn NpuHal + Send + Sync>) -> Result<Self> {
        let devices = Arc::new(RwLock::new(Vec::new()));

        // Discover available NPU devices
        let discovered_devices = hal.discover_devices().await?;
        let mut device_list = devices.write().await;
        for device in discovered_devices {
            device_list.push(device);
        }
        drop(device_list);

        // Create scheduler with discovered devices
        let scheduler = hal.create_scheduler(devices.clone()).await?;

        Ok(NpuManager {
            hal,
            devices,
            scheduler,
        })
    }

    /// Get list of available NPU devices
    pub async fn get_devices(&self) -> Vec<Arc<dyn NpuDevice + Send + Sync>> {
        self.devices.read().await.clone()
    }

    /// Get device by ID
    pub async fn get_device(
        &self,
        device_id: &NpuDeviceId,
    ) -> Option<Arc<dyn NpuDevice + Send + Sync>> {
        let devices = self.devices.read().await;
        devices.iter().find(|d| d.id() == *device_id).cloned()
    }

    /// Submit an inference task to the scheduler
    pub async fn submit_task(&self, task: InferenceTask) -> Result<TaskId> {
        self.scheduler.submit_task(task).await
    }

    /// Cancel a running task
    pub async fn cancel_task(&self, task_id: TaskId) -> Result<()> {
        self.scheduler.cancel_task(task_id).await
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: TaskId) -> Option<TaskStatus> {
        self.scheduler.get_task_status(task_id).await
    }

    /// Get system-wide NPU usage statistics
    pub async fn get_usage_stats(&self) -> NpuUsageStats {
        self.scheduler.get_usage_stats().await
    }
}

/// Initialize the NPU subsystem with default (mock) implementation
pub async fn init_npu_subsystem() -> Result<NpuManager> {
    #[cfg(feature = "npu")]
    {
        // Try to detect real NPU hardware first
        if let Ok(hal) = detect_hardware_hal().await {
            log::info!("Detected real NPU hardware, using hardware HAL");
            NpuManager::new(hal).await
        } else {
            log::info!("No NPU hardware detected, using mock implementation");
            let mock_hal = Arc::new(mock::MockNpuHal::new().await?);
            NpuManager::new(mock_hal).await
        }
    }

    #[cfg(not(feature = "npu"))]
    {
        anyhow::bail!("NPU feature not enabled. Add 'npu' to features in Cargo.toml")
    }
}

/// Initialize NPU subsystem with explicit mock implementation
pub async fn init_mock_npu_subsystem() -> Result<NpuManager> {
    #[cfg(feature = "npu")]
    {
        log::info!("Initializing mock NPU subsystem");
        let mock_hal = Arc::new(mock::MockNpuHal::new().await?);
        NpuManager::new(mock_hal).await
    }

    #[cfg(not(feature = "npu"))]
    {
        anyhow::bail!("NPU feature not enabled. Add 'npu' to features in Cargo.toml")
    }
}

/// Attempt to detect and initialize real NPU hardware
async fn detect_hardware_hal() -> Result<Arc<dyn NpuHal + Send + Sync>> {
    // This would contain platform-specific detection logic
    // For now, we'll implement basic detection stubs

    #[cfg(target_os = "macos")]
    {
        // Try to detect Apple Neural Engine
        if let Ok(hal) = detect_apple_neural_engine().await {
            return Ok(hal);
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Try to detect Intel NPU via OpenVINO
        if let Ok(hal) = detect_intel_npu().await {
            return Ok(hal);
        }
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        // Try to detect NVIDIA GPU as NPU fallback
        if let Ok(hal) = detect_nvidia_gpu().await {
            return Ok(hal);
        }
    }

    anyhow::bail!("No supported NPU hardware detected")
}

#[cfg(target_os = "macos")]
async fn detect_apple_neural_engine() -> Result<Arc<dyn NpuHal + Send + Sync>> {
    #[cfg(feature = "apple_neural_engine")]
    {
        use crate::npu::drivers::AppleNeuralEngineHal;
        log::info!("Detected Apple Neural Engine");
        let hal = AppleNeuralEngineHal::new().await?;
        return Ok(Arc::new(hal) as Arc<dyn NpuHal + Send + Sync>);
    }

    anyhow::bail!("Apple Neural Engine detection not yet supported")
}

#[cfg(target_os = "linux")]
async fn detect_intel_npu() -> Result<Arc<dyn NpuHal + Send + Sync>> {
    // Stub implementation - would use OpenVINO
    anyhow::bail!("Intel NPU detection not yet implemented")
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
async fn detect_nvidia_gpu() -> Result<Arc<dyn NpuHal + Send + Sync>> {
    // Stub implementation - would use CUDA
    anyhow::bail!("NVIDIA GPU detection not yet implemented")
}
