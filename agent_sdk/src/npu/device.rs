//! NPU Device interface and management

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use crate::npu::hal::{MemoryHandle, ModelHandle};
use crate::npu::{
    DeviceHealth, InferenceRequest, InferenceResponse, MemoryRegion, NpuCapabilities, NpuDeviceId,
    NpuDeviceType, NpuVendor, PowerState,
};

/// Core NPU device interface
#[async_trait]
pub trait NpuDevice: Send + Sync {
    /// Get unique device identifier
    fn id(&self) -> NpuDeviceId;

    /// Get device information
    fn info(&self) -> NpuDeviceInfo;

    /// Get device capabilities
    fn capabilities(&self) -> Arc<NpuCapabilities>;

    /// Initialize the device
    async fn init(&self) -> Result<()>;

    /// Shutdown the device
    async fn shutdown(&self) -> Result<()>;

    /// Execute inference on this device
    async fn execute_inference(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Load a model onto this device
    async fn load_model(&self, model_path: &str) -> Result<ModelHandle>;

    /// Unload a model from this device
    async fn unload_model(&self, handle: ModelHandle) -> Result<()>;

    /// Check if device is available for tasks
    async fn is_available(&self) -> bool;

    /// Get current device health status
    async fn get_health(&self) -> Result<DeviceHealth>;

    /// Get current power state
    async fn get_power_state(&self) -> Result<PowerState>;

    /// Set power state
    async fn set_power_state(&self, state: PowerState) -> Result<()>;

    /// Get memory information
    async fn get_memory_info(&self) -> Result<Vec<MemoryRegion>>;

    /// Allocate device memory
    async fn allocate_memory(&self, size_bytes: u64) -> Result<MemoryHandle>;

    /// Free device memory
    async fn free_memory(&self, handle: MemoryHandle) -> Result<()>;

    /// Get current utilization (0.0 to 1.0)
    async fn get_utilization(&self) -> f64;

    /// Get temperature in Celsius
    async fn get_temperature(&self) -> f32;

    /// Reset the device (for error recovery)
    async fn reset(&self) -> Result<()>;
}

/// Static device information
#[derive(Debug, Clone)]
pub struct NpuDeviceInfo {
    pub id: NpuDeviceId,
    pub name: String,
    pub device_type: NpuDeviceType,
    pub vendor: NpuVendor,
    pub driver_version: String,
    pub firmware_version: Option<String>,
    pub serial_number: Option<String>,
    pub pci_id: Option<String>,
    pub numa_node: Option<u32>,
}

impl NpuDeviceInfo {
    pub fn new(
        id: NpuDeviceId,
        name: String,
        device_type: NpuDeviceType,
        vendor: NpuVendor,
    ) -> Self {
        Self {
            id,
            name,
            device_type,
            vendor,
            driver_version: "1.0.0".to_string(),
            firmware_version: None,
            serial_number: None,
            pci_id: None,
            numa_node: None,
        }
    }

    pub fn with_driver_version(mut self, version: String) -> Self {
        self.driver_version = version;
        self
    }

    pub fn with_firmware_version(mut self, version: String) -> Self {
        self.firmware_version = Some(version);
        self
    }

    pub fn with_serial_number(mut self, serial: String) -> Self {
        self.serial_number = Some(serial);
        self
    }

    pub fn with_pci_id(mut self, pci_id: String) -> Self {
        self.pci_id = Some(pci_id);
        self
    }

    pub fn with_numa_node(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }
}

/// Device discovery and enumeration
pub struct DeviceDiscovery;

impl DeviceDiscovery {
    /// Discover all available NPU devices on the system
    pub async fn discover_all() -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>> {
        let mut devices = Vec::new();

        // Platform-specific device discovery
        #[cfg(target_os = "macos")]
        {
            if let Ok(mut apple_devices) = Self::discover_apple_devices().await {
                devices.append(&mut apple_devices);
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(mut intel_devices) = Self::discover_intel_devices().await {
                devices.append(&mut intel_devices);
            }

            if let Ok(mut nvidia_devices) = Self::discover_nvidia_devices().await {
                devices.append(&mut nvidia_devices);
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(mut intel_devices) = Self::discover_intel_devices().await {
                devices.append(&mut intel_devices);
            }

            if let Ok(mut nvidia_devices) = Self::discover_nvidia_devices().await {
                devices.append(&mut nvidia_devices);
            }
        }

        Ok(devices)
    }

    #[cfg(target_os = "macos")]
    async fn discover_apple_devices() -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>> {
        // Stub implementation for Apple Neural Engine discovery
        // In real implementation, this would use Metal or Core ML APIs
        Ok(Vec::new())
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    async fn discover_intel_devices() -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>> {
        // Stub implementation for Intel NPU discovery
        // In real implementation, this would use OpenVINO APIs
        Ok(Vec::new())
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    async fn discover_nvidia_devices() -> Result<Vec<Arc<dyn NpuDevice + Send + Sync>>> {
        // Stub implementation for NVIDIA GPU discovery
        // In real implementation, this would use CUDA APIs
        Ok(Vec::new())
    }
}

/// Device manager for tracking and managing multiple NPU devices
pub struct DeviceManager {
    devices: tokio::sync::RwLock<
        std::collections::HashMap<NpuDeviceId, Arc<dyn NpuDevice + Send + Sync>>,
    >,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            devices: tokio::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Add a device to the manager
    pub async fn add_device(&self, device: Arc<dyn NpuDevice + Send + Sync>) {
        let id = device.id();
        self.devices.write().await.insert(id, device);
    }

    /// Remove a device from the manager
    pub async fn remove_device(
        &self,
        device_id: &NpuDeviceId,
    ) -> Option<Arc<dyn NpuDevice + Send + Sync>> {
        self.devices.write().await.remove(device_id)
    }

    /// Get a device by ID
    pub async fn get_device(
        &self,
        device_id: &NpuDeviceId,
    ) -> Option<Arc<dyn NpuDevice + Send + Sync>> {
        self.devices.read().await.get(device_id).cloned()
    }

    /// Get all devices
    pub async fn get_all_devices(&self) -> Vec<Arc<dyn NpuDevice + Send + Sync>> {
        self.devices.read().await.values().cloned().collect()
    }

    /// Get available devices (ready for work)
    pub async fn get_available_devices(&self) -> Vec<Arc<dyn NpuDevice + Send + Sync>> {
        let devices = self.get_all_devices().await;
        let mut available = Vec::new();

        for device in devices {
            if device.is_available().await {
                available.push(device);
            }
        }

        available
    }

    /// Get devices by type
    pub async fn get_devices_by_type(
        &self,
        device_type: &NpuDeviceType,
    ) -> Vec<Arc<dyn NpuDevice + Send + Sync>> {
        let devices = self.get_all_devices().await;
        devices
            .into_iter()
            .filter(|d| d.info().device_type == *device_type)
            .collect()
    }

    /// Initialize all devices
    pub async fn init_all_devices(&self) -> Result<()> {
        let devices = self.get_all_devices().await;
        for device in devices {
            if let Err(e) = device.init().await {
                log::error!("Failed to initialize device {}: {}", device.id(), e);
            }
        }
        Ok(())
    }

    /// Shutdown all devices
    pub async fn shutdown_all_devices(&self) -> Result<()> {
        let devices = self.get_all_devices().await;
        for device in devices {
            if let Err(e) = device.shutdown().await {
                log::error!("Failed to shutdown device {}: {}", device.id(), e);
            }
        }
        Ok(())
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}
