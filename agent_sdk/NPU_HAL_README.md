# NebulaOS NPU Hardware Abstraction Layer (HAL)

The NPU HAL provides a unified, hardware-agnostic interface for Neural Processing Units in NebulaOS. It enables AI agents to leverage specialized AI/ML hardware while maintaining compatibility across different NPU vendors and types.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NebulaOS Agent SDK                      │
├─────────────────────────────────────────────────────────────┤
│                       NPU Manager                          │
├─────────────────────────────────────────────────────────────┤
│    NPU HAL     │     Scheduler     │     Capabilities      │
├─────────────────────────────────────────────────────────────┤
│  Device Driver Interface (NpuDriver trait)                 │
├─────────────────────────────────────────────────────────────┤
│     Mock NPU    │   Apple Neural   │   Intel NPU   │ GPU   │
│   (Testing)     │     Engine        │   (OpenVINO)  │(CUDA) │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. NPU Manager (`NpuManager`)
Central coordinator for all NPU operations:
- Device discovery and management
- Task scheduling and resource allocation  
- System-wide statistics and monitoring

### 2. Hardware Abstraction Layer (`NpuHal`)
Provides hardware-agnostic interface:
- Device discovery and initialization
- Scheduler creation and management
- Hardware capability reporting

### 3. Device Interface (`NpuDevice`)
Individual NPU device management:
- Model loading/unloading
- Inference execution
- Memory management
- Power and thermal monitoring

### 4. Task Scheduler (`NpuScheduler`)
Manages inference workload distribution:
- Priority-based task queuing
- Resource allocation optimization
- Load balancing across devices

## 🚀 Quick Start

### Basic Usage

```rust
use nebula_agent_sdk::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize NPU subsystem (tries real hardware, falls back to mock)
    let npu_manager = init_npu_subsystem().await?;
    
    // Get available devices
    let devices = npu_manager.get_devices().await;
    println!("Found {} NPU devices", devices.len());
    
    // Check device capabilities
    if let Some(device) = devices.first() {
        let caps = device.capabilities();
        println!("Max batch size: {}", caps.max_batch_size());
        println!("Available memory: {} GB", 
                 caps.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0));
    }
    
    Ok(())
}
```

### Mock Development Mode

For development and testing without real NPU hardware:

```rust
// Explicitly use mock NPU subsystem
let npu_manager = init_mock_npu_subsystem().await?;
```

### Submitting Inference Tasks

```rust
use nebula_agent_sdk::npu::*;
use std::time::Duration;

// Create inference request
let request = InferenceRequest {
    model_path: "my_model.onnx".to_string(),
    inputs: vec![InferenceInput {
        data: input_bytes,
        shape: vec![1, 224, 224, 3],
        data_type: DataType::Float32,
    }],
    timeout: Duration::from_secs(30),
    priority: TaskPriority::Normal,
    agent_id: Some(agent_id),
    metadata: HashMap::new(),
};

// Create resource requirements
let resource_allocation = ResourceAllocation {
    device_id: device.id(),
    compute_units: vec![ComputeUnit::TensorCore],
    memory_bytes: 512 * 1024 * 1024, // 512MB
    power_budget_watts: 20.0,
    timeout: Duration::from_secs(30),
};

// Create and submit task
let task = InferenceTask {
    id: task_id,
    request,
    priority: TaskPriority::Normal,
    resource_requirements: resource_allocation,
    scheduling_hints: SchedulingHints::default(),
};

let task_id = npu_manager.submit_task(task).await?;
```

## 🎯 Supported NPU Types

### Current Support Status

| NPU Type | Status | Implementation |
|----------|--------|----------------|
| Mock NPU | ✅ Complete | Full mock implementation for testing |
| Apple Neural Engine | 🚧 Planned | Metal Performance Shaders / Core ML |
| Intel NPU | 🚧 Planned | OpenVINO Runtime |
| NVIDIA GPU | 🚧 Planned | CUDA / TensorRT |
| AMD GPU | 🚧 Planned | ROCm / MIGraphX |
| Qualcomm Hexagon | 🚧 Planned | Qualcomm Neural SDK |
| Google Edge TPU | 🚧 Planned | Edge TPU Runtime |

### Hardware Detection

The HAL automatically detects available NPU hardware at runtime:

```rust
// Detection priority order:
1. Apple Neural Engine (macOS)
2. Intel NPU (Linux/Windows) 
3. NVIDIA GPU (Linux/Windows)
4. AMD GPU (Linux/Windows)
5. Fallback to CPU
```

## 📊 Device Capabilities

### Compute Capabilities
```rust
let caps = device.capabilities();

// Core information
let max_batch = caps.max_batch_size();
let concurrent = caps.supports_concurrent_inference();
let tensor_cores = caps.get_core_count(&ComputeUnit::TensorCore);

// Data type support
let supports_fp16 = caps.supports_data_type(&DataType::Float16);
let supports_int8 = caps.supports_data_type(&DataType::Int8);
```

### Memory Management
```rust
// Memory capabilities
let total_memory = caps.available_memory();
let unified_memory = caps.supports_memory_type(&MemoryType::Unified);

// Allocate device memory
let handle = device.allocate_memory(1024 * 1024).await?; // 1MB
// ... use memory ...
device.free_memory(handle).await?;
```

### Model Support
```rust
// Check model format support
let onnx_support = caps.supports_model_format(&ModelFormat::Onnx);
let dynamic_loading = caps.model_support.dynamic_loading;

// Load and manage models
let model_handle = device.load_model("model.onnx").await?;
// ... run inference ...
device.unload_model(model_handle).await?;
```

## 🔍 Monitoring and Diagnostics

### Device Health Monitoring
```rust
let health = device.get_health().await?;
println!("Device healthy: {}", health.is_healthy);
println!("Temperature: {}°C", health.temperature_celsius);
println!("Power consumption: {}W", health.power_consumption_watts);
println!("Utilization: {:.1}%", device.get_utilization().await * 100.0);
```

### System Statistics
```rust
let stats = npu_manager.get_usage_stats().await;
println!("Total devices: {}", stats.total_devices);
println!("Active devices: {}", stats.active_devices);
println!("Compute utilization: {:.1}%", stats.compute_utilization * 100.0);
println!("Memory utilization: {:.1}%", stats.memory_utilization * 100.0);
```

### Task Management
```rust
// Submit task and monitor status
let task_id = npu_manager.submit_task(task).await?;

// Check task status
if let Some(status) = npu_manager.get_task_status(task_id).await {
    match status {
        TaskStatus::Queued => println!("Task is waiting in queue"),
        TaskStatus::Running => println!("Task is currently executing"),
        TaskStatus::Completed => println!("Task completed successfully"),
        TaskStatus::Failed(error) => println!("Task failed: {}", error),
        _ => println!("Task status: {:?}", status),
    }
}

// Cancel task if needed
npu_manager.cancel_task(task_id).await?;
```

## ⚡ Performance Features

### Priority-Based Scheduling
```rust
// Task priorities (highest to lowest)
TaskPriority::Critical   // System-critical tasks
TaskPriority::High       // High-priority user tasks  
TaskPriority::Normal     // Default priority
TaskPriority::Low        // Background processing
TaskPriority::Background // Batch jobs
```

### Resource Optimization
```rust
// Scheduling hints for optimal placement
let hints = SchedulingHints {
    preferred_devices: vec![NpuDeviceType::AppleNeuralEngine],
    avoid_devices: vec![slow_device_id],
    required_memory_type: Some(MemoryType::Dedicated),
    min_tops: Some(10.0), // Minimum 10 TOPS performance
    max_latency: Some(Duration::from_millis(100)),
};
```

### Batch Processing
```rust
// Leverage hardware batch capabilities
let batch_size = caps.max_batch_size();
let request = InferenceRequest {
    inputs: create_batch_inputs(batch_size),
    // ... other fields
};
```

## 🧪 Testing and Development

### Running Tests
```bash
# Run all NPU HAL tests
cargo test --features npu

# Run the interactive demo
cargo run --example npu_demo --features npu
```

### Mock Implementation
The mock implementation provides:
- ✅ Complete API compatibility
- ✅ Realistic performance simulation  
- ✅ Error condition testing
- ✅ Resource tracking
- ✅ Concurrent operation support

### Development Environment
```toml
# Cargo.toml features for development
[features]
default = ["ai", "npu"]
npu = ["libc", "uuid", "thiserror", "rand"]
```

## 🔮 Future Roadmap

### Phase 1: Foundation (✅ Complete)
- [x] Core HAL architecture
- [x] Mock implementation
- [x] Device management
- [x] Task scheduling
- [x] Resource allocation

### Phase 2: Hardware Integration (🚧 In Progress)
- [ ] Apple Neural Engine support
- [ ] Intel NPU via OpenVINO
- [ ] NVIDIA GPU via CUDA
- [ ] Hardware detection logic
- [ ] Performance benchmarking

### Phase 3: Advanced Features (📋 Planned)
- [ ] Dynamic model optimization
- [ ] Multi-device inference
- [ ] Model quantization
- [ ] Streaming inference
- [ ] Power management

### Phase 4: System Integration (📋 Planned)
- [ ] NebulaOS kernel integration
- [ ] Real-time scheduling guarantees
- [ ] Security and isolation
- [ ] Network-distributed NPUs
- [ ] Cloud NPU federation

## 🤝 Contributing

### Adding New Hardware Support

1. Implement the `NpuDriver` trait for your hardware
2. Add device detection logic
3. Create hardware-specific tests
4. Update documentation

Example structure:
```rust
// src/npu/drivers/my_npu.rs
pub struct MyNpuDriver {
    // Hardware-specific fields
}

#[async_trait]
impl NpuDriver for MyNpuDriver {
    async fn init(&mut self) -> Result<()> {
        // Initialize hardware
    }
    
    async fn load_model(&self, path: &str) -> Result<ModelHandle> {
        // Load model to hardware
    }
    
    // ... implement other methods
}
```

### Testing Guidelines
- Always test with mock implementation first
- Add hardware-specific integration tests
- Test error conditions and recovery
- Verify resource cleanup

## 📄 License

Licensed under MIT License. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the NebulaOS ecosystem*
