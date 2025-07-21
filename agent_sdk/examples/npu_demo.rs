//! NPU Hardware Abstraction Layer Demo
//!
//! This example demonstrates how to use the NPU HAL to:
//! - Initialize mock NPU devices
//! - Submit inference tasks
//! - Monitor NPU utilization
//! - Manage device resources

use anyhow::Result;
use nebula_agent_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🚀 NebulaOS NPU Hardware Abstraction Layer Demo");

    // Initialize the NPU subsystem with mock devices
    let npu_manager = init_mock_npu_subsystem().await?;
    println!("✅ NPU Manager initialized successfully");

    // Get list of available NPU devices
    let devices = npu_manager.get_devices().await;
    println!("📋 Found {} NPU device(s):", devices.len());

    for device in &devices {
        let info = device.info();
        let capabilities = device.capabilities();
        let health = device.get_health().await?;

        println!("  🔷 Device: {} ({})", info.name, info.id);
        println!("    - Type: {:?}", info.device_type);
        println!("    - Vendor: {:?}", info.vendor);
        println!("    - Driver: {}", info.driver_version);
        println!("    - Available: {}", device.is_available().await);
        println!(
            "    - Health: {} ({}°C, {}W)",
            if health.is_healthy {
                "✅ Healthy"
            } else {
                "❌ Unhealthy"
            },
            health.temperature_celsius,
            health.power_consumption_watts
        );
        println!(
            "    - Utilization: {:.1}%",
            device.get_utilization().await * 100.0
        );
        println!(
            "    - Memory: {:.1} GB available",
            capabilities.memory.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Demonstrate task submission
    if let Some(device) = devices.first() {
        println!("\n🧠 Testing inference task submission...");

        // Create a mock inference task
        use nebula_agent_sdk::npu::{
            InferenceRequest, InferenceTask, ResourceAllocation, SchedulingHints,
        };
        use std::collections::HashMap;
        use std::time::Duration;

        let request = InferenceRequest {
            model_path: "mock_model.onnx".to_string(),
            inputs: vec![nebula_agent_sdk::npu::InferenceInput {
                data: vec![1, 2, 3, 4],
                shape: vec![1, 4],
                data_type: nebula_agent_sdk::npu::DataType::Float32,
            }],
            timeout: Duration::from_secs(30),
            priority: nebula_agent_sdk::npu::TaskPriority::Normal,
            agent_id: Some(1),
            metadata: HashMap::new(),
        };

        let resource_allocation = ResourceAllocation {
            device_id: device.id(),
            compute_units: vec![nebula_agent_sdk::npu::ComputeUnit::TensorCore],
            memory_bytes: 1024 * 1024, // 1MB
            power_budget_watts: 15.0,
            timeout: Duration::from_secs(30),
        };

        let task = InferenceTask {
            id: 1,
            request,
            priority: nebula_agent_sdk::npu::TaskPriority::Normal,
            resource_requirements: resource_allocation,
            scheduling_hints: SchedulingHints::default(),
        };

        // Submit the task
        match npu_manager.submit_task(task).await {
            Ok(task_id) => {
                println!("✅ Task submitted with ID: {}", task_id);

                // Check task status
                if let Some(status) = npu_manager.get_task_status(task_id).await {
                    println!("📊 Task status: {:?}", status);
                }
            }
            Err(e) => {
                println!("❌ Failed to submit task: {}", e);
            }
        }
    }

    // Get system-wide usage statistics
    println!("\n📊 NPU System Statistics:");
    let stats = npu_manager.get_usage_stats().await;
    println!("  - Total devices: {}", stats.total_devices);
    println!("  - Active devices: {}", stats.active_devices);
    println!(
        "  - Compute utilization: {:.1}%",
        stats.compute_utilization * 100.0
    );
    println!(
        "  - Memory utilization: {:.1}%",
        stats.memory_utilization * 100.0
    );
    println!(
        "  - Power consumption: {:.1}W",
        stats.power_consumption_watts
    );
    println!("  - Queued tasks: {}", stats.queued_tasks);

    // Test device capabilities
    if let Some(device) = devices.first() {
        println!("\n🔧 Device Capabilities Demo:");
        let caps = device.capabilities();

        println!("  Compute:");
        println!("    - Max batch size: {}", caps.max_batch_size());
        println!(
            "    - Concurrent inference: {}",
            caps.supports_concurrent_inference()
        );
        println!(
            "    - Tensor cores: {}",
            caps.get_core_count(&nebula_agent_sdk::npu::ComputeUnit::TensorCore)
        );

        println!("  Memory:");
        println!(
            "    - Available: {:.1} GB",
            caps.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "    - Unified memory: {}",
            caps.supports_memory_type(&nebula_agent_sdk::npu::MemoryType::Unified)
        );

        println!("  Models:");
        println!(
            "    - ONNX support: {}",
            caps.supports_model_format(&nebula_agent_sdk::npu::ModelFormat::Onnx)
        );
        println!(
            "    - Dynamic loading: {}",
            caps.model_support.dynamic_loading
        );

        // Test model loading
        println!("\n📥 Testing model loading...");
        match device.load_model("test_model.onnx").await {
            Ok(handle) => {
                println!("✅ Model loaded with handle: {:?}", handle);

                // Unload the model
                if let Err(e) = device.unload_model(handle).await {
                    println!("⚠️  Failed to unload model: {}", e);
                } else {
                    println!("✅ Model unloaded successfully");
                }
            }
            Err(e) => {
                println!("❌ Failed to load model: {}", e);
            }
        }

        // Test memory allocation
        println!("\n💾 Testing memory allocation...");
        let memory_size = 1024 * 1024; // 1MB
        match device.allocate_memory(memory_size).await {
            Ok(handle) => {
                println!(
                    "✅ Allocated {} bytes with handle: {:?}",
                    memory_size, handle
                );

                // Free the memory
                if let Err(e) = device.free_memory(handle).await {
                    println!("⚠️  Failed to free memory: {}", e);
                } else {
                    println!("✅ Memory freed successfully");
                }
            }
            Err(e) => {
                println!("❌ Failed to allocate memory: {}", e);
            }
        }
    }

    println!("\n🎉 NPU HAL Demo completed successfully!");
    Ok(())
}
