//! Apple Neural Engine Demo
//!
//! This example demonstrates Apple Neural Engine detection and capabilities

use anyhow::Result;
use nebula_agent_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🍎 NebulaOS Apple Neural Engine Demo");

    // Try to detect real NPU hardware first (including Apple Neural Engine)
    let npu_manager = init_npu_subsystem().await?;
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
        println!("    - Firmware: {:?}", info.firmware_version);
        println!("    - Available: {}", device.is_available().await);
        println!(
            "    - Health: {} ({}°C, {:.1}W)",
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

        // Show Apple Neural Engine specific details
        if info.device_type == nebula_agent_sdk::npu::NpuDeviceType::AppleNeuralEngine {
            println!("  🧠 Apple Neural Engine Specifications:");
            println!("    - Peak TOPS: {:.1}", capabilities.performance.peak_tops);
            println!(
                "    - Memory Bandwidth: {:.1} GB/s",
                capabilities.performance.memory_bandwidth_gbps
            );
            println!(
                "    - Tensor Cores: {}",
                capabilities.get_core_count(&nebula_agent_sdk::npu::ComputeUnit::TensorCore)
            );
            println!(
                "    - Vector Cores: {}",
                capabilities.get_core_count(&nebula_agent_sdk::npu::ComputeUnit::VectorCore)
            );
            println!(
                "    - Unified Memory: {} GB",
                capabilities.memory.total_memory_bytes / (1024 * 1024 * 1024)
            );
            println!(
                "    - Core ML Support: {}",
                capabilities.supports_model_format(&nebula_agent_sdk::npu::ModelFormat::CoreMl)
            );
            println!(
                "    - Mixed Precision: {}",
                capabilities.compute.mixed_precision
            );
        }

        println!(
            "    - Memory: {:.1} GB available",
            capabilities.memory.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Test inference if we have Apple Neural Engine
    if let Some(device) = devices
        .iter()
        .find(|d| d.info().device_type == nebula_agent_sdk::npu::NpuDeviceType::AppleNeuralEngine)
    {
        println!("\n🧠 Testing Apple Neural Engine inference...");

        use nebula_agent_sdk::npu::{DataType, InferenceInput, InferenceRequest};
        use std::collections::HashMap;
        use std::time::Duration;

        // Create test input (4 float32 values)
        let test_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let input_bytes: Vec<u8> = test_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        let request = InferenceRequest {
            model_path: "apple_neural_test.mlmodel".to_string(),
            inputs: vec![InferenceInput {
                data: input_bytes,
                shape: vec![1, 4],
                data_type: DataType::Float32,
            }],
            timeout: Duration::from_secs(5),
            priority: nebula_agent_sdk::npu::TaskPriority::High,
            agent_id: Some(1),
            metadata: HashMap::new(),
        };

        match device.execute_inference(request).await {
            Ok(response) => {
                println!("✅ Apple Neural Engine inference completed!");
                println!("  - Execution time: {:?}", response.execution_time);
                println!(
                    "  - Output data size: {} bytes",
                    response.outputs[0].data.len()
                );

                // Convert output back to float32 for display
                let output_floats: Vec<f32> = response.outputs[0]
                    .data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                println!("  - Output values: {:?}", output_floats);
            }
            Err(e) => {
                println!("❌ Apple Neural Engine inference failed: {}", e);
            }
        }

        // Test model operations
        println!("\n📥 Testing Apple Neural Engine model operations...");
        match device.load_model("test_model.mlmodel").await {
            Ok(handle) => {
                println!("✅ Model loaded successfully with handle: {:?}", handle);

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
    }

    // Get system statistics
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

    println!("\n🎉 Apple Neural Engine demo completed successfully!");
    Ok(())
}
