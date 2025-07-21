use env_logger;
use filesystem_agent::predictor::AccessPredictor;
use filesystem_agent::types::{FileAccessEvent, AccessType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("🚀 Testing ONNX ML Model Integration");
    println!("=====================================");
    
    // Initialize the predictor
    let mut predictor = AccessPredictor::new().await?;
    println!("✅ Predictor initialized successfully!");
    
    // Test the ML model if available
    #[cfg(feature = "mock")]
    {
        match predictor.test_ml_model().await {
            Ok(()) => println!("🎉 ML model test completed successfully!"),
            Err(e) => println!("⚠️ ML model test failed (this is expected if model file not found): {}", e),
        }
    }
    
    println!("\n📊 Testing prediction capabilities...");
    
    // Create some sample file access events
    let sample_files = vec![
        "/project/src/main.rs",
        "/project/src/lib.rs", 
        "/project/Cargo.toml",
        "/project/README.md",
        "/project/tests/integration.rs",
    ];
    
    // Record some access events
    for file_path in &sample_files {
        let event = FileAccessEvent::new(file_path.to_string())
            .with_access_type(AccessType::Read);
        
        predictor.record_access(&event).await?;
        println!("📝 Recorded access to: {}", file_path);
    }
    
    println!("\n🔮 Testing predictions...");
    
    // Test predictions for various files
    let test_files = vec![
        "/project/src/main.rs",      // Known file
        "/project/src/utils.rs",     // Similar file in same directory
        "/project/Dockerfile",       // New file in project root
        "/home/user/document.txt",   // Unrelated file
    ];
    
    for file_path in &test_files {
        match predictor.predict_access(file_path).await {
            Ok(probability) => {
                println!("🎯 Prediction for '{}': {:.4} ({:.1}%)", 
                        file_path, probability, probability * 100.0);
            },
            Err(e) => {
                println!("❌ Failed to predict for '{}': {}", file_path, e);
            }
        }
    }
    
    println!("\n🔗 Testing related file predictions...");
    
    // Test related file predictions
    match predictor.predict_related_files("/project/src/main.rs").await {
        Ok(related_files) => {
            println!("🎯 Related files for '/project/src/main.rs':");
            for (file_path, probability) in &related_files {
                println!("  - {} (probability: {:.3})", file_path, probability);
            }
        },
        Err(e) => {
            println!("❌ Failed to predict related files: {}", e);
        }
    }
    
    println!("\n✨ Test completed successfully!");
    Ok(())
}
