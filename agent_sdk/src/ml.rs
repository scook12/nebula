//! ML module for the NebulaOS Agent SDK
//!
//! Provides AI/ML inference capabilities using ONNX models
//! with support for NPU acceleration and model management.

use crate::types::*;
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};

#[cfg(feature = "ai")]
use {
    ndarray::Array1,
    ort::{Environment, Session, Value},
};

/// Configuration for loading and running ML models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub model_path: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub precision: Precision,
    pub batch_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "default_model".to_string(),
            model_path: "model.onnx".to_string(),
            input_shape: vec![1, 3, 224, 224], // Default image input shape
            output_shape: vec![1, 1000],       // Default classification output
            precision: Precision::FP32,
            batch_size: 1,
        }
    }
}

/// Request for inference operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub task_id: TaskId,
    pub model_id: String,
    pub input_data: Vec<u8>,
    pub input_shape: Vec<usize>,
    pub priority: InferencePriority,
}

/// Response from inference operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub task_id: TaskId,
    pub success: bool,
    pub output_data: Vec<u8>,
    pub output_shape: Vec<usize>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Handler for ML operations
pub struct MLHandler {
    #[cfg(feature = "ai")]
    sessions: std::collections::HashMap<String, Session>,
    #[cfg(feature = "ai")]
    environment: Option<std::sync::Arc<Environment>>,
    models: std::collections::HashMap<String, ModelConfig>,
}

impl MLHandler {
    /// Create a new ML handler
    pub fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "ai")]
            sessions: std::collections::HashMap::new(),
            #[cfg(feature = "ai")]
            environment: None,
            models: std::collections::HashMap::new(),
        })
    }

    /// Initialize the ONNX runtime environment
    #[cfg(feature = "ai")]
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing ONNX Runtime environment");

        let env = Environment::builder()
            .with_name("NebulaOSAgent")
            .build()?
            .into_arc();

        self.environment = Some(env);
        info!("✅ ONNX Runtime initialized successfully");
        Ok(())
    }

    #[cfg(not(feature = "ai"))]
    pub async fn initialize(&mut self) -> Result<()> {
        warn!("AI features not enabled - ML operations will be mocked");
        Ok(())
    }

    /// Load a model from file
    pub async fn load_model(&mut self, config: ModelConfig) -> Result<()> {
        info!(
            "Loading model: {} from {}",
            config.model_id, config.model_path
        );

        #[cfg(feature = "ai")]
        {
            if let Some(ref env) = self.environment {
                // Check if model file exists
                if !std::path::Path::new(&config.model_path).exists() {
                    return Err(anyhow::anyhow!(
                        "Model file not found: {}",
                        config.model_path
                    ));
                }

                // Load the ONNX model
                let session = ort::SessionBuilder::new(env)?
                    .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                    .with_model_from_file(&config.model_path)?;

                // Validate model inputs/outputs
                let inputs = &session.inputs;
                let outputs = &session.outputs;

                debug!(
                    "Model inputs: {:?}",
                    inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
                );
                debug!(
                    "Model outputs: {:?}",
                    outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
                );

                self.sessions.insert(config.model_id.clone(), session);
                info!("✅ Model {} loaded successfully", config.model_id);
            } else {
                return Err(anyhow::anyhow!("ONNX environment not initialized"));
            }
        }

        self.models.insert(config.model_id.clone(), config);
        Ok(())
    }

    /// Run inference on loaded model
    pub async fn run_inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = std::time::Instant::now();
        debug!("Running inference for model: {}", request.model_id);

        #[cfg(feature = "ai")]
        {
            if let Some(session) = self.sessions.get(&request.model_id) {
                let result = self.run_onnx_inference(session, &request).await;
                let latency = start_time.elapsed().as_millis() as u64;

                match result {
                    Ok(output_data) => {
                        info!("✅ Inference completed in {}ms", latency);
                        Ok(InferenceResponse {
                            task_id: request.task_id,
                            success: true,
                            output_data,
                            output_shape: vec![], // TODO: Get actual output shape
                            latency_ms: latency,
                            error: None,
                        })
                    }
                    Err(e) => {
                        error!("❌ Inference failed: {}", e);
                        Ok(InferenceResponse {
                            task_id: request.task_id,
                            success: false,
                            output_data: vec![],
                            output_shape: vec![],
                            latency_ms: latency,
                            error: Some(e.to_string()),
                        })
                    }
                }
            } else {
                Err(anyhow::anyhow!("Model {} not loaded", request.model_id))
            }
        }

        #[cfg(not(feature = "ai"))]
        {
            // Mock inference for testing without AI features
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let latency = start_time.elapsed().as_millis() as u64;

            Ok(InferenceResponse {
                task_id: request.task_id,
                success: true,
                output_data: vec![0.5f32.to_ne_bytes().to_vec(); 10].concat(), // Mock output
                output_shape: vec![1, 10],
                latency_ms: latency,
                error: None,
            })
        }
    }

    #[cfg(feature = "ai")]
    async fn run_onnx_inference(
        &self,
        session: &Session,
        request: &InferenceRequest,
    ) -> Result<Vec<u8>> {
        // Convert input data to ndarray
        let input_len = request.input_shape.iter().product::<usize>();

        // Assume f32 input data for simplicity
        let input_f32: Vec<f32> = request
            .input_data
            .chunks_exact(4)
            .take(input_len)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        if input_f32.len() != input_len {
            return Err(anyhow::anyhow!("Input data length mismatch"));
        }

        // Create input tensor
        let input_tensor =
            ndarray::Array::from_shape_vec(request.input_shape.clone(), input_f32)?.into_dyn();

        // Run inference
        let allocator = session.allocator();
        let tensor_ref = input_tensor.try_into()?;
        let input_value = Value::from_array(allocator, &tensor_ref)?;

        let outputs = session.run(vec![input_value])?;

        if outputs.is_empty() {
            return Err(anyhow::anyhow!("No output from model"));
        }

        // Extract output data
        let output_tensor = outputs[0].try_extract::<f32>()?;
        let output_data: Vec<u8> = output_tensor
            .view()
            .iter()
            .flat_map(|&x| x.to_ne_bytes())
            .collect();

        Ok(output_data)
    }

    /// Get information about loaded models
    pub fn get_loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Check if a model is loaded
    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// Unload a model
    pub async fn unload_model(&mut self, model_id: &str) -> Result<()> {
        #[cfg(feature = "ai")]
        self.sessions.remove(model_id);

        self.models.remove(model_id);
        info!("Model {} unloaded", model_id);
        Ok(())
    }
}

impl Default for MLHandler {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Helper functions for common ML operations
pub mod helpers {
    use super::*;

    /// Convert image bytes to tensor format
    pub fn image_to_tensor(
        image_data: &[u8],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<Vec<f32>> {
        if image_data.len() != width * height * channels {
            return Err(anyhow::anyhow!("Image data size mismatch"));
        }

        // Convert u8 to f32 and normalize to [0, 1]
        let tensor: Vec<f32> = image_data
            .iter()
            .map(|&pixel| pixel as f32 / 255.0)
            .collect();

        Ok(tensor)
    }

    /// Convert text to simple token IDs (mock tokenization)
    pub fn text_to_tokens(text: &str, max_length: usize) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.chars().take(max_length).map(|c| c as u32).collect();

        // Pad to max_length
        tokens.resize(max_length, 0);
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ml_handler_creation() {
        let mut handler = MLHandler::new().unwrap();
        handler.initialize().await.unwrap();
        assert!(handler.get_loaded_models().is_empty());
    }

    #[tokio::test]
    async fn test_mock_inference() {
        let handler = MLHandler::new().unwrap();

        let request = InferenceRequest {
            task_id: 1,
            model_id: "test_model".to_string(),
            input_data: vec![0; 100],
            input_shape: vec![1, 25],
            priority: InferencePriority::Normal,
        };

        // This should work even without loading a model in mock mode
        #[cfg(not(feature = "ai"))]
        {
            let response = handler.run_inference(request).await.unwrap();
            assert!(response.success);
            assert_eq!(response.task_id, 1);
        }
    }

    #[test]
    fn test_image_to_tensor() {
        let image_data = vec![128; 32 * 32 * 3]; // 32x32 RGB image
        let tensor = helpers::image_to_tensor(&image_data, 32, 32, 3).unwrap();

        assert_eq!(tensor.len(), 32 * 32 * 3);
        assert!((tensor[0] - 128.0 / 255.0).abs() < 0.001);
    }

    #[test]
    fn test_text_to_tokens() {
        let tokens = helpers::text_to_tokens("Hello", 10);
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 'H' as u32);
        assert_eq!(tokens[5], 0); // Padding
    }
}
