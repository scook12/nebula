use std::collections::HashMap;
use log::{info, debug, warn};
use lru::LruCache;
use std::num::NonZeroUsize;
use chrono::{DateTime, Utc, Timelike, Datelike};
use ndarray::Array1;

use ort::{Session, Value};

use crate::types::{
    FileAccessEvent, FileAccessPattern
};

// AI-powered predictor using ONNX Runtime for real ML inference
pub struct AccessPredictor {
    access_patterns: HashMap<String, FileAccessPattern>,
    directory_patterns: HashMap<String, Vec<String>>,  // directory -> frequently accessed files
    extension_patterns: HashMap<String, Vec<String>>,  // extension -> related files
    temporal_cache: LruCache<String, f32>,             // file -> recent prediction score
    user_session_start: DateTime<Utc>,
    
    #[cfg(feature = "mock")]
    ml_session: Option<Session>,
}

impl AccessPredictor {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing AI-powered access predictor");
        
        #[cfg(feature = "mock")]
        let ml_session = Self::init_ml_model().await.ok();
        
        Ok(Self {
            access_patterns: HashMap::new(),
            directory_patterns: HashMap::new(),
            extension_patterns: HashMap::new(),
            temporal_cache: LruCache::new(NonZeroUsize::new(1000).unwrap()),
            user_session_start: Utc::now(),
            
            #[cfg(feature = "mock")]
            ml_session,
        })
    }
    
    #[cfg(feature = "mock")]
    async fn init_ml_model() -> Result<Session, Box<dyn std::error::Error + Send + Sync>> {
        info!("Loading ONNX model for file access prediction");
        
        let model_path = "simple_file_access_predictor.onnx";
        
        // Check if model file exists
        if !std::path::Path::new(model_path).exists() {
            warn!("ONNX model not found at {}, using fallback heuristics", model_path);
            return Err("Model file not found".into());
        }
        
        match ort::Environment::builder().with_name("fileSysPredictorEnv").build() {
            Ok(environment) => {
                let environment = environment.into_arc();
                
                // Load the ONNX model with detailed error handling
                match ort::SessionBuilder::new(&environment)
                    .and_then(|builder| builder.with_optimization_level(ort::GraphOptimizationLevel::Level3))
                    .and_then(|builder| builder.with_model_from_file(model_path)) {
                    Ok(session) => {
                        // Validate model inputs and outputs
                        let inputs = &session.inputs;
                        let outputs = &session.outputs;
                        
                        info!("📊 Model inputs: {:?}", inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
                        info!("📊 Model outputs: {:?}", outputs.iter().map(|o| &o.name).collect::<Vec<_>>());
                        
                        // Verify we have the expected input structure
                        if inputs.is_empty() {
                            warn!("⚠️ Model has no inputs - this might cause issues");
                        }
                        
                        if outputs.is_empty() {
                            warn!("⚠️ Model has no outputs - this might cause issues");
                        }
                        
                        info!("✅ ONNX model loaded successfully!");
                        Ok(session)
                    },
                    Err(e) => {
                        warn!("❌ Failed to load ONNX model: {}", e);
                        Err(format!("Model loading failed: {}", e).into())
                    }
                }
            },
            Err(e) => {
                warn!("❌ Failed to create ONNX environment: {}", e);
                Err(format!("Environment creation failed: {}", e).into())
            }
        }
    }
    
    pub async fn record_access(&mut self, event: &FileAccessEvent) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Recording file access: {}", event.path);
        
        // Update access pattern for this file
        let pattern = self.access_patterns
            .entry(event.path.clone())
            .or_insert_with(|| FileAccessPattern::new(event.path.clone()));
        
        pattern.record_access(event.timestamp);
        
        // Update directory patterns
        if let Some(parent) = std::path::Path::new(&event.path).parent() {
            let dir_str = parent.to_string_lossy().to_string();
            self.directory_patterns
                .entry(dir_str)
                .or_insert_with(Vec::new)
                .push(event.path.clone());
        }
        
        // Update extension patterns
        if let Some(ref ext) = event.file_extension {
            self.extension_patterns
                .entry(ext.clone())
                .or_insert_with(Vec::new)
                .push(event.path.clone());
        }
        
        // Update correlations between files
        self.update_file_correlations(&event.path, event.timestamp).await?;
        
        Ok(())
    }
    
    pub async fn predict_access(&self, file_path: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Predicting access for: {}", file_path);
        
        // Try ML model first if available
        #[cfg(feature = "mock")]
        if let Some(ref session) = self.ml_session {
            if let Ok(ml_prediction) = self.predict_with_ml_model(session, file_path) {
                debug!("🧠 ML prediction for {}: {:.4}", file_path, ml_prediction);
                return Ok(ml_prediction);
            }
        }
        
        // Fallback to heuristic-based prediction
        if let Some(pattern) = self.access_patterns.get(file_path) {
            let heuristic_prediction = pattern.calculate_access_probability(Utc::now());
            debug!("📊 Heuristic prediction for {}: {:.4}", file_path, heuristic_prediction);
            Ok(heuristic_prediction)
        } else {
            // For new files, use heuristic-based prediction
            let new_file_prediction = self.predict_new_file_access(file_path).await?;
            debug!("🆕 New file prediction for {}: {:.4}", file_path, new_file_prediction);
            Ok(new_file_prediction)
        }
    }
    
    #[cfg(feature = "mock")]
    fn predict_with_ml_model(&self, session: &Session, file_path: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        // Generate feature vector for ML model
        let features = self.generate_feature_vector(file_path, Utc::now());
        
        // Ensure we have exactly 8 features (pad or truncate as needed)
        const EXPECTED_FEATURES: usize = 8;
        let mut feature_vec = features.to_vec();
        feature_vec.resize(EXPECTED_FEATURES, 0.0);
        let features = ndarray::Array1::from(feature_vec);
        
        debug!("🔢 Feature vector for {}: {:?}", file_path, features.as_slice().unwrap());
        
        // Prepare input for ONNX model
        let input_tensor = ndarray::Array2::from_shape_vec(
            (1, EXPECTED_FEATURES), 
            features.to_vec()
        )?.into_dyn();
        
        // Get model input and output information
        let inputs = &session.inputs;
        let outputs = &session.outputs;
        
        if inputs.is_empty() {
            return Err("Model has no inputs defined".into());
        }
        
        if outputs.is_empty() {
            return Err("Model has no outputs defined".into());
        }
        
        // Get the first input name (assuming single input)
        let input_name = &inputs[0].name;
        
        debug!("🎯 Running inference with input '{}'", input_name);
        
        // Run inference with proper input mapping
        let allocator = session.allocator();
        let tensor_ref = input_tensor.try_into()?;
        let input_value = Value::from_array(allocator, &tensor_ref)?;
        
        match session.run(vec![input_value]) {
            Ok(outputs) => {
                if outputs.is_empty() {
                    return Err("Model produced no outputs".into());
                }
                
                // Extract the first output (assuming single output)
                let output_tensor = outputs[0].try_extract::<f32>()?;
                let probability = output_tensor.view().first().copied().unwrap_or(0.0);
                
                debug!("🎯 Raw model output: {:.6}", probability);
                
                // Ensure probability is in valid range
                let clamped_probability = probability.max(0.0).min(1.0);
                
                if (probability - clamped_probability).abs() > 0.001 {
                    warn!("⚠️ Model output {:.6} was clamped to {:.6}", probability, clamped_probability);
                }
                
                Ok(clamped_probability)
            },
            Err(e) => {
                warn!("❌ ONNX inference failed: {}", e);
                Err(format!("Inference failed: {}", e).into())
            }
        }
    }
    
    pub async fn predict_related_files(&self, accessed_path: &str) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error + Send + Sync>> {
        let mut predictions = Vec::new();
        
        // Strategy 1: Files in the same directory
        if let Some(parent) = std::path::Path::new(accessed_path).parent() {
            let dir_str = parent.to_string_lossy().to_string();
            if let Some(dir_files) = self.directory_patterns.get(&dir_str) {
                for file_path in dir_files {
                    if file_path != accessed_path {
                        let similarity = self.calculate_directory_similarity(accessed_path, file_path);
                        if similarity > 0.3 {
                            predictions.push((file_path.clone(), similarity * 0.7));
                        }
                    }
                }
            }
        }
        
        // Strategy 2: Files with same extension
        let accessed_ext = std::path::Path::new(accessed_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());
            
        if let Some(ext) = accessed_ext {
            if let Some(ext_files) = self.extension_patterns.get(&ext) {
                for file_path in ext_files {
                    if file_path != accessed_path {
                        let base_prob = 0.4;
                        let recency_factor = self.calculate_recency_factor(file_path);
                        predictions.push((file_path.clone(), base_prob * recency_factor));
                    }
                }
            }
        }
        
        // Strategy 3: Files with established correlations
        if let Some(pattern) = self.access_patterns.get(accessed_path) {
            for (related_file, correlation) in &pattern.related_files {
                if correlation > &0.5 {
                    predictions.push((related_file.clone(), *correlation * 0.8));
                }
            }
        }
        
        // Remove duplicates and sort by probability
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        predictions.dedup_by(|a, b| a.0 == b.0);
        predictions.truncate(10); // Top 10 predictions
        
        Ok(predictions)
    }
    
    async fn predict_new_file_access(&self, file_path: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let mut score = 0.0;
        
        // Factor 1: Directory activity
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            let dir_str = parent.to_string_lossy().to_string();
            if let Some(dir_files) = self.directory_patterns.get(&dir_str) {
                let recent_activity = dir_files.len() as f32 / 100.0; // normalized
                score += recent_activity * 0.3;
            }
        }
        
        // Factor 2: Extension popularity
        let file_ext = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());
            
        if let Some(ext) = file_ext {
            if let Some(ext_files) = self.extension_patterns.get(&ext) {
                let ext_popularity = (ext_files.len() as f32 / 50.0).min(1.0);
                score += ext_popularity * 0.2;
            }
        }
        
        // Factor 3: Time of day pattern
        let time_score = self.calculate_time_of_day_score();
        score += time_score * 0.2;
        
        // Factor 4: Session context (files accessed in current session)
        score += self.calculate_session_context_score() * 0.3;
        
        Ok(score.min(1.0))
    }
    
    async fn update_file_correlations(&mut self, accessed_file: &str, access_time: DateTime<Utc>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Look for files accessed within a time window (e.g., 5 minutes)
        let time_window = chrono::Duration::minutes(5);
        
        // Collect correlations to update in a separate pass to avoid borrow conflicts
        let mut correlations_to_update = Vec::new();
        
        for (other_file, pattern) in &self.access_patterns {
            if other_file == accessed_file {
                continue;
            }
            
            // Check if this file was accessed recently
            let time_diff = (access_time - pattern.last_access).abs();
            if time_diff <= time_window {
                let current_correlation = pattern.related_files
                    .get(accessed_file)
                    .copied()
                    .unwrap_or(0.0);
                
                let new_correlation = (current_correlation + 0.1).min(1.0);
                correlations_to_update.push((other_file.clone(), accessed_file.to_string(), new_correlation));
            }
        }
        
        // Now update the correlations
        for (other_file, accessed_file_key, correlation) in correlations_to_update {
            // Update forward correlation
            if let Some(pattern) = self.access_patterns.get_mut(&other_file) {
                pattern.related_files.insert(accessed_file_key.clone(), correlation);
            }
            
            // Update reverse correlation  
            if let Some(accessed_pattern) = self.access_patterns.get_mut(&accessed_file_key) {
                let reverse_correlation = accessed_pattern.related_files
                    .get(&other_file)
                    .copied()
                    .unwrap_or(0.0);
                
                let new_reverse = (reverse_correlation + 0.1).min(1.0);
                accessed_pattern.related_files.insert(other_file, new_reverse);
            }
        }
        
        Ok(())
    }
    
    fn calculate_directory_similarity(&self, file1: &str, file2: &str) -> f32 {
        let path1 = std::path::Path::new(file1);
        let path2 = std::path::Path::new(file2);
        
        // Same directory = high similarity
        if path1.parent() == path2.parent() {
            return 0.8;
        }
        
        // Same extension in nearby directories
        let ext1 = path1.extension().and_then(|e| e.to_str());
        let ext2 = path2.extension().and_then(|e| e.to_str());
        
        if ext1 == ext2 && ext1.is_some() {
            // Calculate directory path similarity
            let p1_components: Vec<&str> = file1.split('/').collect();
            let p2_components: Vec<&str> = file2.split('/').collect();
            
            let common_prefix = p1_components.iter()
                .zip(p2_components.iter())
                .take_while(|(a, b)| a == b)
                .count();
                
            let total_components = (p1_components.len() + p2_components.len()) / 2;
            let similarity = common_prefix as f32 / total_components as f32;
            
            return similarity * 0.6;
        }
        
        0.1 // Minimal similarity for unrelated files
    }
    
    fn calculate_recency_factor(&self, file_path: &str) -> f32 {
        if let Some(pattern) = self.access_patterns.get(file_path) {
            let time_since_access = (Utc::now() - pattern.last_access).num_seconds() as f32;
            // Decay factor: more recent = higher factor
            (-time_since_access / 7200.0).exp() // 2 hour half-life
        } else {
            0.5 // Default for unknown files
        }
    }
    
    fn calculate_time_of_day_score(&self) -> f32 {
        let current_hour = Utc::now().hour();
        
        // Typical work hours have higher scores
        match current_hour {
            9..=17 => 0.8,   // Work hours
            18..=22 => 0.6,  // Evening
            7..=8 => 0.5,    // Early morning
            23..=24 | 0..=6 => 0.2,  // Night/very early morning
            _ => 0.3,
        }
    }
    
    fn calculate_session_context_score(&self) -> f32 {
        let session_duration = (Utc::now() - self.user_session_start).num_minutes() as f32;
        
        // Active sessions have higher context scores
        match session_duration {
            0.0..=30.0 => 0.9,    // Very active
            30.0..=120.0 => 0.7,  // Active
            120.0..=480.0 => 0.4, // Moderate
            _ => 0.1,             // Long idle
        }
    }
    
    pub fn generate_feature_vector(&self, file_path: &str, current_time: DateTime<Utc>) -> Array1<f32> {
        // Generate feature vector for ML model with consistent 8 features
        let mut features = Vec::new();
        
        // Time-based features (3 features)
        features.push(current_time.hour() as f32 / 24.0);  // Hour normalized [0,1]
        features.push(current_time.minute() as f32 / 60.0);  // Minute normalized [0,1]
        features.push((current_time.weekday().num_days_from_monday() as f32) / 7.0);  // Day of week [0,1]
        
        // File path features (2 features)
        let depth = (file_path.matches('/').count() as f32 / 10.0).min(1.0);  // Path depth normalized
        features.push(depth);
        
        let has_extension = std::path::Path::new(file_path).extension().is_some() as i32 as f32;
        features.push(has_extension);  // Binary: has extension or not
        
        // Historical features (2 features)
        if let Some(pattern) = self.access_patterns.get(file_path) {
            let access_score = ((pattern.access_count as f32).ln() / 10.0).min(1.0);  // Log-normalized access count
            features.push(access_score);
            
            let interval_score = (pattern.average_interval.unwrap_or(86400.0) / 86400.0).min(1.0);  // Interval in days, capped at 1
            features.push(interval_score);
        } else {
            features.push(0.0);  // No historical access
            features.push(0.0);  // No historical intervals
        }
        
        // Directory activity (1 feature)
        let dir_activity = if let Some(parent) = std::path::Path::new(file_path).parent() {
            let dir_str = parent.to_string_lossy().to_string();
            self.directory_patterns.get(&dir_str)
                .map(|files| (files.len() as f32 / 100.0).min(1.0))  // Normalized directory activity
                .unwrap_or(0.0)
        } else {
            0.0
        };
        features.push(dir_activity);
        
        // Ensure we have exactly 8 features
        assert_eq!(features.len(), 8, "Feature vector must have exactly 8 elements");
        
        Array1::from(features)
    }
    
    #[cfg(feature = "mock")]
    pub async fn test_ml_model(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref session) = self.ml_session {
            info!("🧪 Testing ML model with sample data...");
            
            // Test with a few sample file paths
            let test_files = vec![
                "/project/src/main.rs",
                "/project/Cargo.toml", 
                "/project/src/lib.rs",
                "/project/tests/integration.rs",
            ];
            
            for file_path in test_files {
                match self.predict_with_ml_model(session, file_path) {
                    Ok(prediction) => {
                        info!("✅ Test prediction for '{}': {:.4}", file_path, prediction);
                    },
                    Err(e) => {
                        warn!("❌ Test failed for '{}': {}", file_path, e);
                        return Err(e);
                    }
                }
            }
            
            info!("🎉 ML model test completed successfully!");
            Ok(())
        } else {
            Err("ML model not loaded".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AccessType;
    
    #[tokio::test]
    async fn test_predictor_initialization() {
        let predictor = AccessPredictor::new().await;
        assert!(predictor.is_ok());
    }
    
    #[tokio::test]
    async fn test_record_and_predict() {
        let mut predictor = AccessPredictor::new().await.unwrap();
        
        let event = FileAccessEvent::new("/test/file.rs".to_string())
            .with_access_type(AccessType::Read);
            
        predictor.record_access(&event).await.unwrap();
        
        let prediction = predictor.predict_access("/test/file.rs").await.unwrap();
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }
    
    #[tokio::test] 
    async fn test_related_file_prediction() {
        let mut predictor = AccessPredictor::new().await.unwrap();
        
        // Record accesses to establish patterns
        let files = vec![
            "/project/main.rs",
            "/project/lib.rs", 
            "/project/utils.rs",
        ];
        
        for file in &files {
            let event = FileAccessEvent::new(file.to_string());
            predictor.record_access(&event).await.unwrap();
        }
        
        let related = predictor.predict_related_files("/project/main.rs").await.unwrap();
        assert!(!related.is_empty());
        
        // Should predict other Rust files in the same directory
        let rust_files: Vec<_> = related.iter()
            .filter(|(path, _)| path.ends_with(".rs"))
            .collect();
        assert!(!rust_files.is_empty());
    }
}
