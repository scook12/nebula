use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAccessEvent {
    pub path: String,
    pub timestamp: DateTime<Utc>,
    pub access_type: AccessType,
    pub file_size: Option<u64>,
    pub file_extension: Option<String>,
    pub directory_depth: u32,
}

impl FileAccessEvent {
    pub fn new(path: String) -> Self {
        let file_extension = std::path::Path::new(&path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());
            
        let directory_depth = path.matches('/').count() as u32;
        
        Self {
            path,
            timestamp: Utc::now(),
            access_type: AccessType::Read,
            file_size: None,
            file_extension,
            directory_depth,
        }
    }
    
    pub fn with_size(mut self, size: u64) -> Self {
        self.file_size = Some(size);
        self
    }
    
    pub fn with_access_type(mut self, access_type: AccessType) -> Self {
        self.access_type = access_type;
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AccessType {
    Read,
    Write,
    Execute,
    Create,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub file_path: String,
    pub probability: f32,
    pub confidence: f32,
    pub prediction_type: PredictionType,
    pub features: PredictionFeatures,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PredictionType {
    SequentialAccess,   // Files accessed after current file
    RelatedAccess,      // Files in same directory or project
    TemporalPattern,    // Files accessed at similar times
    UserPattern,        // Files matching user's typical workflow
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionFeatures {
    pub time_since_last_access: Option<i64>,  // seconds
    pub access_frequency: f32,                // accesses per day
    pub directory_similarity: f32,            // 0.0 to 1.0
    pub extension_match: bool,
    pub file_size_category: FileSizeCategory,
    pub time_of_day_score: f32,              // how typical is this time for user
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FileSizeCategory {
    Small,    // < 1MB
    Medium,   // 1MB - 100MB
    Large,    // 100MB - 1GB
    XLarge,   // > 1GB
}

impl FileSizeCategory {
    pub fn from_size(size: u64) -> Self {
        match size {
            0..=1_048_576 => Self::Small,
            1_048_577..=104_857_600 => Self::Medium,
            104_857_601..=1_073_741_824 => Self::Large,
            _ => Self::XLarge,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAccessPattern {
    pub path: String,
    pub access_count: u32,
    pub last_access: DateTime<Utc>,
    pub access_times: Vec<DateTime<Utc>>,
    pub average_interval: Option<f32>,  // average seconds between accesses
    pub related_files: HashMap<String, f32>,  // path -> correlation score
}

impl FileAccessPattern {
    pub fn new(path: String) -> Self {
        Self {
            path,
            access_count: 0,
            last_access: Utc::now(),
            access_times: Vec::new(),
            average_interval: None,
            related_files: HashMap::new(),
        }
    }
    
    pub fn record_access(&mut self, timestamp: DateTime<Utc>) {
        self.access_count += 1;
        self.last_access = timestamp;
        self.access_times.push(timestamp);
        
        // Keep only recent accesses (last 100)
        if self.access_times.len() > 100 {
            self.access_times.drain(0..self.access_times.len() - 100);
        }
        
        // Update average interval
        if self.access_times.len() > 1 {
            let intervals: Vec<f32> = self.access_times
                .windows(2)
                .map(|window| (window[1] - window[0]).num_seconds() as f32)
                .collect();
            
            self.average_interval = Some(intervals.iter().sum::<f32>() / intervals.len() as f32);
        }
    }
    
    pub fn calculate_access_probability(&self, current_time: DateTime<Utc>) -> f32 {
        if self.access_count == 0 {
            return 0.0;
        }
        
        // Factor 1: Recency (more recent = higher probability)
        let time_since_last = (current_time - self.last_access).num_seconds() as f32;
        let recency_score = (-time_since_last / 3600.0).exp(); // decay over hours
        
        // Factor 2: Frequency (more frequent = higher probability)
        let frequency_score = (self.access_count as f32).ln() / 10.0;
        
        // Factor 3: Pattern regularity
        let pattern_score = match self.average_interval {
            Some(interval) if interval > 0.0 => {
                let expected_next = self.last_access + chrono::Duration::seconds(interval as i64);
                let time_diff = (current_time - expected_next).num_seconds().abs() as f32;
                (-time_diff / interval).exp()
            }
            _ => 0.0,
        };
        
        // Combine factors
        let combined_score = recency_score * 0.4 + frequency_score * 0.3 + pattern_score * 0.3;
        combined_score.min(1.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_file_access_event_creation() {
        let event = FileAccessEvent::new("/home/user/test.rs".to_string());
        assert_eq!(event.path, "/home/user/test.rs");
        assert_eq!(event.file_extension, Some("rs".to_string()));
        assert_eq!(event.directory_depth, 3);
        assert_eq!(event.access_type, AccessType::Read);
    }
    
    #[test]
    fn test_file_size_category() {
        assert!(matches!(FileSizeCategory::from_size(500_000), FileSizeCategory::Small));
        assert!(matches!(FileSizeCategory::from_size(50_000_000), FileSizeCategory::Medium));
        assert!(matches!(FileSizeCategory::from_size(500_000_000), FileSizeCategory::Large));
        assert!(matches!(FileSizeCategory::from_size(2_000_000_000), FileSizeCategory::XLarge));
    }
    
    #[test]
    fn test_access_pattern_probability() {
        let mut pattern = FileAccessPattern::new("/test/file.txt".to_string());
        
        let base_time = Utc::now();
        pattern.record_access(base_time);
        
        // Should have low probability for immediate re-access
        let prob1 = pattern.calculate_access_probability(base_time + chrono::Duration::seconds(10));
        assert!(prob1 < 0.5);
        
        // Add more accesses to establish pattern
        for i in 1..5 {
            pattern.record_access(base_time + chrono::Duration::hours(i));
        }
        
        // Should have higher probability now
        let prob2 = pattern.calculate_access_probability(base_time + chrono::Duration::hours(5));
        assert!(prob2 > prob1);
    }
}
