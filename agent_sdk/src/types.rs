//! Type definitions for the NebulaOS Agent SDK

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Result type for SDK operations
pub type Result<T> = anyhow::Result<T>;

/// Error type for SDK operations
pub type Error = anyhow::Error;

/// Unique identifier for agents
pub type AgentId = usize;

/// Unique identifier for inference tasks
pub type TaskId = usize;

/// Unique identifier for NPU devices
pub type NPUId = usize;

/// Unique identifier for models
pub type ModelId = String;

/// Agent metadata and state
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub id: AgentId,
    pub name: String,
    pub process_id: Option<u32>,
    pub status: AgentStatus,
    pub capabilities: AgentCapabilities,
    pub resource_usage: ResourceUsage,
    pub message_queue: VecDeque<crate::message::Message>,
    pub created_at: Instant,
    pub last_activity: Instant,
}

impl AgentContext {
    pub fn new(id: AgentId, name: String) -> Self {
        let now = Instant::now();
        Self {
            id,
            name,
            process_id: None,
            status: AgentStatus::Initializing,
            capabilities: AgentCapabilities::default(),
            resource_usage: ResourceUsage::default(),
            message_queue: VecDeque::new(),
            created_at: now,
            last_activity: now,
        }
    }

    pub fn update_activity(&mut self) {
        self.last_activity = Instant::now();
    }
}

/// Current status of an agent
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentStatus {
    Initializing,
    Ready,
    Busy,
    Error(String),
    Shutdown,
}

/// What capabilities does this agent have?
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub can_inference: bool,
    pub can_training: bool,
    pub supported_models: Vec<String>,
    pub max_tensor_size: usize,
    pub preferred_npu: Option<NPUId>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub memory_mb: u64,
    pub npu_utilization: f32,
    pub inference_count: u64,
    pub total_inference_time: Duration,
    pub message_count: u64, // Added for testing
}

/// Inference task submitted to the scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTask {
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub model_id: ModelId,
    pub input_data: Vec<u8>,
    pub priority: InferencePriority,
    pub max_latency: Option<Duration>,
    pub submitted_at: u64, // Unix timestamp
}

/// Priority levels for inference tasks
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InferencePriority {
    Low,
    Normal,
    High,
    Realtime,
}

/// Result of an inference operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub task_id: TaskId,
    pub success: bool,
    pub output_data: Vec<u8>,
    pub latency: Duration,
    pub error: Option<String>,
}

/// NPU device capabilities and status
#[derive(Debug, Clone)]
pub struct NPUDevice {
    pub id: NPUId,
    pub name: String,
    pub capabilities: NPUCapabilities,
    pub status: NPUStatus,
    pub current_model: Option<ModelId>,
    pub allocated_to: Option<AgentId>,
}

/// What can this NPU do?
#[derive(Debug, Clone, Default)]
pub struct NPUCapabilities {
    pub max_memory_mb: u64,
    pub supported_precision: Vec<Precision>,
    pub max_batch_size: usize,
    pub ops_per_second: u64,
}

/// Current status of an NPU
#[derive(Debug, Clone, PartialEq)]
pub enum NPUStatus {
    Idle,
    Busy,
    Error(String),
    Maintenance,
}

/// Precision levels supported by NPU
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
    INT4,
}

/// Pool of available NPU devices
#[derive(Debug, Clone)]
pub struct NPUPool {
    pub devices: Vec<NPUDevice>,
    pub task_queue: VecDeque<InferenceTask>,
}

impl NPUPool {
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            task_queue: VecDeque::new(),
        }
    }

    pub fn add_device(&mut self, device: NPUDevice) {
        self.devices.push(device);
    }

    pub fn allocate_npu(&mut self, agent_id: AgentId) -> Option<NPUId> {
        for device in &mut self.devices {
            if device.status == NPUStatus::Idle && device.allocated_to.is_none() {
                device.allocated_to = Some(agent_id);
                device.status = NPUStatus::Busy;
                return Some(device.id);
            }
        }
        None
    }

    pub fn deallocate_npu(&mut self, npu_id: NPUId) -> bool {
        if let Some(device) = self.devices.iter_mut().find(|d| d.id == npu_id) {
            device.allocated_to = None;
            device.status = NPUStatus::Idle;
            device.current_model = None;
            true
        } else {
            false
        }
    }
}

impl Default for NPUPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_context_creation() {
        let ctx = AgentContext::new(1, "test_agent".to_string());
        assert_eq!(ctx.id, 1);
        assert_eq!(ctx.name, "test_agent");
        assert_eq!(ctx.status, AgentStatus::Initializing);
    }

    #[test]
    fn test_npu_pool_allocation() {
        let mut pool = NPUPool::new();

        let npu = NPUDevice {
            id: 0,
            name: "TestNPU".to_string(),
            capabilities: NPUCapabilities::default(),
            status: NPUStatus::Idle,
            current_model: None,
            allocated_to: None,
        };

        pool.add_device(npu);

        let allocated_id = pool.allocate_npu(1);
        assert_eq!(allocated_id, Some(0));

        // Should not allocate the same NPU again
        let second_allocation = pool.allocate_npu(2);
        assert_eq!(second_allocation, None);

        // Deallocate and try again
        assert!(pool.deallocate_npu(0));
        let third_allocation = pool.allocate_npu(2);
        assert_eq!(third_allocation, Some(0));
    }

    #[test]
    fn test_agent_capabilities() {
        let caps = AgentCapabilities {
            can_inference: true,
            can_training: false,
            supported_models: vec!["gpt".to_string(), "bert".to_string()],
            max_tensor_size: 1024 * 1024,
            preferred_npu: Some(0),
        };

        assert!(caps.can_inference);
        assert!(!caps.can_training);
        assert_eq!(caps.supported_models.len(), 2);
    }
}
