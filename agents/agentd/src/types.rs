//! Type definitions for the NebulaOS Agent System

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

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
    pub message_queue: VecDeque<AgentMessage>,
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
}

/// Messages passed between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: AgentId,
    pub to: AgentId,
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: u64, // Unix timestamp
}

/// Types of messages agents can send
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Basic text/data message
    Data,
    /// Request for inference
    InferenceRequest,
    /// Response containing inference results
    InferenceResponse,
    /// System control message
    Control,
    /// Error notification
    Error,
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

/// Commands that can be sent to the agent scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentCommand {
    /// Register a new agent
    Register {
        name: String,
        capabilities: AgentCapabilities,
    },
    /// Unregister an agent
    Unregister {
        agent_id: AgentId,
    },
    /// Send a message to another agent
    SendMessage {
        to: AgentId,
        message: AgentMessage,
    },
    /// Submit an inference task
    SubmitInference {
        task: InferenceTask,
    },
    /// Query agent status
    GetStatus {
        agent_id: Option<AgentId>, // None = get all agents
    },
    /// Load a model into the NPU pool
    LoadModel {
        model_id: ModelId,
        model_data: Vec<u8>,
    },
}

/// Responses from the agent scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentResponse {
    /// Successful operation
    Success,
    /// Agent registered with this ID
    Registered { agent_id: AgentId },
    /// Message received
    Message { message: AgentMessage },
    /// Inference task result
    InferenceResult { result: InferenceResult },
    /// Status information
    Status { agents: Vec<AgentStatus> },
    /// Error occurred
    Error { message: String },
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

/// Current NPU status
#[derive(Debug, Clone, PartialEq)]
pub enum NPUStatus {
    Idle,
    Loading,
    Running,
    Error(String),
}

/// Supported number precisions
#[derive(Debug, Clone, PartialEq)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
    INT4,
}

/// Pool of available NPU devices
#[derive(Debug, Default)]
pub struct NPUPool {
    pub devices: HashMap<NPUId, NPUDevice>,
    pub allocation_map: HashMap<AgentId, NPUId>,
    pub task_queue: VecDeque<InferenceTask>,
}

impl NPUPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_device(&mut self, device: NPUDevice) {
        self.devices.insert(device.id, device);
    }

    pub fn allocate_npu(&mut self, agent_id: AgentId) -> Option<NPUId> {
        // Find first available NPU
        for (npu_id, device) in &mut self.devices {
            if device.status == NPUStatus::Idle && device.allocated_to.is_none() {
                device.allocated_to = Some(agent_id);
                self.allocation_map.insert(agent_id, *npu_id);
                return Some(*npu_id);
            }
        }
        None
    }

    pub fn deallocate_npu(&mut self, agent_id: AgentId) {
        if let Some(npu_id) = self.allocation_map.remove(&agent_id) {
            if let Some(device) = self.devices.get_mut(&npu_id) {
                device.allocated_to = None;
                device.status = NPUStatus::Idle;
            }
        }
    }
}
