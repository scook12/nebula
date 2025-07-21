//! Core Agent Scheme implementation for NebulaOS

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use log::{info, warn, debug};

#[cfg(feature = "redox")]
use {
    redox_scheme::{scheme::SchemeSync, CallerCtx, OpenResult, Response, SignalBehavior, Socket},
    redox_syscall::{error::*, flag::*, schemev2::NewFdFlags, Error},
};

use crate::types::*;

/// Handle for open agent scheme resources
#[cfg(feature = "redox")]
#[derive(Debug, Default)]
pub struct AgentHandle {
    pub handle_type: HandleType,
    pub buffer: Vec<u8>,
    pub agent_id: Option<AgentId>,
    pub flags: usize,
}

/// Types of handles in the agent scheme
#[cfg(feature = "redox")]
#[derive(Debug, Clone)]
pub enum HandleType {
    Control,
    Agent { agent_id: AgentId },
    Inference,
    Status,
}

#[cfg(feature = "redox")]
impl Default for HandleType {
    fn default() -> Self {
        HandleType::Control
    }
}

/// Agent scheme for Redox
#[cfg(feature = "redox")]
pub struct AgentScheme<'socket> {
    agents: HashMap<AgentId, AgentContext>,
    handles: HashMap<usize, AgentHandle>,
    npu_pool: NPUPool,
    next_agent_id: AtomicUsize,
    next_handle_id: AtomicUsize,
    next_task_id: AtomicUsize,
    socket: &'socket Socket,
}

/// Agent scheme for mock testing
#[cfg(feature = "mock")]
pub struct AgentScheme {
    agents: HashMap<AgentId, AgentContext>,
    npu_pool: NPUPool,
    next_agent_id: AtomicUsize,
    next_task_id: AtomicUsize,
}

// Common trait for both implementations
pub trait AgentSchemeImpl {
    fn register_agent(&mut self, name: String, capabilities: AgentCapabilities) -> Result<AgentId, String>;
    fn send_message(&mut self, message: AgentMessage) -> Result<(), String>;
    fn receive_message(&mut self, agent_id: AgentId) -> Option<AgentMessage>;
    fn process_inference_tasks(&mut self);
}

// Redox implementation
#[cfg(feature = "redox")]
impl<'socket> AgentScheme<'socket> {
    pub fn new(socket: &'socket Socket) -> Self {
        let mut scheme = Self {
            agents: HashMap::new(),
            handles: HashMap::new(),
            npu_pool: NPUPool::new(),
            next_agent_id: AtomicUsize::new(1),
            next_handle_id: AtomicUsize::new(1),
            next_task_id: AtomicUsize::new(1),
            socket,
        };
        scheme.init_mock_npus();
        info!("AgentScheme initialized with {} NPU devices", scheme.npu_pool.devices.len());
        scheme
    }

    fn init_mock_npus(&mut self) {
        let npu1 = NPUDevice {
            id: 0,
            name: "MockNPU-0".to_string(),
            capabilities: NPUCapabilities {
                max_memory_mb: 8192,
                supported_precision: vec![Precision::FP32, Precision::FP16],
                max_batch_size: 32,
                ops_per_second: 1000000,
            },
            status: NPUStatus::Idle,
            current_model: None,
            allocated_to: None,
        };
        self.npu_pool.add_device(npu1);
    }
}

// Mock implementation  
#[cfg(feature = "mock")]
impl AgentScheme {
    pub fn mock_new() -> Self {
        let mut scheme = Self {
            agents: HashMap::new(),
            npu_pool: NPUPool::new(),
            next_agent_id: AtomicUsize::new(1),
            next_task_id: AtomicUsize::new(1),
        };
        scheme.init_mock_npus();
        scheme
    }

    fn init_mock_npus(&mut self) {
        let npu1 = NPUDevice {
            id: 0,
            name: "MockNPU-0".to_string(),
            capabilities: NPUCapabilities {
                max_memory_mb: 8192,
                supported_precision: vec![Precision::FP32, Precision::FP16],
                max_batch_size: 32,
                ops_per_second: 1000000,
            },
            status: NPUStatus::Idle,
            current_model: None,
            allocated_to: None,
        };
        self.npu_pool.add_device(npu1);
    }

    pub fn mock_register_agent(&mut self, name: String) -> Result<AgentId, String> {
        let capabilities = AgentCapabilities {
            can_inference: true,
            can_training: false,
            supported_models: vec!["test_model".to_string()],
            max_tensor_size: 1024 * 1024,
            preferred_npu: None,
        };
        self.register_agent(name, capabilities)
    }

    pub fn mock_send_message(&mut self, from: AgentId, to: AgentId, payload: &[u8]) -> Result<(), String> {
        let message = AgentMessage {
            from,
            to,
            message_type: MessageType::Data,
            payload: payload.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.send_message(message)
    }

    pub fn mock_receive_message(&mut self, agent_id: AgentId) -> Result<Vec<u8>, String> {
        if let Some(message) = self.receive_message(agent_id) {
            Ok(message.payload)
        } else {
            Err("No messages available".to_string())
        }
    }
}

// Common implementation for both
macro_rules! impl_agent_scheme_common {
    ($type:ty) => {
        impl $type {
            pub fn register_agent(&mut self, name: String, capabilities: AgentCapabilities) -> Result<AgentId, String> {
                let agent_id = self.next_agent_id.fetch_add(1, Ordering::SeqCst);
                let mut agent = AgentContext::new(agent_id, name.clone());
                agent.capabilities = capabilities;
                agent.status = AgentStatus::Ready;

                if agent.capabilities.can_inference {
                    if let Some(npu_id) = self.npu_pool.allocate_npu(agent_id) {
                        info!("Allocated NPU {} to agent {} ({})", npu_id, agent_id, name);
                    } else {
                        warn!("No available NPU for agent {} ({})", agent_id, name);
                    }
                }

                self.agents.insert(agent_id, agent);
                info!("Registered agent {} with ID {}", name, agent_id);
                Ok(agent_id)
            }

            pub fn send_message(&mut self, message: AgentMessage) -> Result<(), String> {
                let target_agent = self.agents.get_mut(&message.to)
                    .ok_or_else(|| format!("Target agent {} not found", message.to))?;

                target_agent.message_queue.push_back(message.clone());
                target_agent.update_activity();

                debug!("Sent message from agent {} to agent {}", message.from, message.to);
                Ok(())
            }

            pub fn receive_message(&mut self, agent_id: AgentId) -> Option<AgentMessage> {
                if let Some(agent) = self.agents.get_mut(&agent_id) {
                    agent.update_activity();
                    agent.message_queue.pop_front()
                } else {
                    None
                }
            }

            pub fn process_inference_tasks(&mut self) {
                if let Some(task) = self.npu_pool.task_queue.pop_front() {
                    if let Some(agent) = self.agents.get_mut(&task.agent_id) {
                        let result = InferenceResult {
                            task_id: task.task_id,
                            success: true,
                            output_data: b"mock_inference_result".to_vec(),
                            latency: std::time::Duration::from_millis(50),
                            error: None,
                        };

                        let message = AgentMessage {
                            from: 0,
                            to: task.agent_id,
                            message_type: MessageType::InferenceResponse,
                            payload: bincode::serialize(&result).unwrap_or_default(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        };

                        agent.message_queue.push_back(message);
                        agent.status = AgentStatus::Ready;
                        agent.resource_usage.inference_count += 1;

                        debug!("Completed inference task {} for agent {}", task.task_id, task.agent_id);
                    }
                }
            }
        }
    };
}

#[cfg(feature = "redox")]
impl_agent_scheme_common!(AgentScheme<'_>);

#[cfg(feature = "mock")]
impl_agent_scheme_common!(AgentScheme);

// Redox SchemeSync implementation
#[cfg(feature = "redox")]
impl<'socket> SchemeSync for AgentScheme<'socket> {
    fn open(&mut self, path: &str, flags: usize, _ctx: &CallerCtx) -> Result<OpenResult> {
        let handle_id = self.next_handle_id.fetch_add(1, Ordering::SeqCst);
        let mut handle = AgentHandle::default();
        handle.flags = flags;

        match path {
            "register" => {
                handle.handle_type = HandleType::Control;
                info!("Opened agent registration handle {}", handle_id);
            },
            "status" => {
                handle.handle_type = HandleType::Status;
                info!("Opened status handle {}", handle_id);
            },
            _ => return Err(Error::new(EINVAL))
        }

        self.handles.insert(handle_id, handle);

        Ok(OpenResult::ThisScheme {
            number: handle_id,
            flags: NewFdFlags::empty(),
        })
    }

    fn write(&mut self, id: usize, buf: &[u8], _offset: u64, _flags: u32, _ctx: &CallerCtx) -> Result<usize> {
        let _handle = self.handles.get_mut(&id).ok_or(Error::new(EBADF))?;
        // Basic implementation - just return success for now
        self.process_inference_tasks();
        Ok(buf.len())
    }

    fn read(&mut self, id: usize, buf: &mut [u8], _offset: u64, _flags: u32, _ctx: &CallerCtx) -> Result<usize> {
        let _handle = self.handles.get_mut(&id).ok_or(Error::new(EBADF))?;
        // Basic implementation - return empty for now
        Ok(0)
    }

    fn on_close(&mut self, id: usize) {
        if let Some(_handle) = self.handles.remove(&id) {
            debug!("Closed handle {}", id);
        }
    }
}
