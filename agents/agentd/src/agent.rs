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
        let handle = self.handles.get_mut(&id).ok_or(Error::new(EBADF))?;
        
        match handle.handle_type {
            HandleType::Control => {
                // Parse incoming command
                let command: AgentCommand = bincode::deserialize(buf)
                    .map_err(|_| Error::new(EINVAL))?;
                
                let response = match command {
                    AgentCommand::Register { name, capabilities } => {
                        match self.register_agent(name, capabilities) {
                            Ok(agent_id) => AgentResponse::Registered { agent_id },
                            Err(msg) => AgentResponse::Error { message: msg },
                        }
                    },
                    AgentCommand::SendMessage { to, message } => {
                        match self.send_message(message) {
                            Ok(()) => AgentResponse::Success,
                            Err(msg) => AgentResponse::Error { message: msg },
                        }
                    },
                    AgentCommand::SubmitInference { task } => {
                        self.npu_pool.task_queue.push_back(task);
                        AgentResponse::Success
                    },
                    AgentCommand::GetStatus { agent_id } => {
                        let statuses = if let Some(id) = agent_id {
                            if let Some(agent) = self.agents.get(&id) {
                                vec![agent.status.clone()]
                            } else {
                                vec![]
                            }
                        } else {
                            self.agents.values().map(|a| a.status.clone()).collect()
                        };
                        AgentResponse::Status { agents: statuses }
                    },
                    AgentCommand::LoadModel { model_id, model_data } => {
                        // For now, just acknowledge model loading
                        debug!("Model {} loaded ({} bytes)", model_id, model_data.len());
                        AgentResponse::Success
                    },
                    AgentCommand::Unregister { agent_id } => {
                        if let Some(_agent) = self.agents.remove(&agent_id) {
                            self.npu_pool.deallocate_npu(agent_id);
                            info!("Unregistered agent {}", agent_id);
                            AgentResponse::Success
                        } else {
                            AgentResponse::Error { message: format!("Agent {} not found", agent_id) }
                        }
                    },
                };
                
                // Store response in handle buffer for reading
                handle.buffer = bincode::serialize(&response)
                    .map_err(|_| Error::new(EINVAL))?;
            },
            HandleType::Agent { agent_id } => {
                // Direct agent communication - add to message queue
                if let Some(agent) = self.agents.get_mut(&agent_id) {
                    // For now, just add raw data to buffer for the agent to read
                    agent.message_queue.extend(buf.iter().cloned());
                    agent.update_activity();
                } else {
                    return Err(Error::new(ENOENT));
                }
            },
            HandleType::Inference => {
                // Parse inference request
                let task: InferenceTask = bincode::deserialize(buf)
                    .map_err(|_| Error::new(EINVAL))?;
                self.npu_pool.task_queue.push_back(task);
            },
            HandleType::Status => {
                // Status endpoint is read-only
                return Err(Error::new(EPERM));
            },
        }
        
        // Process any pending inference tasks
        self.process_inference_tasks();
        
        Ok(buf.len())
    }

    fn read(&mut self, id: usize, buf: &mut [u8], _offset: u64, flags: u32, _ctx: &CallerCtx) -> Result<usize> {
        let handle = self.handles.get_mut(&id).ok_or(Error::new(EBADF))?;
        
        match handle.handle_type {
            HandleType::Control => {
                // Return response from buffer
                if !handle.buffer.is_empty() {
                    let len = std::cmp::min(buf.len(), handle.buffer.len());
                    buf[..len].copy_from_slice(&handle.buffer[..len]);
                    handle.buffer.drain(..len);
                    Ok(len)
                } else if (flags as usize) & O_NONBLOCK == O_NONBLOCK {
                    Err(Error::new(EAGAIN))
                } else {
                    Err(Error::new(EWOULDBLOCK))
                }
            },
            HandleType::Agent { agent_id } => {
                // Read messages for this agent
                if let Some(agent) = self.agents.get_mut(&agent_id) {
                    if let Some(message) = agent.message_queue.pop_front() {
                        let serialized = bincode::serialize(&message)
                            .map_err(|_| Error::new(EINVAL))?;
                        let len = std::cmp::min(buf.len(), serialized.len());
                        buf[..len].copy_from_slice(&serialized[..len]);
                        agent.update_activity();
                        Ok(len)
                    } else if (flags as usize) & O_NONBLOCK == O_NONBLOCK {
                        Err(Error::new(EAGAIN))
                    } else {
                        Err(Error::new(EWOULDBLOCK))
                    }
                } else {
                    Err(Error::new(ENOENT))
                }
            },
            HandleType::Inference => {
                // Return inference results
                if !handle.buffer.is_empty() {
                    let len = std::cmp::min(buf.len(), handle.buffer.len());
                    buf[..len].copy_from_slice(&handle.buffer[..len]);
                    handle.buffer.drain(..len);
                    Ok(len)
                } else if (flags as usize) & O_NONBLOCK == O_NONBLOCK {
                    Err(Error::new(EAGAIN))
                } else {
                    Err(Error::new(EWOULDBLOCK))
                }
            },
            HandleType::Status => {
                // Return JSON status of all agents
                let status_info = serde_json::json!({
                    "agents": self.agents.len(),
                    "npus": self.npu_pool.devices.len(),
                    "pending_tasks": self.npu_pool.task_queue.len(),
                    "agent_list": self.agents.iter().map(|(id, agent)| {
                        serde_json::json!({
                            "id": id,
                            "name": agent.name,
                            "status": format!("{:?}", agent.status),
                            "inference_count": agent.resource_usage.inference_count,
                        })
                    }).collect::<Vec<_>>()
                });
                
                let status_str = status_info.to_string();
                let status_bytes = status_str.as_bytes();
                let len = std::cmp::min(buf.len(), status_bytes.len());
                buf[..len].copy_from_slice(&status_bytes[..len]);
                Ok(len)
            },
        }
    }

    fn on_close(&mut self, id: usize) {
        if let Some(_handle) = self.handles.remove(&id) {
            debug!("Closed handle {}", id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test capabilities
    fn test_capabilities() -> AgentCapabilities {
        AgentCapabilities {
            can_inference: true,
            can_training: false,
            supported_models: vec!["test_model".to_string()],
            max_tensor_size: 1024 * 1024,
            preferred_npu: None,
        }
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_mock_new() {
        let scheme = AgentScheme::mock_new();
        assert_eq!(scheme.agents.len(), 0);
        assert_eq!(scheme.npu_pool.devices.len(), 1); // One mock NPU should be initialized
        assert_eq!(scheme.next_agent_id.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_agent_registration() {
        let mut scheme = AgentScheme::mock_new();
        
        // Test successful registration
        let agent_id = scheme.mock_register_agent("Test Agent".to_string()).unwrap();
        assert_eq!(agent_id, 1); // First agent should get ID 1
        assert!(scheme.agents.contains_key(&agent_id));
        
        let agent = scheme.agents.get(&agent_id).unwrap();
        assert_eq!(agent.name, "Test Agent");
        assert_eq!(agent.status, AgentStatus::Ready);
        assert!(agent.capabilities.can_inference);
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_register_multiple_agents() {
        let mut scheme = AgentScheme::mock_new();
        
        let agent1_id = scheme.mock_register_agent("Agent 1".to_string()).unwrap();
        let agent2_id = scheme.mock_register_agent("Agent 2".to_string()).unwrap();
        
        assert_eq!(agent1_id, 1);
        assert_eq!(agent2_id, 2);
        assert_eq!(scheme.agents.len(), 2);
        
        assert!(scheme.agents.contains_key(&agent1_id));
        assert!(scheme.agents.contains_key(&agent2_id));
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_send_receive_message() {
        let mut scheme = AgentScheme::mock_new();
        let agent_id1 = scheme.mock_register_agent("Agent 1".to_string()).unwrap();
        let agent_id2 = scheme.mock_register_agent("Agent 2".to_string()).unwrap();

        // Send message from agent 1 to agent 2
        let test_payload = b"Hello, Agent 2!";
        scheme.mock_send_message(agent_id1, agent_id2, test_payload).unwrap();
        
        // Agent 2 should have a message in its queue
        let agent2 = scheme.agents.get(&agent_id2).unwrap();
        assert_eq!(agent2.message_queue.len(), 1);
        
        // Receive the message
        let received_payload = scheme.mock_receive_message(agent_id2).unwrap();
        assert_eq!(received_payload, test_payload);
        
        // Message queue should now be empty
        let agent2 = scheme.agents.get(&agent_id2).unwrap();
        assert_eq!(agent2.message_queue.len(), 0);
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_send_to_nonexistent_agent() {
        let mut scheme = AgentScheme::mock_new();
        let agent_id1 = scheme.mock_register_agent("Agent 1".to_string()).unwrap();
        
        // Try to send message to non-existent agent
        let result = scheme.mock_send_message(agent_id1, 999, b"Hello");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_receive_message_no_messages() {
        let mut scheme = AgentScheme::mock_new();
        let agent_id = scheme.mock_register_agent("Test Agent".to_string()).unwrap();
        
        // Try to receive message when queue is empty
        let result = scheme.mock_receive_message(agent_id);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "No messages available");
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_receive_from_nonexistent_agent() {
        let mut scheme = AgentScheme::mock_new();
        
        // Try to receive message from non-existent agent
        let result = scheme.receive_message(999);
        assert!(result.is_none());
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_process_inference_tasks_empty_queue() {
        let mut scheme = AgentScheme::mock_new();
        let _agent_id = scheme.mock_register_agent("Test Agent".to_string()).unwrap();
        
        // Process inference tasks when no tasks are queued
        scheme.process_inference_tasks();
        
        // Should not crash or cause issues
        assert_eq!(scheme.npu_pool.task_queue.len(), 0);
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_multiple_message_queue() {
        let mut scheme = AgentScheme::mock_new();
        let agent_id1 = scheme.mock_register_agent("Agent 1".to_string()).unwrap();
        let agent_id2 = scheme.mock_register_agent("Agent 2".to_string()).unwrap();

        // Send multiple messages
        scheme.mock_send_message(agent_id1, agent_id2, b"Message 1").unwrap();
        scheme.mock_send_message(agent_id1, agent_id2, b"Message 2").unwrap();
        scheme.mock_send_message(agent_id1, agent_id2, b"Message 3").unwrap();
        
        // Agent 2 should have 3 messages
        let agent2 = scheme.agents.get(&agent_id2).unwrap();
        assert_eq!(agent2.message_queue.len(), 3);
        
        // Receive messages in FIFO order
        let msg1 = scheme.mock_receive_message(agent_id2).unwrap();
        let msg2 = scheme.mock_receive_message(agent_id2).unwrap();
        let msg3 = scheme.mock_receive_message(agent_id2).unwrap();
        
        assert_eq!(msg1, b"Message 1");
        assert_eq!(msg2, b"Message 2");
        assert_eq!(msg3, b"Message 3");
        
        // Queue should be empty now
        let result = scheme.mock_receive_message(agent_id2);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_npu_allocation() {
        let mut scheme = AgentScheme::mock_new();
        
        // Register an agent that can do inference
        let agent_id = scheme.register_agent(
            "Inference Agent".to_string(),
            test_capabilities()
        ).unwrap();
        
        // Check that NPU was allocated
        let npu = scheme.npu_pool.devices.get(&0).unwrap();
        assert_eq!(npu.allocated_to, Some(agent_id));
        // NPU should still be Idle status but allocated to the agent
        assert_eq!(npu.status, NPUStatus::Idle);
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_agent_capabilities() {
        let mut scheme = AgentScheme::mock_new();
        
        let custom_capabilities = AgentCapabilities {
            can_inference: false,
            can_training: true,
            supported_models: vec!["custom_model".to_string(), "another_model".to_string()],
            max_tensor_size: 2048 * 1024,
            preferred_npu: Some(0),
        };
        
        let agent_id = scheme.register_agent(
            "Custom Agent".to_string(),
            custom_capabilities.clone()
        ).unwrap();
        
        let agent = scheme.agents.get(&agent_id).unwrap();
        assert_eq!(agent.capabilities.can_inference, false);
        assert_eq!(agent.capabilities.can_training, true);
        assert_eq!(agent.capabilities.supported_models.len(), 2);
        assert_eq!(agent.capabilities.max_tensor_size, 2048 * 1024);
        assert_eq!(agent.capabilities.preferred_npu, Some(0));
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_message_type_preservation() {
        let mut scheme = AgentScheme::mock_new();
        let agent_id1 = scheme.mock_register_agent("Agent 1".to_string()).unwrap();
        let agent_id2 = scheme.mock_register_agent("Agent 2".to_string()).unwrap();

        // Send message using direct method to test message type
        let message = AgentMessage {
            from: agent_id1,
            to: agent_id2,
            message_type: MessageType::InferenceRequest,
            payload: b"test payload".to_vec(),
            timestamp: 123456789,
        };
        
        scheme.send_message(message.clone()).unwrap();
        
        // Receive and verify message type is preserved
        let received_message = scheme.receive_message(agent_id2).unwrap();
        assert_eq!(received_message.message_type, MessageType::InferenceRequest);
        assert_eq!(received_message.from, agent_id1);
        assert_eq!(received_message.to, agent_id2);
        assert_eq!(received_message.payload, b"test payload");
    }
}
