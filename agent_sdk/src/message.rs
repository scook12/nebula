//! Message module for the NebulaOS Agent SDK
//!
//! Defines message structures, types, and message handling traits
//! for agents to communicate with each other.

use crate::types::*;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Represents a message exchanged between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub from: AgentId,
    pub to: AgentId,
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: u64, // Unix timestamp
}

/// Types of messages agents can send
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Trait defining message handling behavior for agents
#[async_trait]
pub trait MessageHandler {
    /// Handle reception of a message
    async fn handle_message(
        &self,
        context: Arc<Mutex<AgentContext>>,
        message: Message,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    struct TestHandler;

    #[async_trait]
    impl MessageHandler for TestHandler {
        async fn handle_message(
            &self,
            context: Arc<Mutex<AgentContext>>,
            _message: Message,
        ) -> Result<()> {
            let mut ctx = context.lock().unwrap();
            ctx.last_activity = Instant::now();
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_message_handling() {
        let context = Arc::new(Mutex::new(AgentContext::new(1, "test_agent".to_string())));
        let handler = TestHandler;

        let message = Message {
            from: 0,
            to: 1,
            message_type: MessageType::Data,
            payload: vec![1, 2, 3],
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        assert!(handler
            .handle_message(context.clone(), message)
            .await
            .is_ok());
        assert_eq!(context.lock().unwrap().message_queue.len(), 0);
    }
}
