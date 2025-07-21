//! Runtime module for the NebulaOS Agent SDK
//!
//! Manages the execution of multiple agents, providing lifecycle
//! management, resource allocation, and coordination between agents.

use crate::ml::MLHandler;
use crate::prelude::*;
use crate::types::AgentContext;
use anyhow::Result;
use std::sync::{Arc, Mutex};

pub struct NebulaRuntime {
    agents: Vec<Agent>,
    ml_handler: Arc<tokio::sync::Mutex<MLHandler>>,
}

impl NebulaRuntime {
    /// Create a new Nebula runtime
    pub async fn new() -> Result<Self> {
        let ml_handler = Arc::new(tokio::sync::Mutex::new(MLHandler::new()?));
        ml_handler.lock().await.initialize().await?;

        Ok(Self {
            agents: Vec::new(),
            ml_handler,
        })
    }

    /// Register a new agent with the runtime
    pub fn register_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
    }

    /// Start all registered agents
    pub async fn start_agents(&self) -> Result<()> {
        for agent in &self.agents {
            agent.initialize().await?;
            agent.start(TestHandler {}).await?; // Use a test handler or a real one
        }
        Ok(())
    }

    /// Shutdown all agents gracefully
    pub async fn shutdown_agents(&self) -> Result<()> {
        for agent in &self.agents {
            agent.shutdown().await?;
        }
        Ok(())
    }

    /// Get the ML handler for inference operations
    pub fn get_ml_handler(&self) -> Arc<tokio::sync::Mutex<MLHandler>> {
        self.ml_handler.clone()
    }
}

/// Temporary test handler for agent execution
pub struct TestHandler;

#[async_trait::async_trait]
impl AgentHandler for TestHandler {
    async fn handle_message(
        &self,
        context: Arc<Mutex<AgentContext>>,
        _message: Message,
    ) -> Result<()> {
        let mut ctx = context.lock().unwrap();
        ctx.update_activity();
        Ok(())
    }
}
