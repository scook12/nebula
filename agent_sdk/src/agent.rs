//! Agent module for the NebulaOS Agent SDK
//!
//! Defines the core agent structure, lifecycle management, and
//! capabilities description for agents running on NebulaOS.

use crate::message::{Message, MessageHandler};
use crate::types::*;
use anyhow::Result;
use log::{debug, error, info};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Configuration options for an agent
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub name: String,
    pub capabilities: AgentCapabilities,
    pub max_message_queue_size: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "unnamed_agent".to_string(),
            capabilities: AgentCapabilities::default(),
            max_message_queue_size: 1000,
        }
    }
}

/// The main interface for an agent
#[derive(Clone)]
pub struct Agent {
    pub id: AgentId,
    pub config: AgentConfig,
    context: Arc<Mutex<AgentContext>>,
    message_tx: mpsc::UnboundedSender<Message>,
    message_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<Message>>>>,
}

impl Agent {
    /// Create a new agent with the given configuration
    pub fn new(id: AgentId, config: AgentConfig) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        Self {
            id,
            config: config.clone(),
            context: Arc::new(Mutex::new(AgentContext::new(id, config.name.clone()))),
            message_tx,
            message_rx: Arc::new(Mutex::new(Some(message_rx))),
        }
    }

    /// Get a reference to the agent's context (for reading/writing state)
    pub fn context(&self) -> Arc<Mutex<AgentContext>> {
        self.context.clone()
    }

    /// Send a message to this agent
    pub fn send_message(&self, message: Message) -> Result<()> {
        self.message_tx.send(message)?;
        debug!("Message sent to agent {}", self.id);
        Ok(())
    }

    /// Initialize the agent and prepare for execution
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing agent: {} (ID: {})", self.config.name, self.id);
        {
            let mut ctx = self.context.lock().unwrap();
            ctx.status = AgentStatus::Ready;
            ctx.capabilities = self.config.capabilities.clone();
        }
        Ok(())
    }

    /// Start the agent's main execution loop
    pub async fn start<H>(&self, handler: H) -> Result<()>
    where
        H: AgentHandler + Send + 'static,
    {
        let context = self.context.clone();
        let message_rx = {
            let mut rx_guard = self.message_rx.lock().unwrap();
            rx_guard
                .take()
                .ok_or_else(|| anyhow::anyhow!("Agent already started"))?
        };

        // Spawn the message processing task
        let handler_context = context.clone();
        tokio::spawn(async move {
            Self::message_loop(handler_context, message_rx, handler).await;
        });

        info!("Agent {} started successfully", self.config.name);
        Ok(())
    }

    /// Internal message processing loop
    async fn message_loop<H>(
        context: Arc<Mutex<AgentContext>>,
        mut message_rx: mpsc::UnboundedReceiver<Message>,
        handler: H,
    ) where
        H: AgentHandler,
    {
        while let Some(message) = message_rx.recv().await {
            debug!("Processing message: {:?}", message.message_type);

            // Update agent activity
            {
                let mut ctx = context.lock().unwrap();
                ctx.update_activity();
                ctx.status = AgentStatus::Busy;
            }

            // Handle the message
            if let Err(e) = handler.handle_message(context.clone(), message).await {
                error!("Error handling message: {}", e);
                let mut ctx = context.lock().unwrap();
                ctx.status = AgentStatus::Error(e.to_string());
            } else {
                let mut ctx = context.lock().unwrap();
                ctx.status = AgentStatus::Ready;
            }
        }

        info!("Agent message loop ended");
        let mut ctx = context.lock().unwrap();
        ctx.status = AgentStatus::Shutdown;
    }

    /// Get the current status of the agent
    pub fn status(&self) -> AgentStatus {
        let ctx = self.context.lock().unwrap();
        ctx.status.clone()
    }

    /// Shutdown the agent gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down agent: {}", self.config.name);
        {
            let mut ctx = self.context.lock().unwrap();
            ctx.status = AgentStatus::Shutdown;
        }
        // Close the message channel to stop the message loop
        drop(self.message_tx.clone());
        Ok(())
    }
}

/// Builder pattern for creating agents with fluent API
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    /// Create a new agent builder with the given name
    pub fn new(name: &str) -> Self {
        Self {
            config: AgentConfig {
                name: name.to_string(),
                ..AgentConfig::default()
            },
        }
    }

    /// Set the agent's capabilities
    pub fn with_capabilities(mut self, capabilities: AgentCapabilities) -> Self {
        self.config.capabilities = capabilities;
        self
    }

    /// Set the maximum message queue size
    pub fn with_message_queue_size(mut self, size: usize) -> Self {
        self.config.max_message_queue_size = size;
        self
    }

    /// Enable AI inference capability
    pub fn with_ai_inference(mut self) -> Self {
        self.config.capabilities.can_inference = true;
        self
    }

    /// Enable training capability  
    pub fn with_training(mut self) -> Self {
        self.config.capabilities.can_training = true;
        self
    }

    /// Add supported model types
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        self.config.capabilities.supported_models = models;
        self
    }

    /// Build the agent with the specified configuration
    pub fn build(self, id: AgentId) -> Agent {
        Agent::new(id, self.config)
    }
}

/// Trait defining the behavior of an agent
#[async_trait::async_trait]
pub trait AgentHandler {
    /// Handle an incoming message
    async fn handle_message(
        &self,
        context: Arc<Mutex<AgentContext>>,
        message: Message,
    ) -> Result<()>;

    /// Optional: Handle agent initialization
    async fn on_initialize(&self, _context: Arc<Mutex<AgentContext>>) -> Result<()> {
        Ok(())
    }

    /// Optional: Handle agent shutdown
    async fn on_shutdown(&self, _context: Arc<Mutex<AgentContext>>) -> Result<()> {
        Ok(())
    }
}

/// Runtime for managing multiple agents
pub struct AgentRuntime {
    agents: Vec<Agent>,
    next_agent_id: AgentId,
}

impl AgentRuntime {
    /// Create a new agent runtime
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            next_agent_id: 1,
        }
    }

    /// Register a new agent with the runtime
    pub fn register_agent(&mut self, config: AgentConfig) -> Agent {
        let agent = Agent::new(self.next_agent_id, config);
        self.next_agent_id += 1;
        self.agents.push(agent.clone());
        agent
    }

    /// Start all registered agents
    pub async fn start_all<H>(&self, handler: H) -> Result<()>
    where
        H: AgentHandler + Clone + Send + 'static,
    {
        for agent in &self.agents {
            agent.initialize().await?;
            agent.start(handler.clone()).await?;
        }
        info!("Started {} agents", self.agents.len());
        Ok(())
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: AgentId) -> Option<&Agent> {
        self.agents.iter().find(|agent| agent.id == id)
    }

    /// Get all agents
    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    /// Shutdown all agents
    pub async fn shutdown_all(&self) -> Result<()> {
        for agent in &self.agents {
            agent.shutdown().await?;
        }
        info!("All agents shut down");
        Ok(())
    }
}

impl Default for AgentRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::MessageType;

    struct TestHandler;

    #[async_trait::async_trait]
    impl AgentHandler for TestHandler {
        async fn handle_message(
            &self,
            context: Arc<Mutex<AgentContext>>,
            message: Message,
        ) -> Result<()> {
            let mut ctx = context.lock().unwrap();
            ctx.resource_usage.message_count += 1;
            debug!("Test handler processed message: {:?}", message.message_type);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_agent_creation() {
        let agent = AgentBuilder::new("test_agent").with_ai_inference().build(1);

        assert_eq!(agent.id, 1);
        assert_eq!(agent.config.name, "test_agent");

        let ctx = agent.context.lock().unwrap();
        assert_eq!(ctx.id, 1);
        assert_eq!(ctx.name, "test_agent");
    }

    #[tokio::test]
    async fn test_agent_message_handling() {
        let agent = AgentBuilder::new("test_agent").build(1);

        agent.initialize().await.unwrap();
        agent.start(TestHandler).await.unwrap();

        let message = Message {
            from: 0,
            to: 1,
            message_type: MessageType::Data,
            payload: b"test".to_vec(),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        agent.send_message(message).unwrap();

        // Wait a bit for message processing
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let ctx = agent.context.lock().unwrap();
        assert_eq!(ctx.resource_usage.message_count, 1);
    }
}
