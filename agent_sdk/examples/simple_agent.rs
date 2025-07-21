//! Simple Agent Example
//!
//! This example demonstrates how to create a basic agent using the NebulaOS Agent SDK.

use nebula_agent_sdk::prelude::*;
use std::sync::{Arc, Mutex};

/// A simple echo agent that responds to messages
#[derive(Clone)]
struct EchoAgent;

#[async_trait::async_trait]
impl AgentHandler for EchoAgent {
    async fn handle_message(
        &self,
        context: Arc<Mutex<AgentContext>>,
        message: Message,
    ) -> AnyResult<()> {
        info!("EchoAgent received message: {:?}", message.message_type);

        // Update context
        {
            let mut ctx = context.lock().unwrap();
            ctx.update_activity();
            ctx.resource_usage.message_count += 1;
        }

        // Echo the message back (in a real scenario, you'd send to another agent)
        match message.message_type {
            MessageType::Data => {
                let payload_str = String::from_utf8_lossy(&message.payload);
                info!("📨 Echo: {}", payload_str);
            }
            MessageType::InferenceRequest => {
                info!("🧠 Received inference request - processing...");
                // Here you could integrate with ML models
            }
            _ => {
                info!("📋 Received message of type: {:?}", message.message_type);
            }
        }

        Ok(())
    }

    async fn on_initialize(&self, context: Arc<Mutex<AgentContext>>) -> AnyResult<()> {
        let ctx = context.lock().unwrap();
        info!("🚀 EchoAgent '{}' initialized!", ctx.name);
        Ok(())
    }

    async fn on_shutdown(&self, context: Arc<Mutex<AgentContext>>) -> AnyResult<()> {
        let ctx = context.lock().unwrap();
        info!("🛑 EchoAgent '{}' shutting down", ctx.name);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    // Initialize the SDK
    let mut runtime = nebula_agent_sdk::init_with_logger("info").await?;

    println!("🌟 NebulaOS Agent SDK Example");
    println!("==============================");

    // Create agent configuration
    let agent_config = AgentBuilder::new("echo_agent")
        .with_ai_inference()
        .with_models(vec!["echo_model".to_string()])
        .build(1);

    // Register the agent with runtime
    runtime.register_agent(agent_config.clone());

    println!("✅ Agent registered: {}", agent_config.config.name);

    // Initialize and start the agent
    agent_config.initialize().await?;
    agent_config.start(EchoAgent).await?;

    println!("🚀 Agent started successfully");

    // Send some test messages
    let test_messages = vec![
        Message {
            from: 0,
            to: 1,
            message_type: MessageType::Data,
            payload: b"Hello from NebulaOS!".to_vec(),
            timestamp: chrono::Utc::now().timestamp() as u64,
        },
        Message {
            from: 0,
            to: 1,
            message_type: MessageType::InferenceRequest,
            payload: b"inference_data".to_vec(),
            timestamp: chrono::Utc::now().timestamp() as u64,
        },
    ];

    for message in test_messages {
        println!("📤 Sending message: {:?}", message.message_type);
        agent_config.send_message(message)?;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Wait a bit for messages to process
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Check agent status
    println!("📊 Agent Status: {:?}", agent_config.status());

    // Show resource usage
    {
        let context = agent_config.context();
        let ctx = context.lock().unwrap();
        println!(
            "📈 Messages processed: {}",
            ctx.resource_usage.message_count
        );
        println!(
            "⏰ Agent uptime: {:?}",
            ctx.last_activity.duration_since(ctx.created_at)
        );
    }

    // Test ML handler
    let ml_handler = runtime.get_ml_handler();
    let ml = ml_handler.lock().await;
    println!("🧠 ML models loaded: {:?}", ml.get_loaded_models());

    // Graceful shutdown
    println!("🔄 Shutting down...");
    agent_config.shutdown().await?;
    runtime.shutdown_agents().await?;

    println!("✨ Example completed successfully!");
    Ok(())
}
