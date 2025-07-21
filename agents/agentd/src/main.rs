//! NebulaOS Agent Scheduler Daemon (agentd)
//! 
//! This daemon provides kernel-level AI agent management for NebulaOS.
//! It handles agent registration, scheduling, communication, and resource allocation.

#![cfg_attr(feature = "redox", feature(int_roundings, let_chains))]

use log::{info, error};

#[cfg(feature = "redox")]
use log::warn;

#[cfg(feature = "redox")]
use {
    std::sync::Mutex,
    redox_event::{EventFlags, EventQueue},
    redox_scheme::{wrappers::ReadinessBased, Socket},
};

mod agent;
mod types;

pub use agent::AgentScheme;
pub use types::*;

fn main() {
    env_logger::init();
    info!("Starting NebulaOS Agent Scheduler Daemon v{}", env!("CARGO_PKG_VERSION"));

    #[cfg(feature = "redox")]
    {
        redox_daemon::Daemon::new(move |daemon| {
            match run_daemon(daemon) {
                Ok(()) => {
                    info!("agentd shutting down gracefully");
                    std::process::exit(0);
                },
                Err(error) => {
                    error!("agentd failed: {}", error);
                    std::process::exit(1);
                }
            }
        }).expect("agentd: failed to daemonize");
    }

    #[cfg(feature = "mock")]
    {
        info!("Running in mock mode (non-Redox system)");
        match run_mock() {
            Ok(()) => info!("Mock agentd completed successfully"),
            Err(error) => error!("Mock agentd failed: {}", error),
        }
    }
}

#[cfg(feature = "redox")]
fn run_daemon(daemon: redox_daemon::Daemon) -> anyhow::Result<()> {
    redox_event::user_data! {
        enum EventSource {
            AgentSocket,
        }
    }

    // Create agent scheme socket
    let agent_socket = Socket::nonblock("agent")
        .map_err(|e| anyhow::anyhow!("failed to create agent scheme: {}", e))?;
    
    let agent_scheme = Mutex::new(AgentScheme::new(&agent_socket));
    let mut agent_handler = ReadinessBased::new(&agent_socket, 16);

    info!("Agent scheme initialized on socket 'agent'");
    
    // Signal daemon is ready
    daemon.ready().map_err(|e| anyhow::anyhow!("daemon ready failed: {}", e))?;

    // Create event listener
    let mut event_queue = EventQueue::<EventSource>::new()
        .map_err(|e| anyhow::anyhow!("failed to create event queue: {}", e))?;
    
    event_queue
        .subscribe(
            agent_socket.inner().raw(),
            EventSource::AgentSocket,
            EventFlags::READ,
        )
        .map_err(|e| anyhow::anyhow!("failed to subscribe agent socket: {}", e))?;

    // Drop root privileges if possible
    libredox::call::setrens(0, 0).ok();

    info!("NebulaOS Agent Daemon is ready!");

    // Main event loop
    let mut agent_eof = false;
    while !agent_eof {
        let Some(event_res) = event_queue.next() else {
            break;
        };
        let event = event_res.map_err(|e| anyhow::anyhow!("error in event queue: {}", e))?;

        match event.user_data {
            EventSource::AgentSocket => {
                // Handle agent scheme events
                if !agent_eof {
                    // 1. Read requests
                    match agent_handler.read_requests() {
                        Ok(true) => {} // Success
                        Ok(false) => {
                            warn!("Agent socket EOF");
                            agent_eof = true;
                        }
                        Err(err) => return Err(anyhow::anyhow!("read_requests error: {}", err)),
                    }
                }

                // 2. Process requests
                agent_handler.process_requests(|| agent_scheme.lock().unwrap());

                // 3. Poll blocking requests
                agent_handler
                    .poll_all_requests(|| agent_scheme.lock().unwrap())
                    .map_err(|e| anyhow::anyhow!("poll_all_requests error: {}", e))?;

                // 4. Write responses
                match agent_handler.write_responses() {
                    Ok(true) => {} // Success
                    Ok(false) => {
                        warn!("Agent socket write EOF");
                        agent_eof = true;
                    }
                    Err(err) => return Err(anyhow::anyhow!("write_responses error: {}", err)),
                }
            }
        }
    }

    info!("Agent daemon main loop ended");
    Ok(())
}

#[cfg(feature = "mock")]
fn run_mock() -> anyhow::Result<()> {
    info!("Mock agent daemon running...");
    
    // Create a simple mock agent scheme for testing
    let mut agent_scheme = AgentScheme::mock_new();
    
    info!("Mock agent scheme created");
    info!("Available operations:");
    info!("  - Agent registration simulation");
    info!("  - Basic message passing test");
    info!("  - Inference task simulation");

    // Simulate some agent operations
    let agent1_id = agent_scheme.mock_register_agent("test_agent_1".to_string())
        .map_err(|e| anyhow::anyhow!(e))?;
    let agent2_id = agent_scheme.mock_register_agent("test_agent_2".to_string())
        .map_err(|e| anyhow::anyhow!(e))?;
    
    info!("Registered agents: {} and {}", agent1_id, agent2_id);
    
    // Simulate message passing
    agent_scheme.mock_send_message(agent1_id, agent2_id, b"Hello from agent 1")
        .map_err(|e| anyhow::anyhow!(e))?;
    
    let msg = agent_scheme.mock_receive_message(agent2_id)
        .map_err(|e| anyhow::anyhow!(e))?;
    info!("Agent {} received: {:?}", agent2_id, String::from_utf8_lossy(&msg));

    info!("Mock operations completed successfully");
    Ok(())
}
