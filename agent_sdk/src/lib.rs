//! NebulaOS Agent SDK
//!
//! This SDK provides a simple and powerful framework for building AI agents
//! that can run on NebulaOS. It handles agent lifecycle, communication,
//! ML model integration, and system resource management.

pub mod agent;
pub mod message;
pub mod ml;
pub mod runtime;
pub mod types;

// NPU Hardware Abstraction Layer (optional)
#[cfg(feature = "npu")]
pub mod npu;

// Re-export the main types and traits for easy access
pub use agent::{Agent, AgentBuilder, AgentConfig, AgentHandler, AgentRuntime};
pub use message::{Message, MessageHandler, MessageType};
pub use ml::{InferenceRequest, InferenceResponse, MLHandler, ModelConfig};
pub use runtime::NebulaRuntime;
pub use types::*;

// Explicit re-export for commonly used types
pub use types::{AgentCapabilities, AgentContext, AgentStatus};

// Prelude for convenient imports
pub mod prelude {
    pub use crate::agent::{Agent, AgentBuilder, AgentConfig, AgentHandler};
    pub use crate::message::{Message, MessageHandler, MessageType};
    pub use crate::ml::{InferenceRequest, InferenceResponse, MLHandler, ModelConfig};
    pub use crate::runtime::NebulaRuntime;
    pub use crate::types::{
        AgentCapabilities, AgentContext, AgentId, AgentStatus, Error, Result, TaskId,
    };
    pub use anyhow::Result as AnyResult;
    pub use log::{debug, error, info, warn};
    pub use tokio;

    // NPU HAL exports (when enabled)
    #[cfg(feature = "npu")]
    pub use crate::npu::{
        init_mock_npu_subsystem, init_npu_subsystem, NpuDevice, NpuDeviceId, NpuDeviceType,
        NpuManager,
    };
}

/// SDK version information
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the SDK with default configuration
pub async fn init() -> Result<NebulaRuntime> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("NebulaOS Agent SDK v{} initialized", SDK_VERSION);
    NebulaRuntime::new().await
}

/// Initialize the SDK with custom logging
pub async fn init_with_logger(log_level: &str) -> Result<NebulaRuntime> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();
    log::info!(
        "NebulaOS Agent SDK v{} initialized with log level: {}",
        SDK_VERSION,
        log_level
    );
    NebulaRuntime::new().await
}
