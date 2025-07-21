//! NPU Device Drivers
//!
//! This module contains hardware-specific drivers for different NPU types

#[cfg(all(target_os = "macos", feature = "apple_neural_engine"))]
pub mod apple_neural_engine;

#[cfg(all(target_os = "macos", feature = "apple_neural_engine"))]
pub mod apple_neural_device;

#[cfg(all(target_os = "macos", feature = "apple_neural_engine"))]
pub mod apple_neural_hal;

#[cfg(all(target_os = "macos", feature = "apple_neural_engine"))]
pub use apple_neural_engine::AppleNeuralEngineDriver;

#[cfg(all(target_os = "macos", feature = "apple_neural_engine"))]
pub use apple_neural_device::AppleNeuralDevice;

#[cfg(all(target_os = "macos", feature = "apple_neural_engine"))]
pub use apple_neural_hal::AppleNeuralEngineHal;
