pub mod filesystem;
pub mod predictor;
pub mod types;

pub use predictor::AccessPredictor;
pub use types::{FileAccessEvent, AccessType, FileAccessPattern};

#[cfg(feature = "mock")]
pub use filesystem::{FilesystemScheme, FileHandle, PrefetchCache};
