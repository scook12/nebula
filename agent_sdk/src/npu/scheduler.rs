//! NPU Scheduler interface and implementation

use crate::npu::{
    InferenceResponse, InferenceTask, NpuDeviceId, NpuUsageStats, TaskPriority, TaskStatus,
};
use crate::types::TaskId;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// NPU Task Scheduler trait
#[async_trait]
pub trait NpuScheduler: Send + Sync {
    /// Submit a new inference task to the scheduler
    async fn submit_task(&self, task: InferenceTask) -> Result<TaskId>;

    /// Cancel a running or queued task
    async fn cancel_task(&self, task_id: TaskId) -> Result<()>;

    /// Get the status of a task
    async fn get_task_status(&self, task_id: TaskId) -> Option<TaskStatus>;

    /// Get system usage statistics
    async fn get_usage_stats(&self) -> NpuUsageStats;
}

/// Mock implementation of an NPU Scheduler
#[derive(Default, Clone)]
pub struct MockScheduler {
    tasks: Arc<RwLock<HashMap<TaskId, TaskStatus>>>,
}

#[async_trait]
impl NpuScheduler for MockScheduler {
    async fn submit_task(&self, task: InferenceTask) -> Result<TaskId> {
        let id = self.tasks.read().await.len();
        self.tasks.write().await.insert(id, TaskStatus::Queued);
        Ok(id)
    }

    async fn cancel_task(&self, task_id: TaskId) -> Result<()> {
        self.tasks.write().await.remove(&task_id);
        Ok(())
    }

    async fn get_task_status(&self, task_id: TaskId) -> Option<TaskStatus> {
        self.tasks.read().await.get(&task_id).cloned()
    }

    async fn get_usage_stats(&self) -> NpuUsageStats {
        let tasks = self.tasks.read().await.len();
        NpuUsageStats {
            total_devices: 1,
            active_devices: 0,
            compute_utilization: 0.0,
            memory_utilization: 0.0,
            power_consumption_watts: 0.0,
            tasks_completed_last_minute: tasks as u64,
            average_task_time: std::time::Duration::from_secs(0),
            queued_tasks: tasks as usize,
        }
    }
}
