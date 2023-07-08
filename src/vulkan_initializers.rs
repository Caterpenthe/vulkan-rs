use ash::vk;
use std::sync::Arc;

pub fn semaphore_create_info() -> vk::SemaphoreCreateInfo {
    vk::SemaphoreCreateInfo::builder().build()
}

pub(crate) fn submit_info() -> vk::SubmitInfo {
    vk::SubmitInfo::builder().build()
}
