use ash::{vk, Entry};

/// Encapsulates a single frame buffer attachment
pub struct FramebufferAttachment {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
    subresource_range: vk::ImageSubresourceRange,
    description: vk::AttachmentDescription,
}
