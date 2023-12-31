use std::ffi::c_void;
use std::sync::Arc;
use ash::vk;
use crate::vulkan_device::VulkanDevice;

pub struct TextureData {
    pub device: Arc<VulkanDevice>,

    pub image: vk::Image,
    pub image_layout: vk::ImageLayout,
    pub device_memory: vk::DeviceMemory,
    pub view: vk::ImageView,

    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub descriptor: vk::DescriptorImageInfo,
    pub format: vk::Format,

    pub sampler: vk::Sampler,
}

pub trait TextureBase {
    fn update_descriptor(&self);
    fn destroy(&self);
    fn load_ktx_file(&mut self, filename: &str, ktx_texture: Vec<*mut c_void>);
    fn load_from_file(&mut self, filename: &str);
    fn from_buffer(buffer: &[u8], format: vk::Format, tex_usage: vk::ImageUsageFlags);
}

pub struct Texture {
    data: TextureData,
}

pub struct Texture2D {
    data: TextureData,
}

