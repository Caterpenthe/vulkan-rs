use ash::prelude::VkResult;
use ash::vk;

pub static DEFAULT_FENCE_TIMEOUT: u64 = 100000000000;

pub fn get_supported_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> VkResult<vk::Format> {
    let depth_formats = [
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D32_SFLOAT,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D16_UNORM_S8_UINT,
        vk::Format::D16_UNORM,
    ];
    for &format in depth_formats.iter() {
        let format_props =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };
        if format_props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Ok(format);
        }
    }
    return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
}

pub fn get_assets_path() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets");
    path
}

pub fn get_shader_path() -> std::path::PathBuf {
    let mut path = get_assets_path();
    path.push("shaders");
    path
}

pub struct SimpleStat {
    start_time: i64,
    stat_name: String,
}

impl SimpleStat {
    pub fn new(name: &str) -> Self {
        SimpleStat {
            start_time: puffin::now_ns(),
            stat_name: String::from(name),
        }
    }
}

impl Drop for SimpleStat {
    fn drop(&mut self) {
        let end_time = puffin::now_ns();
        let duration = end_time - self.start_time;
        println!(
            "{}: {} ms",
            self.stat_name,
            duration as f64 / 1000.0 / 1000.0
        );
    }
}
