use ash::prelude::VkResult;
use ash::{extensions::khr, vk, Entry};
use std::sync::Arc;

pub struct Surface {
    pub raw: vk::SurfaceKHR,
    pub loader: khr::Surface,
}

impl Surface {
    pub fn create(
        instance: &ash::Instance,
        entry: &Entry,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
    ) -> VkResult<Arc<Self>> {
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, display, window, None)? };

        let loader = khr::Surface::new(&entry, &instance);

        Ok(Arc::new(Self {
            raw: surface,
            loader,
        }))
    }
}
