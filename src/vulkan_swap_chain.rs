use crate::vulkan_device::VulkanDevice;
use crate::vulkan_surface::Surface;
use ash::extensions::khr;
use ash::extensions::khr::Swapchain;
use ash::prelude::VkResult;
use ash::{vk, Device, Entry};
use std::sync::Arc;

pub struct SwapChainBuffer {
    pub image: vk::Image,
    pub view: vk::ImageView,
}

pub struct VulkanSwapChain {
    pub raw: vk::SwapchainKHR,
    pub loader: Swapchain,
    pub device: Arc<VulkanDevice>,
    pub surface: Arc<Surface>,
    pub images: Vec<vk::Image>,
    pub buffers: Vec<SwapChainBuffer>,
    pub queue_node_index: u32,
    pub color_format: vk::Format,
}

impl VulkanSwapChain {
    pub fn create(
        instance: Arc<ash::Instance>,
        device: Arc<VulkanDevice>,
        surface: Arc<Surface>,

        width: u32,
        height: u32,
        vsync: bool,
        full_screen: bool,
    ) -> VkResult<Self> {
        // Store the current swap chain handle so we can use it later on to ease up recreation
        let old_swapchain = vk::SwapchainKHR::null();

        let surface_capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(device.physical_device, surface.raw)
        }?;

        let swapchain_extent = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D { width, height },
            _ => surface_capabilities.current_extent,
        };
        //Select a present mode for the swapchain

        //The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
        //This mode waits for the vertical blank ("v-sync")
        let mut swapchain_present_mode = vk::PresentModeKHR::FIFO;
        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(device.physical_device, surface.raw)
        }?;
        //If v-sync is not requested, try to find a mailbox mode
        //It's the lowest latency non-tearing present mode available
        if (!vsync) {
            for present_mode in present_modes {
                if present_mode == vk::PresentModeKHR::MAILBOX {
                    swapchain_present_mode = vk::PresentModeKHR::MAILBOX;
                    break;
                } else if (swapchain_present_mode != vk::PresentModeKHR::MAILBOX) {
                    swapchain_present_mode = vk::PresentModeKHR::IMMEDIATE;
                }
            }
        }

        // Determine the number of images
        let mut desired_number_of_swapchain_images = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_number_of_swapchain_images > surface_capabilities.max_image_count
        {
            desired_number_of_swapchain_images = surface_capabilities.max_image_count;
        }

        // Find the transformation of the surface
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            // We prefer a non-rotated transform
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        // Find a supported composite alpha format (not all devices support alpha opaque)
        let mut composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;
        // Simply select the first composite alpha format available
        let composite_alpha_flags = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ];
        for &composite_alpha_flag in &composite_alpha_flags {
            if surface_capabilities
                .supported_composite_alpha
                .contains(composite_alpha_flag)
            {
                composite_alpha = composite_alpha_flag;
                break;
            };
        }

        let mut image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        // Enable transfer source on swap chain images if supported
        if surface_capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }

        let surface_format = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(device.physical_device, surface.raw)
                .unwrap()[0]
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder().
            surface(surface.raw).
            min_image_count(desired_number_of_swapchain_images).
            image_format(surface_format.format).
            image_color_space(surface_format.color_space).
            image_extent(swapchain_extent).
            image_usage(image_usage).
            pre_transform(pre_transform).
            image_array_layers(1).
            image_sharing_mode(vk::SharingMode::EXCLUSIVE).
            // queue_family_indices(&[0]).
            present_mode(swapchain_present_mode).
            // // Setting oldSwapchain to the previous swap chain handle disables the ability to use a mailbox present mode
            old_swapchain(old_swapchain).
            // Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
            clipped(true).
            composite_alpha(composite_alpha);

        let swapchain_loader = khr::Swapchain::new(&instance, &device.logical_device);
        let swap_chain =
            unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        // // If an existing swap chain is re-created, destroy the old swap chain
        // // This also cleans up all the presentable images
        // if old_swapchain != vk::SwapchainKHR::null() {
        //     unsafe { fns.destroy_swapchain(old_swapchain, None) };
        // }

        // Get the swap chain images
        let images: Vec<vk::Image> = unsafe { swapchain_loader.get_swapchain_images(swap_chain) }?;

        // Get the swap chain buffers containing the image and imageview
        let mut buffers: Vec<SwapChainBuffer> = vec![];
        for image in images.clone() {
            let image_view = unsafe {
                device.logical_device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build(),
                    None,
                )
            }?;
            buffers.push(SwapChainBuffer {
                image,
                view: image_view,
            });
        }

        // Get available queue family properties
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(device.physical_device) };
        assert!(queue_family_properties.len() >= 1);
        let queue_count = queue_family_properties.len() as u32;

        // Iterate over each queue to learn whether it supports presenting:
        // Find a queue with present support
        // Will be used to present the swap chain images to the windowing system
        let mut supports_present = vec![false; queue_count as usize];
        for i in 0..queue_count {
            supports_present[i as usize] = unsafe {
                surface.loader.get_physical_device_surface_support(
                    device.physical_device,
                    i,
                    surface.raw,
                )
            }?;
        }

        // Search for a graphics and a present queue in the array of queue
        // families, try to find one that supports both
        let mut graphics_queue_node_index = u32::MAX;
        let mut present_queue_node_index = u32::MAX;
        for i in 0..queue_count {
            if queue_family_properties[i as usize]
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                if graphics_queue_node_index == u32::MAX {
                    graphics_queue_node_index = i;
                }

                if supports_present[i as usize] {
                    graphics_queue_node_index = i;
                    present_queue_node_index = i;
                    break;
                }
            }
        }
        if present_queue_node_index == u32::MAX {
            // If there's no queue that supports both present and graphics
            // try to find a separate present queue
            for i in 0..queue_count {
                if supports_present[i as usize] {
                    present_queue_node_index = i;
                    break;
                }
            }
        }

        // Exit if either a graphics or a presenting queue hasn't been found
        if graphics_queue_node_index == u32::MAX || present_queue_node_index == u32::MAX {
            panic!("Could not find a graphics and/or presenting queue!");
        }

        // todo: Add support for separate graphics and presenting queue
        if graphics_queue_node_index != present_queue_node_index {
            panic!("Separate graphics and presenting queues are not supported yet!");
        }

        let queue_node_index = graphics_queue_node_index;

        Ok(VulkanSwapChain {
            raw: swap_chain,
            loader: swapchain_loader,
            device,
            surface,
            images,
            buffers,
            queue_node_index,
            color_format: surface_format.format,
        })
    }

    pub fn acquire_next_image(
        &self,
        present_complete_semaphore: vk::Semaphore,
    ) -> VkResult<(u32, bool)> {
        unsafe {
            self.loader.acquire_next_image(
                self.raw,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            )
        }
    }

    pub fn queue_present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> VkResult<bool> {
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&[wait_semaphore])
            .swapchains(&[self.raw])
            .image_indices(&[image_index])
            .build();
        unsafe { self.loader.queue_present(queue, &present_info) }
    }
}

pub struct SwapchainDesc {
    pub vsync: bool,
    pub full_screen: bool,
}
