/*
* Vulkan Example Base struct
*
* Copyright (C) by Caterpenthe
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

use crate::benchmark::Benchmark;
use crate::camera::Camera;
use crate::command_line_parser::CommandLineParser;
use crate::vulkan_device::VulkanDevice;
use crate::vulkan_surface::Surface;
use crate::vulkan_swap_chain::{SwapchainDesc, VulkanSwapChain};
use crate::vulkan_ui_overlay::UIOverlay;
use ash::extensions::ext::DebugUtils;
use ash::prelude::*;
use ash::{vk, Entry, Instance};
use glam::Vec2;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::sync::Arc;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Fullscreen, Window, WindowBuilder};

pub static ARGS: Vec<String> = vec![];

pub struct Semaphores {
    // Swap chain image presentation
    present_complete: vk::Semaphore,
    // Command buffer submission and execution
    render_complete: vk::Semaphore,
}

impl Default for Semaphores {
    fn default() -> Self {
        Semaphores {
            present_complete: vk::Semaphore::null(),
            render_complete: vk::Semaphore::null(),
        }
    }
}

pub enum FullscreenMode {
    Borderless,

    /// Seems to be the only way for stutter-free rendering on Nvidia + Win10.
    Exclusive,
}

/// Example settings that can be changed e.g. by command line arguments
pub struct Settings {
    /// Activates validation layers (and message output) when set to true
    pub validation: bool,
    /// Set to true if fullscreen mode has been requested via command line
    pub fullscreen: Option<FullscreenMode>,
    /// Set to true if v-sync will be forced for the swapchain
    pub vsync: bool,
    /// Enable UI overlay
    pub overlay: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            validation: true,
            fullscreen: None,
            vsync: false,
            overlay: false,
        }
    }
}

pub struct DepthStencil {
    pub image: vk::Image,
    pub mem: vk::DeviceMemory,
    pub view: vk::ImageView,
}

pub struct GamePadState {
    axis_left: Vec2,
    axis_right: Vec2,
}

impl Default for GamePadState {
    fn default() -> Self {
        GamePadState {
            axis_left: Vec2::new(0.0, 0.0),
            axis_right: Vec2::new(0.0, 0.0),
        }
    }
}

#[derive(Default)]
pub struct MouseButtons {
    left: bool,
    right: bool,
    middle: bool,
}

pub trait Example: vk::ExtendsPhysicalDeviceFeatures2 {
    fn init(base: &mut ExampleApp) -> Self;

    fn prepare(&mut self, base: &mut ExampleApp) -> VkResult<()>;

    fn build_command_buffers(&mut self, base: &mut ExampleApp) {}

    fn setup_depth_stencil(
        depth_format: vk::Format,
        device: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> VkResult<DepthStencil> {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(depth_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);

        let image = unsafe { device.logical_device.create_image(&image_create_info, None) }?;
        let mem_reqs = unsafe { device.logical_device.get_image_memory_requirements(image) };

        let mem_alloc = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(device.get_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?)
            .build();

        let mem = unsafe { device.logical_device.allocate_memory(&mem_alloc, None) }?;

        unsafe { device.logical_device.bind_image_memory(image, mem, 0) }?;

        let mut aspect_mask = vk::ImageAspectFlags::DEPTH;
        if depth_format >= vk::Format::D16_UNORM_S8_UINT {
            aspect_mask |= vk::ImageAspectFlags::STENCIL;
        }
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .image(image)
            .format(depth_format)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .aspect_mask(aspect_mask)
                    .build(),
            );

        let view = unsafe {
            device
                .logical_device
                .create_image_view(&image_view_create_info, None)
        }?;

        Ok(DepthStencil { image, mem, view })
    }

    fn setup_semaphores(device: &VulkanDevice) -> VkResult<Semaphores> {
        let mut semaphores = Semaphores::default();
        let semaphore_create_info = crate::vulkan_initializers::semaphore_create_info();
        // Create a semaphore used to synchronize image presentation
        // Ensures that the image is displayed before we start submitting new commands to the queue

        if let Ok(semaphore) = unsafe {
            device
                .logical_device
                .create_semaphore(&semaphore_create_info, None)
        } {
            semaphores.present_complete = semaphore;
        }
        // Create a semaphore used to synchronize command submission
        // Ensures that the image is not presented until all commands have been submitted and executed
        if let Ok(semaphore) = unsafe {
            device
                .logical_device
                .create_semaphore(&semaphore_create_info, None)
        } {
            semaphores.render_complete = semaphore;
        };
        Ok(semaphores)
    }
    fn setup_command_pool(
        device: &VulkanDevice,
        swapchain: &VulkanSwapChain,
    ) -> VkResult<vk::CommandPool> {
        let cmd_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(swapchain.queue_node_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe {
            device
                .logical_device
                .create_command_pool(&cmd_pool_create_info, None)
        }
    }

    fn setup_render_pass(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> VkResult<vk::RenderPass> {
        let attachments: Vec<vk::AttachmentDescription> = vec![
            // Color attachment
            vk::AttachmentDescription::builder()
                .format(color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build(),
            // Depth attachment
            vk::AttachmentDescription::builder()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ];

        let color_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let depth_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_reference])
            .depth_stencil_attachment(&depth_reference)
            .build();

        // Subpass dependencies for layout transitions
        let dependencies: Vec<vk::SubpassDependency> = vec![
            vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .build(),
            vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .build(),
        ];

        let subpasses = [subpass_description];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        unsafe {
            device
                .logical_device
                .create_render_pass(&render_pass_info, None)
        }
    }

    fn create_pipeline_cache(device: &VulkanDevice) -> VkResult<vk::PipelineCache> {
        let pipeline_cache_info = vk::PipelineCacheCreateInfo::builder();
        unsafe {
            device
                .logical_device
                .create_pipeline_cache(&pipeline_cache_info, None)
        }
    }

    fn setup_frame_buffers(
        device: &VulkanDevice,
        depth_stencil: &DepthStencil,
        render_pass: vk::RenderPass,
        swap_chain: &VulkanSwapChain,
        width: u32,
        height: u32,
    ) -> VkResult<Vec<vk::Framebuffer>> {
        let mut frame_buffers: Vec<vk::Framebuffer> = vec![];

        for i in 0..swap_chain.buffers.len() {
            let attachments = [swap_chain.buffers[i].view, depth_stencil.view];
            let frame_buffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1);
            frame_buffers.push(unsafe {
                device
                    .logical_device
                    .create_framebuffer(&frame_buffer_info, None)
            }?);
        }

        Ok(frame_buffers)
    }

    fn create_command_buffers(
        device: &VulkanDevice,
        swapchain: &VulkanSwapChain,
        cmd_pool: vk::CommandPool,
    ) -> VkResult<Vec<vk::CommandBuffer>> {
        let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain.images.len() as u32);
        unsafe {
            device
                .logical_device
                .allocate_command_buffers(&cmd_buf_allocate_info)
        }
    }

    fn setup_synchronization_primitives(
        device: &VulkanDevice,
        draw_cmd_buffers: &Vec<vk::CommandBuffer>,
    ) -> VkResult<Vec<vk::Fence>> {
        // Wait fences to sync command buffer access
        let mut fences: Vec<vk::Fence> = vec![];
        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..draw_cmd_buffers.len() {
            fences.push(unsafe { device.logical_device.create_fence(&fence_create_info, None) }?);
        }
        Ok(fences)
    }

    fn get_enabled_features(
        physical_device: &vk::PhysicalDevice,
        enabled_features: &mut vk::PhysicalDeviceFeatures,
    ) {
    }
    fn get_enabled_extensions(
        physical_device: &vk::PhysicalDevice,
        enabled_device_extensions: &mut Vec<&'static CStr>,
    ) {
    }

    fn get_next_chain<T: vk::ExtendsPhysicalDeviceFeatures2>() -> Option<T>;
}

pub struct RenderBackend {
    /// Vulkan instance, stores all per-application states
    instance: Arc<Instance>,
    /// Encapsulated physical and logical vulkan device
    pub vulkan_device: Arc<VulkanDevice>,
    /// Command buffer pool
    pub cmd_pool: vk::CommandPool,
    /// Descriptor set pool
    pub descriptor_pool: vk::DescriptorPool,
    /// Pipeline cache object
    pub pipeline_cache: vk::PipelineCache,
    /// Command buffers used for rendering
    pub draw_cmd_buffers: Vec<vk::CommandBuffer>,
    /// Global render pass for frame buffer writes
    pub render_pass: vk::RenderPass,
    /// Depth buffer format (selected during Vulkan initialization)
    depth_format: vk::Format,
    /// Wraps the swap chain to present images (framebuffers) to the windowing system
    pub swapchain: VulkanSwapChain,
    /// Pipeline stages used to wait at for graphics queue submissions
    submit_pipeline_stages: vk::PipelineStageFlags,
    /// Contains command buffers and semaphores to be presented to the queue
    submit_info: vk::SubmitInfo,
    /// Synchronization semaphores
    semaphores: Semaphores,
    depth_stencil: DepthStencil,
    wait_fences: Vec<vk::Fence>,
    debug_call_back: vk::DebugUtilsMessengerEXT,
    /// List of available frame buffers (same as number of swap chain images)
    pub frame_buffers: Vec<vk::Framebuffer>,
    /// Active frame buffer index
    pub current_buffer: u32,
    /// List of shader modules created (stored for cleanup)
    shader_modules: Vec<vk::ShaderModule>,
}

pub struct PhantomChain();
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PhantomChain {}

pub struct ExampleApp {
    pub dest_width: u32,
    pub dest_height: u32,
    pub resizing: bool,
    /// Frame counter to display fps
    pub frame_counter: i64,
    pub last_fps: f64,
    pub last_timestamp: i64,
    pub prev_end: i64,
    pub supported_instance_extensions: Vec<&'static CStr>,
    /// Set of device extensions to be enabled for this example (must be set in the derived constructor)
    pub enabled_device_extensions: Vec<&'static CStr>,
    pub enabled_instance_extensions: Vec<&'static CStr>,
    /// Set of physical device features to be enabled for this example (must be set in the derived constructor)
    pub enabled_features: vk::PhysicalDeviceFeatures,

    pub swapchain_desc: SwapchainDesc,

    pub prepared: bool,

    pub shader_dir: String,
    pub resized: bool,
    pub view_updated: bool,
    pub width: u32,
    pub height: u32,
    pub ui_overlay: UIOverlay,
    pub command_line_parser: CommandLineParser,
    /// Last frame time measured using a high performance timer (if available)
    pub frame_timer: f32,
    pub benchmark: Option<Arc<Benchmark>>,
    pub settings: Settings,
    pub default_clear_color: Option<Arc<vk::ClearColorValue>>,
    // Defines a frame rate independent timer value clamped from -1.0...1.0
    // For use in animations, rotations, etc.
    pub timer: f32,
    // Multiplier for speeding up (or slowing down) the global timer
    pub timer_speed: f32,
    pub paused: bool,
    pub camera: Camera,
    pub mouse_pos: Vec2,
    pub title: CString,
    pub name: CString,
    pub api_version: u32,
    pub gamepad_state: GamePadState,
    pub mouse_buttons: MouseButtons,

    pub backend: Option<RenderBackend>,
    pub window: Arc<Window>,
    event_loop: Option<EventLoop<()>>,
}

impl ExampleApp {
    fn create_instance(&mut self, entry: &Entry, window: Arc<Window>) -> VkResult<Instance> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name(self.name.as_c_str())
            .engine_name(self.name.as_c_str())
            .api_version(self.api_version);

        // let mut instance_extensions = vec![ash::extensions::khr::Surface::name().as_ptr()];
        // #[cfg(windows)]
        // {
        //     instance_extensions.push(ash::extensions::khr::Win32Surface::name().as_ptr());
        // }
        //
        // if cfg!(windows) {
        //     instance_extensions.push(ash::extensions::khr::Win32Surface::name().as_ptr());
        // }

        let mut instance_extensions =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();

        // Get extensions supported by the instance and store for later use
        if let Ok(extension_properties) = entry.enumerate_instance_extension_properties(None) {
            for extension in extension_properties.iter() {
                let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                self.supported_instance_extensions.push(extension_name);
            }
        }

        // Enabled requested instance extensions
        if !self.enabled_instance_extensions.is_empty() {
            for i in 0..self.enabled_instance_extensions.len() {
                let enabled_instance_extension = &self.enabled_instance_extensions[i];
                if self
                    .supported_instance_extensions
                    .iter()
                    .any(|extension_name| enabled_instance_extension == extension_name)
                {
                    instance_extensions.push(enabled_instance_extension.as_ptr());
                } else {
                    println!(
                        "Enabled instance extension \"{}\" is not present at instance level",
                        enabled_instance_extension.to_str().unwrap()
                    );
                }
            }
        }

        // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
        // Note that on Android this layer requires at least NDK r20
        let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let enabled_layers = [validation_layer_name.as_ptr()];

        let mut instance_create_info =
            vk::InstanceCreateInfo::builder().application_info(&app_info);

        if !instance_extensions.is_empty() {
            if self.settings.validation {
                instance_extensions.push(DebugUtils::name().as_ptr());
                // SRS - Dependency when VK_EXT_DEBUG_MARKER is enabled
            }
            instance_create_info =
                instance_create_info.enabled_extension_names(&instance_extensions);
        }

        if self.settings.validation {
            // Check if this layer is available at instance level
            if let Ok(layer_properties) = entry.enumerate_instance_layer_properties() {
                let mut validation_layer_present = false;
                for layer_property in layer_properties.iter() {
                    let layer_name = unsafe { CStr::from_ptr(layer_property.layer_name.as_ptr()) };
                    if layer_name == validation_layer_name.as_c_str() {
                        validation_layer_present = true;
                        break;
                    }
                }
                if validation_layer_present {
                    instance_create_info =
                        instance_create_info.enabled_layer_names(&enabled_layers);
                } else {
                    println!("Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled");
                }
            }
        }

        unsafe { entry.create_instance(&instance_create_info, None) }
    }
    fn init_render_backend<E>(&mut self, window: Arc<Window>) -> VkResult<()>
    where
        E: Example + vk::ExtendsPhysicalDeviceFeatures2,
    {
        let _ = crate::tools::SimpleStat::new("init_render_backend");

        let entry = Entry::linked();

        // Vulkan instance
        let instance = Arc::new(self.create_instance(&entry, window.clone())?);
        println!("init_render_backend:Instance created");

        // If requested, we enable the default validation layers for debugging
        let mut debug_call_back = vk::DebugUtilsMessengerEXT::null();
        if self.settings.validation {
            debug_call_back = crate::vulkan_debug::debug::setup_debugging(&entry, &instance)?;
        }

        println!("init_render_backend:Debug callback created");

        let surface = Surface::create(
            &instance,
            &entry,
            window.raw_display_handle(),
            window.raw_window_handle(),
        )?;

        println!("init_render_backend:Surface created");

        // Physical device
        let physical_device = self.get_physical_device(&instance)?;

        println!("init_render_backend:Physical device created");

        // Vulkan device creation
        // This is handled by a separate class that gets a logical device representation
        // and encapsulates functions related to a device
        // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation
        E::get_enabled_features(&physical_device, &mut self.enabled_features);
        // Derived examples can enable extensions based on the list of supported extensions read from the physical device
        E::get_enabled_extensions(&physical_device, &mut self.enabled_device_extensions);

        let vulkan_device = VulkanDevice::create(
            instance.clone(),
            physical_device,
            &self.enabled_features,
            &mut self.enabled_device_extensions,
            E::get_next_chain::<E>(),
            None,
            true,
        )?;

        println!("init_render_backend:Vulkan device created");

        let swapchain = VulkanSwapChain::create(
            instance.clone(),
            vulkan_device.clone(),
            surface,
            self.width,
            self.height,
            self.swapchain_desc.vsync,
            self.swapchain_desc.full_screen,
        )?;

        println!("init_render_backend:Swapchain created");

        // Find a suitable depth format
        let depth_format = crate::tools::get_supported_depth_format(&instance, physical_device)?;

        // Create synchronization objects
        let semaphores = E::setup_semaphores(&vulkan_device)?;

        println!("init_render_backend:Semaphores created");

        // Set up submit info structure
        // Semaphores will stay the same during application lifetime
        // Command buffer submission info is set by each example
        let submit_pipeline_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(std::slice::from_ref(&submit_pipeline_stages))
            .wait_semaphores(std::slice::from_ref(&semaphores.present_complete))
            .signal_semaphores(std::slice::from_ref(&semaphores.render_complete))
            .build();

        let cmd_pool = E::setup_command_pool(&vulkan_device, &swapchain)?;
        let depth_stencil =
            E::setup_depth_stencil(depth_format, &vulkan_device, self.width, self.height)?;

        let render_pass =
            E::setup_render_pass(&vulkan_device, swapchain.color_format, depth_format)?;

        let pipeline_cache = E::create_pipeline_cache(&vulkan_device)?;

        let frame_buffers = E::setup_frame_buffers(
            &vulkan_device,
            &depth_stencil,
            render_pass,
            &swapchain,
            self.width,
            self.height,
        )?;

        let draw_cmd_buffers = E::create_command_buffers(&vulkan_device, &swapchain, cmd_pool)?;

        let wait_fences = E::setup_synchronization_primitives(&vulkan_device, &draw_cmd_buffers)?;

        self.backend = Some(RenderBackend {
            instance,
            vulkan_device,
            depth_format,
            cmd_pool,
            swapchain,
            submit_pipeline_stages,
            submit_info,
            semaphores,
            depth_stencil,
            render_pass,
            pipeline_cache,
            frame_buffers,
            draw_cmd_buffers,
            wait_fences,
            debug_call_back,
            current_buffer: 0,
            descriptor_pool: vk::DescriptorPool::null(),
            shader_modules: vec![],
        });

        println!("Render backend initialized");
        Ok(())
    }
    fn get_physical_device(&self, instance: &Instance) -> VkResult<vk::PhysicalDevice> {
        // GPU selection
        // Select physical device to be used for the Vulkan example
        // Defaults to the first device unless specified by command line
        let selected_device = 0;
        // todo: add command line argument for device selection
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        Ok(physical_devices[selected_device])
    }

    pub fn render_loop<FrameFn>(&mut self, frame_fn: &mut FrameFn) -> VkResult<()>
    where
        FrameFn: FnMut(&mut Self),
    {
        self.dest_width = self.width;
        self.dest_height = self.height;

        self.last_timestamp = puffin::now_ns();

        self.prev_end = self.last_timestamp;
        let mut quit_message_received = false;

        let mut event_loop = self.event_loop.take().unwrap();
        let mut events = Vec::new();
        while !quit_message_received {
            let ui_wants_mouse = false;
            event_loop.run_return(|event, _, control_flow| {
                *control_flow = ControlFlow::Poll;

                let mut allow_event = true;
                match &event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                            quit_message_received = true;
                        }
                        WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. }
                            if ui_wants_mouse =>
                        {
                            allow_event = false;
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if let Some(key_code) = input.virtual_keycode {
                                if input.state == ElementState::Pressed {
                                    match key_code {
                                        VirtualKeyCode::W => {}
                                        VirtualKeyCode::A => {}
                                        VirtualKeyCode::S => {}
                                        VirtualKeyCode::D => {}
                                        _ => {}
                                    }
                                }
                            }
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }

                if allow_event {
                    events.push(event.to_static());
                }
            });

            self.next_frame(frame_fn);
        }
        self.event_loop = Some(event_loop);

        //Flush device to make sure all resources can be freed
        unsafe {
            self.backend
                .as_mut()
                .unwrap()
                .vulkan_device
                .logical_device
                .device_wait_idle()
        }
    }

    fn render<FrameFn>(&mut self, frame_fn: &mut FrameFn)
    where
        FrameFn: FnMut(&mut Self),
    {
        frame_fn(self);
    }

    fn next_frame<FrameFn>(&mut self, frame_fn: &mut FrameFn)
    where
        FrameFn: FnMut(&mut Self),
    {
        // puffin::profile_scope!("frame");

        let start = puffin::now_ns();
        if self.view_updated {
            self.view_updated = false;
            self.view_changed();
        }

        self.render(frame_fn);
        self.frame_counter += 1;
        let end = puffin::now_ns();

        self.frame_timer = (end - start) as f32 / 1000000.0;

        self.camera.update(self.frame_timer);
        if self.camera.moving() {
            self.view_updated = true;
        }

        //Convert to clamped timer value
        if !self.paused {
            self.timer += self.timer_speed * self.frame_timer;
            if self.timer > 1.0 {
                self.timer -= 1.0;
            }
        }

        let fps_timer =
            chrono::Duration::nanoseconds(end - self.last_timestamp).num_milliseconds() as f64;
        if fps_timer > 1000.0 {
            self.last_fps = self.frame_counter as f64 * (1000.0 / fps_timer);
            self.window.set_title(&format!(
                "{} - {:.2} fps",
                self.title.to_string_lossy(),
                self.last_fps
            ));

            self.frame_counter = 0;
            self.last_timestamp = end;
        }

        self.prev_end = end;
    }

    fn view_changed(&mut self) {}

    pub fn builder<E>() -> ExampleAppBuilder<E>
    where
        E: Example,
    {
        ExampleAppBuilder::new()
    }
}

pub struct ExampleAppBuilder<E>
where
    E: Example,
{
    settings: Settings,
    width: u32,
    height: u32,
    camera: Camera,
    swapchain_desc: SwapchainDesc,
    shader_dir: String,
    name: CString,
    title: CString,

    enabled_instance_extensions: Vec<&'static CStr>,

    window_builder: WindowBuilder,

    _marker: PhantomData<E>,
}

impl<E> Default for ExampleAppBuilder<E>
where
    E: Example,
{
    fn default() -> Self {
        ExampleAppBuilder {
            settings: Settings::default(),
            width: 1280,
            height: 720,
            camera: Camera::new(),
            name: CString::new(String::from("Vulkan Example")).unwrap(),
            title: CString::new(String::from("Vulkan Example")).unwrap(),
            swapchain_desc: SwapchainDesc {
                vsync: false,
                full_screen: false,
            },
            shader_dir: String::from("shaders"),

            enabled_instance_extensions: vec![],

            window_builder: WindowBuilder::new(),

            _marker: PhantomData,
        }
    }
}

impl<E> ExampleAppBuilder<E>
where
    E: Example,
{
    pub fn build(self) -> VkResult<ExampleApp> {
        let mut window_builder = self.window_builder;
        window_builder = window_builder
            .with_title(self.name.to_str().unwrap_or_default())
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
        let event_loop = EventLoop::new();
        if let Some(fullscreen) = &self.settings.fullscreen {
            window_builder = window_builder.with_fullscreen(match fullscreen {
                FullscreenMode::Borderless => Some(Fullscreen::Borderless(None)),
                FullscreenMode::Exclusive => Some(Fullscreen::Exclusive(
                    event_loop
                        .primary_monitor()
                        .expect("at least one monitor")
                        .video_modes()
                        .next()
                        .expect("at least one video mode"),
                )),
            });
        }
        let window = Arc::new(window_builder.build(&event_loop).expect("window"));

        let mut vulkan_example = ExampleApp {
            dest_width: 0,
            dest_height: 0,
            resizing: false,
            shader_dir: "glsl".to_string(),
            frame_counter: 0,
            last_fps: 0.0,
            last_timestamp: 0,
            prev_end: 0,

            supported_instance_extensions: vec![],

            enabled_device_extensions: vec![],
            enabled_instance_extensions: self.enabled_instance_extensions,
            enabled_features: vk::PhysicalDeviceFeatures::builder().build(),

            resized: false,
            view_updated: false,

            prepared: false,

            width: self.width,
            height: self.height,
            ui_overlay: UIOverlay {},

            command_line_parser: CommandLineParser {},
            frame_timer: 1.0,
            benchmark: None,
            settings: self.settings,
            default_clear_color: None,
            timer: 0.0,
            timer_speed: 0.25,
            paused: false,
            camera: self.camera,
            mouse_pos: Vec2::new(0.0, 0.0),
            title: self.title,
            name: self.name,
            api_version: ash::vk::API_VERSION_1_0,
            gamepad_state: Default::default(),
            mouse_buttons: Default::default(),

            swapchain_desc: SwapchainDesc {
                vsync: false,
                full_screen: false,
            },

            backend: None,
            window: window.clone(),
            event_loop: Some(event_loop),
        };

        vulkan_example.init_render_backend::<E>(window)?;
        Ok(vulkan_example)
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn enabled_instance_extensions(
        mut self,
        enabled_instance_extensions: Vec<&'static CStr>,
    ) -> Self {
        self.enabled_instance_extensions = enabled_instance_extensions;
        self
    }

    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.title = CString::new(String::from(name)).unwrap();
        self
    }

    pub fn title(mut self, title: &str) -> Self {
        self.title = CString::new(String::from(title)).unwrap();
        self
    }

    pub fn shader_dir(mut self, shader_dir: String) -> Self {
        self.shader_dir = shader_dir;
        self
    }

    pub fn camera(mut self, camera: Camera) -> Self {
        self.camera = camera;
        self
    }
}
