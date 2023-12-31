
/*
 * Shows how to load and display an animated scene from a glTF file using vertex skinning
 * See the accompanying README.md for a short tutorial on the data structures and functions required for vertex skinning
 *
 * For details on how glTF 2.0 works, see the official spec at https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 */

#![windows_subsystem = "windows"]

use ash::prelude::VkResult;
use ash::util::Align;
use ash::vk;
use ash::vk::{ExtendsPhysicalDeviceFeatures2, PhysicalDevice, PhysicalDeviceFeatures};
use glam::{Vec3, Vec4};
use memoffset::offset_of;
use std::ffi::CStr;
use std::mem::{align_of, size_of};
use std::rc::Rc;
use std::sync::Arc;
use vulkan_rs::camera::{Camera, CameraType};
use vulkan_rs::vulkan_device::VulkanDevice;
use vulkan_rs::vulkan_example_base::*;

struct Vertices {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct Indices {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    count: u32,
}

struct Material{
    base_color_factor : Vec4,
    base_color_texture_index :usize,
}

struct Image {
    texture: 
}

type NodeRef = Option<Rc<Node>>;
struct Node {
    name: String,
    parent: NodeRef,
    children: Vec<NodeRef>,
    local_matrix: glam::Mat4,
    world_matrix: glam::Mat4,
    skin: Option<usize>,
    mesh: Option<usize>,
}
struct VulkanGLTFModel {
    vulkan_device: Arc<VulkanDevice>,
    copy_queue: vk::Queue,

    vertices : Vertices,
    indices : Indices,

}

static USE_STAGING: bool = false;
static DRAW_EVERY_FRAME: bool = false;

/// Vertex layout used in this example
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: Vec4,
    color: Vec4,
}
/// Vertex Buffer and attributes
struct VertexBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// Uniform buffer block object
struct UniformBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    descriptor: vk::DescriptorBufferInfo,
}

/// For simplicity we will use the same uniform block layout as in the shader:
//
//	layout(set = 0, binding = 0) uniform UBO
//	{
//		mat4 projectionMatrix;
//		mat4 modelMatrix;
//		mat4 viewMatrix;
//	} ubo;
//
/// This way we can just memcopy the ubo data to the ubo
/// Note: You should use data types that align with the GPU in order to avoid manual padding (vec4, mat4)
#[derive(Debug, Clone, Copy)]
struct UboVS {
    projection_matrix: glam::Mat4,
    view_matrix: glam::Mat4,
    model_matrix: glam::Mat4,
}

/// Index buffer
struct IndexBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    count: u32,
}

pub struct GLTFSkinning {
    /// Vertex buffer and attributes
    vertices: VertexBuffer,
    /// Index buffer
    indices: IndexBuffer,
    /// Uniform buffer block object
    uniform_buffer_vs: UniformBuffer,

    ubo_vs: UboVS,

    /// The pipeline layout is used by a pipeline to access the descriptor sets.
    ///
    /// It defines interface (without binding any actual data) between the shader stages used by the pipeline and the shader resources.
    ///
    /// A pipeline layout can be shared among multiple pipelines as long as their interfaces match.
    pipeline_layout: vk::PipelineLayout,

    /// Pipelines (often called "pipeline state objects") are used to bake all states that affect a pipeline.
    ///
    /// While in OpenGL every state can be changed at (almost) any time, Vulkan requires to layout the graphics (and compute) pipeline states upfront.
    ///
    /// So for each combination of non-dynamic pipeline states you need a new pipeline (there are a few exceptions to this not discussed here).
    ///
    /// Even though this adds a new dimension of planing ahead, it's a great opportunity for performance optimizations by the driver.
    pipeline: vk::Pipeline,

    /// The descriptor set layout describes the shader binding layout (without actually referencing descriptor)
    ///
    /// Like the pipeline layout it's pretty much a blueprint and can be used with different descriptor sets as long as their layout matches.
    descriptor_set_layout: vk::DescriptorSetLayout,

    /// The descriptor set stores the resources bound to the binding points in a shader.
    ///
    /// It connects the binding points of the different shaders with the buffers and images used for those bindings.
    descriptor_set: vk::DescriptorSet,

    // Synchronization primitives
    // Synchronization is an important concept of Vulkan that OpenGL mostly hid away. Getting this right is crucial to using Vulkan.
    /// Semaphores
    ///
    /// Used to coordinate operations within the graphics queue and ensure correct command ordering
    present_complete_semaphore: vk::Semaphore,
    render_complete_semaphore: vk::Semaphore,

    /// Fences
    ///
    /// Used to check the completion of queue operations (e.g. command buffer execution)
    queue_complete_fences: Vec<vk::Fence>,
}

impl Default for GLTFSkinning {
    fn default() -> Self {
        GLTFSkinning {
            vertices: VertexBuffer {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
            },
            indices: IndexBuffer {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                count: 0,
            },
            uniform_buffer_vs: UniformBuffer {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                descriptor: vk::DescriptorBufferInfo::default(),
            },
            ubo_vs: UboVS {
                projection_matrix: glam::Mat4::IDENTITY,
                view_matrix: glam::Mat4::IDENTITY,
                model_matrix: glam::Mat4::IDENTITY,
            },
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_set: vk::DescriptorSet::null(),
            present_complete_semaphore: vk::Semaphore::null(),
            render_complete_semaphore: vk::Semaphore::null(),
            queue_complete_fences: vec![],
        }
    }
}

impl Example for GLTFSkinning {
    fn init(_context: &mut ExampleContext) -> Self {
        Self::default()
    }

    fn prepare(&mut self, context: &mut ExampleContext) -> VkResult<()> {
        if let ExampleContext {
            camera,
            shader_dir,
            backend:
            Some(RenderBackend {
                     vulkan_device: device,
                     draw_cmd_buffers,
                     cmd_pool,
                     render_pass,
                     descriptor_pool,
                     pipeline_cache,
                     ..
                 }),
            ..
        } = context
        {
            self.prepare_synchronization_primitives(device, draw_cmd_buffers)?;
            self.prepare_vertices(device, cmd_pool, USE_STAGING)?;
            self.prepare_uniform_buffers(device)?;
            self.update_uniform_buffers(device, camera)?;
            self.setup_descriptor_set_layout(device)?;
            self.prepare_pipelines(device, shader_dir, render_pass, pipeline_cache)?;
            self.setup_descriptor_pool(device, descriptor_pool)?;
            self.setup_descriptor_set(device, descriptor_pool)?;
            if !DRAW_EVERY_FRAME {
                self.build_command_buffers(context);
            }
        } else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }

        Ok(())
    }

    fn build_command_buffers(&mut self, context: &mut ExampleContext) {
        if let ExampleContext {
            width,
            height,
            backend:
            Some(RenderBackend {
                     vulkan_device: device,
                     draw_cmd_buffers,
                     render_pass,
                     frame_buffers,
                     ..
                 }),
            ..
        } = context
        {
            let cmd_buf_info = vk::CommandBufferBeginInfo::builder();

            // Set clear values for all framebuffer attachments with loadOp set to clear
            // We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.2, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            for (i, draw_cmd_buffer) in draw_cmd_buffers.iter().enumerate() {
                unsafe {
                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(*render_pass)
                        .framebuffer(frame_buffers[i])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: *width,
                                height: *height,
                            },
                        })
                        .clear_values(&clear_values);

                    device
                        .logical_device
                        .begin_command_buffer(*draw_cmd_buffer, &cmd_buf_info)
                        .expect("Failed to begin command buffer");

                    // Start the first sub pass specified in our default render pass setup by the context class
                    // This will clear the color and depth attachment
                    device.logical_device.cmd_begin_render_pass(
                        *draw_cmd_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );

                    // Update dynamic viewport state
                    let viewports = [vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: *width as f32,
                        height: *height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }];
                    device
                        .logical_device
                        .cmd_set_viewport(*draw_cmd_buffer, 0, &viewports);

                    // Update dynamic scissor state
                    let scissors = [vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: *width,
                            height: *height,
                        },
                    }];
                    device
                        .logical_device
                        .cmd_set_scissor(*draw_cmd_buffer, 0, &scissors);

                    // Bind descriptor sets describing shader binding points
                    device.logical_device.cmd_bind_descriptor_sets(
                        *draw_cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.descriptor_set],
                        &[],
                    );

                    // Bind the rendering pipeline
                    // The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
                    device.logical_device.cmd_bind_pipeline(
                        *draw_cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline,
                    );

                    // Bind triangle vertex buffer (contains position and colors)
                    let vertex_buffers = [self.vertices.buffer];
                    device.logical_device.cmd_bind_vertex_buffers(
                        *draw_cmd_buffer,
                        0,
                        &vertex_buffers,
                        &[0],
                    );

                    // Bind triangle index buffer
                    device.logical_device.cmd_bind_index_buffer(
                        *draw_cmd_buffer,
                        self.indices.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    // Draw indexed triangle
                    device.logical_device.cmd_draw_indexed(
                        *draw_cmd_buffer,
                        self.indices.count,
                        1,
                        0,
                        0,
                        1,
                    );

                    device.logical_device.cmd_end_render_pass(*draw_cmd_buffer);

                    // Ending the render pass will add an implicit barrier transitioning the frame buffer color attachment to
                    // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting it to the windowing system
                    device
                        .logical_device
                        .end_command_buffer(*draw_cmd_buffer)
                        .expect("Failed to end command buffer");
                }
            }
        }
    }

    fn setup_depth_stencil(
        depth_format: vk::Format,
        device: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> VkResult<DepthStencil> {
        // Create an optimal image used as the depth stencil attachment

        let image_info = vk::ImageCreateInfo::builder()
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
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.logical_device.create_image(&image_info, None) }?;

        // Allocate memory for the image (device local) and bind it to our image
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

        // Create a view for the depth stencil image
        // Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
        // This allows for multiple views of one image with differing ranges (e.g. for different layers)

        let mut aspect_mask = vk::ImageAspectFlags::DEPTH;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT)
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

    fn setup_render_pass(
        device: &VulkanDevice,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> VkResult<vk::RenderPass> {
        // This example will use a single render pass with one subpass

        // Descriptors for the attachments used by this renderpass
        let attachments: Vec<vk::AttachmentDescription> = vec![
            // Color attachment
            vk::AttachmentDescription::builder()
                .format(color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
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
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ];

        // Setup attachment references
        let color_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let depth_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        // Setup a single subpass reference
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_reference])
            .depth_stencil_attachment(&depth_reference)
            .build();

        // Setup subpass dependencies
        // These will add the implicit attachment layout transitions specified by the attachment descriptions
        // The actual usage layout is preserved through the layout specified in the attachment reference
        // Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
        // srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
        // Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)

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

        let subpasses = [subpass];
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

    fn get_enabled_features(
        _physical_device: &PhysicalDevice,
        _enabled_features: &mut PhysicalDeviceFeatures,
    ) {
    }

    fn get_enabled_extensions(
        _physical_device: &PhysicalDevice,
        _enabled_device_extensions: &mut Vec<&'static CStr>,
    ) {
    }

    fn get_next_chain<T: ExtendsPhysicalDeviceFeatures2>() -> Option<T> {
        None
    }
}

impl GLTFSkinning {
    fn prepare_synchronization_primitives(
        &mut self,
        device: &VulkanDevice,
        draw_cmd_buffers: &Vec<vk::CommandBuffer>,
    ) -> VkResult<()> {
        // Semaphores (Used for correct command ordering)
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        // Semaphore used to ensures that image presentation is complete before starting to submit again
        self.present_complete_semaphore = unsafe {
            device
                .logical_device
                .create_semaphore(&semaphore_create_info, None)?
        };

        // Semaphore used to ensures that all commands submitted have been finished before submitting the image to the queue
        self.render_complete_semaphore = unsafe {
            device
                .logical_device
                .create_semaphore(&semaphore_create_info, None)?
        };

        // Fences (Used to check draw command buffer completion)
        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..draw_cmd_buffers.len() {
            self.queue_complete_fences.push(unsafe {
                device
                    .logical_device
                    .create_fence(&fence_create_info, None)?
            });
        }
        Ok(())
    }

    // Prepare vertex and index buffers for an indexed triangle
    // Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
    fn prepare_vertices(
        &mut self,
        device: &VulkanDevice,
        cmd_pool: &vk::CommandPool,
        use_staging_buffers: bool,
    ) -> VkResult<()> {
        // A note on memory management in Vulkan in general:
        //	This is a very complex topic and while it's fine for an example application to to small individual memory allocations that is not
        //	what should be done a real-world application, where you should allocate large chunks of memory at once instead.

        // Setup vertices
        let vertex_buffer = [
            Vertex {
                pos: Vec4::new(-1.0, 1.0, 0.0, 1.0),
                color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            },
            Vertex {
                pos: Vec4::new(1.0, 1.0, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            },
            Vertex {
                pos: Vec4::new(0.0, -1.0, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
        ];
        let vertex_buffer_size = size_of::<Vertex>() as u64 * vertex_buffer.len() as u64;

        // Setup indices
        let index_buffer = [0u32, 1, 2];
        self.indices.count = index_buffer.len() as u32;
        let index_buffer_size = size_of::<u32>() as u64 * index_buffer.len() as u64;

        if use_staging_buffers {
            // Static data like vertex and index buffer should be stored on the device memory
            // for optimal (and fastest) access by the GPU
            //
            // To achieve this we use so-called "staging buffers" :
            // - Create a buffer that's visible to the host (and can be mapped)
            // - Copy the data to this buffer
            // - Create another buffer that's local on the device (VRAM) with the same size
            // - Copy the data from the host to the device using a command buffer
            // - Delete the host visible (staging) buffer
            // - Use the device local buffers for rendering

            struct StagingBuffer {
                memory: vk::DeviceMemory,
                buffer: vk::Buffer,
            }

            struct StagingBuffers {
                vertices: StagingBuffer,
                indices: StagingBuffer,
            }

            let mut staging_buffers = StagingBuffers {
                vertices: StagingBuffer {
                    memory: vk::DeviceMemory::null(),
                    buffer: vk::Buffer::null(),
                },
                indices: StagingBuffer {
                    memory: vk::DeviceMemory::null(),
                    buffer: vk::Buffer::null(),
                },
            };

            // Vertex buffer
            let mut vertex_buffer_info = vk::BufferCreateInfo::builder()
                .size(vertex_buffer_size)
                // Buffer is used as the copy source
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);

            staging_buffers.vertices.buffer = unsafe {
                device
                    .logical_device
                    .create_buffer(&vertex_buffer_info, None)
                    .unwrap()
            };
            let mem_reqs = unsafe {
                device
                    .logical_device
                    .get_buffer_memory_requirements(staging_buffers.vertices.buffer)
            };
            // Request a host visible memory type that can be used to copy our data do
            // Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer

            let mem_alloc = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    device.memory_properties,
                ));
            staging_buffers.vertices.memory = unsafe {
                device
                    .logical_device
                    .allocate_memory(&mem_alloc, None)
                    .expect("Vertices allocation failed")
            };
            // Map and copy
            unsafe {
                let vertex_ptr = device
                    .logical_device
                    .map_memory(
                        staging_buffers.vertices.memory,
                        0,
                        vertex_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Map memory failed.");
                let mut vertex_slice =
                    Align::new(vertex_ptr, align_of::<Vertex>() as u64, vertex_buffer_size);
                vertex_slice.copy_from_slice(&vertex_buffer);

                device
                    .logical_device
                    .unmap_memory(staging_buffers.vertices.memory);

                device
                    .logical_device
                    .bind_buffer_memory(
                        staging_buffers.vertices.buffer,
                        staging_buffers.vertices.memory,
                        0,
                    )
                    .expect("Binding vertex buffer memory failed.");
            };
            // Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
            vertex_buffer_info = vertex_buffer_info
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER);
            self.vertices.buffer = unsafe {
                device
                    .logical_device
                    .create_buffer(&vertex_buffer_info, None)
                    .expect("Vertex buffer creation failed.")
            };
            let mem_reqs = unsafe {
                device
                    .logical_device
                    .get_buffer_memory_requirements(self.vertices.buffer)
            };
            let mem_alloc = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    device.memory_properties,
                ));
            self.vertices.memory = unsafe {
                device
                    .logical_device
                    .allocate_memory(&mem_alloc, None)
                    .expect("Vertex buffer allocation failed.")
            };
            unsafe {
                device
                    .logical_device
                    .bind_buffer_memory(self.vertices.buffer, self.vertices.memory, 0)
                    .expect("Binding vertex buffer memory failed.");
            };

            // Index buffer
            let mut index_buffer_info = vk::BufferCreateInfo::builder()
                .size(index_buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            // Copy index data to a buffer visible to the host (staging buffer)
            staging_buffers.indices.buffer = unsafe {
                device
                    .logical_device
                    .create_buffer(&index_buffer_info, None)
                    .expect("Index buffer creation failed.")
            };
            let mem_reqs = unsafe {
                device
                    .logical_device
                    .get_buffer_memory_requirements(staging_buffers.indices.buffer)
            };
            let mem_alloc = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    device.memory_properties,
                ));
            staging_buffers.indices.memory = unsafe {
                device
                    .logical_device
                    .allocate_memory(&mem_alloc, None)
                    .expect("Index buffer allocation failed.")
            };
            unsafe {
                device
                    .logical_device
                    .bind_buffer_memory(
                        staging_buffers.indices.buffer,
                        staging_buffers.indices.memory,
                        0,
                    )
                    .expect("Binding index buffer memory failed.");
            }

            // Create destination buffer with device only visibility
            index_buffer_info = index_buffer_info
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER);
            self.indices.buffer = unsafe {
                device
                    .logical_device
                    .create_buffer(&index_buffer_info, None)
                    .expect("Index buffer creation failed.")
            };
            let mem_reqs = unsafe {
                device
                    .logical_device
                    .get_buffer_memory_requirements(self.indices.buffer)
            };
            let mem_alloc = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    device.memory_properties,
                ));
            self.indices.memory = unsafe {
                device
                    .logical_device
                    .allocate_memory(&mem_alloc, None)
                    .expect("Index buffer allocation failed.")
            };
            unsafe {
                let index_ptr = device
                    .logical_device
                    .map_memory(
                        staging_buffers.indices.memory,
                        0,
                        index_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Map memory failed.");

                let mut index_slice =
                    Align::new(index_ptr, align_of::<u32>() as u64, index_buffer_size);
                index_slice.copy_from_slice(&index_buffer);

                device
                    .logical_device
                    .unmap_memory(staging_buffers.indices.memory);

                device
                    .logical_device
                    .bind_buffer_memory(self.indices.buffer, self.indices.memory, 0)
                    .expect("Binding index buffer memory failed.");
            }

            // Buffer copies have to be submitted to a queue, so we need a command buffer for them
            // Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
            let copy_cmd = get_command_buffer(device, *cmd_pool, true)
                .expect("Command buffer allocation failed.");

            // Put buffer region copies into command buffer

            // Vertex buffer
            let copy_region = vk::BufferCopy::builder().size(vertex_buffer_size).build();
            let copy_regions = [copy_region];

            unsafe {
                device.logical_device.cmd_copy_buffer(
                    copy_cmd,
                    staging_buffers.vertices.buffer,
                    self.vertices.buffer,
                    &copy_regions,
                );
            }

            // Index buffer
            let copy_region = vk::BufferCopy::builder().size(index_buffer_size).build();
            let copy_regions = [copy_region];

            unsafe {
                device.logical_device.cmd_copy_buffer(
                    copy_cmd,
                    staging_buffers.indices.buffer,
                    self.indices.buffer,
                    &copy_regions,
                );
            }

            // Flushing the command buffer will also submit it to the queue and uses a fence to ensure that all commands have been executed before returning
            flush_command_buffer(&device, device.queue, *cmd_pool, copy_cmd)
                .expect("Failed to flush command buffer");

            // Destroy staging buffers
            // Note: Staging buffer must not be deleted before the copies have been submitted and executed
            unsafe {
                device
                    .logical_device
                    .destroy_buffer(staging_buffers.vertices.buffer, None);
                device
                    .logical_device
                    .free_memory(staging_buffers.vertices.memory, None);
                device
                    .logical_device
                    .destroy_buffer(staging_buffers.indices.buffer, None);
                device
                    .logical_device
                    .free_memory(staging_buffers.indices.memory, None);
            }
        } else {
            // Don't use staging
            // Create host-visible buffers only and use these for rendering. This is not advised and will usually result in lower rendering performance

            // Vertex buffer
            let vertex_buffer_info = vk::BufferCreateInfo::builder()
                .size(vertex_buffer_size)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER);

            // Copy vertex data to a buffer visible to the host
            self.vertices.buffer = unsafe {
                device
                    .logical_device
                    .create_buffer(&vertex_buffer_info, None)
                    .expect("Vertex buffer creation failed.")
            };
            let mem_reqs = unsafe {
                device
                    .logical_device
                    .get_buffer_memory_requirements(self.vertices.buffer)
            };
            let mem_alloc = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    device.memory_properties,
                ));
            self.vertices.memory = unsafe {
                device
                    .logical_device
                    .allocate_memory(&mem_alloc, None)
                    .expect("Vertex buffer allocation failed.")
            };
            unsafe {
                let vertex_ptr = device
                    .logical_device
                    .map_memory(
                        self.vertices.memory,
                        0,
                        vertex_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Map memory failed.");

                let mut vertex_slice =
                    Align::new(vertex_ptr, align_of::<Vertex>() as u64, vertex_buffer_size);
                vertex_slice.copy_from_slice(&vertex_buffer);

                device.logical_device.unmap_memory(self.vertices.memory);

                device
                    .logical_device
                    .bind_buffer_memory(self.vertices.buffer, self.vertices.memory, 0)
                    .expect("Binding vertex buffer memory failed.");
            }

            // Index buffer
            let index_buffer_info = vk::BufferCreateInfo::builder()
                .size(index_buffer_size)
                .usage(vk::BufferUsageFlags::INDEX_BUFFER);

            // Copy index data to a buffer visible to the host
            self.indices.buffer = unsafe {
                device
                    .logical_device
                    .create_buffer(&index_buffer_info, None)
                    .expect("Index buffer creation failed.")
            };
            let mem_reqs = unsafe {
                device
                    .logical_device
                    .get_buffer_memory_requirements(self.indices.buffer)
            };
            let mem_alloc = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_reqs.size)
                .memory_type_index(get_memory_type_index(
                    mem_reqs.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    device.memory_properties,
                ));
            self.indices.memory = unsafe {
                device
                    .logical_device
                    .allocate_memory(&mem_alloc, None)
                    .expect("Index buffer allocation failed.")
            };
            unsafe {
                let index_ptr = device
                    .logical_device
                    .map_memory(
                        self.indices.memory,
                        0,
                        index_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Map memory failed.");

                let mut index_slice =
                    Align::new(index_ptr, align_of::<u32>() as u64, index_buffer_size);
                index_slice.copy_from_slice(&index_buffer);

                device.logical_device.unmap_memory(self.indices.memory);

                device
                    .logical_device
                    .bind_buffer_memory(self.indices.buffer, self.indices.memory, 0)
                    .expect("Binding index buffer memory failed.");
            }
        }
        Ok(())
    }

    fn prepare_uniform_buffers(&mut self, device: &VulkanDevice) -> VkResult<()> {
        // Prepare and initialize a uniform buffer block containing shader uniforms
        // Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks

        // Vertex shader uniform buffer block
        let mut alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(0)
            .memory_type_index(0);
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size_of::<UboVS>() as u64)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);

        // Create a new buffer
        let buffer = unsafe { device.logical_device.create_buffer(&buffer_info, None)? };

        // Get memory requirements including size, alignment and memory type
        let mem_reqs = unsafe { device.logical_device.get_buffer_memory_requirements(buffer) };
        alloc_info = alloc_info.allocation_size(mem_reqs.size);
        // Get the memory type index that supports host visible memory access
        // Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
        // We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
        // Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular context
        alloc_info = alloc_info.memory_type_index(get_memory_type_index(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device.memory_properties,
        ));
        // Allocate memory for the uniform buffer
        let memory = unsafe { device.logical_device.allocate_memory(&alloc_info, None)? };
        // Bind memory to buffer
        unsafe {
            device
                .logical_device
                .bind_buffer_memory(buffer, memory, 0)?;
        }

        // Store information in the uniform's descriptor that is used by the descriptor set
        let descriptor = vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(0)
            .range(size_of::<UboVS>() as u64)
            .build();

        self.uniform_buffer_vs = UniformBuffer {
            buffer,
            memory,
            descriptor,
        };

        Ok(())
    }

    fn update_uniform_buffers(&mut self, device: &VulkanDevice, camera: &Camera) -> VkResult<()> {
        self.ubo_vs = UboVS {
            projection_matrix: camera.matrices.perspective,
            view_matrix: camera.matrices.view,
            model_matrix: glam::Mat4::IDENTITY,
        };

        // Map uniform buffer and update it
        unsafe {
            let data_ptr = device.logical_device.map_memory(
                self.uniform_buffer_vs.memory,
                0,
                size_of::<UboVS>() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )?;
            // data_ptr.copy_from_nonoverlapping(&self.ubo_vs, 1);
            let mut x = Align::new(
                data_ptr,
                align_of::<UboVS>() as u64,
                size_of::<UboVS>() as u64,
            );
            x.copy_from_slice(std::slice::from_ref(&self.ubo_vs));
            device
                .logical_device
                .unmap_memory(self.uniform_buffer_vs.memory);
        }

        Ok(())
    }

    fn setup_descriptor_set_layout(&mut self, device: &VulkanDevice) -> VkResult<()> {
        // Setup layout of descriptors used in this example
        // Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
        // So every shader binding should map to one descriptor set layout binding

        // Binding 0: Uniform buffer (Vertex shader)
        let layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()];

        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        let descriptor_set_layouts = [unsafe {
            device
                .logical_device
                .create_descriptor_set_layout(&descriptor_layout_info, None)?
        }];

        // Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
        // In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

        let pipeline_layout = unsafe {
            device
                .logical_device
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        self.descriptor_set_layout = descriptor_set_layouts[0];
        self.pipeline_layout = pipeline_layout;

        Ok(())
    }

    fn prepare_pipelines(
        &mut self,
        device: &VulkanDevice,
        shader_dir: &String,
        render_pass: &vk::RenderPass,
        pipeline_cache: &vk::PipelineCache,
    ) -> VkResult<()> {
        // Create the graphics pipeline used in this example
        // Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
        // A pipeline is then stored and hashed on the GPU making pipeline changes very fast
        // Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

        let mut pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            // The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
            .layout(self.pipeline_layout)
            // Renderpass this pipeline is attached to
            .render_pass(*render_pass);

        // Construct the different states making up the pipeline

        // Input assembly state describes how primitives are assembled
        // This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Rasterization state
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .depth_bias_enable(false)
            .line_width(1.0);

        // Color blend state describes how blend factors are calculated (if used)
        // We need one blend attachment state per color attachment (even if blending is not used
        let blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA) //That's very important, because this may cause the color to be transparent
            .build()];
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachment_states);

        // Viewport state sets the number of viewports and scissor used in this pipeline
        // Note: This is actually overridden by the dynamic states (see below)
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        // Enable dynamic states
        // Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
        // To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
        // For this example we will set the viewport and scissor using dynamic states
        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        // Depth and stencil state containing depth and stencil compare and test operations
        // We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
        let mut depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .back(
                vk::StencilOpState::builder()
                    .fail_op(vk::StencilOp::KEEP)
                    .pass_op(vk::StencilOp::KEEP)
                    .compare_op(vk::CompareOp::ALWAYS)
                    .build(),
            );
        let front = depth_stencil_state.back.clone();
        depth_stencil_state = depth_stencil_state.front(front);

        // Multi sampling state
        // This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Vertex input descriptions
        // Specifies the vertex input parameters for a pipeline

        // Vertex input binding
        // This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
        let vertex_input_bindings = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];

        // Input attribute bindings describe shader attribute locations and memory layouts
        let vertex_input_attributes = [
            // These match the following shader layout (see triangle.vert):
            //	layout (location = 0) in vec3 inPos;
            //	layout (location = 1) in vec3 inColor;
            // Attribute location 0: Position
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32)
                .build(),
            // Attribute location 1: Color
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32)
                .build(),
        ];

        // Vertex input state used for pipeline creation
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_input_bindings)
            .vertex_attribute_descriptions(&vertex_input_attributes);

        let mut vert_shader_path = vulkan_rs::tools::get_shader_path();
        vert_shader_path.push(shader_dir.clone() + "/triangle/triangle.vert.spv");
        let mut frag_shader_path = vulkan_rs::tools::get_shader_path();
        frag_shader_path.push(shader_dir.clone() + "/triangle/triangle.frag.spv");
        let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        // Shaders
        let shader_stages = [
            // Vertex shader
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(
                    load_spirv_shader(device, &vert_shader_path)
                        .expect("Failed to load vertex shader"),
                )
                .name(&name)
                .build(),
            // Fragment shader
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(
                    load_spirv_shader(device, &frag_shader_path)
                        .expect("Failed to load fragment shader"),
                )
                .name(&name)
                .build(),
        ];

        // Set pipeline shader stage info
        pipeline_create_info = pipeline_create_info
            .stages(&shader_stages)
            // Assign the pipeline states to the pipeline creation info structure
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .rasterization_state(&rasterization_state)
            .color_blend_state(&color_blend_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .viewport_state(&viewport_state)
            .dynamic_state(&dynamic_state_info);

        let create_infos = [pipeline_create_info.build()];

        // Create rendering pipeline using the specified states
        let pipelines = unsafe {
            device
                .logical_device
                .create_graphics_pipelines(*pipeline_cache, &create_infos, None)
        }
            .expect("Failed to create graphics pipeline");

        // Shader modules are no longer needed once the graphics pipeline has been created
        unsafe {
            device
                .logical_device
                .destroy_shader_module(shader_stages[0].module, None);
            device
                .logical_device
                .destroy_shader_module(shader_stages[1].module, None);
        }
        self.pipeline = pipelines[0];

        Ok(())
    }

    fn setup_descriptor_pool(
        &mut self,
        device: &VulkanDevice,
        descriptor_pool: &mut vk::DescriptorPool,
    ) -> VkResult<()> {
        // We need to tell the API the number of max. requested descriptors per type
        let pool_sizes = [
            // This example only uses one descriptor type (uniform buffer) and only requests one descriptor of this type
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        ];

        // Create the global descriptor pool
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            // Set the max. number of sets that can be requested
            .max_sets(1);

        *descriptor_pool = unsafe {
            device
                .logical_device
                .create_descriptor_pool(&descriptor_pool_info, None)?
        };

        Ok(())
    }

    fn setup_descriptor_set(
        &mut self,
        device: &VulkanDevice,
        descriptor_pool: &vk::DescriptorPool,
    ) -> VkResult<()> {
        let descriptor_set_layouts = [self.descriptor_set_layout];

        // Allocate a new descriptor set from the global descriptor pool
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(*descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_set = unsafe {
            device
                .logical_device
                .allocate_descriptor_sets(&alloc_info)?[0]
        };

        // Update the descriptor set determining the shader binding points
        // For every binding point used in a shader there needs to be one
        // descriptor set matching that binding point

        let write_descriptor_sets = [
            // Binding 0 : Uniform buffer
            //	Variable descriptor count using descriptorCount and pImageInfo
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&[self.uniform_buffer_vs.descriptor])
                .build(),
        ];

        unsafe {
            device
                .logical_device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        };
        self.descriptor_set = descriptor_set;

        Ok(())
    }

    fn record_submit_command_buffer(&mut self, context: &mut ExampleContext) {
        if let ExampleContext {
            width,
            height,
            backend:
            Some(RenderBackend {
                     vulkan_device: device,
                     draw_cmd_buffers,
                     render_pass,
                     frame_buffers,
                     current_buffer,
                     ..
                 }),
            ..
        } = context
        {
            // Use a fence to wait until the command buffer has finished execution before using it again
            unsafe {
                device
                    .logical_device
                    .wait_for_fences(
                        std::slice::from_ref(&self.queue_complete_fences[*current_buffer as usize]),
                        true,
                        u64::MAX,
                    )
                    .expect("Failed to wait for fence");
                device
                    .logical_device
                    .reset_fences(std::slice::from_ref(
                        &self.queue_complete_fences[*current_buffer as usize],
                    ))
                    .expect("Failed to reset fences");
            }

            let cmd_buf_info = vk::CommandBufferBeginInfo::builder();

            // Set clear values for all framebuffer attachments with loadOp set to clear
            // We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let i = *current_buffer as usize;
            let draw_cmd_buffer = draw_cmd_buffers[i];
            unsafe {
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(*render_pass)
                    .framebuffer(frame_buffers[i])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: *width,
                            height: *height,
                        },
                    })
                    .clear_values(&clear_values);

                device
                    .logical_device
                    .begin_command_buffer(draw_cmd_buffer, &cmd_buf_info)
                    .expect("Failed to begin command buffer");

                // Start the first sub pass specified in our default render pass setup by the context class
                // This will clear the color and depth attachment
                device.logical_device.cmd_begin_render_pass(
                    draw_cmd_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                // Update dynamic viewport state
                let viewports = [vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: *width as f32,
                    height: *height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }];
                device
                    .logical_device
                    .cmd_set_viewport(draw_cmd_buffer, 0, &viewports);

                // Update dynamic scissor state
                let scissors = [vk::Extent2D {
                    width: *width,
                    height: *height,
                }
                    .into()];
                device
                    .logical_device
                    .cmd_set_scissor(draw_cmd_buffer, 0, &scissors);

                // Bind descriptor sets describing shader binding points
                device.logical_device.cmd_bind_descriptor_sets(
                    draw_cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );

                // Bind the rendering pipeline
                // The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
                device.logical_device.cmd_bind_pipeline(
                    draw_cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline,
                );

                // Bind triangle vertex buffer (contains position and colors)
                let vertex_buffers = [self.vertices.buffer];
                device.logical_device.cmd_bind_vertex_buffers(
                    draw_cmd_buffer,
                    0,
                    &vertex_buffers,
                    &[0],
                );

                // Bind triangle index buffer
                device.logical_device.cmd_bind_index_buffer(
                    draw_cmd_buffer,
                    self.indices.buffer,
                    0,
                    vk::IndexType::UINT32,
                );

                // Draw indexed triangle
                device.logical_device.cmd_draw_indexed(
                    draw_cmd_buffer,
                    self.indices.count,
                    1,
                    0,
                    0,
                    1,
                );

                device.logical_device.cmd_end_render_pass(draw_cmd_buffer);

                // Ending the render pass will add an implicit barrier transitioning the frame buffer color attachment to
                // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting it to the windowing system
                device
                    .logical_device
                    .end_command_buffer(draw_cmd_buffer)
                    .expect("Failed to end command buffer");
            }
        }
    }

    fn draw(&mut self, context: &mut ExampleContext) {
        let res = context
            .backend
            .as_mut()
            .unwrap()
            .swapchain
            .acquire_next_image(self.present_complete_semaphore);
        match res {
            Ok((index, _)) => {
                context.backend.as_mut().unwrap().current_buffer = index;
            }
            _ => {
                panic!("Failed to acquire next image");
            }
        };

        if DRAW_EVERY_FRAME {
            self.record_submit_command_buffer(context);
        }

        if let ExampleContext {
            backend:
            Some(RenderBackend {
                     vulkan_device: device,
                     draw_cmd_buffers,
                     swapchain,
                     current_buffer,
                     ..
                 }),
            ..
        } = context
        {
            if !DRAW_EVERY_FRAME {
                // Use a fence to wait until the command buffer has finished execution before using it again
                unsafe {
                    device
                        .logical_device
                        .wait_for_fences(
                            std::slice::from_ref(
                                &self.queue_complete_fences[*current_buffer as usize],
                            ),
                            true,
                            u64::MAX,
                        )
                        .expect("Failed to wait for fence");
                    device
                        .logical_device
                        .reset_fences(std::slice::from_ref(
                            &self.queue_complete_fences[*current_buffer as usize],
                        ))
                        .expect("Failed to reset fences");
                }
            }
            // Pipeline stage at which the queue submission will wait (via pWaitSemaphores)
            let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            // The submit info structure specifies a command buffer queue submission batch
            let submit_info = vk::SubmitInfo::builder()
                .wait_dst_stage_mask(&wait_dst_stage_mask)
                // Submit the currently active command buffer
                .command_buffers(&[draw_cmd_buffers[*current_buffer as usize]])
                // Semaphore(s) to wait upon before the submitted command buffer starts executing
                .wait_semaphores(&[self.present_complete_semaphore])
                // Semaphore(s) to be signaled when command buffers have completed
                .signal_semaphores(&[self.render_complete_semaphore])
                .build();

            // Submit to the graphics queue passing a wait fence
            unsafe {
                device
                    .logical_device
                    .queue_submit(
                        device.queue,
                        &[submit_info],
                        self.queue_complete_fences[*current_buffer as usize],
                    )
                    .expect("Failed to execute queue submit.");
            }

            // Present the current buffer to the swap chain
            // Pass the semaphore signaled by the command buffer submission from the submit info as the wait semaphore for swap chain presentation
            // This ensures that the image is not presented to the windowing system until all commands have been submitted
            match swapchain.queue_present(
                device.queue,
                *current_buffer,
                self.render_complete_semaphore,
            ) {
                Ok(_) => {}
                _ => {
                    panic!("Failed to present queue");
                }
            }
        }
    }
}

// This function is used to request a device memory type that supports all the property flags we request (e.g. device local, host visible)
// Upon success it will return the index of the memory type that fits our requested memory properties
// This is necessary as implementations can offer an arbitrary number of memory types with different
// memory properties.
// You can check http://vulkan.gpuinfo.org/ for details on different memory configurations
fn get_memory_type_index(
    mut type_bits: u32,
    properties: vk::MemoryPropertyFlags,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
) -> u32 {
    // Iterate over all memory types available for the device used in this example
    for i in 0..device_memory_properties.memory_type_count {
        if (type_bits & 1) == 1 {
            if (device_memory_properties.memory_types[i as usize].property_flags & properties)
                == properties
            {
                return i;
            }
        }
        type_bits = type_bits >> 1;
    }

    panic!("Could not find a suitable memory type!");
}

fn get_command_buffer(
    device: &VulkanDevice,
    cmd_pool: vk::CommandPool,
    begin: bool,
) -> VkResult<vk::CommandBuffer> {
    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe {
        device
            .logical_device
            .allocate_command_buffers(&cmd_buf_allocate_info)?
    };
    if begin {
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .logical_device
                .begin_command_buffer(cmd_buffers[0], &cmd_buf_begin_info)?;
        }
    }
    Ok(cmd_buffers[0])
}

fn flush_command_buffer(
    device: &VulkanDevice,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
) -> VkResult<()> {
    unsafe {
        device.logical_device.end_command_buffer(command_buffer)?;
    }

    let cmd_bufs = [command_buffer];
    let cmd_submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);

    // Create fence to ensure that the command buffer has finished executing
    let fence_ci = vk::FenceCreateInfo::builder();
    let fence = unsafe { device.logical_device.create_fence(&fence_ci, None)? };

    // Submit to the queue
    unsafe {
        device
            .logical_device
            .queue_submit(queue, &[cmd_submit_info.build()], fence)?;
    };
    // Wait for the fence to signal that command buffer has finished executing
    unsafe {
        device.logical_device.wait_for_fences(
            &[fence],
            true,
            vulkan_rs::tools::DEFAULT_FENCE_TIMEOUT,
        )?;
    }

    unsafe {
        device.logical_device.destroy_fence(fence, None);
    }

    unsafe {
        device
            .logical_device
            .free_command_buffers(cmd_pool, &cmd_bufs);
    }

    Ok(())
}

fn load_spirv_shader(device: &VulkanDevice, path: &std::path::Path) -> VkResult<vk::ShaderModule> {
    let mut spv_file = std::fs::File::open(path).expect("Failed to load SPIR-V shader");
    let spv_bytes = ash::util::read_spv(&mut spv_file).expect("Failed to load SPIR-V shader");
    let shader_info = vk::ShaderModuleCreateInfo::builder().code(&spv_bytes);
    unsafe {
        device
            .logical_device
            .create_shader_module(&shader_info, None)
    }
}

fn main() -> VkResult<()> {
    let width = 1280;
    let height = 720;
    let mut camera = Camera::new();
    camera.camera_type = CameraType::LookAt;
    camera.set_position(Vec3::new(0.0, 0.0, -1.5));
    camera.set_rotation(Vec3::new(0.0, 0.0, 0.0));
    camera.set_perspective(90.0, width as f32 / height as f32, 1.0, 256.0);
    let mut context: ExampleContext = ExampleContext::builder::<GLTFSkinning>()
        .title("Vulkan Example - Basic indexed triangle")
        .settings(Settings {
            validation: true,
            fullscreen: None,
            vsync: false,
            overlay: false,
        })
        .width(width)
        .height(height)
        .camera(camera)
        .build::<PhantomFeatures>()?;
    let mut example = GLTFSkinning::init(&mut context);
    let mut frame_fn = |context: &mut ExampleContext| {
        if !context.prepared {
            example.prepare(context).expect("Failed to prepare example");
            context.prepared = true;
        }
        example.draw(context);
    };
    context.render_loop(&mut frame_fn)
}
