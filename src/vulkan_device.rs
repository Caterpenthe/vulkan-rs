use ash::prelude::VkResult;
use ash::vk::{self, PhysicalDeviceMemoryProperties, PhysicalDeviceProperties, QueueFlags};
use ash::{Device, Instance};
use std::ffi::CStr;
use std::ops::BitOr;
use std::sync::Arc;

pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
    pub transfer: u32,
}

pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

pub struct VulkanDevice {
    /// Physical device representation */
    pub physical_device: vk::PhysicalDevice,
    /// Logical device representation (application's view of the device) */
    pub logical_device: Arc<Device>,
    /// Handle to the device graphics queue that command buffers are submitted to
    pub queue: vk::Queue,
    /// Properties of the physical device including limits that the application can check against */
    properties: vk::PhysicalDeviceProperties,
    /// Features of the physical device that an application can use to check if a feature is supported */
    features: vk::PhysicalDeviceFeatures,
    /// Features that have been enabled for use on the physical device */
    enabled_features: vk::PhysicalDeviceFeatures,
    /// Memory types and heaps of the physical device */
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Queue family properties of the physical device */
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
    /// List of extensions supported by the device */
    supported_extensions: Vec<&'static CStr>,
    /// Default command pool for the graphics queue family index */
    command_pool: vk::CommandPool,
    /// Set to true when the debug marker extension is detected */
    enable_debug_markers: bool,
    /// Contains queue family indices */
    pub queue_family_indices: QueueFamilyIndices,
}

impl VulkanDevice {
    pub fn create<T: vk::ExtendsPhysicalDeviceFeatures2>(
        instance: Arc<Instance>,
        physical_device: vk::PhysicalDevice,

        enabled_features: &vk::PhysicalDeviceFeatures,
        enabled_extensions: &mut Vec<&CStr>,
        next_chain: Option<&mut T>,

        requested_queue_types: Option<vk::QueueFlags>,
        use_swap_chain: bool,
    ) -> VkResult<Arc<Self>> {
        let properties;
        let features;
        let memory_properties;
        let queue_family_properties: Vec<vk::QueueFamilyProperties>;
        let mut enable_debug_markers = false;
        let logical_device;
        let command_pool;
        let queue;

        let queue_family_indices = QueueFamilyIndices {
            graphics: 0,
            compute: 0,
            transfer: 0,
        };

        // Store Properties features, limits and properties of the physical device for later use
        // Device properties also contain limits and sparse properties
        properties = unsafe { instance.get_physical_device_properties(physical_device) };

        // Features should be checked by the examples before using them
        features = unsafe { instance.get_physical_device_features(physical_device) };

        // Memory properties are used regularly for creating all kinds of buffers
        memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Queue family properties, used for setting up requested queues upon device creation
        queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        // Get list of supported extensions
        let supported_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }?
                .iter()
                .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
                .collect::<Vec<_>>();

        // Desired queues need to be requested upon logical device creation
        // Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
        // requests different queue types
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = vec![];
        // Get queue family indices for the requested queue family types
        // Note that the indices may overlap depending on the implementation
        let default_queue_priority = 0.0f32;

        // Graphics queue
        let mut requested_queue_types = requested_queue_types;
        if requested_queue_types.is_none() {
            requested_queue_types = Some(vk::QueueFlags::GRAPHICS.bitor(vk::QueueFlags::COMPUTE));
        };
        let requested_queue_types = requested_queue_types.unwrap();

        let mut queue_family_index: QueueFamilyIndices = QueueFamilyIndices {
            graphics: 0,
            compute: 0,
            transfer: 0,
        };

        if requested_queue_types.contains(vk::QueueFlags::GRAPHICS) {
            queue_family_index.graphics =
                get_queue_family_index(vk::QueueFlags::GRAPHICS, &queue_family_properties);

            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index.graphics)
                .queue_priorities(std::slice::from_ref(&default_queue_priority))
                .build();
            queue_create_infos.push(queue_create_info);
        } else {
            queue_family_index.graphics = 0;
        }

        // Dedicated compute queue
        if requested_queue_types.contains(vk::QueueFlags::COMPUTE) {
            queue_family_index.compute =
                get_queue_family_index(vk::QueueFlags::COMPUTE, &queue_family_properties);
            if queue_family_index.compute != queue_family_index.graphics {
                // If compute family index differs, we need an additional queue create info for the compute queue
                let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_index.compute)
                    .queue_priorities(std::slice::from_ref(&default_queue_priority))
                    .build();
                queue_create_infos.push(queue_create_info);
            }
        } else {
            // Else we use the same queue
            queue_family_index.compute = queue_family_index.graphics;
        }
        // Dedicated transfer queue
        if requested_queue_types.contains(vk::QueueFlags::TRANSFER) {
            queue_family_index.transfer =
                get_queue_family_index(vk::QueueFlags::TRANSFER, &queue_family_properties);
            if queue_family_index.transfer != queue_family_index.graphics
                && queue_family_index.transfer != queue_family_index.compute
            {
                // If compute family index differs, we need an additional queue create info for the compute queue
                let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_index.transfer)
                    .queue_priorities(std::slice::from_ref(&default_queue_priority))
                    .build();
                queue_create_infos.push(queue_create_info);
            }
        } else {
            // Else we use the same queue
            queue_family_index.transfer = queue_family_index.graphics;
        }

        // Create the logical device representation
        let mut device_extensions = enabled_extensions;
        if use_swap_chain {
            // If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
            device_extensions.push(ash::extensions::khr::Swapchain::name());
        }

        let extension_names = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        let mut device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(enabled_features)
            .enabled_extension_names(extension_names.as_slice());

        // If a pNext(Chain) has been passed, we need to add it to the device creation info
        let mut physical_device_features2: vk::PhysicalDeviceFeatures2;
        if let Some(next_chain) = next_chain {
            physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
                .features(*enabled_features)
                .push_next(next_chain)
                .build();
            device_create_info = device_create_info.push_next(&mut physical_device_features2);
        }

        // Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
        if extension_supported(
            ash::extensions::ext::DebugUtils::name(),
            &supported_extensions,
        ) {
            device_extensions.push(ash::extensions::ext::DebugUtils::name());
            enable_debug_markers = true;
        }

        let extension_names;
        if device_extensions.len() > 0 {
            for ext in device_extensions.iter() {
                if !supported_extensions.contains(ext) {
                    panic!(
                        "Enabled device extension {} is not present at device level",
                        ext.to_str().unwrap()
                    );
                }
            }

            extension_names = device_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            device_create_info =
                device_create_info.enabled_extension_names(extension_names.as_slice());
        }

        logical_device = Arc::new(unsafe {
            instance.create_device(physical_device, &device_create_info, None)?
        });

        // Create a default command pool for graphics command buffers
        command_pool = create_command_pool(queue_family_index.graphics, None, &*logical_device)?;

        // Get a graphics queue from the device
        queue = unsafe { logical_device.get_device_queue(queue_family_indices.graphics, 0) };

        Ok(Arc::new(VulkanDevice {
            physical_device,
            logical_device,
            queue,
            properties,
            features,
            enabled_features: enabled_features.clone(),
            memory_properties,
            queue_family_properties,
            supported_extensions,
            command_pool,
            enable_debug_markers,
            queue_family_indices,
        }))
    }

    pub fn get_memory_type(
        &self,
        type_bits: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> VkResult<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_bits & (1 << i)) > 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }
        Err(vk::Result::ERROR_UNKNOWN)
    }
}

fn create_command_pool(
    queue_family_index: u32,
    create_flags: Option<vk::CommandPoolCreateFlags>,
    logical_device: &ash::Device,
) -> VkResult<vk::CommandPool> {
    let mut create_flags = create_flags;
    if create_flags.is_none() {
        create_flags = Some(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    }
    let create_flags = create_flags.unwrap();

    let cmd_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .flags(create_flags)
        .build();

    unsafe { logical_device.create_command_pool(&cmd_pool_info, None) }
}

fn extension_supported(extension: &CStr, supported_extensions: &Vec<&'static CStr>) -> bool {
    for supported_extension in supported_extensions.iter() {
        if *supported_extension == extension {
            return true;
        }
    }
    false
}

fn get_queue_family_index(
    queue_flags: QueueFlags,
    queue_family_properties: &Vec<vk::QueueFamilyProperties>,
) -> u32 {
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS) {
        for (i, properties) in queue_family_properties.iter().enumerate() {
            if properties.queue_flags.contains(QueueFlags::COMPUTE)
                && !properties.queue_flags.contains(QueueFlags::GRAPHICS)
            {
                return i as u32;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if queue_flags.contains(QueueFlags::TRANSFER)
        && !queue_flags.contains(QueueFlags::GRAPHICS)
        && !queue_flags.contains(QueueFlags::COMPUTE)
    {
        for (i, properties) in queue_family_properties.iter().enumerate() {
            if properties.queue_flags.contains(QueueFlags::TRANSFER)
                && !properties.queue_flags.contains(QueueFlags::GRAPHICS)
                && !properties.queue_flags.contains(QueueFlags::COMPUTE)
            {
                return i as u32;
            }
        }
    }

    // For other queue types or if no separate compute queue is present, return the first one to support the requested flags
    for (i, properties) in queue_family_properties.iter().enumerate() {
        if properties.queue_flags.contains(queue_flags) {
            return i as u32;
        }
    }

    panic!("Could not find a matching queue family index")
}
