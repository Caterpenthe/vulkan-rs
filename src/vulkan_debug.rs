pub mod debug {
    use ash::{Entry, vk};
    pub use ash::Instance;
    use std::borrow::Cow;
    use std::ffi::CStr;
    use std::os::raw::c_char;
    use ash::extensions::ext::DebugUtils;
    use ash::prelude::VkResult;

    pub fn message_callback(
        flags: vk::DebugReportFlagsEXT,
        obj_type: vk::DebugReportObjectTypeEXT,
        src_object: u64,
        location: usize,
        msg_code: i32,
        layer_prefix: *const c_char,
        msg: *const c_char,
        user_data: *mut std::ffi::c_void,
    ) -> u32 {
        let message = unsafe { CStr::from_ptr(msg) }.to_str().unwrap();

        #[allow(clippy::if_same_then_else)]
        if message.starts_with("Validation Error: [ VUID-VkWriteDescriptorSet-descriptorType-00322")
            || message
                .starts_with("Validation Error: [ VUID-VkWriteDescriptorSet-descriptorType-02752")
        {
            // Validation layers incorrectly report an error in pushing immutable sampler descriptors.
            //
            // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushDescriptorSetKHR.html
            // This documentation claims that it's necessary to push immutable samplers.
        } else if message.starts_with("Validation Performance Warning") {
        } else if message.starts_with("Validation Warning: [ VUID_Undefined ]") {
            log::warn!("{}\n", message);
        } else {
            log::error!("{}\n", message);
        }

        ash::vk::FALSE
    }
    pub fn setup_debugging(entry: &Entry, instance: &Instance) -> VkResult<vk::DebugUtilsMessengerEXT> {
        let debug_call_back;
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(crate::vulkan_debug::debug::vulkan_debug_callback));

        let debug_utils_loader = DebugUtils::new(&entry, &instance);
        unsafe {
            debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)?;
        }
        Ok(debug_call_back)
    }
    pub fn free_debug_callback(instacne: Instance) {}
    pub unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }
}
