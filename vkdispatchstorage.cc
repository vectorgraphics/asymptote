/**
 * @file vkdispatchstorage.cc
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief Storage for vulkan dynamic loader, as required by
 *        https://github.com/KhronosGroup/Vulkan-Hpp#extensions--per-device-function-pointers
 */

#include "vk.h"

#if defined(HAVE_VULKAN)
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif
