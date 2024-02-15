#pragma once

#if defined(HAVE_VULKAN)

// We undefined NDEBUG for common.h, but some files
// do not use common.h, causing includes
// to be a mix of NDEBUG-ed vulkan header and those without
#undef NDEBUG
#include <vulkan/vulkan.hpp>

namespace camp
{

namespace vkutils
{

/** Asserts vulkan result is successful, or throws error otherwise */
void checkVkResult(vk::Result const& result);

}
}
#endif
