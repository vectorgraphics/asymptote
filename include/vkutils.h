#pragma once

#if defined(HAVE_VULKAN)

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