#pragma once

#include "common.h"

#if !defined(FOR_SHARED) && defined(HAVE_LIBGLM) && \
  defined(HAVE_LIBGLFW) && (defined(HAVE_LIBVULKAN) || defined(HAVE_LIBGL) || defined(HAVE_LIBOSMESA))

namespace camp {

/**
 * Dynamically probe for Vulkan availability at runtime.
 * Returns true if the Vulkan loader library can be loaded and
 * vkCreateInstance is resolvable.  On success, the Vulkan dispatch
 * table is initialised so that AsyVkRender can use it without
 * linking -lvulkan at load time.
 *
 * If this returns false the application should fall back to OpenGL.
 */
bool tryLoadVulkan();

/**
 * Release the dynamically-loaded Vulkan library (if any).
 */
void unloadVulkan();

} // namespace camp

#endif
