#pragma once

#include "common.h"

#if !defined(FOR_SHARED) && defined(HAVE_LIBGLM) && \
  defined(HAVE_LIBGLFW) && (defined(HAVE_LIBVULKAN) || defined(HAVE_LIBGL) || defined(HAVE_LIBOSMESA))

namespace camp {

bool tryLoadVulkan();
bool tryLoadOpenGL();
void createRenderer();
void initRenderer(const char* format = nullptr);  // Format-aware initialization
void unloadVulkan();
void unloadOpenGL();

} // namespace camp

#endif
