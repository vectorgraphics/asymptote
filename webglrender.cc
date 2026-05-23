/*****
 * webglrender.cc
 * Lightweight renderer for WebGL (html) and v3d output.
 * Does not require Vulkan or OpenGL libraries.
 *****/

#include "common.h"

#ifdef HAVE_LIBGLM

#include "webglrender.h"

namespace camp
{

void AsyWebGLRender::render(RenderFunctionArgs const& args)
{
  copyRenderArgs(args);

  fullWidth = (int)ceil(args.width);
  fullHeight = (int)ceil(args.height);
}

} // namespace camp

#endif // HAVE_LIBGLM
