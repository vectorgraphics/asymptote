/*****
 * webglrender.cc
 * Lightweight renderer for WebGL (html) and v3d output.
 * Does not require Vulkan or OpenGL libraries.
 *****/

#include "common.h"

#ifdef HAVE_LIBGLM

#include "webglrender.h"

using namespace glm;

namespace camp
{

void AsyWebGLRender::render(RenderFunctionArgs const& args)
{
  copyRenderArgs(args);

  fullWidth = (int)ceil(args.width);
  fullHeight = (int)ceil(args.height);

  Width = fullWidth;
  Height = fullHeight;

  // Initialize camera state so that getProjViewMat() returns a valid matrix.
  // This is needed for bbox2::Bounds which uses Transform2T(getProjViewMat(), v).
  X = Y = cx = cy = 0;
  rotateMat = dmat4(1.0);
  Zoom = Zoom0;

  double cz = 0.5 * (Zmin + Zmax);
  viewMat = translate(translate(dmat4(1.0), dvec3(cx, cy, cz)) * rotateMat, dvec3(0, 0, -cz));

  setProjection();
  updateModelViewData();
}

} // namespace camp

#endif // HAVE_LIBGLM
