/*****
 * webglrender.h
 * Lightweight renderer for WebGL (html) and v3d output.
 * Does not require Vulkan or OpenGL libraries.
 *****/

#pragma once

#include "common.h"
#include "renderBase.h"

namespace camp
{

/**
 * AsyWebGLRender - A minimal renderer for WebGL/v3d export.
 *
 * This class handles the initial setup phase for WebGL (html) and v3d output
 * formats. It sets up dimensions, camera parameters, lighting, and other state
 * needed by jsfile.cc and v3dfile.cc to generate the output files.
 *
 * Unlike AsyVkRender and AsyGLRender, this class does NOT require any GPU
 * libraries (Vulkan, OpenGL, GLFW). It simply configures the base class state
 * and returns early, as the actual rendering is done client-side by JavaScript.
 */
class AsyWebGLRender : public AsyRender
{
public:
  AsyWebGLRender() = default;
  ~AsyWebGLRender() override = default;

  // Implementation of base class pure virtual function
  void render(RenderFunctionArgs const& args) override;

  // Pure virtual implementations (no-op for WebGL renderer)
  void drawFrame() override {}
  void exportHandler(int = 0) override {}
  void Export(int imageIndex = 0) override {}

protected:
  // No Vulkan or OpenGL dependencies - pure CPU-based setup
};

} // namespace camp
