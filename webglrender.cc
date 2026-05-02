/*****
 * webglrender.cc
 * Lightweight renderer for WebGL (html) and v3d output.
 * Does not require Vulkan or OpenGL libraries.
 *****/

#include "webglrender.h"
#include "settings.h"
#include "picture.h"
#include "interact.h"

using settings::getSetting;
using namespace glm;

namespace camp
{

void AsyWebGLRender::render(RenderFunctionArgs const& args)
{
  bool v3d = args.format == "v3d";
  bool webgl = args.format == "html";
  bool format3d = webgl || v3d;

  // Store picture and basic state
  pic = args.pic;
  Prefix = args.prefix;
  Format = args.format;
  remesh = true;
  nlights = args.nlightsin;
  Lights = args.lights;
  LightsDiffuse = args.diffuse;
  Oldpid = args.oldpid;

  // Camera and view parameters
  Angle = args.angle * radians;
  lastzoom = 0;
  Zoom0 = std::fpclassify(args.zoom) == FP_NORMAL ? args.zoom : 1.0;
  Shift = args.shift / args.zoom;
  Margin = args.margin;

  // Background color
  for (int i = 0; i < 4; i++)
    Background[i] = static_cast<float>(args.background[i]);

  ViewExport = args.view;
  View = args.view && !settings::getSetting<bool>("offscreen");

  title = std::string(PACKAGE_NAME) + ": " + args.prefix.c_str();

  // Scene bounds
  Xmin = args.m.getx();
  Xmax = args.M.getx();
  Ymin = args.m.gety();
  Ymax = args.M.gety();
  Zmin = args.m.getz();
  Zmax = args.M.getz();

  haveScene = Xmin < Xmax && Ymin < Ymax && Zmin < Zmax;
  orthographic = Angle == 0.0;
  H = orthographic ? 0.0 : -tan(0.5 * Angle) * Zmax;
  Xfactor = Yfactor = 1.0;

  // Transform matrices
  for (int i = 0; i < 16; ++i)
    T[i] = args.t[i];

  for (int i = 0; i < 16; ++i)
    Tup[i] = args.tup[i];

  if (!(initialized && interact::interactive)) {
    antialias = settings::getSetting<Int>("antialias") > 1;

    // Calculate expand factor - for format3d, use 1.0 (no expansion)
    double expand;
    if (format3d)
      expand = 1.0;
    else {
      expand = settings::getSetting<double>("render");
      if (expand < 0)
        expand *= (Format.empty() || Format == "eps" || Format == "pdf") ? -2.0 : -1.0;
      if (antialias)
        expand *= 2.0;
    }

    oWidth = args.width;
    oHeight = args.height;
    Aspect = args.width / args.height;

    fullWidth = (int)ceil(expand * args.width);
    fullHeight = (int)ceil(expand * args.height);

    if (format3d) {
      Width = fullWidth;
      Height = fullHeight;
    } else {
      // For non-format3d, we would need screen dimensions from GLFW/Vulkan/OpenGL
      // This path should not be reached for WebGL renderer
      Width = fullWidth;
      Height = fullHeight;
    }

    // Reset camera state for format3d output
    home(format3d);

    // For WebGL/v3d output, we just need to set up dimensions and return.
    // The actual rendering is done client-side by JavaScript/WebGL.
    if (format3d) {
      remesh = true;
#ifdef HAVE_RENDERER
      if (threads)
        format3dWait = true;
#endif
      return;
    }

    ArcballFactor = 1 + 8.0 * hypot(Margin.getx(), Margin.gety()) / hypot(Width, Height);
    Aspect = ((double)Width) / Height;

    setosize();
  }

  clearMaterials();
  initialized = true;

  // For format3d output, we're done - just set up dimensions and return.
  // The calling code (picture.cc) will then create jsfile/gzv3dfile to write the output.
}

} // namespace camp
