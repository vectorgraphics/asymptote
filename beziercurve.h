/*****
 * beziercurve.h
 * Author: John C. Bowman
 *
 * Render a Bezier curve.
 *****/

#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <cstring>

// NOTE: glrender.h is included from bezierpatch.h (which is included by
// beziercurve users) after bbox2.h has established HAVE_GL via common.h.
// But beziercurve.h can be included before bezierpatch.h, so include it here
// too (after common.h would have been pulled via vkrender.h/drawelement.h).
#if defined(HAVE_GL) && defined(HAVE_VULKAN)
#include "glrender.h"
#endif

namespace camp {

#ifdef HAVE_VULKAN

extern const double Fuzz;
extern const double Fuzz2;

extern int MaterialIndex;

struct BezierCurve
{
  VertexBuffer data;
  double res,res2;
  bool Onscreen;

  BezierCurve() : Onscreen(true) {}

  void init(double res);

  // Approximate bounds by bounding box of control polyhedron.
  bool offscreen(size_t n, const triple *v) {
    if(bbox2(n,v).offscreen()) {
      Onscreen=false;
      return true;
    }
    return false;
  }

  void render(const triple *p, bool straight);
  void render(const triple *p, std::uint32_t I0, std::uint32_t I1);

  void append() {
#if defined(HAVE_VULKAN) && defined(HAVE_GL)
    if(gl::glFallback) {
      vertexBuffer& dst=material1Data;
      size_t offset=dst.vertices.size();
      dst.vertices.resize(offset+data.materialVertices.size());
      std::memcpy(&dst.vertices[offset],data.materialVertices.data(),
                  data.materialVertices.size()*sizeof(vertexData));
      for(auto idx : data.indices)
        dst.indices.push_back((GLuint)(idx+offset));
      return;
    }
#endif
    vkLineData.extendMaterial(data);
  }

  void notRendered() {
#if defined(HAVE_VULKAN) && defined(HAVE_GL)
    if(gl::glFallback) {
      material1Data.rendered=false;
      return;
    }
#endif
    vkLineData.renderCount=0;
  }

  void queue(const triple *g, bool straight, double ratio) {
    data.clear();
    notRendered();
    Onscreen=true;
    init(pixelResolution*ratio);
    render(g,straight);
  }
};

struct Pixel
{
  VertexBuffer data;

  void append() {
#if defined(HAVE_VULKAN) && defined(HAVE_GL)
    if(gl::glFallback) {
      vertexBuffer& dst=material0Data;
      size_t offset=dst.vertices0.size();
      dst.vertices0.resize(offset+data.pointVertices.size());
      std::memcpy(&dst.vertices0[offset],data.pointVertices.data(),
                  data.pointVertices.size()*sizeof(vertexData0));
      for(auto idx : data.indices)
        dst.indices.push_back((GLuint)(idx+offset));
      return;
    }
#endif
    vkPointData.extendPoint(data);
  }

  void notRendered() {
#if defined(HAVE_VULKAN) && defined(HAVE_GL)
    if(gl::glFallback) {
      material0Data.rendered=false;
      return;
    }
#endif
    vkPointData.renderCount=0;
  }

  void queue(const triple& p, double width) {
    data.clear();
    notRendered();
    MaterialIndex=materialIndex;
    data.indices.push_back(data.addVertex(PointVertex{p,(float)width,MaterialIndex}));
    append();
  }

  void draw();
};

#endif

} //namespace camp

#endif
