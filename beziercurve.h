/*****
 * beziercurve.h
 * Author: John C. Bowman
 *
 * Render a Bezier curve.
 *****/

#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

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
    lineData.extendMaterial(data);
  }

  void notRendered() {
    lineData.renderCount=0;
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
    pointData.extendPoint(data);
  }

  void notRendered() {
    pointData.renderCount=0;
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
