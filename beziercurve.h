/*****
 * beziercurve.h
 * Author: John C. Bowman
 *
 * Render a Bezier curve.
 *****/

#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include "drawelement.h"

namespace camp {

#ifdef HAVE_GL

static const double pixel=0.5; // Adaptive rendering constant.

extern const double Fuzz;
extern const double Fuzz2;

struct BezierCurve
{
  static std::vector<GLfloat> buffer;
  static std::vector<GLint> indices;
  triple u,v,w;
  GLuint nvertices;
  double epsilon;
  double res,res2;
  triple Min,Max;
  
  BezierCurve() : nvertices(0) {}
  
  void init(double res, const triple& Min, const triple& Max);
    
// Store the vertex v and its normal vector n in the buffer.
  GLuint vertex(const triple &v) {
    buffer.push_back(v.getx());
    buffer.push_back(v.gety());
    buffer.push_back(v.getz());
    return nvertices++;
  }
  
// Approximate bounds by bounding box of control polyhedron.
  bool offscreen(size_t n, const triple *v) {
    double x,y,z;
    double X,Y,Z;
    
    boundstriples(x,y,z,X,Y,Z,4,v);
    return
      X < Min.getx() || x > Max.getx() ||
      Y < Min.gety() || y > Max.gety() ||
      Z < Min.getz() || z > Max.getz();
  }
  
  void render(const triple *p, GLuint I0, GLuint I1);
  void render(const triple *p, bool straight);
  
  void clear() {
    nvertices=0;
    buffer.clear();
    indices.clear();
  }
  
  ~BezierCurve() {
    clear();
  }
  
  void draw();
  
  void render(const triple *g, bool straight, double ratio,
              const triple& Min, const triple& Max) {
    init(pixel*ratio,Min,Max);
    render(g,straight);
  }
  
  void draw(const triple *g, bool straight, double ratio,
            const triple& Min, const triple& Max) {
    render(g,straight,ratio,Min,Max);
    draw();
  }
};

#endif

} //namespace camp

#endif
