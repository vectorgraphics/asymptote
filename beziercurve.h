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

#ifdef HAVE_LIBGLM

extern const double Fuzz;
extern const double Fuzz2;

class vertexData1 {
public:
  GLfloat position[3];
  GLint material;
  GLint center; // Index to center of billboard label
  vertexData1() {};
  vertexData1(const triple& v) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    material=drawElement::materialIndex;
    center=0;
  }
  vertexData1(const triple& v, billboard_t) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    material=drawElement::materialIndex;
    center=drawElement::centerIndex;
  }
};

class pixelData {
public:
  GLfloat position[3];
  GLint  material;
  GLfloat width;
  pixelData() {};
  pixelData(const triple& v, double width) : width(width) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    material=drawElement::materialIndex;
  }
};

struct BezierCurve
{
  static std::vector<vertexData1> vertexbuffer;
  static std::vector<GLuint> indices;
  double res,res2;
  pair Min,Max;
  typedef GLuint vertexFunction(const triple &v);
  vertexFunction *pvertex;

  static GLuint vertsBufferIndex; 
  static GLuint elemBufferIndex; 
  
  std::vector<GLuint> *pindices;
  bool Offscreen;
  
  BezierCurve() {}
  
  void init(double res, const pair& Min, const pair& Max,
            bool billboard=false);
    
// Store the vertex v in the buffer.
  static GLuint vertex(const triple &v) {
    size_t nvertices=vertexbuffer.size();
    vertexbuffer.push_back(vertexData1(v));
    return nvertices;
  }
  
  static GLuint bvertex(const triple &v) {
    size_t nvertices=vertexbuffer.size();
    vertexbuffer.push_back(vertexData1(v,billboard));
    return nvertices;
  }
  
  // Approximate bounds by bounding box of control polyhedron.
  bool offscreen(size_t n, const triple *v) {
    double x,y;
    double X,Y;

    X=x=v[0].getx();
    Y=y=v[0].gety();
    
    for(size_t i=1; i < n; ++i) {
      triple V=v[i];
      double vx=V.getx();
      double vy=V.gety();
      if(vx < x) x=vx;
      else if(vx > X) X=vx;
      if(vy < y) y=vy;
      else if(vy > Y) Y=vy;
    }

    if(X >= Min.getx() && x <= Max.getx() &&
       Y >= Min.gety() && y <= Max.gety())
      return false;

    return Offscreen=true;
  }

  static void clear() {
    vertexbuffer.clear();
    indices.clear();
  }
  
  ~BezierCurve() {}
  
  void render(const triple *p, GLuint I0, GLuint I1);
  void render(const triple *p, bool straight);
  
  bool queue(const triple *g, bool straight, double ratio,
             const pair& Min, const pair& Max, bool billboard=false) {
    init(pixel*ratio,Min,Max,billboard);
    render(g,straight);
    return Offscreen;
  }
  
  void draw();
};

struct Pixel
{
  static std::vector<pixelData> vertexbuffer;
  
// Store the vertex v in the buffer.
  static void vertex(const triple &v, double width) {
    vertexbuffer.push_back(pixelData(v,width));
  }
  
  static void clear() {
    vertexbuffer.clear();
  }
  
  Pixel() {}
  ~Pixel() {}
  
  void queue(const triple& p, double width);
  void draw();
};

#endif

} //namespace camp

#endif
