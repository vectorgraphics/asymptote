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

extern const double Fuzz;
extern const double Fuzz2;

class vertexData1 {
public:
  GLfloat position[3];
  GLint  material;
  vertexData1() {};
  vertexData1(const triple& v) {
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
  GLuint nvertices;
  double res,res2;
  triple Min,Max;

  static GLuint vertsBufferIndex; 
  static GLuint elemBufferIndex; 
  
  BezierCurve() : nvertices(0) {}
  
  void init(double res, const triple& Min, const triple& Max);
    
// Store the vertex v in the buffer.
  GLuint vertex(const triple &v) {
    vertexbuffer.push_back(vertexData1(v));
    return nvertices++;
  }
  
  void createBuffers() {
    glGenBuffers(1,&vertsBufferIndex);
    glGenBuffers(1,&elemBufferIndex);

    //vbo
    registerBuffer(vertexbuffer,vertsBufferIndex);

    //ebo
    registerBuffer(indices,elemBufferIndex);
  }
  
// Approximate bounds by bounding box of control polyhedron.
  bool offscreen(size_t n, const triple *v) {
    double x,y,z;
    double X,Y,Z;
    
    boundstriples(x,y,z,X,Y,Z,n,v);
    return
      X < Min.getx() || x > Max.getx() ||
      Y < Min.gety() || y > Max.gety() ||
      Z < Min.getz() || z > Max.getz();
  }
  
  void clear() {
    nvertices=0;
    vertexbuffer.clear();
    indices.clear();
    
    glDeleteBuffers(1,&vertsBufferIndex);
    glDeleteBuffers(1,&elemBufferIndex);
  }
  
  ~BezierCurve() {}
  
  void render(const triple *p, GLuint I0, GLuint I1);
  void render(const triple *p, bool straight);
  
  void queue(const triple *g, bool straight, double ratio,
              const triple& Min, const triple& Max) {
    init(pixel*ratio,Min,Max);
    render(g,straight);
  }
  
  void draw();
  void draw(const triple *g, bool straight, double ratio,
            const triple& Min, const triple& Max) {
    queue(g,straight,ratio,Min,Max);
    draw();
  }
};

struct Pixel
{
  Pixel() {}
  ~Pixel() {}
  
  void draw(const triple& p);
};

#endif

} //namespace camp

#endif
