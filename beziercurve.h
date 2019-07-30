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
  triple Min,Max;

  static GLuint vertsBufferIndex; 
  static GLuint elemBufferIndex; 
  
  void init(double res, const triple& Min, const triple& Max);
    
// Store the vertex v in the buffer.
  static GLuint vertex(const triple &v) {
    size_t nvertices=vertexbuffer.size();
    vertexbuffer.push_back(vertexData1(v));
    return nvertices;
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
  
  static void clear() {
    vertexbuffer.clear();
    indices.clear();
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
