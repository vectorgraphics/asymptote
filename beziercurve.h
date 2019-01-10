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

struct BezierCurve
{
  static std::vector<GLfloat> buffer;
  static std::vector<GLuint> indices;
  GLuint nvertices;
  double res,res2;
  triple Min,Max;

  static std::array<GLuint,1> vertsBufferIndex; 
  static std::array<GLuint,1> elemBufferIndex; 
  
  BezierCurve() : nvertices(0) {}
  
  void init(double res, const triple& Min, const triple& Max);
    
// Store the vertex v in the buffer.
  GLuint vertex(const triple &v) {
    buffer.push_back(v.getx());
    buffer.push_back(v.gety());
    buffer.push_back(v.getz());
    return nvertices++;
  }
  
  void createBuffers() {
    glGenBuffers(1,vertsBufferIndex.data());
    glGenBuffers(1,elemBufferIndex.data());

    auto registerBufferFloat=[&](std::vector<GLfloat>& buffervector, GLuint bufferIndex)
    {
      if (!buffervector.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER,bufferIndex);
        glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*buffervector.size(),buffervector.data(),GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER,0);
      }
    };

    auto registerBufferUint=[&](std::vector<GLuint>& buffervector, GLuint bufferIndex)
    {
      if (!buffervector.empty()) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bufferIndex);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(GLuint)*buffervector.size(),buffervector.data(),GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
      }
    };

    //vbo
    registerBufferFloat(buffer,vertsBufferIndex[0]);

    //ebo
    registerBufferUint(indices,elemBufferIndex[0]);
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
    buffer.clear();
    indices.clear();
    
    glDeleteBuffers(1,vertsBufferIndex.data());
    glDeleteBuffers(1,elemBufferIndex.data());
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
