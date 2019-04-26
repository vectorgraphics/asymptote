/*****
 * drawbezierpatch.cc
 * Author: John C. Bowman
 *
 * Render a Bezier curve.
 *****/

#include "bezierpatch.h"
#include "beziercurve.h"

namespace camp {

#ifdef HAVE_GL

extern GLint noColorShader;
extern void setUniforms(GLint shader); 

std::vector<vertexData1> BezierCurve::vertexbuffer;
std::vector<GLuint> BezierCurve::indices;

GLuint BezierCurve::vertsBufferIndex;
GLuint BezierCurve::elemBufferIndex;

void BezierCurve::init(double res, const triple& Min, const triple& Max)
{
  this->res=res;
  res2=res*res;
  this->Min=Min;
  this->Max=Max;

  const size_t nbuffer=10000;
  vertexbuffer.reserve(nbuffer);
  indices.reserve(nbuffer);
}

// Use a uniform partition to draw a Bezier patch.
// p is an array of 4 triples representing the control points.
// Ii are the vertices indices.
void BezierCurve::render(const triple *p, GLuint I0, GLuint I1)
{
  triple p0=p[0];
  triple p1=p[1];
  triple p2=p[2];
  triple p3=p[3];
  if(Distance1(p0,p1,p2,p3) < res2) { // Segment is flat
    triple P[]={p0,p3};
    if(!offscreen(2,P)) {
      indices.push_back(I0);
      indices.push_back(I1);
    }
  } else { // Segment is not flat
    if(offscreen(4,p)) return;
    triple m0=0.5*(p0+p1);
    triple m1=0.5*(p1+p2);
    triple m2=0.5*(p2+p3);
    triple m3=0.5*(m0+m1);
    triple m4=0.5*(m1+m2);
    triple m5=0.5*(m3+m4);
      
    triple s0[]={p0,m0,m3,m5};
    triple s1[]={m5,m4,m2,p3};
      
    GLuint i0=vertex(m5);
      
    render(s0,I0,i0);
    render(s1,i0,I1);
  }
}

void BezierCurve::render(const triple *p, bool straight) 
{
  GLuint i0=vertex(p[0]);
  GLuint i3=vertex(p[3]);
    
  if(straight) {
    indices.push_back(i0);
    indices.push_back(i3);
  } else
    render(p,i0,i3);
}
  
void BezierCurve::draw()
{
  const size_t size=sizeof(GLfloat);
  static const size_t bytestride=sizeof(vertexData1);

  GLuint vao;
  glGenVertexArrays(1,&vao);
  glBindVertexArray(vao);

  createBuffers();
    
  camp::setUniforms(noColorShader); 
  
  const GLint posAttrib=glGetAttribLocation(noColorShader, "position");
  const GLint materialAttrib=glGetAttribLocation(noColorShader,"material");

  glBindBuffer(GL_ARRAY_BUFFER,vertsBufferIndex);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,elemBufferIndex);

  glVertexAttribPointer(posAttrib,3,GL_FLOAT,GL_FALSE,bytestride,(void *) 0);
  glEnableVertexAttribArray(posAttrib);

  glVertexAttribIPointer(materialAttrib,1,GL_INT,bytestride,(void *) (3*size));
  glEnableVertexAttribArray(materialAttrib);

  glFlush(); // Workaround broken MSWindows drivers for Intel GPU
  glDrawElements(GL_LINES,indices.size(),GL_UNSIGNED_INT,(void*)(0));
  
  glDisableVertexAttribArray(posAttrib);
  glDisableVertexAttribArray(materialAttrib);
  
  glBindBuffer(GL_ARRAY_BUFFER,0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
  glUseProgram(0);

  glBindVertexArray(0);
  glDeleteVertexArrays(1,&vao);
  
  clear();
}

void Pixel::draw(const triple& p)
{
  vertexData1 point(p);

  const size_t size=sizeof(GLfloat);
  static const size_t bytestride=sizeof(vertexData1);

  GLuint vbo;
  glGenBuffers(1,&vbo);
  
  GLuint vao;
  glGenVertexArrays(1,&vao);
  glBindVertexArray(vao);

  camp::setUniforms(noColorShader); 
  
  const GLint posAttrib=glGetAttribLocation(noColorShader, "position");
  const GLint materialAttrib=glGetAttribLocation(noColorShader,"material");

  glBindBuffer(GL_ARRAY_BUFFER,vbo);
  glBufferData(GL_ARRAY_BUFFER,bytestride,&point,GL_STATIC_DRAW);

  glVertexAttribPointer(posAttrib,3,GL_FLOAT,GL_FALSE,0,(void*)(0));
  glEnableVertexAttribArray(posAttrib);
  
  glVertexAttribIPointer(materialAttrib,1,GL_INT,bytestride,(void *) (3*size));
  glEnableVertexAttribArray(materialAttrib);
  
  glDrawArrays(GL_POINTS,0,1);

  glDisableVertexAttribArray(posAttrib);
  glDisableVertexAttribArray(materialAttrib);
  
  glBindBuffer(GL_ARRAY_BUFFER,0);
  glUseProgram(0);

  glBindVertexArray(0);
  glDeleteVertexArrays(1,&vao);
}

#endif

} //namespace camp
