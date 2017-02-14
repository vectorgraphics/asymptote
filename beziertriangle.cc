/*****
 * drawbeziertriangle.cc
 * Authors: Jesse Frohlich and John C. Bowman
 *
 * Render a Bezier triangle.
 *****/

#include "drawsurface.h"

namespace camp {

#ifdef HAVE_GL

std::vector<GLfloat> BezierTriangle::buffer;
std::vector<GLfloat> BezierTriangle::Buffer;
std::vector<GLuint> BezierTriangle::indices;
std::vector<GLuint> BezierTriangle::Indices;
std::vector<GLfloat> BezierTriangle::tbuffer;
std::vector<GLuint> BezierTriangle::tindices;
std::vector<GLfloat> BezierTriangle::tBuffer;
std::vector<GLuint> BezierTriangle::tIndices;

GLuint BezierTriangle::nvertices=0;
GLuint BezierTriangle::ntvertices=0;
GLuint BezierTriangle::Nvertices=0;
GLuint BezierTriangle::Ntvertices=0;

extern const double Fuzz2;

#ifdef __MSDOS__
const double FillFactor=1.0;
#else
const double FillFactor=0.1;
#endif

void BezierTriangle::init(double res, const triple& Min, const triple& Max,
                          bool transparent, GLfloat *colors)
{
  empty=false;
  this->res=res;
  res2=res*res;
  Epsilon=FillFactor*res;
  this->Min=Min;
  this->Max=Max;
    
  const size_t nbuffer=10000;
  if(transparent) {
    tbuffer.reserve(nbuffer);
    tindices.reserve(nbuffer);
    pindices=&tindices;
    pvertex=&tvertex;
    if(colors) {
      tBuffer.reserve(nbuffer);
      tIndices.reserve(nbuffer);
      pindices=&tIndices;
      pVertex=&tVertex;
    }
  } else {
    buffer.reserve(nbuffer);
    indices.reserve(nbuffer);
    pindices=&indices;
    pvertex=&vertex;
    if(colors) {
      Buffer.reserve(nbuffer);
      Indices.reserve(nbuffer);
      pindices=&Indices;
      pVertex=&Vertex;
    }
  }
}
  
// Uses a uniform partition to draw a Bezier triangle.
// p is an array of 10 triples representing the control points.
// Pi are the (possibly) adjusted vertices indexed by Ii.
// The 'flati' are flatness flags for each boundary.
void BezierTriangle::render(const triple *p,
                            GLuint I0, GLuint I1, GLuint I2,
                            triple P0, triple P1, triple P2,
                            bool flat0, bool flat1, bool flat2,
                            GLfloat *C0, GLfloat *C1, GLfloat *C2)
{
  if(Distance(p) < res2) { // Triangle is flat
    triple P[]={P0,P1,P2};
    if(!offscreen(3,P)) {
      std::vector<GLuint> &p=*pindices;
      p.push_back(I0);
      p.push_back(I1);
      p.push_back(I2);
    }
  } else { // Triangle is not flat
    if(offscreen(10,p)) return;
    /* Control points are indexed as follows:

       Coordinate
        Index

                                  030
                                   9
                                   /\
                                  /  \
                                 /    \
                                /      \
                               /        \
                          021 +          + 120
                           5 /            \ 8
                            /              \
                           /                \
                          /                  \
                         /                    \
                    012 +          +           + 210
                     2 /          111           \ 7
                      /            4             \
                     /                            \
                    /                              \
                   /                                \
                  /__________________________________\
                003         102           201        300
                 0           1             3          6


       Subdivision:
                                   P2
                                   030
                                   /\
                                  /  \
                                 /    \
                                /      \
                               /        \
                              /    up    \
                             /            \
                            /              \
                        p1 /________________\ p0
                          /\               / \
                         /  \             /   \
                        /    \           /     \
                       /      \  center /       \
                      /        \       /         \
                     /          \     /           \
                    /    left    \   /    right    \
                   /              \ /               \
                  /________________V_________________\
                003               p2                300
                P0                                    P1
    */

    // Subdivide triangle
    triple l003=p[0];
    triple p102=p[1];
    triple p012=p[2];
    triple p201=p[3];
    triple p111=p[4];
    triple p021=p[5];
    triple r300=p[6];
    triple p210=p[7];
    triple p120=p[8];
    triple u030=p[9];

    triple u021=0.5*(u030+p021);
    triple u120=0.5*(u030+p120);

    triple p033=0.5*(p021+p012);
    triple p231=0.5*(p120+p111);
    triple p330=0.5*(p120+p210);

    triple p123=0.5*(p012+p111);

    triple l012=0.5*(p012+l003);
    triple p312=0.5*(p111+p201);
    triple r210=0.5*(p210+r300);

    triple l102=0.5*(l003+p102);
    triple p303=0.5*(p102+p201);
    triple r201=0.5*(p201+r300);

    triple u012=0.5*(u021+p033);
    triple u210=0.5*(u120+p330);
    triple l021=0.5*(p033+l012);
    triple p4xx=0.5*p231+0.25*(p111+p102);
    triple r120=0.5*(p330+r210);
    triple px4x=0.5*p123+0.25*(p111+p210);
    triple pxx4=0.25*(p021+p111)+0.5*p312;
    triple l201=0.5*(l102+p303);
    triple r102=0.5*(p303+r201);

    triple l210=0.5*(px4x+l201); // =c120
    triple r012=0.5*(px4x+r102); // =c021
    triple l300=0.5*(l201+r102); // =r003=c030

    triple r021=0.5*(pxx4+r120); // =c012
    triple u201=0.5*(u210+pxx4); // =c102
    triple r030=0.5*(u210+r120); // =u300=c003

    triple u102=0.5*(u012+p4xx); // =c201
    triple l120=0.5*(l021+p4xx); // =c210
    triple l030=0.5*(u012+l021); // =u003=c300

    triple l111=0.5*(p123+l102);
    triple r111=0.5*(p312+r210);
    triple u111=0.5*(u021+p231);
    triple c111=0.25*(p033+p330+p303+p111);

    // A kludge to remove subdivision cracks, only applied the first time
    // an edge is found to be flat before the rest of the subpatch is.
    triple p2=0.5*(P1+P0);
    if(!flat0) {
      if((flat0=Distance1(l003,p102,p201,r300) < res2))
        p2 += Epsilon*unit(l300-c111);
      else p2=l300;
    }

    triple p1=0.5*(P2+P0);
    if(!flat1) {
      if((flat1=Distance1(l003,p012,p021,u030) < res2))
        p1 += Epsilon*unit(l030-c111);
      else p1=l030;
    }

    triple p0=0.5*(P2+P1);
    if(!flat2) {
      if((flat2=Distance1(r300,p210,p120,u030) < res2))
        p0 += Epsilon*unit(r030-c111);
      else p0=r030;
    }

    triple l[]={l003,l102,l012,l201,l111,l021,l300,l210,l120,l030}; // left
    triple r[]={l300,r102,r012,r201,r111,r021,r300,r210,r120,r030}; // right
    triple u[]={l030,u102,u012,u201,u111,u021,r030,u210,u120,u030}; // up
    triple c[]={r030,u201,r021,u102,c111,r012,l030,l120,l210,l300}; // center

    triple n0=normal(l300,r012,r021,r030,u201,u102,l030);
    triple n1=normal(r030,u201,u102,l030,l120,l210,l300);
    triple n2=normal(l030,l120,l210,l300,r012,r021,r030);
          
    if(C0) {
      GLfloat c0[4],c1[4],c2[4];
      for(int i=0; i < 4; ++i) {
        c0[i]=0.5*(C1[i]+C2[i]);
        c1[i]=0.5*(C0[i]+C2[i]);
        c2[i]=0.5*(C0[i]+C1[i]);
      }
      
      GLuint i0=pVertex(p0,n0,c0);
      GLuint i1=pVertex(p1,n1,c1);
      GLuint i2=pVertex(p2,n2,c2);
          
      render(l,I0,i2,i1,P0,p2,p1,flat0,flat1,false,C0,c2,c1);
      render(r,i2,I1,i0,p2,P1,p0,flat0,false,flat2,c2,C1,c0);
      render(u,i1,i0,I2,p1,p0,P2,false,flat1,flat2,c1,c0,C2);
      render(c,i0,i1,i2,p0,p1,p2,false,false,false,c0,c1,c2);
    } else {
      GLuint i0=pvertex(p0,n0);
      GLuint i1=pvertex(p1,n1);
      GLuint i2=pvertex(p2,n2);
          
      render(l,I0,i2,i1,P0,p2,p1,flat0,flat1,false);
      render(r,i2,I1,i0,p2,P1,p0,flat0,false,flat2);
      render(u,i1,i0,I2,p1,p0,P2,false,flat1,flat2);
      render(c,i0,i1,i2,p0,p1,p2,false,false,false);
    }
  }
}

void BezierTriangle::render(const triple *p, bool straight, GLfloat *c0)
{
  triple p0=p[0];
  epsilon=0;
  for(int i=1; i < 10; ++i)
    epsilon=max(epsilon,abs2(p[i]-p0));
  
  epsilon *= Fuzz2;
    
  GLuint I0,I1,I2;
    
  triple p6=p[6];
  triple p9=p[9];
    
  triple n0=normal(p9,p[5],p[2],p0,p[1],p[3],p6);
  triple n1=normal(p0,p[1],p[3],p6,p[7],p[8],p9);    
  triple n2=normal(p6,p[7],p[8],p9,p[5],p[2],p0);
    
  if(c0) {
    GLfloat *c1=c0+4;
    GLfloat *c2=c0+8;
    
    I0=pVertex(p0,n0,c0);
    I1=pVertex(p6,n1,c1);
    I2=pVertex(p9,n2,c2);
    
    if(!straight)
      render(p,I0,I1,I2,p0,p6,p9,false,false,false,c0,c1,c2);
  } else {
    I0=pvertex(p0,n0);
    I1=pvertex(p6,n1);
    I2=pvertex(p9,n2);
    
    if(!straight)
      render(p,I0,I1,I2,p0,p6,p9,false,false,false);
  }
    
  if(straight) {
    pindices->push_back(I0);
    pindices->push_back(I1);
    pindices->push_back(I2);
  }
}
    
void BezierTriangle::draw()
{
  if(empty) return;
  size_t stride=6;
  size_t Stride=10;
  size_t size=sizeof(GLfloat);
  size_t bytestride=stride*size;
  size_t Bytestride=Stride*size;
    
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_VERTEX_ARRAY);
  
  if(indices.size()) {
    glVertexPointer(3,GL_FLOAT,bytestride,&buffer[0]);
    glNormalPointer(GL_FLOAT,bytestride,&buffer[3]);
    glDrawElements(GL_TRIANGLES,indices.size(),GL_UNSIGNED_INT,&indices[0]);
  }
  
  if(Indices.size()) {
    glEnableClientState(GL_COLOR_ARRAY);
    glEnable(GL_COLOR_MATERIAL);
    glVertexPointer(3,GL_FLOAT,Bytestride,&Buffer[0]);
    glNormalPointer(GL_FLOAT,Bytestride,&Buffer[3]);
    glColorPointer(4,GL_FLOAT,Bytestride,&Buffer[6]);
    glDrawElements(GL_TRIANGLES,Indices.size(),GL_UNSIGNED_INT,&Indices[0]);
    glDisable(GL_COLOR_MATERIAL);
    glDisableClientState(GL_COLOR_ARRAY);
  }
  
  if(tindices.size()) {
    B=&tbuffer[0]; 
    tstride=stride;
    qsort(&tindices[0],tindices.size()/3,3*sizeof(GLuint),compare);
    glVertexPointer(3,GL_FLOAT,bytestride,&tbuffer[0]);
    glNormalPointer(GL_FLOAT,bytestride,&tbuffer[3]);
    glDrawElements(GL_TRIANGLES,tindices.size(),GL_UNSIGNED_INT,&tindices[0]);
  }
  
  if(tIndices.size()) {
    B=&tBuffer[0];
    tstride=Stride;
    qsort(&tIndices[0],tIndices.size()/3,3*sizeof(GLuint),compare);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnable(GL_COLOR_MATERIAL);
    glVertexPointer(3,GL_FLOAT,Bytestride,&tBuffer[0]);
    glNormalPointer(GL_FLOAT,Bytestride,&tBuffer[3]);
    glColorPointer(4,GL_FLOAT,Bytestride,&tBuffer[6]);
    glDrawElements(GL_TRIANGLES,tIndices.size(),GL_UNSIGNED_INT,&tIndices[0]);
    glDisable(GL_COLOR_MATERIAL);
    glDisableClientState(GL_COLOR_ARRAY);
  }
  
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  
  clear();
}

#endif

} //namespace camp
