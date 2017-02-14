/*****
 * drawbezierpatch.cc
 * Author: John C. Bowman
 *
 * Render a Bezier patch.
 *****/

#include "bezierpatch.h"

namespace camp {

#ifdef HAVE_GL

std::vector<GLfloat> BezierPatch::buffer;
std::vector<GLfloat> BezierPatch::Buffer;
std::vector<GLuint> BezierPatch::indices;
std::vector<GLuint> BezierPatch::Indices;
std::vector<GLfloat> BezierPatch::tbuffer;
std::vector<GLuint> BezierPatch::tindices;
std::vector<GLfloat> BezierPatch::tBuffer;
std::vector<GLuint> BezierPatch::tIndices;

GLuint BezierPatch::nvertices=0;
GLuint BezierPatch::ntvertices=0;
GLuint BezierPatch::Nvertices=0;
GLuint BezierPatch::Ntvertices=0;

extern const double Fuzz2;

#ifdef __MSDOS__      
const double FillFactor=0.2;
#else
const double FillFactor=0.1;
#endif

void BezierPatch::init(double res, const triple& Min, const triple& Max,
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
    
// Use a uniform partition to draw a Bezier patch.
// p is an array of 16 triples representing the control points.
// Pi are the (possibly) adjusted vertices indexed by Ii.
// The 'flati' are flatness flags for each boundary.
void BezierPatch::render(const triple *p,
                         GLuint I0, GLuint I1, GLuint I2, GLuint I3,
                         triple P0, triple P1, triple P2, triple P3,
                         bool flat0, bool flat1, bool flat2, bool flat3,
                         GLfloat *C0, GLfloat *C1, GLfloat *C2, GLfloat *C3)
{
  if(Distance(p) < res2) { // Patch is flat
    triple P[]={P0,P1,P2,P3};
    if(!offscreen(4,P)) {
      std::vector<GLuint> &p=*pindices;
      p.push_back(I0);
      p.push_back(I1);
      p.push_back(I2);
      p.push_back(I0);
      p.push_back(I2);
      p.push_back(I3);
    }
  } else { // Patch is not flat
    if(offscreen(16,p)) return;
    /* Control points are indexed as follows:
         
       Coordinate
       +-----
        Index
         

       03    13    23    33
       +-----+-----+-----+
       |3    |7    |11   |15
       |     |     |     |
       |02   |12   |22   |32
       +-----+-----+-----+
       |2    |6    |10   |14
       |     |     |     |
       |01   |11   |21   |31
       +-----+-----+-----+
       |1    |5    |9    |13
       |     |     |     |
       |00   |10   |20   |30
       +-----+-----+-----+
       0     4     8     12
         

       Subdivision:
       P refers to a corner
       m refers to a midpoint
       s refers to a subpatch
         
                m2
       +--------+--------+
       |P3      |      P2|
       |        |        |
       |   s3   |   s2   |
       |        |        |
       |        |m4      |
     m3+--------+--------+m1
       |        |        |
       |        |        |
       |   s0   |   s1   |
       |        |        |
       |P0      |      P1|
       +--------+--------+
                m0
    */

    triple p0=p[0];
    triple p3=p[3];
    triple p12=p[12];
    triple p15=p[15];
      
    Split3 c0(p0,p[1],p[2],p3);
    Split3 c1(p[4],p[5],p[6],p[7]);
    Split3 c2(p[8],p[9],p[10],p[11]);
    Split3 c3(p12,p[13],p[14],p15);

    Split3 c4(p0,p[4],p[8],p12);
    Split3 c5(c0.m0,c1.m0,c2.m0,c3.m0);
    Split3 c6(c0.m3,c1.m3,c2.m3,c3.m3);
    Split3 c7(c0.m5,c1.m5,c2.m5,c3.m5);
    Split3 c8(c0.m4,c1.m4,c2.m4,c3.m4);
    Split3 c9(c0.m2,c1.m2,c2.m2,c3.m2);
    Split3 c10(p3,p[7],p[11],p15);

    triple s0[]={p0,c0.m0,c0.m3,c0.m5,c4.m0,c5.m0,c6.m0,c7.m0,
                 c4.m3,c5.m3,c6.m3,c7.m3,c4.m5,c5.m5,c6.m5,c7.m5};
    triple s1[]={c4.m5,c5.m5,c6.m5,c7.m5,c4.m4,c5.m4,c6.m4,c7.m4,
                 c4.m2,c5.m2,c6.m2,c7.m2,p12,c3.m0,c3.m3,c3.m5};
    triple s2[]={c7.m5,c8.m5,c9.m5,c10.m5,c7.m4,c8.m4,c9.m4,c10.m4,
                 c7.m2,c8.m2,c9.m2,c10.m2,c3.m5,c3.m4,c3.m2,p15};
    triple s3[]={c0.m5,c0.m4,c0.m2,p3,c7.m0,c8.m0,c9.m0,c10.m0,
                 c7.m3,c8.m3,c9.m3,c10.m3,c7.m5,c8.m5,c9.m5,c10.m5};
      
    triple m4=s0[15];
      
    triple n0=normal(s0[0],s0[4],s0[8],s0[12],s0[13],s0[14],s0[15]);
    if(n0 == 0.0) {
      n0=normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
      if(n0 == 0.0) n0=normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
    }
      
    triple n1=normal(s1[12],s1[13],s1[14],s1[15],s1[11],s1[7],s1[3]);
    if(n1 == 0.0) {
      n1=normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
      if(n1 == 0.0) n1=normal(s1[0],s1[4],s1[8],s1[12],s1[11],s1[7],s1[3]);
    }
      
    triple n2=normal(s2[15],s2[11],s2[7],s2[3],s2[2],s2[1],s2[0]);
    if(n2 == 0.0) {
      n2=normal(s2[15],s2[11],s2[7],s2[3],s2[4],s2[8],s2[12]);
      if(n2 == 0.0) n2=normal(s2[12],s2[13],s2[14],s2[15],s2[2],s2[1],s2[0]);
    }
      
    triple n3=normal(s3[3],s3[2],s3[1],s3[0],s3[4],s3[8],s3[12]);
    if(n3 == 0.0) {
      n3=normal(s3[3],s3[2],s3[1],s3[0],s3[13],s3[14],s3[15]);
      if(n3 == 0.0) n3=normal(s3[15],s3[11],s3[7],s3[3],s3[4],s3[8],s3[12]);
    }
      
    triple n4=normal(s2[3],s2[2],s2[1],m4,s2[4],s2[8],s2[12]);
      
    // A kludge to remove subdivision cracks, only applied the first time
    // an edge is found to be flat before the rest of the subpatch is.
      
    triple m0=0.5*(P0+P1);
    if(!flat0) {
      if((flat0=Distance1(p0,p[4],p[8],p12) < res2))
        m0 -= Epsilon*unit(derivative(s1[0],s1[1],s1[2],s1[3]));
      else m0=s0[12];
    }
      
    triple m1=0.5*(P1+P2);
    if(!flat1) {
      if((flat1=Distance1(p12,p[13],p[14],p15) < res2))
        m1 -= Epsilon*unit(derivative(s2[12],s2[8],s2[4],s2[0]));
      else m1=s1[15];
    }
      
    triple m2=0.5*(P2+P3);
    if(!flat2) {
      if((flat2=Distance1(p15,p[11],p[7],p3) < res2))
        m2 -= Epsilon*unit(derivative(s2[3],s2[2],s2[1],s1[0]));
      else m2=s2[3];
    }
      
    triple m3=0.5*(P3+P0);
    if(!flat3) {
      if((flat3=Distance1(p0,p[1],p[2],p3) < res2))
        m3 -= Epsilon*unit(derivative(s3[0],s3[4],s3[8],s3[12]));
      else m3=s3[0];
    }
      
      
    if(C0) {
      GLfloat c0[4],c1[4],c2[4],c3[4],c4[4];
      for(size_t i=0; i < 4; ++i) {
        c0[i]=0.5*(C0[i]+C1[i]);
        c1[i]=0.5*(C1[i]+C2[i]);
        c2[i]=0.5*(C2[i]+C3[i]);
        c3[i]=0.5*(C3[i]+C0[i]);
        c4[i]=0.5*(c0[i]+c2[i]);
      }
      
      GLuint i0=pVertex(m0,n0,c0);
      GLuint i1=pVertex(m1,n1,c1);
      GLuint i2=pVertex(m2,n2,c2);
      GLuint i3=pVertex(m3,n3,c3);
      GLuint i4=pVertex(m4,n4,c4);
      render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3,
             C0,c0,c4,c3);
      render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false,
             c0,C1,c1,c4);
      render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false,
             c4,c1,C2,c2);
      render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3,
             c3,c4,c2,C3);
    } else {
      GLuint i0=pvertex(m0,n0);
      GLuint i1=pvertex(m1,n1);
      GLuint i2=pvertex(m2,n2);
      GLuint i3=pvertex(m3,n3);
      GLuint i4=pvertex(m4,n4);
      render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3);
      render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false);
      render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false);
      render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3);
    }
  }
}

void BezierPatch::render(const triple *p, bool straight, GLfloat *c0)
{
  triple p0=p[0];
  epsilon=0;
  for(unsigned i=1; i < 16; ++i)
    epsilon=max(epsilon,abs2(p[i]-p0));
  
  epsilon *= Fuzz2;
    
  triple p3=p[3];
  triple p12=p[12];
  triple p15=p[15];

  triple n0=normal(p3,p[2],p[1],p0,p[4],p[8],p12);
  if(n0 == 0.0) n0=normal(p3,p[2],p[1],p0,p[13],p[14],p15);
  if(n0 == 0.0) n0=normal(p15,p[11],p[7],p3,p[4],p[8],p12);
    
  triple n1=normal(p0,p[4],p[8],p12,p[13],p[14],p15);
  if(n1 == 0.0) n1=normal(p0,p[4],p[8],p12,p[11],p[7],p3);
  if(n1 == 0.0) n1=normal(p3,p[2],p[1],p0,p[13],p[14],p15);
    
  triple n2=normal(p12,p[13],p[14],p15,p[11],p[7],p3);
  if(n2 == 0.0) n2=normal(p12,p[13],p[14],p15,p[2],p[1],p0);
  if(n2 == 0.0) n2=normal(p0,p[4],p[8],p12,p[11],p[7],p3);
    
  triple n3=normal(p15,p[11],p[7],p3,p[2],p[1],p0);
  if(n3 == 0.0) n3=normal(p15,p[11],p[7],p3,p[4],p[8],p12);
  if(n3 == 0.0) n3=normal(p12,p[13],p[14],p15,p[2],p[1],p0);
    
  GLuint I0,I1,I2,I3;
    
  if(c0) {
    GLfloat *c1=c0+4;
    GLfloat *c2=c0+8;
    GLfloat *c3=c0+12;
    
    I0=pVertex(p0,n0,c0);
    I1=pVertex(p12,n1,c1);
    I2=pVertex(p15,n2,c2);
    I3=pVertex(p3,n3,c3);
      
    if(!straight)
      render(p,I0,I1,I2,I3,p0,p12,p15,p3,false,false,false,false,
             c0,c1,c2,c3);
  } else {
    I0=pvertex(p0,n0);
    I1=pvertex(p12,n1);
    I2=pvertex(p15,n2);
    I3=pvertex(p3,n3);
    
    if(!straight)
      render(p,I0,I1,I2,I3,p0,p12,p15,p3,false,false,false,false);
  }
    
  if(straight) {
    pindices->push_back(I0);
    pindices->push_back(I1);
    pindices->push_back(I2);
    pindices->push_back(I0);
    pindices->push_back(I2);
    pindices->push_back(I3);
  }
}

void BezierPatch::draw()
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
