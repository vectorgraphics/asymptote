
/*****
 * drawbeziertriangle.cc
 * Authors: Jesse Frohlich and John C. Bowman
 *
 * Render a Bezier triangle.
 *****/

#include "drawsurface.h"

namespace camp {

#ifdef HAVE_GL

static const double pixel=0.5; // Adaptive rendering constant.

extern const double Fuzz;
extern const double Fuzz2;

// return the maximum perpendicular distance squared of points c0 and c1
// from z0--z1.
inline double Distance1(const triple& z0, const triple& c0,
                            const triple& c1, const triple& z1)
{
  triple Z0=c0-z0;
  triple Q=unit(z1-z0);
  triple Z1=c1-z0;
  return max(abs2(Z0-dot(Z0,Q)*Q),abs2(Z1-dot(Z1,Q)*Q));
}

// return the perpendicular distance squared of a point z from the plane
// through u with unit normal n.
inline double Distance2(const triple& z, const triple& u, const triple& n)
{
  double d=dot(z-u,n);
  return d*d;
}
  
// Returns one-sixth of the second derivative of the Bezier curve defined
// by a,b,c,d at 0. 
inline triple bezierPP(triple a, triple b, triple c) {
  return a+c-2.0*b;
}

// Returns one-third of the third derivative of the Bezier curve defined by
// a,b,c,d.
inline triple bezierPPP(triple a, triple b, triple c, triple d) {
  return d-a+3.0*(b-c);
}

#ifdef __MSDOS__      
const double FillFactor=1.0;
#else
const double FillFactor=0.1;
#endif      

struct RenderTriangle
{
  std::vector<GLfloat> buffer;
  std::vector<GLint> indices;
  triple u,v,w;
  GLuint nvertices;
  double epsilon;
  double Epsilon;
  double res,res2;
  triple Min,Max;
  
  void init(double res, const triple& Min, const triple& Max) {
    this->res=res;
    res2=res*res;
    Epsilon=FillFactor*res;
    this->Min=Min;
    this->Max=Max;
    
    const size_t nbuffer=10000;
    buffer.reserve(nbuffer);
    indices.reserve(nbuffer);
    nvertices=0;
  }
    
  void clear() {
    buffer.clear();
    indices.clear();
  }
  
// Store the vertex v and its normal vector n in the buffer.
  GLuint vertex(const triple &v, const triple& n) {
    buffer.push_back(v.getx());
    buffer.push_back(v.gety());
    buffer.push_back(v.getz());
    
    buffer.push_back(n.getx());
    buffer.push_back(n.gety());
    buffer.push_back(n.getz());
    
    return nvertices++;
  }
  
// Store the vertex v and its normal vector n and colour in the buffer.
  GLuint vertex(const triple& V, const triple& n, GLfloat *c) {
    int rc=vertex(V,n);
    buffer.push_back(c[0]);
    buffer.push_back(c[1]);
    buffer.push_back(c[2]);
    buffer.push_back(c[3]);
    return rc;
  }
  
  triple normal(triple left3, triple left2, triple left1, triple middle,
                triple right1, triple right2, triple right3) {
    triple rp=right1-middle;
    triple lp=left1-middle;
    triple n=triple(rp.gety()*lp.getz()-rp.getz()*lp.gety(),
                    rp.getz()*lp.getx()-rp.getx()*lp.getz(),
                    rp.getx()*lp.gety()-rp.gety()*lp.getx());
    if(abs2(n) > epsilon)
      return unit(n);
    
    triple lpp=bezierPP(middle,left1,left2);
    triple rpp=bezierPP(middle,right1,right2);
    n=cross(rpp,lp)+cross(rp,lpp);
    if(abs2(n) > epsilon)
      return unit(n);

    triple lppp=bezierPPP(middle,left1,left2,left3);
    triple rppp=bezierPPP(middle,right1,right2,right3);
    return unit(9.0*cross(rpp,lpp)+
                3.0*(cross(rp,lppp)+cross(rppp,lp)+
                     cross(rppp,lpp)+cross(rpp,lppp))+
                cross(rppp,lppp));
  }

  inline double Distance(const triple *p)
  {
    triple p0=p[0];
    triple p6=p[6];
    triple p9=p[9];

    // Only the internal point is tested for deviance from the triangle
    // formed by the vertices. We assume that the Jacobian is nonzero so
    // that we only need to calculate the perpendicular distance of the
    // internal point from this triangle.  
    double d=Distance2(p[4],p0,normal(p9,p[5],p[2],p0,p[1],p[3],p6));

    // Determine how straight the edges are.
    d=max(d,Distance1(p0,p[1],p[3],p6));
    d=max(d,Distance1(p0,p[2],p[5],p9));
    return max(d,Distance1(p6,p[7],p[8],p9));
  }

// Approximate bounds by bounding box of control polyhedron.
  bool offscreen(int n, const triple *v) {
    double x,y,z;
    double X,Y,Z;
    
    boundstriples(x,y,z,X,Y,Z,n,v);
    return
      X < Min.getx() || x > Max.getx() ||
      Y < Min.gety() || y > Max.gety() ||
      Z < Min.getz() || z > Max.getz();
  }
  
// Uses a uniform partition to draw a Bezier triangle.
// p is an array of 10 triples representing the control points.
// Pi is the full precision value indexed by Ii.
// The 'flati' are flatness flags for each boundary.
  void render(const triple *p,
              GLuint I0, GLuint I1, GLuint I2,
              triple P0, triple P1, triple P2,
              bool flat0, bool flat1, bool flat2,
              GLfloat *C0=NULL, GLfloat *C1=NULL, GLfloat *C2=NULL)
  {
    if(Distance(p) < res2) { // Triangle is flat
      triple P[]={P0,P1,P2};
      if(!offscreen(3,P)) {
        indices.push_back(I0);
        indices.push_back(I1);
        indices.push_back(I2);
      }
    } else { // Triangle is not flat
      if(offscreen(10,p)) return;
      /*    Naming Convention:
       
                                   P2
                                   030
                                   /\
                                  /  \
                                 /    \
                                /      \
                               /   up   \
                              /          \
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
      
        GLuint i0=vertex(p0,n0,c0);
        GLuint i1=vertex(p1,n1,c1);
        GLuint i2=vertex(p2,n2,c2);
          
        render(l,I0,i2,i1,P0,p2,p1,flat0,flat1,false,C0,c2,c1);
        render(r,i2,I1,i0,p2,P1,p0,flat0,false,flat2,c2,C1,c0);
        render(u,i1,i0,I2,p1,p0,P2,false,flat1,flat2,c1,c0,C2);
        render(c,i0,i1,i2,p0,p1,p2,false,false,false,c0,c1,c2);
      } else {
        GLuint i0=vertex(p0,n0);
        GLuint i1=vertex(p1,n1);
        GLuint i2=vertex(p2,n2);
          
        render(l,I0,i2,i1,P0,p2,p1,flat0,flat1,false);
        render(r,i2,I1,i0,p2,P1,p0,flat0,false,flat2);
        render(u,i1,i0,I2,p1,p0,P2,false,flat1,flat2);
        render(c,i0,i1,i2,p0,p1,p2,false,false,false);
      }
    }
  }

// n is the maximum depth
  void render(const triple *p, bool straight, GLfloat *c0=NULL) {
    triple p0=p[0];
    epsilon=0;
    for(int i=1; i < 10; ++i)
      epsilon=max(epsilon,abs2(p[i]-p0));
  
    epsilon *= Fuzz2;
    
    GLuint i0,i1,i2;
    
    triple p6=p[6];
    triple p9=p[9];
    
    triple n0=normal(p9,p[5],p[2],p0,p[1],p[3],p6);
    triple n1=normal(p0,p[1],p[3],p6,p[7],p[8],p9);    
    triple n2=normal(p6,p[7],p[8],p9,p[5],p[2],p0);
    
    if(c0) {
      GLfloat *c1=c0+4;
      GLfloat *c2=c0+8;
    
      i0=vertex(p0,n0,c0);
      i1=vertex(p6,n1,c1);
      i2=vertex(p9,n2,c2);
    
      if(!straight)
        render(p,i0,i1,i2,p0,p6,p9,false,false,false,c0,c1,c2);
    } else {
      i0=vertex(p0,n0);
      i1=vertex(p6,n1);
      i2=vertex(p9,n2);
    
      if(!straight)
        render(p,i0,i1,i2,p0,p6,p9,false,false,false);
    }
    
    if(straight) {
      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i2);
    }
    
    size_t stride=(c0 ? 10 : 6)*sizeof(GL_FLOAT);

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    if(c0) glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3,GL_FLOAT,stride,&buffer[0]);
    glNormalPointer(GL_FLOAT,stride,&buffer[3]);
    if(c0) glColorPointer(4,GL_FLOAT,stride,&buffer[6]);
    glDrawElements(GL_TRIANGLES,indices.size(),GL_UNSIGNED_INT,&indices[0]);
    if(c0) glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
  }
};

RenderTriangle R;

void bezierTriangle(const triple *g, bool straight, double ratio,
                    const triple& Min, const triple& Max, GLfloat *colors)
{
  R.init(pixel*ratio,Min,Max);
  R.render(g,straight,colors);
  R.clear();
}

#endif

} //namespace camp

