/*****
 * drawbezierpatch.cc
 * Author: John C. Bowman
 *
 * Render a Bezier patch.
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
  return abs2(dot(z-u,n)*n);
}
  
// Returns one-third of the first derivative of the Bezier curve defined by
// a,b,c,d at 0.
inline triple bezierP(triple a, triple b) {
  return b-a;
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

struct RenderPatch
{
  std::vector<GLfloat> buffer;
  std::vector<GLint> indices;
  triple u,v,w;
  GLuint nvertices;
  double cx,cy,cz;
  double epsilon;
  double Epsilon;
  double res,res2;
  bool billboard;
  
  void init(double res, bool havebillboard, const triple& center) {
    this->res=res;
    res2=res*res;
    Epsilon=FillFactor*res;

    const size_t nbuffer=10000;
    buffer.reserve(nbuffer);
    indices.reserve(nbuffer);
    nvertices=0;
    
    billboard=havebillboard;
    if(billboard) {
      cx=center.getx();
      cy=center.gety();
      cz=center.getz();

      gl::projection P=gl::camera(false);
      w=unit(P.camera-P.target);
      v=unit(perp(P.up,w));
      u=cross(v,w);
    }
  }
    
  void clear() {
    buffer.clear();
    indices.clear();
  }
  
// Store the vertex v and its normal vector n in the buffer.
  GLuint vertex(const triple& V, const triple& n) {
    if(billboard) {
      double x=V.getx()-cx;
      double y=V.gety()-cy;
      double z=V.getz()-cz;
      buffer.push_back(cx+u.getx()*x+v.getx()*y+w.getx()*z);
      buffer.push_back(cy+u.gety()*x+v.gety()*y+w.gety()*z);
      buffer.push_back(cz+u.getz()*x+v.getz()*y+w.getz()*z);
    } else {
      buffer.push_back(V.getx());
      buffer.push_back(V.gety());
      buffer.push_back(V.getz());
    }
    
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
  
  triple normal0(triple left3, triple left2, triple left1, triple middle,
                 triple right1, triple right2, triple right3) {
    //cout << "normal0 called." << endl;
    // Lots of repetition here.
    // TODO: Check if lp,rp,lpp,rpp should be manually inlined (i.e., is the
    // third order normal usually computed when normal0() is called?).
    triple lp=bezierP(middle,left1);
    triple rp=bezierP(middle,right1);
    triple lpp=bezierPP(middle,left1,left2);
    triple rpp=bezierPP(middle,right1,right2);
    triple n1=cross(rpp,lp)+cross(rp,lpp);
    if(abs2(n1) > epsilon) {
      return unit(n1);
    } else {
      triple lppp=bezierPPP(middle,left1,left2,left3);
      triple rppp=bezierPPP(middle,right1,right2,right3);
      triple n2= 9.0*cross(rpp,lpp)+
        3.0*(cross(rp,lppp)+cross(rppp,lp)+
             cross(rppp,lpp)+cross(rpp,lppp))+
        cross(rppp,lppp);
      return unit(n2);
    }
  }

  triple normal(triple left3, triple left2, triple left1, triple middle,
                triple right1, triple right2, triple right3) {
    triple bu=right1-middle;
    triple bv=left1-middle;
    triple n=triple(bu.gety()*bv.getz()-bu.getz()*bv.gety(),
                    bu.getz()*bv.getx()-bu.getx()*bv.getz(),
                    bu.getx()*bv.gety()-bu.gety()*bv.getx());
    return abs2(n) > epsilon ? unit(n) :
      normal0(left3,left2,left1,middle,right1,right2,right3);
  }
  
  double Distance(const triple *p) {
    triple p0=p[0];
    triple p3=p[3];
    triple p12=p[12];
    triple p15=p[15];
    
    triple n1=normal(p0,p[4],p[8],p12,p[13],p[14],p15);
    triple n3=normal(p15,p[11],p[7],p3,p[2],p[1],p0);
    if(n1 == 0.0) n1=n3;
    if(n3 == 0.0) n3=n1;

    // Determine how flat each subtriangle of the patch is.
    double d=Distance2(p[5],p12,n1);
    d=max(d,Distance2(p[9],p12,n1));
    d=max(d,Distance2(p[10],p12,n1));
    
    d=max(d,Distance2(p[5],p3,n3));
    d=max(d,Distance2(p[6],p3,n3));
    d=max(d,Distance2(p[10],p3,n3));
    
    // Determine how straight the edges are.
    d=max(d,Distance1(p0,p[1],p[2],p3));
    d=max(d,Distance1(p0,p[4],p[8],p12));
    d=max(d,Distance1(p3,p[7],p[11],p15));
    return max(d,Distance1(p12,p[13],p[14],p15));
  }
  
  void mesh(const triple *p, const GLuint *I)
  {
    // Draw the frame of the control points of a cubic Bezier mesh
    indices.push_back(I[0]);
    indices.push_back(I[1]);
    indices.push_back(I[2]);
    indices.push_back(I[0]);
    indices.push_back(I[2]);
    indices.push_back(I[3]);
  }
  
  struct Split3 {
    triple m0,m2,m3,m4,m5;
    Split3() {}
    Split3(triple z0, triple c0, triple c1, triple z1) {
      m0=0.5*(z0+c0);
      triple m1=0.5*(c0+c1);
      m2=0.5*(c1+z1);
      m3=0.5*(m0+m1);
      m4=0.5*(m1+m2);
      m5=0.5*(m3+m4);
    }
  };

  // Uses a uniform partition to draw a Bezier patch.
  // p is an array of 16 triples representing the control points.
  // Pi is the full precision value indexed by Ii.
  // The 'flati' are flatness flags for each boundary.
  void render(const triple *p,
              GLuint I0, GLuint I1, GLuint I2, GLuint I3,
              triple P0, triple P1, triple P2, triple P3,
              bool flat0, bool flat1, bool flat2, bool flat3,
              GLfloat *C0=NULL, GLfloat *C1=NULL, GLfloat *C2=NULL,
              GLfloat *C3=NULL)
  {
    if(Distance(p) < res2) { // Patch is flat
      GLuint I[]={I0,I1,I2,I3};
      mesh(p,I);
    } else { // Patch is not flat
        /* Control points are labelled as follows:
         
          Coordinates
          +
          Ordering
         
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
         
         Key points and patch sections:
         P refers to a corner
         m refers to a midpoint
         s refers to a patch section
         
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
      if(n0 == 0.0) n0=normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
      if(n0 == 0.0) n0=normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
      
      triple n1=normal(s1[12],s1[13],s1[14],s1[15],s1[11],s1[7],s1[3]);
      if(n1 == 0.0) n1=normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
      if(n1 == 0.0) n1=normal(s1[0],s1[4],s1[8],s1[12],s1[11],s1[7],s1[3]);
      
      triple n2=normal(s2[15],s2[11],s2[7],s2[3],s2[2],s2[1],s2[0]);
      if(n2 == 0.0) n2=normal(s2[15],s2[11],s2[7],s2[3],s2[4],s2[8],s2[12]);
      if(n2 == 0.0) n2=normal(s2[12],s2[13],s2[14],s2[15],s2[2],s2[1],s2[0]);
      
      triple n3=normal(s3[3],s3[2],s3[1],s3[0],s3[4],s3[8],s3[12]);
      if(n3 == 0.0) n3=normal(s3[3],s3[2],s3[1],s3[0],s3[13],s3[14],s3[15]);
      if(n3 == 0.0) n3=normal(s3[15],s3[11],s3[7],s3[3],s3[4],s3[8],s3[12]);
      
      triple n4=normal(s2[3],s2[2],s2[1],m4,s2[4],s2[8],s2[12]);
      
      triple m0,m1,m2,m3;
      
      // A kludge to remove subdivision cracks, only applied the first time
      // an edge is found to be flat before the rest of the subpatch is.
      if(flat0)
        m0=0.5*(P0+P1);
      else {
        if((flat0=Distance1(p0,p[4],p[8],p12) < res2))
          m0=0.5*(P0+P1)+Epsilon*unit(s0[12]-s2[3]);
        else
          m0=s0[12];
      }
      
      if(flat1)
        m1=0.5*(P1+P2);
      else {
        if((flat1=Distance1(p12,p[13],p[14],p15) < res2))
          m1=0.5*(P1+P2)+Epsilon*unit(s1[15]-s3[0]);
        else
          m1=s1[15];
      }
      
      if(flat2)
        m2=0.5*(P2+P3);
      else {
        if((flat2=Distance1(p15,p[11],p[7],p3) < res2))
          m2=0.5*(P2+P3)+Epsilon*unit(s2[3]-s0[12]);
        else
          m2=s2[3];
      }
      
      if(flat3)
        m3=0.5*(P3+P0);
      else {
        if((flat3=Distance1(p3,p[2],p[1],p0) < res2))
         m3=0.5*(P3+P0)+Epsilon*unit(s3[0]-s1[15]);
        else
          m3=s3[0];
      }
      
      if(C0) {
        GLfloat c0[4],c1[4],c2[4],c3[4],c4[4];
        for(int i=0; i < 4; ++i) {
          c0[i]=0.5*(C0[i]+C1[i]);
          c1[i]=0.5*(C1[i]+C2[i]);
          c2[i]=0.5*(C2[i]+C3[i]);
          c3[i]=0.5*(C3[i]+C0[i]);
          c4[i]=0.5*(c0[i]+c2[i]);
        }
      
        GLuint i0=vertex(m0,n0,c0);
        GLuint i1=vertex(m1,n1,c1);
        GLuint i2=vertex(m2,n2,c2);
        GLuint i3=vertex(m3,n3,c3);
        GLuint i4=vertex(m4,n4,c4);
        render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3,
               C0,c0,c4,c3);
        render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false,
               c0,C1,c1,c4);
        render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false,
               c4,c1,C2,c2);
        render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3,
               c3,c4,c2,C3);
      } else {
        GLuint i0=vertex(m0,n0);
        GLuint i1=vertex(m1,n1);
        GLuint i2=vertex(m2,n2);
        GLuint i3=vertex(m3,n3);
        GLuint i4=vertex(m4,n4);
        render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3);
        render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false);
        render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false);
        render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3);
      }
    }
  }

  void render(const triple *p, bool straight, GLfloat *c0=NULL) {
    triple p0=p[0];
    epsilon=0;
    for(int i=1; i < 16; ++i)
      epsilon=max(epsilon,abs2(p[i]-p0));
  
    epsilon *= Fuzz2;
    
    GLuint i0,i1,i2,i3;
    
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
    
    if(c0) {
      GLfloat *c1=c0+4;
      GLfloat *c2=c0+8;
      GLfloat *c3=c0+12;
    
      i0=vertex(p0,n0,c0);
      i1=vertex(p12,n1,c1);
      i2=vertex(p15,n2,c2);
      i3=vertex(p3,n3,c3);
      
      if(!straight)
        render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false,
        c0,c1,c2,c3);
    } else {
      i0=vertex(p0,n0);
      i1=vertex(p12,n1);
      i2=vertex(p15,n2);
      i3=vertex(p3,n3);
    
      if(!straight)
        render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false);
    }
    
    if(straight) {
      GLuint I[]={i0,i1,i2,i3};
      mesh(p,I);
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

static RenderPatch R;

void bezierPatch(const triple *g, bool straight, double ratio,
                 bool havebillboard, triple center, GLfloat *colors)
{
  R.init(pixel*ratio,havebillboard,center);
  R.render(g,straight,colors);
  R.clear();
}

#endif

} //namespace camp
