/*****
 * bezierpatch.h
 * Authors: John C. Bowman and Jesse Frohlich
 *
 * Render Bezier patches and triangles.
 *****/

#ifndef BEZIERPATCH_H
#define BEZIERPATCH_H

#include "bbox2.h"

namespace camp {

#ifdef HAVE_LIBGLM

struct BezierPatch
{
  VertexBuffer data;

  bool transparent;
  bool color;
  double epsilon;
  double Epsilon;
  double res2;
  double Res2; // Reduced resolution for Bezier triangles flatness test.
  // typedef std::uint32_t (vertexBuffer::*vertexFunction)(const triple &v,
  //                                                const triple& n);
  // vertexFunction pvertex; // pointer to vertex function to use (transparent or not)
  bool Onscreen;

  BezierPatch() : transparent(false), color(false), Onscreen(true) {}

  void init(double res);

  void init(double res, float *colors) {
    transparent=false;
    color=colors;
    init(res);
  }

  triple normal(triple left3, triple left2, triple left1, triple middle,
                triple right1, triple right2, triple right3) {
    triple lp=3.0*(left1-middle);
    triple rp=3.0*(right1-middle);

    triple n=cross(rp,lp);
    if(abs2(n) > epsilon)
      return n;

    triple lpp=bezierPP(middle,left1,left2);
    triple rpp=bezierPP(middle,right1,right2);

    n=cross(rpp,lp)+cross(rp,lpp);
    if(abs2(n) > epsilon)
      return n;

    triple lppp=bezierPPP(middle,left1,left2,left3);
    triple rppp=bezierPPP(middle,right1,right2,right3);

    n=cross(rpp,lpp)+cross(rppp,lp)+cross(rp,lppp);
    if(abs2(n) > epsilon)
      return n;

    n=cross(rppp,lpp)+cross(rpp,lppp);
    if(abs2(n) > epsilon)
      return n;

    return cross(rppp,lppp);
  }

  // Return the differential of the Bezier curve p0,p1,p2,p3 at 0
  triple differential(triple p0, triple p1, triple p2, triple p3) {
    triple p=p1-p0;
    if(abs2(p) > epsilon)
      return p;

    p=bezierPP(p0,p1,p2);
    if(abs2(p) > epsilon)
      return p;

    return bezierPPP(p0,p1,p2,p3);
  }

  // Determine the flatness of a Bezier patch.
  pair Distance(const triple *p) {
    triple p0=p[0];
    triple p3=p[3];
    triple p12=p[12];
    triple p15=p[15];

    // Check the horizontal flatness.
    double h=Flatness(p0,p12,p3,p15);
    // Check straightness of the horizontal edges and interior control curves.
    h=std::max(h,Straightness(p0,p[4],p[8],p12));
    h=std::max(h,Straightness(p[1],p[5],p[9],p[13]));
    h=std::max(h,Straightness(p[2],p[6],p[10],p[14]));
    h=std::max(h,Straightness(p3,p[7],p[11],p15));

    // Check the vertical flatness.
    double v=Flatness(p0,p3,p12,p15);
    // Check straightness of the vertical edges and interior control curves.
    v=std::max(v,Straightness(p0,p[1],p[2],p3));
    v=std::max(v,Straightness(p[4],p[5],p[6],p[7]));
    v=std::max(v,Straightness(p[8],p[9],p[10],p[11]));
    v=std::max(v,Straightness(p12,p[13],p[14],p15));

    return pair(h,v);
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

  // Approximate bounds by bounding box of control polyhedron.
  bool offscreen(size_t n, const triple *v) {
    if(bbox2(n,v).offscreen()) {
      Onscreen=false;
      return true;
    }
    return false;
  }

  virtual void render(const triple *p, bool straight, float *c0=NULL);
  void render(const triple *p,
              std::uint32_t I0, std::uint32_t I1, std::uint32_t I2, std::uint32_t I3,
              triple P0, triple P1, triple P2, triple P3,
              bool flat0, bool flat1, bool flat2, bool flat3,
              float *C0=NULL, float *C1=NULL, float *C2=NULL,
              float *C3=NULL);

  void append() {
    if(transparent)
      transparentData.extendColor(data);
    else {
      if(color)
        colorData.extendColor(data);
      else
        materialData.extendMaterial(data);
    }
  }

  virtual void notRendered() {
    if(transparent)
      transparentData.renderCount=0;
    else {
      if(color)
        colorData.renderCount=0;
      else
        materialData.renderCount=0;
    }
  }

  void queue(const triple *g, bool straight, double ratio, bool Transparent,
             float *colors=NULL) {
    data.clear();
    Onscreen=true;
    transparent=Transparent;
    color=colors;
    notRendered();
    init(pixelResolution*ratio);
    render(g,straight,colors);
  }

};

struct BezierTriangle : public BezierPatch {
public:
  BezierTriangle() : BezierPatch() {}

  double Distance(const triple *p) {
    triple p0=p[0];
    triple p6=p[6];
    triple p9=p[9];

    // Check how far the internal point is from the centroid of the vertices.
    double d=abs2((p0+p6+p9)*third-p[4]);

    // Determine how straight the edges are.
    d=std::max(d,Straightness(p0,p[1],p[3],p6));
    d=std::max(d,Straightness(p0,p[2],p[5],p9));
    return std::max(d,Straightness(p6,p[7],p[8],p9));
  }

  void render(const triple *p, bool straight, float *c0=NULL);
  void render(const triple *p,
              std::uint32_t I0, std::uint32_t I1, std::uint32_t I2,
              triple P0, triple P1, triple P2,
              bool flat0, bool flat1, bool flat2,
              float *C0=NULL, float *C1=NULL, float *C2=NULL);
};

// triangle groups (can mix vertex dependent and material index color)
struct Triangles : public BezierPatch {
public:
  Triangles() : BezierPatch() {}

  void queue(size_t nP, const triple* P, size_t nN, const triple* N,
             size_t nC, const prc::RGBAColour* C, size_t nI,
             const uint32_t (*PI)[3], const uint32_t (*NI)[3],
             const uint32_t (*CI)[3], bool transparent);

  void append() {
    if(transparent)
      transparentData.extendColor(data);
    else
      triangleData.extendColor(data);
  }

  void notRendered() {
    if(transparent)
      transparentData.renderCount=0;
    else
      triangleData.renderCount=0;
  }
};

#endif

} //namespace camp

#endif
