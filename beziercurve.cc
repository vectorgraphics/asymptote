/*****
 * drawbezierpatch.cc
 * Author: John C. Bowman
 *
 * Render a Bezier curve.
 *****/

#include "bezierpatch.h"
#include "beziercurve.h"

namespace camp {

#ifdef HAVE_VULKAN

void BezierCurve::init(double res)
{
  this->res=res;
  res2=res*res;

  MaterialIndex=vk->materialIndex;
}

inline triple normal(triple bP, triple bPP)
{
  return dot(bP,bP)*bPP-dot(bP,bPP)*bP;
}

void BezierCurve::render(const triple *p, bool straight)
{
  triple p0=p[0];
  triple p3=p[3];
  triple n0,n1;

  if(straight) {
    n0=n1=triple(0.0,0.0,1.0);
  } else {
    triple p1=p[1];
    triple p2=p[2];

    n0=normal(p1-p0,p0+p2-2.0*p1);
    n1=normal(p3-p2,p3+p1-2.0*p2);
  }

  size_t i0=data.addVertex(MaterialVertex{p0,n0,MaterialIndex});
  size_t i3=data.addVertex(MaterialVertex{p3,n1,MaterialIndex});

  if(straight) {
    data.indices.push_back(i0);
    data.indices.push_back(i3);
  } else
    render(p,i0,i3);
  append();
}

// Use a uniform partition to draw a Bezier curve.
// p is an array of 4 triples representing the control points.
// Ii are the vertex indices.
void BezierCurve::render(const triple *p, GLuint I0, GLuint I1)
{
  triple p0=p[0];
  triple p1=p[1];
  triple p2=p[2];
  triple p3=p[3];
  if(Straightness(p0,p1,p2,p3) < res2) { // Segment is flat
    triple P[]={p0,p3};
    if(!offscreen(2,P)) {
      data.indices.push_back(I0);
      data.indices.push_back(I1);
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

    triple n0=normal(bezierPh(p0,p1,p2,p3),bezierPPh(p0,p1,p2,p3));
    size_t i0=data.addVertex(MaterialVertex{m5,n0,MaterialIndex});

    render(s0,I0,i0);
    render(s1,i0,I1);
  }
}

#endif

} //namespace camp
