#pragma once

#include "pair.h"
#include "triple.h"
#include "glmCommon.h"
#include "render.h"

namespace camp {

#ifdef HAVE_LIBGLM
class bbox2 {
public:
  // Bitwise AND of the clip-space outcodes of the points seen so far: bit i
  // is set when every point lies outside clipping plane i.
  unsigned outside;

  bbox2(size_t n, const triple *v) {
    Bounds(v[0]);
    for(size_t i=1; i < n; ++i)
      bounds(v[i]);
  }

  bbox2(const triple& m, const triple& M) {
    Bounds(m);
    bounds(triple(m.getx(),m.gety(),M.getz()));
    bounds(triple(m.getx(),M.gety(),m.getz()));
    bounds(triple(m.getx(),M.gety(),M.getz()));
    bounds(triple(M.getx(),m.gety(),m.getz()));
    bounds(triple(M.getx(),m.gety(),M.getz()));
    bounds(triple(M.getx(),M.gety(),m.getz()));
    bounds(M);
  }

  // take account of object bounds
  bbox2(const triple& m, const triple& M, const triple& BB) {
    Bounds(billboardTransform(BB,m));
    bounds(billboardTransform(BB,triple(m.getx(),m.gety(),M.getz())));
    bounds(billboardTransform(BB,triple(m.getx(),M.gety(),m.getz())));
    bounds(billboardTransform(BB,triple(m.getx(),M.gety(),M.getz())));
    bounds(billboardTransform(BB,triple(M.getx(),m.gety(),m.getz())));
    bounds(billboardTransform(BB,triple(M.getx(),m.gety(),M.getz())));
    bounds(billboardTransform(BB,triple(M.getx(),M.gety(),m.getz())));
    bounds(billboardTransform(BB,M));
  }

  // Is the convex hull of the 3D points offscreen, that is, entirely outside
  // one of the clipping planes? The test is done in homogeneous clip
  // coordinates rather than after the perspective division, so that it
  // remains valid for points behind the eye.
  bool offscreen() {
    return outside != 0;
  }

  // Return the clip-space outcode of v.
  static unsigned outcode(const triple& v) {
#ifdef HAVE_RENDERER
    const double *t=glm::value_ptr(getProjViewMat());
    double vx=v.getx(), vy=v.gety(), vz=v.getz();
    double x=t[0]*vx+t[4]*vy+t[8]*vz+t[12];
    double y=t[1]*vx+t[5]*vy+t[9]*vz+t[13];
    double z=t[2]*vx+t[6]*vy+t[10]*vz+t[14];
    double w=t[3]*vx+t[7]*vy+t[11]*vz+t[15];
    // Allow a small tolerance at the sides of the viewport.
    double W=(1.0+1.0e-2)*w;
    // The near plane is z=0 for [0,1] clip-space depth but z=-w for [-1,1]
    // depth; testing against z=-w is conservative in both cases.
    return (x < -W ? 1 : 0) | (x > W ? 2 : 0) |
      (y < -W ? 4 : 0) | (y > W ? 8 : 0) |
      (z < -w ? 16 : 0) | (z > w ? 32 : 0);
#else
    return 0;
#endif
  }

  void Bounds(const triple& v) {
    outside=outcode(v);
  }

  void bounds(const triple& v) {
    outside &= outcode(v);
  }
};
#endif // HAVE_LIBGLM

}
