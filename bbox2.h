#pragma once

#include "pair.h"
#include "triple.h"
#include "vkrender.h"
#include "glmCommon.h"

namespace camp {

class bbox2 {
public:
  double x,y,X,Y;
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
    Bounds(vk->billboardTransform(BB,m));
    bounds(vk->billboardTransform(BB,triple(m.getx(),m.gety(),M.getz())));
    bounds(vk->billboardTransform(BB,triple(m.getx(),M.gety(),m.getz())));
    bounds(vk->billboardTransform(BB,triple(m.getx(),M.gety(),M.getz())));
    bounds(vk->billboardTransform(BB,triple(M.getx(),m.gety(),m.getz())));
    bounds(vk->billboardTransform(BB,triple(M.getx(),m.gety(),M.getz())));
    bounds(vk->billboardTransform(BB,triple(M.getx(),M.gety(),m.getz())));
    bounds(vk->billboardTransform(BB,M));
  }

// Is 2D bounding box formed by projecting 3d points in vector v offscreen?
  bool offscreen() {
    double eps=1.0e-2;
    double min=-1.0-eps;
    double max=1.0+eps;
    return X < min || x > max || Y < min || y > max;
  }

  void Bounds(const triple& v) {
    pair V=Transform2T(glm::value_ptr(vk->projViewMat),v);
    x=X=V.getx();
    y=Y=V.gety();
  }

  void bounds(const triple& v) {
    pair V=Transform2T(glm::value_ptr(vk->projViewMat),v);
    double a=V.getx();
    double b=V.gety();
    if(a < x) x=a;
    else if(a > X) X=a;
    if(b < y) y=b;
    else if(b > Y) Y=b;
  }
};

}
