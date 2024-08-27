/*****
 * drawclipbegin.h
 * John Bowman
 *
 * Begin clip of picture to specified path.
 *****/

#ifndef DRAWCLIPBEGIN_H
#define DRAWCLIPBEGIN_H

#include "bezierpatch.h"
#include "drawelement.h"
#include "path.h"
#include "drawpath.h"

namespace camp {

class clipFace : public gc {
public:
  size_t ncontrols;
  triple *controls;
  bool straight;

  string wrongsize() {
  return (ncontrols == 16 ? "4x4" : "triangular")+string(" array of triples");
  }

  clipFace(vm::array *P, bool straight) :
    straight(straight) {

    size_t n=checkArray(P);
    if(n != 4) wrongsize();

    vm::array *P0=vm::read<vm::array*>(P,0);
    ncontrols=checkArray(P0) == 4 ? 16 : 10;
    size_t k=0;
    controls=new triple[ncontrols];
    for(size_t i=0; i < 4; ++i) {
      vm::array *Pi=vm::read<vm::array*>(P,i);
      size_t n=(ncontrols == 16 ? 4 : i+1);
      if(checkArray(Pi) != n)
        reportError(wrongsize());
      for(unsigned int j=0; j < n; ++j)
        controls[k++]=vm::read<triple>(Pi,j);
    }
  }

  clipFace(const double* t, clipFace *f) :
    ncontrols(f->ncontrols), straight(f->straight) {
    controls=new triple[f->ncontrols];
    for(size_t i=0; i < ncontrols; ++i) {
      controls[i]=t*f->controls[i];
    }
  }

  bool Straight() {
    return straight;
  }

};

typedef mem::vector<clipFace*> clipVolume;

class drawClipBegin : public drawSuperPathPenBase {
  bool gsave;
  bool stroke;
public:
  void noncyclic() {
    reportError("cannot clip to non-cyclic path");
  }

  drawClipBegin(const vm::array& src, bool stroke, pen pentype,
                bool gsave=true, const string& key="") :
    drawElement(key), drawSuperPathPenBase(src,pentype), gsave(gsave),
    stroke(stroke) {
    if(!stroke && !cyclic()) noncyclic();
  }

  virtual ~drawClipBegin() {}

  bool beginclip() {return true;}

  void bounds(bbox& b, iopipestream& iopipe, boxvector& vbox,
              bboxlist& bboxstack) {
    bboxstack.push_back(b);
    bbox bpath;
    if(stroke) strokebounds(bpath);
    else drawSuperPathPenBase::bounds(bpath,iopipe,vbox,bboxstack);
    bboxstack.push_back(bpath);
  }

  bool begingroup() {return true;}

  bool svg() {return true;}

  void save(bool b) {
    gsave=b;
  }

  bool draw(psfile *out) {
    if(gsave) out->gsave();
    if(empty()) return true;
    out->beginclip();
    writepath(out,false);
    if(stroke) strokepath(out);
    out->endclip(pentype);
    return true;
  }

  bool write(texfile *out, const bbox& bpath) {
    if(gsave) out->gsave();
    if(empty()) return true;

    if(out->toplevel())
      out->beginpicture(bpath);

    out->begingroup();

    out->beginspecial();
    out->beginraw();
    writeshiftedpath(out);
    if(stroke) strokepath(out);
    out->endclip(pentype);
    out->endraw();
    out->endspecial();

    return true;
  }

  drawElement *transformed(const transform& t)
  {
    return new drawClipBegin(transpath(t),stroke,transpen(t),gsave,KEY);
  }

};

class drawClip3Begin : public drawElement {
  clipVolume *V;
  BezierPatch clipPatch;
  BezierTriangle clipTriangle; // Implement base class here!
public:
  drawClip3Begin(clipVolume *V, const string& key="") :
    drawElement(key), V(V) {}

  drawClip3Begin(const double* t, drawClip3Begin *s) :
    drawElement(s->KEY) {
    V=new clipVolume;
    size_t n=s->V->size();
    for(size_t i=0; i < n; ++i) {
      clipVolume *v=s->V;
      clipFace *f=(*v)[i];
      V->push_back(new clipFace(t,f));
    }
  }

  virtual ~drawClip3Begin() {}

  bool beginclip() {return true;}

  void render(double size2, const triple& Min, const triple& Max,
              double perspective, bool remesh);

  bool begingroup() {return true;}


  drawElement *transformed(const double* t) {
    return new drawClip3Begin(t,this);
  }
};

}

#endif
