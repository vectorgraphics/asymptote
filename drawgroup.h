/*****
 * drawgroup.h
 * John Bowman
 *
 * Group elements in a picture to be deconstructed as a single object.
 *****/

#ifndef DRAWGROUP_H
#define DRAWGROUP_H

#include "drawelement.h"

namespace camp {

class drawBegin : public drawElement {
  int interaction;
  triple center;
  string name;
  double compression;
  bool closed;   // render the surface as one-sided; may yield faster rendering 
  bool tessellate; // use tessellated mesh to store straight patches
  bool dobreak; // force breaking
  bool nobreak; // force grouping for transparent patches
  
  groupmap *g;
public:
  drawBegin(string name, double compression, bool closed, bool tessellate, 
            bool dobreak, bool nobreak) :
    interaction(EMBEDDED), name(name), compression(compression), closed(closed),
    tessellate(tessellate), dobreak(dobreak), nobreak(nobreak) {}
    
  drawBegin(int interaction, triple center, string name, double compression,
             bool closed, bool tessellate, bool dobreak, bool nobreak) :
    interaction(interaction), center(center), name(name), 
    compression(compression), closed(closed),
    tessellate(tessellate), dobreak(dobreak), nobreak(nobreak) {}
  
  virtual ~drawBegin() {}

  bool begingroup() {return true;}
  
  bool write(prcfile *out, unsigned int *count, vm::array *index,
             vm::array *origin, double compressionlimit,
             groupsmap& groups) {
    groupmap& group=groups.back();
    if(name.empty()) name="group";
    groupmap::const_iterator p=group.find(name);
    
    unsigned c=(p != group.end()) ? p->second+1 : 0;
    group[name]=c;
    
    ostringstream buf;
    buf << name;
    if(c > 0) buf << "-" << (c+1);
      
    if(interaction == BILLBOARD) {
      buf << "-" << (*count)++ << "\001";
      index->push((Int) origin->size());
      origin->push(center);
    }
    
    PRCoptions options(closed,tessellate,dobreak,nobreak);
    
    groups.push_back(groupmap());
    out->begingroup(buf.str().c_str(), compression > 0.0 ?
                    max(compression,compressionlimit) : 0.0,&options);
    return true;
  }
  
  drawBegin(const vm::array& t, const drawBegin *s) :
    interaction(s->interaction), name(s->name), compression(s->compression),
    closed(s->closed), tessellate(s->tessellate), dobreak(s->dobreak),
    nobreak(s->nobreak) {
    center=run::operator *(t,s->center);
  }
  
  drawElement *transformed(const array& t) {
    return new drawBegin(t,this);
  }
};

class drawEnd : public drawElement {
public:
  drawEnd() {}
  
  virtual ~drawEnd() {}

  bool endgroup() {return true;}
  
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap& groups) {
    
    groups.pop_back();
    out->endgroup();
    return true;
  }
};

}

GC_DECLARE_PTRFREE(camp::drawBegin);
GC_DECLARE_PTRFREE(camp::drawEnd);

#endif
