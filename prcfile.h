#ifndef PRCFILE_H
#define PRCFILE_H

#include "prc/oPRCFile.h"

namespace camp {

const double scale3D=1.0/settings::cm;

inline RGBAColour rgba(pen p) {
  p.convert();
  p.torgb();
  return RGBAColour(p.red(),p.green(),p.blue(),p.opacity());
}
  
class prcfile : public oPRCFile {
  std::list<PRCentity *> entities;
public:  
  prcfile(string name) : oPRCFile(name.c_str()) {}
  ~prcfile() {

    for(std::list<PRCentity *>::iterator p=entities.begin();
        p != entities.end(); ++p) {
      assert(*p);
      delete *p;
    }
  }

  void begingroup(string name) {}
  void endgroup() {}
  
  void add(PRCentity* e) {
    entities.push_back(e);
    oPRCFile::add(e);
  }
};
  
inline void writeBezierKnots(PRCbitStream &out, uint32_t d, uint32_t n)
{
  out << (double) 1;
  uint32_t stop=d+n;
  for(uint32_t i=1; i < stop; ++i)
    out << (double) ((i+2)/d); // integer division is intentional
  out << (double) ((stop+1)/d);
}
    
class PRCBezierCurve : public PRCcurve
{
  uint32_t d;
  uint32_t n;
public:
  PRCBezierCurve(oPRCFile *p, uint32_t d, uint32_t n, double cP[][3],
                 const RGBAColour &c) :
    PRCcurve(p,d,n,cP,NULL,c,scale3D,false,NULL), d(d), n(n) {}
  PRCBezierCurve(oPRCFile *p, uint32_t d, uint32_t n, double cP[][3],
                 const PRCMaterial &m) :
    PRCcurve(p,d,n,cP,NULL,m,scale3D,false,NULL), d(d), n(n) {}
private:
  void writeKnots(PRCbitStream &out) {
    writeBezierKnots(out,d,n);
  }
};

class PRCBezierSurface : public PRCsurface
{
  uint32_t dU,dV;
  uint32_t nU,nV;
public:
  PRCBezierSurface(oPRCFile *p, uint32_t dU, uint32_t dV, uint32_t nU,
                   uint32_t nV, double cP[][3], const RGBAColour &c,
                   double g=0.0, string name="") :
    PRCsurface(p,dU,dV,nU,nV,cP,NULL,NULL,c,scale3D,false,NULL,g,name.c_str()),
    dU(dU), dV(dV), nU(nU), nV(nV) {}
  PRCBezierSurface(oPRCFile *p, uint32_t dU, uint32_t dV, uint32_t nU,
                   uint32_t nV, double cP[][3], const PRCMaterial &m,
                   double g=0.0, string name="") :
    PRCsurface(p,dU,dV,nU,nV,cP,NULL,NULL,m,scale3D,false,NULL,g,name.c_str()),
    dU(dU), dV(dV), nU(nU), nV(nV) {}
private:
  void writeKnots(PRCbitStream &out) {
    writeBezierKnots(out,dU,nU);
    writeBezierKnots(out,dV,nV);
  }
};

} //namespace camp

#endif
