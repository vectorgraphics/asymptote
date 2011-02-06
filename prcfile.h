#ifndef PRCFILE_H
#define PRCFILE_H

#include "memory.h"
#include "prc/oPRCFile.h"

namespace camp {

inline RGBAColour rgba(pen p) {
  p.convert();
  p.torgb();
  return RGBAColour(p.red(),p.green(),p.blue(),p.opacity());
}
  
static const double inches=72;
static const double cm=inches/2.54;

class prcfile : public oPRCFile {
public:  
  prcfile(string name) : oPRCFile(name.c_str(),10.0/cm) {} // Use bp.
};

} //namespace camp

#endif
