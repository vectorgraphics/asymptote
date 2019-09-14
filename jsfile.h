#ifndef JSFILE_H
#define JSFILE_H

#include <fstream>
#include <glm/glm.hpp>

#include "common.h"
#include "triple.h"
#include "locate.h"
#include "prcfile.h"

namespace gl {
extern glm::mat4 projViewMat;
}

namespace camp {

class jsfile {
  jsofstream out;
  
public:  
  jsfile() {}
  ~jsfile();
  
  void open(string name);
  void copy(string name);
  
  void addPatch(const triple* controls, size_t n, const triple& Min,
                const triple& Max, const prc::RGBAColour *colors);
  
  void addCurve(const triple& z0, const triple& c0,
                const triple& c1, const triple& z1,
                const triple& Min, const triple& Max,
                const prc::RGBAColour color);
  
  void addMaterial(size_t index);
};

} //namespace camp

#endif
