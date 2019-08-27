#ifndef JSFILE_H
#define JSFILE_H

#include <fstream>
#include <glm/glm.hpp>
#include "memory.h"
#include "triple.h"
#include "locate.h"

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
  void addPatch(const triple* controls);
};

} //namespace camp

#endif
