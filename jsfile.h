#ifndef JSFILE_H
#define JSFILE_H

#include <fstream>
#include <glm/glm.hpp>
#include "memory.h"
#include "locate.h"

namespace gl {
extern glm::mat4 projViewMat;
}

namespace camp {

class jsfile {
  jsofstream out;
  
public:  
  void copy(string name) {
    std::ifstream fin(settings::locateFile(name).c_str());
    string s;
    while(getline(fin,s))
      out << s << endl;
  }
  
  jsfile() {}
  
  void open(string name) {
    out.open(name);
    copy(settings::WebGLheader);
    for(size_t i=0; i < 4; ++i) {
      for(size_t j=0; j < 4; ++j)
        out << gl::projViewMat[i][j] << ", ";
      out << endl;
    }
  }
  
  ~jsfile() {
    copy(settings::WebGLfooter);
  }
  
  void addPatch(const triple* controls) {
    out << "P.push([" << endl; 
    for(size_t i=0; i < 15; ++i)
      out << controls[i] << "," << endl;
    out << controls[15] << endl << "]);" << endl << endl;
  }
  
};

} //namespace camp

#endif
