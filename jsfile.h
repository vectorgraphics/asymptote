#ifndef JSFILE_H
#define JSFILE_H

#include <fstream>
#include "memory.h"
#include "locate.h"

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
  
  jsfile(string name) {
    out.open(name);
    copy(settings::WebGLheader);
  }
  
  ~jsfile() {
    copy(settings::WebGLfooter);
  }
  
  void addPatch(const triple* controls) {
    out << "p.push([" << endl; 
    for(size_t i=0; i < 15; ++i)
      out << controls[i] << "," << endl;
    out << controls[15] << endl << "]);" << endl << endl;
  }
  
};

} //namespace camp

#endif
