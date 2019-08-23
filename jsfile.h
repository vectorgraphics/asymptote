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
    out << "pMatrix=new Float32Array([" << endl;
    for(size_t i=0; i < 4; ++i) {
      for(size_t j=0; j < 4; ++j)
        out << gl::projViewMat[i][j] << ", ";
      out << endl;
    }
    out << "]);" << endl;
    out <<
"    var materialIndex = 0;"
"    var objMaterial = new Material("
"      baseColor = [1, 1, 0, 1],"
"      emissive = [0, 0, 0, 1],"
"      specular = [1, 1, 1, 1],"
"      roughness = 0.15,"
"      metallic = 0,"
"      f0 = 0.04"
"    );"
""
"    var L = [0.447735768366173, 0.497260947684137, 0.743144825477394];"
"    var Ambient = [0.1, 0.1, 0.1];"
"    var Diffuse = [0.8, 0.8, 0.8, 1];"
"    var Specular = [0.7, 0.7, 0.7, 1];"
"    var specularfactor = 3;"
""
"    var emissive = [0, 0, 0, 1];"
"    var ambient = [0, 0, 0, 1];"
"    var diffuse = [1, 0, 0, 1];"
"    var specular = [0.75, 0.75, 0.75, 1];"
"    var shininess = 0.5;"
""
"    var cameraPos = vec3.fromValues(0, 0, 2);"
"    var cameraLookAt = vec3.fromValues(0, 0, 0);"
"    var cameraUp = vec3.fromValues(1, 0, 0);"
        << endl;
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
