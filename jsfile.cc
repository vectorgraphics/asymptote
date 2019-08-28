#include "jsfile.h"

#include "settings.h"
#include "glrender.h"

namespace gl {
extern glm::mat4 projViewMat;
};

namespace camp {
  void jsfile::copy(string name) {
    std::ifstream fin(settings::locateFile(name).c_str());
    string s;
    while(getline(fin,s))
      out << s << endl;
  }

  void jsfile::open(string name) {
        out.open(name);
    copy(settings::WebGLheader);
    out <<  "target=" << 0.5*(gl::zmin+gl::zmax) << ";" << endl;

    out << "pMatrix=new Float32Array([" << endl;
    for(size_t i=0; i < 4; ++i) {
      for(size_t j=0; j < 4; ++j)
        out << gl::projViewMat[i][j] << ", ";
      out << endl;
    }
    out << "]);" << endl;
    out <<
      "canvasWidth=" << gl::fullWidth << ";" << endl << 
      "canvasHeight=" << gl::fullHeight << ";" << endl;
    out <<
"     var materialIndex = 0;"
"     var objMaterials = [new Material("
"      baseColor = [1, 1, 0, 1],"
"      emissive = [0, 0, 0, 1],"
"      specular = [1, 1, 1, 1],"
"      roughness = 0.15,"
"      metallic = 0,"
"      f0 = 0.04"
"    )];"
"    var lights = [new Light("
"      type = enumDirectionalLight,"
"      lightColor = [1, 0.87, 0.745],"
"      brightness = 1,"
"      customParam = [0, 0, 1, 0]"
"    )];"
        << endl;
  }

  jsfile::~jsfile() {
        copy(settings::WebGLfooter);
  }

  void jsfile::addPatch(triple const* controls) {
    out << "P.push(new BezierPatch([" << endl; 
    for(size_t i=0; i < 15; ++i)
      out << controls[i] << "," << endl;
    out << controls[15] << endl << "]));" << endl << endl;
  }

};
