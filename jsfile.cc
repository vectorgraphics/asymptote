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
    out << "var materialIndex = 0;\n"
"     var objMaterials = [new Material(\n"
"      baseColor = [1, 1, 0, 1],\n"
"      emissive = [0, 0, 0, 1],\n"
"      specular = [1, 1, 1, 1],\n"
"      roughness = 0.15,\n"
"      metallic = 0,\n"
"      f0 = 0.04\n"
"    )];"
"    var lights = [new Light(\n"
"      type = enumDirectionalLight,\n"
"      lightColor = [1, 0.87, 0.745],\n"
"      brightness = 1,\n"
"      customParam = [0, 0, 1, 0]\n"
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
