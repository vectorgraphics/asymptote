#include "jsfile.h"

#include "settings.h"
#include "glrender.h"
#include "drawelement.h"

namespace gl {
extern glm::mat4 projViewMat;
};


namespace camp {

void jsfile::copy(string name) {
  std::ifstream fin(settings::locateFile(name).c_str());
  string s;
  while(getline(fin,s))
    out << s << newl;
  out.flush();
}

void jsfile::open(string name) {
  out.open(name);
  copy(settings::WebGLheader);
  out <<  "b=[" << gl::xmin << "," << gl::ymin << "," << gl::zmin << "];" 
      << newl;
  out <<  "B=[" << gl::xmax << "," << gl::ymax << "," << gl::zmax << "];" 
      << newl;
  out << "orthographic=" << std::boolalpha << gl::orthographic << ";"
      << newl;
  out << "angle=" << gl::Angle << ";"
      << newl;

  out <<
    "canvasWidth=" << gl::fullWidth << ";" << newl << 
    "canvasHeight=" << gl::fullHeight << ";" << newl << 
    "size2=Math.hypot(canvasWidth,canvasHeight);" << newl <<
    "Zoom0=" << gl::Zoom0 << newl;
  
  out << 
    "    var lights = [new Light(\n"
    "      type = enumDirectionalLight,\n"
    "      lightColor = [1, 0.87, 0.745],\n"
    "      brightness = 1,\n"
    "      customParam = [0, 0, 1, 0]\n"
    "    )];"
      << newl;

}

jsfile::~jsfile() {
  copy(settings::WebGLfooter);
}

void jsfile::addPatch(triple const* controls, const triple& Min,
                      const triple& Max, const prc::RGBAColour *c)
{
  out << "P.push(new BezierPatch([" << newl;
  for(size_t i=0; i < 15; ++i)
    out << controls[i] << "," << newl;
  out << controls[15] << newl << "]," 
      << drawElement::materialIndex << ",";
  out << Min << "," << Max;
  if(c) {
    out << ",[" << newl;
    for(int i=0; i < 4; ++i)
      out << "[" << c[i].R << "," << c[i].G << "," << c[i].B << "," << c[i].A
          << "]," << newl;
    out << "]" << newl;
  }
  out << "));" << newl;
}

void jsfile::addMaterial(size_t index) {
  out << "M.push(new Material("
      << drawElement::material[index]
      << "));" << newl << newl;
}

}
