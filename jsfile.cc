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
}

void jsfile::open(string name) {
  out.open(name);
  copy(settings::WebGLheader);
  out << newl
      <<  "b=[" << gl::xmin << "," << gl::ymin << "," << gl::zmin << "];" 
      << newl;
  out <<  "B=[" << gl::xmax << "," << gl::ymax << "," << gl::zmax << "];" 
      << newl;
  out << "orthographic=" << std::boolalpha << gl::orthographic << ";"
      << newl;
  out << "angle=" << gl::Angle << ";"
      << newl
      << "canvasWidth=" << gl::fullWidth << ";" << newl
      << "canvasHeight=" << gl::fullHeight << ";" << newl
      << "size2=Math.hypot(canvasWidth,canvasHeight);" << newl
      << "Zoom0=" << gl::Zoom0 << ";" << newl;
  
  out << 
    "    var lights = [new Light(\n"
    "      type = enumDirectionalLight,\n"
    "      lightColor = [1, 1, 1],\n"
    "      brightness = 1,\n"
    "      customParam = [0.235702260395516,-0.235702260395516,0.942809041582063, 0]\n"
    "    )];"
      << newl;
}

jsfile::~jsfile() {
  size_t ncenters=drawElement::center.size();
  if(ncenters > 0) {
    out << "Centers=[";
    for(size_t i=0; i < ncenters; ++i)
      out << newl << drawElement::center[i] << ",";
    out << newl << "];" << newl;
  }
  copy(settings::WebGLfooter);
}

void jsfile::addPatch(triple const* controls, size_t n,
                      const triple& Min, const triple& Max,
                      const prc::RGBAColour *c)
{
  out << "P.push(new BezierPatch([" << newl;
  size_t last=n-1;
  for(size_t i=0; i < last; ++i)
    out << controls[i] << "," << newl;
  out << controls[last] << newl << "]," 
      << drawElement::centerIndex << "," << drawElement::materialIndex << ","
      << Min << "," << Max;
  if(c) {
    out << ",[" << newl;
    for(int i=0; i < 4; ++i)
      out << "[" << byte(c[i].R) << "," << byte(c[i].G) << "," << byte(c[i].B)
          << "," << byte(c[i].A) << "]," << newl;
    out << "]" << newl;
  }
  out << "));" << newl << newl;
}

void jsfile::addCurve(const triple& z0, const triple& c0,
                      const triple& c1, const triple& z1,
                      const triple& Min, const triple& Max,
                      const prc::RGBAColour color)
{
  out << "P.push(new BezierCurve([" << newl;
  out << z0 << "," << newl
      << c0 << "," << newl
      << c1 << "," << newl
      << z1 << newl << "],"
      << drawElement::centerIndex << "," << drawElement::materialIndex << ","
      << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addMaterial(size_t index) {
  out << "M.push(new Material("
      << drawElement::material[index]
      << "));" << newl << newl;
}

}
