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
  out.precision(settings::getSetting<Int>("digits"));
  copy(settings::WebGLheader);
  out << newl
      <<  "b=[" << gl::xmin << "," << gl::ymin << "," << gl::zmin << "];" 
      << newl
      <<  "B=[" << gl::xmax << "," << gl::ymax << "," << gl::zmax << "];" 
      << newl
      << "orthographic=" << std::boolalpha << gl::orthographic << ";"
      << newl
      << "angle=" << gl::Angle << ";"
      << newl
      << "canvasWidth=" << gl::fullWidth << ";" << newl
      << "canvasHeight=" << gl::fullHeight << ";" << newl
      << "size2=Math.hypot(canvasWidth,canvasHeight);" << newl
      << "Zoom0=" << gl::Zoom0 << ";" << newl << newl
      << "let lights=[";
  for(size_t i=0; i < gl::nlights; ++i) {
    size_t i4=4*i;
    out << "new Light(" << newl
        << "direction=" << gl::Lights[i] << "," << newl 
        << "color=[" << gl::Diffuse[i4] << "," << gl::Diffuse[i4+1]
        << "," << gl::Diffuse[i4+2] << "])," << newl;
  }
  out << "];" << newl << newl;
  size_t nmaterials=drawElement::material.size();
  out << "let Materials=[";
  for(size_t i=0; i < nmaterials; ++i)
    out << "new Material(" << newl
        << drawElement::material[i]
        << ")," << newl;
  out << "];" << newl << newl;
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
                      const triple& Min, const triple& Max)
{
  out << "P.push(new BezierCurve([" << newl;
  out << z0 << "," << newl
      << c0 << "," << newl
      << c1 << "," << newl
      << z1 << newl << "],"
      << drawElement::centerIndex << "," << drawElement::materialIndex << ","
      << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addCurve(const triple& z0, const triple& z1,
                      const triple& Min, const triple& Max)
{
  out << "P.push(new BezierCurve([" << newl;
  out << z0 << "," << newl
      << z1 << newl << "],"
      << drawElement::centerIndex << "," << drawElement::materialIndex << ","
      << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addPixel(const triple& z0, double width,
                      const triple& Min, const triple& Max)
{
  out << "P.push(new Pixel(" << newl;
  out << z0 << "," << width << "," << newl
      << drawElement::materialIndex << ","
      << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addMaterial(size_t index) {
  out << "Materials.push(new Material(" << newl
       << drawElement::material[index]
      << "));" << newl << newl;
}

}
