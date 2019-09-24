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
  out << "<!DOCTYPE html>" << newl << newl
    
      << "<!-- Use the following line to include this file within another web page:" << newl
      << newl
      << "<object data=\"" << name <<"\" style=\"width:"
      << gl::fullWidth << ";height:" << gl::fullHeight
      << ";position:relative;top:0;left:0;\"></object>" << newl << newl
      << "-->" << newl << newl;

  out.precision(settings::getSetting<Int>("digits"));
  copy(settings::WebGLheader);
  out << newl
      << "canvasWidth=" << gl::fullWidth << ";" << newl
      << "canvasHeight=" << gl::fullHeight << ";" << newl << newl
      <<  "b=[" << gl::xmin << "," << gl::ymin << "," << gl::zmin << "];" 
      << newl
      <<  "B=[" << gl::xmax << "," << gl::ymax << "," << gl::zmax << "];" 
      << newl
      << "orthographic=" << std::boolalpha << gl::orthographic << ";"
      << newl
      << "angle=" << gl::Angle << ";"
      << newl
       << "Zoom0=" << gl::Zoom0 << ";" << newl << newl
      << "size2=Math.hypot(canvasWidth,canvasHeight);" << newl << newl
      << "let lights=[";
  for(size_t i=0; i < gl::nlights; ++i) {
    size_t i4=4*i;
    out << "new Light(" << newl
        << "direction=" << gl::Lights[i] << "," << newl 
        << "color=[" << gl::Diffuse[i4] << "," << gl::Diffuse[i4+1]
        << "," << gl::Diffuse[i4+2] << "])," << newl;
  }
  out << "];" << newl << newl;
  size_t nmaterials=material.size();
  out << "let Materials=[";
  for(size_t i=0; i < nmaterials; ++i)
    out << "new Material(" << newl
        << material[i]
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

void jsfile::addColor(const prc::RGBAColour& c) 
{
  out << "[" << byte(c.R) << "," << byte(c.G) << "," << byte(c.B)
      << "," << byte(c.A) << "]";
}

void jsfile::addIndices(const uint32_t *I) 
{
  out << "[" << I[0] << "," << I[1] << "," << I[2] << "]";
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
      << drawElement::centerIndex << "," << materialIndex << ","
      << Min << "," << Max;
  if(c) {
    out << ",[" << newl;
    for(int i=0; i < 4; ++i) {
      addColor(c[i]);
      out << "," << newl;
    }
    out << "]";
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
      << drawElement::centerIndex << "," << materialIndex << ","
      << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addCurve(const triple& z0, const triple& z1,
                      const triple& Min, const triple& Max)
{
  out << "P.push(new BezierCurve([" << newl;
  out << z0 << "," << newl
      << z1 << newl << "],"
      << drawElement::centerIndex << "," << materialIndex << ","
      << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addPixel(const triple& z0, double width,
                      const triple& Min, const triple& Max)
{
  out << "P.push(new Pixel(" << newl;
  out << z0 << "," << width << "," << newl
      << materialIndex << "," << Min << "," << Max << "));" << newl << newl;
}

void jsfile::addMaterial(size_t index)
{
  out << "Materials.push(new Material(" << newl
       << material[index]
      << "));" << newl << newl;
}

void jsfile::addTriangles(size_t nP, const triple* P, size_t nN,
                          const triple* N, size_t nC, const prc::RGBAColour* C,
                          size_t nI, const uint32_t (*PI)[3],
                          const uint32_t (*NI)[3], const uint32_t (*CI)[3],
                          const triple& Min, const triple& Max)
{
  for(size_t i=0; i < nP; ++i)
    out << "Positions.push(" << P[i] << ");" << newl;
  
  for(size_t i=0; i < nN; ++i)
    out << "Normals.push(" << N[i] << ");" << newl;
  
  for(size_t i=0; i < nC; ++i) {
    out << "Colors.push(";
    addColor(C[i]);
    out << ");" << newl;
  }
  
  for(size_t i=0; i < nI; ++i) {
    out << "Indices.push(["; 
    addIndices(PI[i]);
    out << ",";
    addIndices(NI[i]);
    if(nC) {
      out << ",";
      addIndices(CI[i]);
    }
    out << "]);" << newl;
  }
  out << "P.push(new Triangles("
      << materialIndex << "," << newl
      << Min << "," << Max << "));" << newl;
  out << newl;
}

}
