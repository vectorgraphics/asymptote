#include "jsfile.h"

#include "settings.h"
// #include "glrender.h"
#include "drawelement.h"

using namespace settings;

namespace camp {

const string s="document.asy.";

#ifndef HAVE_LIBGLM
size_t materialIndex=0;
#endif

jsfile::jsfile() : finished(false), fileName("")
{

}

jsfile::jsfile(string name) : finished(false), fileName(name)
{
  open(name);
}

jsfile::~jsfile()
{
  if (!finished)
    {
      finish(fileName);
    }
}

void jsfile::close()
{
  if (!finished)
    {
      finish(fileName);
    }
}

void jsfile::copy(string name, bool header)
{
  std::ifstream fin(locateFile(name).c_str());
  string s;
  if(header) getline(fin,s);
  while(getline(fin,s))
    out << s << newl;
}

void jsfile::header(string name)
{
  out.open(name);
  out << "<!DOCTYPE html>" << newl << newl;
}

void jsfile::meta(string name, bool svg)
{
  out << "<html lang=\"\">" << newl
      << newl
      << "<head>" << newl
      << "<title>" << stripExt(name) << "</title>" << newl
      << newl
      << "<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\"/>" << newl;
  if(svg)
    out << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>";
  else
    out << "<meta name=\"viewport\" content=\"user-scalable=no\"/>";
  out << newl << "<style>" << newl;
  if(svg && !getSetting<bool>("absolute"))
    out << "svg, #container {" << newl
        << "display: block;" << newl
        << "width: 100vw;" << newl
        << "height: 100vh;" << newl
        << "}" << newl;
  out << "body {margin: 0;}" << newl
      << "</style>" << newl;
  if(svg)
    out << "</head>" << newl;
  out << newl;
}

void jsfile::footer(string name)
{
  out << newl << "</body>"
      << newl << newl << "</html>"
      << newl;
  out.flush();
  if(verbose > 0)
    cout << "Wrote " << name << endl;
}

void jsfile::svgtohtml(string prefix)
{
  string name=buildname(prefix,"html");
  header(name);
  meta(name);

  out << "<body>" << newl << newl;
  copy(locateFile(auxname(prefix,"svg")),true);
  footer(name);
  finished=true;
}

void jsfile::comment(string name)
{
#ifdef HAVE_LIBGLM
  out << "<!-- Use the following line to embed this file within another web page:" << newl
      << newl
      << "<iframe src=\"" << name
      << "\" width=\"" << vk->fullWidth
      << "\" height=\"" << vk->fullHeight
      << "\" frameborder=\"0\"></iframe>" << newl
      << newl
      << "-->" << newl << newl;
#endif
}

void jsfile::open(string name)
{
  header(name);
  comment(name);
  meta(name,false);

#ifdef HAVE_LIBGLM
  out.precision(getSetting<Int>("digits"));

  bool ibl=getSetting<bool>("ibl");
  bool webgl2=ibl || getSetting<bool>("webgl2");
  if(ibl)
    out << "<script src=\"https://vectorgraphics.gitlab.io/asymptote/ibl/tinyexr.js\">"
        << newl << "</script>" << newl;

  if(getSetting<bool>("offline")) {
    out << "<script>" << newl;
    copy(locateFile(AsyGL));
    out << newl << "</script>" << newl;
  } else
    out << "<script" << newl << "src=\""
        << getSetting<string>("asygl") << "\">" << newl << "</script>" << newl;

  out << newl << "<script>" << newl;
  out << newl
      << s << "canvasWidth=" << vk->fullWidth << ";" << newl
      << s << "canvasHeight=" << vk->fullHeight << ";" << newl << newl
      << s << "webgl2=" << std::boolalpha << webgl2 << ";"
      << newl
      << s << "ibl=" << std::boolalpha << ibl << ";"
      << newl
      << s << "absolute=" << std::boolalpha << getSetting<bool>("absolute") << ";"
      << newl;
  if(ibl) {
    out << s << "imageURL=\"" << getSetting<string>("imageURL")+"/\";" << newl;
    out << s << "image=\"" << getSetting<string>("image") << "\";" << newl << newl;
  }
  out << newl
      <<  s << "minBound=[" << vk->Xmin << "," << vk->Ymin << "," << vk->Zmin << "];"
      << newl
      <<  s << "maxBound=[" << vk->Xmax << "," << vk->Ymax << "," << vk->Zmax << "];"
      << newl
      << s << "orthographic=" << vk->orthographic << ";"
      << newl
      << s << "angleOfView=" << vk->Angle << ";"
      << newl
      << s << "initialZoom=" << vk->Zoom0 << ";" << newl;
    if(vk->Shift != pair(0.0,0.0))
      out << s << "viewportShift=" << vk->Shift*vk->Zoom0 << ";" << newl;
    out << s << "viewportMargin=" << vk->Margin << ";" << newl << newl
        << s << "zoomFactor=" << getSetting<double>("zoomfactor") << ";" << newl
        << s << "zoomPinchFactor=" << getSetting<double>("zoomPinchFactor") << ";"
      << newl
        << s << "zoomPinchCap=" << getSetting<double>("zoomPinchCap") << ";" << newl
        << s << "zoomStep=" << getSetting<double>("zoomstep") << ";" << newl
        << s << "shiftHoldDistance=" << getSetting<double>("shiftHoldDistance") << ";"
      << newl
        << s << "shiftWaitTime=" << getSetting<double>("shiftWaitTime") << ";"
      << newl
        << s << "vibrateTime=" << getSetting<double>("vibrateTime") << ";"
        << newl << newl;
  out << s << "background=[" << vk->Background[0] << "," << vk->Background[1] << ","
      << vk->Background[2] << "," << vk->Background[3] << "];"
      << newl << newl;
  out << s << "Transform=[" << vk->T[0];
  for(int i=1; i < 16; ++i)
    out << "," << newl << vk->T[i];
  out << "];" << newl << newl;

  for(size_t i=0; i < vk->nlights; ++i) {
    size_t i4=4*i;
    out << "light(" << newl
        << vk->Lights[i] << "," << newl
        << "[" << vk->LightsDiffuse[i4] << "," << vk->LightsDiffuse[i4+1] << ","
        << vk->LightsDiffuse[i4+2] << "]);" << newl;
  }
  out << newl;

  vk->clearCenters();
  vk->clearMaterials();
#endif
}

void jsfile::finish(string name)
{
#ifdef HAVE_LIBGLM
  finished=true;
  size_t ncenters=drawElement::centers.size();
  if(ncenters > 0) {
    out << s << "Centers=[";
    for(size_t i=0; i < ncenters; ++i)
      out << newl << drawElement::centers[i] << ",";
    out << newl << "];" << newl;
  }
  out << "</script>"
      << newl << "</head>"
      << newl << newl << "<body style=\"overflow: hidden;\" onload=\"webGLStart();\">"
      << newl << "<canvas id=\"Asymptote\" width=\""
      << vk->fullWidth << "\" height=\"" <<  vk->fullHeight
      << "\" style=\"border: none; cursor: pointer;\">"
      << newl << "</canvas>";
  footer(name);
#endif
}

void jsfile::addColor(const prc::RGBAColour& c)
{
  out << "[" << c.R << "," << c.G << "," << c.B << "," << c.A << "]";
}

void jsfile::addIndices(const uint32_t *I)
{
  out << "[" << I[0] << "," << I[1] << "," << I[2] << "]";
}

void jsfile::addRawPatch(triple const* controls, size_t n,
                         const prc::RGBAColour *c, size_t nc)
{
  out << "patch([" << newl;
  size_t last=n-1;
  for(size_t i=0; i < last; ++i)
    out << controls[i] << "," << newl;
  out << controls[last] << newl << "],"
      << drawElement::centerIndex << "," << vk->materialIndex;
  if(c) {
    out << ",[" << newl;
    for(size_t i=0; i < nc; ++i) {
      addColor(c[i]);
      out << "," << newl;
    }
    out << "]";
  }
  out << ");" << newl << newl;
}

void jsfile::addCurve(const triple& z0, const triple& c0,
                      const triple& c1, const triple& z1)
{
  out << "curve([" << newl;
  out << z0 << "," << newl
      << c0 << "," << newl
      << c1 << "," << newl
      << z1 << newl << "],"
      << drawElement::centerIndex << "," << vk->materialIndex
      << ");" << newl << newl;
}

void jsfile::addCurve(const triple& z0, const triple& z1)
{
  out << "curve([" << newl;
  out << z0 << "," << newl
      << z1 << newl << "],"
      << drawElement::centerIndex << "," << vk->materialIndex
      << ");" << newl << newl;
}

void jsfile::addPixel(const triple& z0, double width)
{
  out << "pixel(" << newl;
  out << z0 << "," << width << "," << newl << vk->materialIndex
      << ");" << newl << newl;
}

#ifdef HAVE_LIBGLM
void jsfile::addMaterial(Material const& material)
{
  out << "material(" << newl
      << material
      << ");" << newl << newl;
}
#endif

void jsfile::addTriangles(size_t nP, const triple* P, size_t nN,
                          const triple* N, size_t nC, const prc::RGBAColour* C,
                          size_t nI, const uint32_t (*PI)[3],
                          const uint32_t (*NI)[3], const uint32_t (*CI)[3])
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
    const uint32_t *PIi=PI[i];
    const uint32_t *NIi=NI[i];
    bool keepNI=distinct(NIi,PIi);
    bool keepCI=nC && distinct(CI[i],PIi);
    addIndices(PIi);
    if(keepNI || keepCI) {
      out << ",";
      if(keepNI) addIndices(NIi);
    }
    if(keepCI) {
      out << ",";
      addIndices(CI[i]);
    }
    out << "]);" << newl;
  }
  out << "triangles("
      << drawElement::centerIndex << "," << vk->materialIndex
      << ");" << newl << newl;
}

void jsfile::addSphere(const triple& center, double radius)
{
  out << "sphere(" << center << "," << radius << ","
      << drawElement::centerIndex << "," << vk->materialIndex
      << ");" << newl << newl;
}

void jsfile::addHemisphere(const triple& center, double radius,
                           const double& polar, const double& azimuth)
{
  out << "sphere(" << center << "," << radius << ","
      << drawElement::centerIndex << "," << vk->materialIndex
      << "," << newl << "[" << polar << "," << azimuth << "]";
  out << ");" << newl << newl;
}

// core signifies whether to also draw a central line for better small-scale
// rendering.
void jsfile::addCylinder(const triple& center, double radius, double height,
                         const double& polar, const double& azimuth,
                         bool core)
{
  out << "cylinder(" << center << "," << radius << "," << height << ","
      << drawElement::centerIndex << "," << vk->materialIndex
      << "," << newl << "[" << polar << "," << azimuth << "]," << core
      << ");" << newl << newl;
}

void jsfile::addDisk(const triple& center, double radius,
                     const double& polar, const double& azimuth)
{
  out << "disk(" << center << "," << radius << ","
      << drawElement::centerIndex << "," << vk->materialIndex
      << "," << newl << "[" << polar << "," << azimuth << "]"
      << ");" << newl << newl;
}

void jsfile::addTube(const triple *g, double width, bool core)
{
  out << "tube(["
      << g[0] << "," << newl
      << g[1] << "," << newl
      << g[2] << "," << newl
      << g[3] << newl << "],"
      << width << ","
      << drawElement::centerIndex << "," << vk->materialIndex << "," << core
      << ");" << newl << newl;
}

void jsfile::addPatch(triple const* controls, prc::RGBAColour const* c)
{
  addRawPatch(controls,16,c,4);
}

void jsfile::addStraightPatch(triple const* controls,
                              prc::RGBAColour const* c)
{
  addRawPatch(controls,4,c,4);
}

void jsfile::addBezierTriangle(triple const* controls,
                               prc::RGBAColour const* c)
{
  addRawPatch(controls,10,c,3);
}

void jsfile::addStraightBezierTriangle(triple const* controls,
                                       prc::RGBAColour const* c)
{
  addRawPatch(controls,3,c,3);
}

}
