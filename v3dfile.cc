/*
 * v3dfile.cc
 * V3D Export class
 * Written by: Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com> \
 *   and John C. Bowman <bowman@ualberta.ca>
 */

#include "v3dfile.h"
#include "drawelement.h"

namespace camp
{

using settings::getSetting;
using std::make_unique;

absv3dfile::absv3dfile() : finalized(false), singleprecision(false)
{
}

absv3dfile::absv3dfile(bool singleprecision) : finalized(false), singleprecision(singleprecision)
{
}

void absv3dfile::writeInit()
{
  uint32_t doubleprecision = !singleprecision;
  getXDRFile() << v3dVersion << doubleprecision;
  addHeaders();
}

void absv3dfile::addHeaders()
{
  getXDRFile() << v3dtypes::header;
  std::vector<std::unique_ptr<AHeader>> headers;

  headers.emplace_back(make_unique<Uint32Header>(v3dheadertypes::canvasWidth, gl::fullWidth));
  headers.emplace_back(make_unique<Uint32Header>(v3dheadertypes::canvasHeight, gl::fullHeight));
  headers.emplace_back(make_unique<Uint32Header>(v3dheadertypes::absolute, getSetting<bool>("absolute")));
  headers.emplace_back(make_unique<TripleHeader>(v3dheadertypes::b, triple(gl::xmin, gl::ymin, gl::zmin)));
  headers.emplace_back(make_unique<TripleHeader>(v3dheadertypes::B, triple(gl::xmax, gl::ymax, gl::zmax)));
  headers.emplace_back(make_unique<Uint32Header>(v3dheadertypes::orthographic, gl::orthographic));
  headers.emplace_back(make_unique<DoubleFloatHeader>(v3dheadertypes::angle_, gl::Angle));
  headers.emplace_back(make_unique<DoubleFloatHeader>(v3dheadertypes::Zoom0, gl::Zoom0));
  headers.emplace_back(make_unique<PairHeader>(v3dheadertypes::viewportMargin, gl::Margin));

  if (gl::Shift!=pair(0.0,0.0))
  {
    headers.emplace_back(make_unique<PairHeader>(v3dheadertypes::viewportShift, gl::Shift*gl::Zoom0));
  }

  for(size_t i=0; i < gl::nlights; ++i)
  {
    size_t i4=4*i;
    headers.emplace_back(make_unique<LightHeader>(
            gl::Lights[i],
            prc::RGBAColour(gl::Diffuse[i4], gl::Diffuse[i4+1], gl::Diffuse[i4+2], 1.0)
            ));
  }

  headers.emplace_back(make_unique<RGBAHeader>(
          v3dheadertypes::background,
          prc::RGBAColour(gl::Background[0],gl::Background[1],gl::Background[2],gl::Background[3])));

  headers.emplace_back(make_unique<DoubleFloatHeader>(v3dheadertypes::zoomFactor, getSetting<double>("zoomfactor")));
  headers.emplace_back(make_unique<DoubleFloatHeader>(
          v3dheadertypes::zoomPinchFactor, getSetting<double>("zoomPinchFactor")));
  headers.emplace_back(make_unique<DoubleFloatHeader>(
          v3dheadertypes::zoomPinchCap, getSetting<double>("zoomPinchCap")));
  headers.emplace_back(make_unique<DoubleFloatHeader>(v3dheadertypes::zoomStep, getSetting<double>("zoomstep")));
  headers.emplace_back(make_unique<DoubleFloatHeader>(
          v3dheadertypes::shiftHoldDistance, getSetting<double>("shiftHoldDistance")));
  headers.emplace_back(make_unique<DoubleFloatHeader>(
          v3dheadertypes::shiftWaitTime, getSetting<double>("shiftWaitTime")));
  headers.emplace_back(make_unique<DoubleFloatHeader>(
          v3dheadertypes::vibrateTime, getSetting<double>("vibrateTime")));


  getXDRFile() << (uint32_t)headers.size();
  for (auto const& headerObj : headers)
  {
    getXDRFile() << *headerObj;
  }
}

void absv3dfile::addCenters()
{
  getXDRFile() << v3dtypes::centers;
  size_t nelem=drawElement::center.size();
  getXDRFile() << (uint32_t) nelem;
  if (nelem > 0)
    addTriples(drawElement::center.data(), nelem);
}

void absv3dfile::addTriples(triple const* triples, size_t n)
{
  for(size_t i=0; i < n; ++i)
    getXDRFile() << triples[i];
}

void absv3dfile::addColors(prc::RGBAColour const* col, size_t nc)
{
  for(size_t i=0; i < nc; ++i)
    getXDRFile() << col[i];
}


void absv3dfile::addPatch(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  getXDRFile() << (c == nullptr ? v3dtypes::bezierPatch : v3dtypes::bezierPatchColor);
  addTriples(controls, 16);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 4);
  }
}

void absv3dfile::addStraightPatch(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  getXDRFile() << (c == nullptr ? v3dtypes::quad : v3dtypes::quadColor);
  addTriples(controls, 4);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 4);
  }
}

void absv3dfile::addBezierTriangle(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  getXDRFile() << (c == nullptr ? v3dtypes::bezierTriangle : v3dtypes::bezierTriangleColor);
  addTriples(controls, 10);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 3);
  }
}

void absv3dfile::addStraightBezierTriangle(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  getXDRFile() << (c == nullptr ? v3dtypes::triangle : v3dtypes::triangleColor);
  addTriples(controls, 3);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 3);
  }
}

void absv3dfile::addMaterial(Material const& mat)
{
  getXDRFile() << v3dtypes::material_;
  addvec4(mat.diffuse);
  addvec4(mat.emissive);
  addvec4(mat.specular);
  addvec4(mat.parameters);
}

void absv3dfile::addCenterIndexMat()
{
  getXDRFile() << (uint32_t) drawElement::centerIndex << (uint32_t) materialIndex;
}

void absv3dfile::addvec4(glm::vec4 const& vec)
{
  getXDRFile() << static_cast<float>(vec.x) << static_cast<float>(vec.y)
    << static_cast<float>(vec.z) << static_cast<float>(vec.w);
}

void absv3dfile::addHemisphere(triple const& center, double radius, double const& polar, double const& azimuth)
{
  getXDRFile() << v3dtypes::halfSphere << center << radius;
  addCenterIndexMat();
  getXDRFile() << polar << azimuth;
}

void absv3dfile::addSphere(triple const& center, double radius)
{
  getXDRFile() << v3dtypes::sphere << center << radius;
  addCenterIndexMat();
}

void
absv3dfile::addCylinder(triple const& center, double radius, double height, double const& polar, double const& azimuth,
                     bool core)
{
  getXDRFile() << v3dtypes::cylinder << center << radius << height;
  addCenterIndexMat();
  getXDRFile() << polar << azimuth << core;
}

void absv3dfile::addDisk(triple const& center, double radius, double const& polar, double const& azimuth)
{
  getXDRFile() << v3dtypes::disk << center << radius;
  addCenterIndexMat();
  getXDRFile() << polar << azimuth;
}

void absv3dfile::addTube(triple const* g, double width, triple const& Min, triple const& Max, bool core)
{
  getXDRFile() << v3dtypes::tube;
  for (int i=0;i<4;++i)
  {
    getXDRFile() << g[i];
  }
  getXDRFile() << width;
  addCenterIndexMat();
  getXDRFile() << core;
}

void absv3dfile::addTriangles(size_t nP, triple const* P, size_t nN, triple const* N, size_t nC, prc::RGBAColour const* C,
                           size_t nI, uint32_t const (* PI)[3], uint32_t const (* NI)[3], uint32_t const (* CI)[3],
                           triple const& Min, triple const& Max)
{
  getXDRFile() << v3dtypes::triangles;
  getXDRFile() << (uint32_t) nP;
  addTriples(P,nP);
  getXDRFile() << (uint32_t) nN;
  addTriples(N,nN);

  getXDRFile() << (uint32_t) nC;
  if(nC > 0)
    addColors(C,nC);

  getXDRFile() << (uint32_t) nI;

  for(size_t i=0; i < nI; ++i) {
    const uint32_t *PIi=PI[i];
    const uint32_t *NIi=NI[i];
    addIndices(PIi);
    bool keepNI=distinct(NIi,PIi);
    getXDRFile() << (uint32_t) keepNI;
    if(keepNI)
      addIndices(NIi);
    if(nC) {
      const uint32_t *CIi=CI[i];
      bool keepCI=distinct(CIi,PIi);
      getXDRFile() << (uint32_t) keepCI;
      if(keepCI)
        addIndices(CIi);
    }
  }

  getXDRFile() << (uint32_t) materialIndex;
}

void absv3dfile::addIndices(uint32_t const* v)
{
  getXDRFile() << v[0] << v[1] << v[2];
}

void absv3dfile::addCurve(triple const& z0, triple const& c0, triple const& c1, triple const& z1, triple const& Min,
                       triple const& Max)
{
  getXDRFile() << v3dtypes::curve << z0 << c0 << c1 << z1;
  addCenterIndexMat();

}

void absv3dfile::addCurve(triple const& z0, triple const& z1, triple const& Min, triple const& Max)
{
  getXDRFile() << v3dtypes::line << z0 << z1;
  addCenterIndexMat();
}

void absv3dfile::addPixel(triple const& z0, double width, triple const& Min, triple const& Max)
{
 getXDRFile() << v3dtypes::pixel_ << z0 << width;
 getXDRFile() << (uint32_t) materialIndex;
}

void absv3dfile::precision(int digits)
{
  // inert function for override
}

void absv3dfile::finalize()
{
  if (!finalized)
  {
    addCenters();
    finalized=true;
  }
}

// for headers

xdr::oxstream& operator<<(xdr::oxstream& ox, AHeader const& header)
{
  ox << (uint32_t)header.ty << header.getByteSize();
  header.writeContent(ox);
  return ox;
}

// gzv3dfile

xdr::oxstream& gzv3dfile::getXDRFile()
{
  return memxdrfile;
}

gzv3dfile::gzv3dfile(string const& name, bool singleprecision): absv3dfile(singleprecision), memxdrfile(singleprecision), name(name), destroyed(false)
{
  writeInit();
}

gzv3dfile::~gzv3dfile()
{
  close();
}

void gzv3dfile::close()
{
  if (!destroyed)
  {
    finalize();
    memxdrfile.close();
    gzFile fil = gzopen(name.c_str(), "wb9");
    gzwrite(fil, data(), length());
    gzclose(fil);
    destroyed=true;
  }
}

char const* gzv3dfile::data() const
{
  return memxdrfile.stream();
}

size_t const& gzv3dfile::length() const
{
  return memxdrfile.getLength();
}

uint32_t LightHeader::getByteSize() const
{
  return (TRIPLE_DOUBLE_SIZE + RGBA_FLOAT_SIZE)/4;
}

void LightHeader::writeContent(xdr::oxstream& ox) const
{
  ox << direction << color;

}

LightHeader::LightHeader(triple const& direction, prc::RGBAColour const& color) :
        AHeader(v3dheadertypes::light), direction(direction), color(color)
{
}
} //namespace camp
