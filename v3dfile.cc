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



v3dfile::v3dfile(string const& name, open_mode mode) :
  xdrfile(name.c_str(), mode), finished(false)
{
  xdrfile << v3dVersion;
  addHeaders();
}

v3dfile::~v3dfile()
{
  closeFile();
}

void v3dfile::addHeaders()
{
  xdrfile << v3dtypes::header;
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


  xdrfile << (uint32_t)headers.size();
  for (auto const& headerObj : headers)
  {
    xdrfile << *headerObj;
  }
}


void v3dfile::close()
{
  closeFile();
}

void v3dfile::closeFile()
{
  if (!finished)
  {
    finished = true;
    addCenters();
    xdrfile.close();
  }
}

void v3dfile::addCenters()
{
  xdrfile << v3dtypes::centers;
  size_t nelem=drawElement::center.size();
  xdrfile << (uint32_t) nelem;
  if (nelem > 0)
    addTriples(drawElement::center.data(), nelem);
}

void v3dfile::addTriples(triple const* triples, size_t n)
{
  for(size_t i=0; i < n; ++i)
    xdrfile << triples[i];
}

void v3dfile::addColors(prc::RGBAColour const* col, size_t nc)
{
  for(size_t i=0; i < nc; ++i)
    xdrfile << col[i];
}


void v3dfile::addPatch(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  xdrfile << (c == nullptr ? v3dtypes::bezierPatch : v3dtypes::bezierPatchColor);
  addTriples(controls, 16);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 4);
  }
  xdrfile << Min << Max;
}

void v3dfile::addStraightPatch(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  xdrfile << (c == nullptr ? v3dtypes::quad : v3dtypes::quadColor);
  addTriples(controls, 4);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 4);
  }
  xdrfile << Min << Max;
}

void v3dfile::addBezierTriangle(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  xdrfile << (c == nullptr ? v3dtypes::bezierTriangle : v3dtypes::bezierTriangleColor);
  addTriples(controls, 10);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 3);
  }
  xdrfile << Min << Max;
}

void v3dfile::addStraightBezierTriangle(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  xdrfile << (c == nullptr ? v3dtypes::triangle : v3dtypes::triangleColor);
  addTriples(controls, 3);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 3);
  }
  xdrfile << Min << Max;
}

void v3dfile::addMaterial(Material const& mat)
{
  xdrfile << v3dtypes::material_;
  addvec4(mat.diffuse);
  addvec4(mat.emissive);
  addvec4(mat.specular);
  addvec4(mat.parameters);
}

void v3dfile::addCenterIndexMat()
{
  xdrfile << (uint32_t) drawElement::centerIndex << (uint32_t) materialIndex;
}

void v3dfile::addvec4(glm::vec4 const& vec)
{
  xdrfile << static_cast<float>(vec.x) << static_cast<float>(vec.y)
    << static_cast<float>(vec.z) << static_cast<float>(vec.w);
}

void v3dfile::addSphereHalf(triple const& center, double radius, double const& polar, double const& azimuth)
{
  xdrfile << v3dtypes::halfSphere << center << radius;
  addCenterIndexMat();
  xdrfile << polar << azimuth;
}

void v3dfile::addSphere(triple const& center, double radius)
{
  xdrfile << v3dtypes::sphere << center << radius;
  addCenterIndexMat();
}

void
v3dfile::addCylinder(triple const& center, double radius, double height, double const& polar, double const& azimuth,
                     bool core)
{
  xdrfile << v3dtypes::cylinder << center << radius << height;
  addCenterIndexMat();
  xdrfile << polar << azimuth << core;
}

void v3dfile::addDisk(triple const& center, double radius, double const& polar, double const& azimuth)
{
  xdrfile << v3dtypes::disk << center << radius;
  addCenterIndexMat();
  xdrfile << polar << azimuth;
}

void v3dfile::addTube(triple const* g, double width, triple const& Min, triple const& Max, bool core)
{
  xdrfile << v3dtypes::tube;
  for (int i=0;i<4;++i)
  {
    xdrfile << g[i];
  }
  xdrfile << width;
  addCenterIndexMat();
  xdrfile << Min << Max << core;
}

void v3dfile::addTriangles(size_t nP, triple const* P, size_t nN, triple const* N, size_t nC, prc::RGBAColour const* C,
                           size_t nI, uint32_t const (* PI)[3], uint32_t const (* NI)[3], uint32_t const (* CI)[3],
                           triple const& Min, triple const& Max)
{
  xdrfile << v3dtypes::triangles;
  xdrfile << nP;
  addTriples(P,nP);
  xdrfile << nN;
  addTriples(N,nN);

  if (nC > 0 && C != nullptr)
  {
    xdrfile << nC;
    addColors(C,nC);
  }
  else
  {
    uint32_t zero = 0;
    xdrfile << zero;
  }

  xdrfile << nI;

  for(size_t i=0; i < nI; ++i)
  {
    const uint32_t *PIi=PI[i];
    const uint32_t *NIi=NI[i];
    bool keepNI=distinct(NIi,PIi);
    bool keepCI=nC && distinct(CI[i],PIi);

    if (keepNI)
    {
      xdrfile << (keepCI ? v3dTriangleIndexType::index_PosNormColor : v3dTriangleIndexType::index_PosNorm);
      addIndices(PI[i]);
      addIndices(NI[i]);
    }
    else
    {
      xdrfile << (keepCI ? v3dTriangleIndexType::index_PosColor : v3dTriangleIndexType::index_Pos);
      addIndices(PI[i]);
    }

    if (keepCI)
    {
      addIndices(CI[i]);
    }
  }

  xdrfile << (uint32_t) materialIndex << Min << Max;
}

void v3dfile::addIndices(uint32_t const* v)
{
  xdrfile << v[0] << v[1] << v[2];
}

void v3dfile::addCurve(triple const& z0, triple const& c0, triple const& c1, triple const& z1, triple const& Min,
                       triple const& Max)
{
  xdrfile << v3dtypes::curve << z0 << c0 << c1 << z1;
  addCenterIndexMat();
  xdrfile << Min << Max;

}

void v3dfile::addCurve(triple const& z0, triple const& z1, triple const& Min, triple const& Max)
{
  xdrfile << v3dtypes::line << z0 << z1;
  addCenterIndexMat();
  xdrfile << Min << Max;
}

void v3dfile::addPixel(triple const& z0, double width, triple const& Min, triple const& Max)
{
 xdrfile << v3dtypes::pixel_ << z0 << width;
 xdrfile << (uint32_t) materialIndex;
 xdrfile << Min << Max;
}

void v3dfile::precision(int digits)
{
  // inert function for override
}


xdr::oxstream& operator<<(xdr::oxstream& ox, AHeader const& header)
{
  ox << (uint32_t)header.ty << header.getByteSize();
  header.writeContent(ox);
  return ox;
}
} //namespace camp
