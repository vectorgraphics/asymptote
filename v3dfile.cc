//
// Created by Supakorn on 7/24/2021.
//

#include "v3dfile.h"
#include "drawelement.h"

namespace camp
{

std::vector<double> fromTriples(triple const* triples, size_t n);

v3dfile::v3dfile(string const& name, uint32_t const& version, open_mode mode) :
  xdrfile(name.c_str(), mode)
{
  xdrfile << version;
}

v3dfile::~v3dfile()
{
  xdrfile.close();
}

std::vector<double> fromTriples(triple const* triples, size_t n)
{
  std::vector<double> ctlPts;
  for (size_t i=0;i<n;++i)
  {
    auto arr = triples[i].array();
    ctlPts.insert(ctlPts.end(), arr.begin(), arr.end());
  }
  return ctlPts;
}

void v3dfile::addPatch(triple const* controls, size_t n, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c,
                       size_t nc)
{
  if (n == 4 || n == 16) // quad patches
  {
    xdrfile << (c == nullptr ? v3dTypes::bezierPatch_noColor : v3dTypes::bezierPatch);
  }
  else if (n == 3 || n == 10) // triangles
  {
    xdrfile << (c == nullptr ? v3dTypes::bezierTriangle_noColor : v3dTypes::bezierTriangle);
  }
  // xdr does not support 16 bit. Treated as int
  xdrfile << fromTriples(controls, n);
  addCenterIndexMat();

  if (c != nullptr)
  {
    std::vector<double> clrPts;
    for (size_t i=0;i<nc;++i)
    {
      auto arr = c[i].array();
      clrPts.insert(clrPts.end(), arr.begin(), arr.end());
    }
    xdrfile << clrPts;
  }
}

void v3dfile::addMaterial(Material const& mat)
{
  xdrfile << v3dTypes::material_;
  addvec4(mat.diffuse);
  addvec4(mat.emissive);
  addvec4(mat.specular);
  addvec4(mat.parameters);
}

void v3dfile::addCenterIndexMat()
{
  xdrfile << drawElement::centerIndex << materialIndex;
}

void v3dfile::addvec4(glm::vec4 const& vec)
{
  xdrfile << static_cast<float>(vec.x) << static_cast<float>(vec.y)
    << static_cast<float>(vec.z) << static_cast<float>(vec.w);
}

void v3dfile::addSphereHalf(triple const& center, double radius, double const& polar, double const& azimuth)
{
  xdrfile << v3dTypes::halfSphere << center.array() << radius;
  addCenterIndexMat();
  xdrfile << polar << azimuth;
}

void v3dfile::addSphere(triple const& center, double radius)
{
  xdrfile << v3dTypes::sphere << center.array() << radius;
  addCenterIndexMat();
}

void
v3dfile::addCylinder(triple const& center, double radius, double height, double const& polar, double const& azimuth,
                     bool core)
{
  xdrfile << v3dTypes::cylinder << center.array() << radius << height;
  addCenterIndexMat();
  xdrfile << polar << azimuth << core;
}

void v3dfile::addDisk(triple const& center, double radius, double const& polar, double const& azimuth)
{
  xdrfile << v3dTypes::disk << center.array() << radius;
  addCenterIndexMat();
  xdrfile << polar << azimuth;
}

void v3dfile::addTube(triple const* g, double width, triple const& Min, triple const& Max, bool core)
{
  xdrfile << v3dTypes::tube;
  for (int i=0;i<4;++i)
  {
    xdrfile << g[i].array();
  }
  xdrfile << width;
  addCenterIndexMat();
  xdrfile << Min.array() << Max.array() << core;
}

void v3dfile::addTrianglesNoColor(size_t nP, triple const* P, size_t nN, triple const* N, size_t nI,
                                  uint32_t const (* PI)[3], uint32_t const (* NI)[3], triple const& Min,
                                  triple const& Max)
{
  xdrfile << v3dTypes::triangles_noColor;
  xdrfile << fromTriples(P,nP) << fromTriples(N, nN);

  xdrfile << nI;
  for(size_t i=0; i < nI; ++i)
  {
    addIndices(PI[i]);
    addIndices(NI[i]);
  }

  xdrfile << materialIndex << Min.array() << Max.array();
}

void v3dfile::addTriangles(size_t nP, triple const* P, size_t nN, triple const* N, size_t nC, prc::RGBAColour const* C,
                           size_t nI, uint32_t const (* PI)[3], uint32_t const (* NI)[3], uint32_t const (* CI)[3],
                           triple const& Min, triple const& Max)
{
  xdrfile << v3dTypes::triangles;
  xdrfile << fromTriples(P,nP) << fromTriples(N,nN);
  std::vector<double> clrPts;
  for (size_t i=0;i<nC;++i)
  {
    auto arr = C[i].array();
    clrPts.insert(clrPts.end(), arr.begin(), arr.end());
  }
  xdrfile << clrPts;

  xdrfile << nI;
  for(size_t i=0; i < nI; ++i)
  {
    addIndices(PI[i]);
    addIndices(NI[i]);
    addIndices(CI[i]);
  }

  xdrfile << materialIndex << Min.array() << Max.array();
}

void v3dfile::addIndices(uint32_t const* trip)
{
  xdrfile << std::array<uint32_t, 3> {*trip, *(trip+1), *(trip+2)};
}


} //namespace camp