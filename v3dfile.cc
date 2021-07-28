//
// Created by Supakorn on 7/24/2021.
//

#include "v3dfile.h"
#include "drawelement.h"
#include "jsfile.h"

namespace camp
{

v3dfile::v3dfile(string const& name, uint32_t const& version, open_mode mode) :
  xdrfile(name.c_str(), mode)
{
  xdrfile << version;
}

v3dfile::~v3dfile()
{
  xdrfile.close();
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
  xdrfile << (c == nullptr ? v3dTypes::bezierPatch : v3dTypes::bezierPatchColor);
  addTriples(controls, 16);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 16);
  }
}

void v3dfile::addStraightPatch(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{

  xdrfile << (c == nullptr ? v3dTypes::quad : v3dTypes::quadColor);
  addTriples(controls, 4);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 4);
  }
}

void v3dfile::addBezierTriangle(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{

  xdrfile << (c == nullptr ? v3dTypes::bezierTriangle : v3dTypes::bezierTriangleColor);
  addTriples(controls, 10);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 10);
  }
}

void v3dfile::addStraightBezierTriangle(triple const* controls, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c)
{
  xdrfile << (c == nullptr ? v3dTypes::triangle : v3dTypes::triangleColor);
  addTriples(controls, 3);
  addCenterIndexMat();

  if (c != nullptr)
  {
    addColors(c, 3);
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
  xdrfile << v3dTypes::halfSphere << center << radius;
  addCenterIndexMat();
  xdrfile << polar << azimuth;
}

void v3dfile::addSphere(triple const& center, double radius)
{
  xdrfile << v3dTypes::sphere << center << radius;
  addCenterIndexMat();
}

void
v3dfile::addCylinder(triple const& center, double radius, double height, double const& polar, double const& azimuth,
                     bool core)
{
  xdrfile << v3dTypes::cylinder << center << radius << height;
  addCenterIndexMat();
  xdrfile << polar << azimuth << core;
}

void v3dfile::addDisk(triple const& center, double radius, double const& polar, double const& azimuth)
{
  xdrfile << v3dTypes::disk << center << radius;
  addCenterIndexMat();
  xdrfile << polar << azimuth;
}

void v3dfile::addTube(triple const* g, double width, triple const& Min, triple const& Max, bool core)
{
  xdrfile << v3dTypes::tube;
  for (int i=0;i<4;++i)
  {
    xdrfile << g[i];
  }
  xdrfile << width;
  addCenterIndexMat();
  xdrfile << Min << Max << core;
}

void v3dfile::addTriangles(vertexBuffer& vb, triple const& Min, triple const& Max)
{
  auto vertexDataToTriple = [](vertexData& vd)
  {
    return triple(vd.position[0], vd.position[1], vd.position[2]);
  };
  std::vector<triple> trip;
  std::transform(vb.vertices.begin(), vb.vertices.end(), std::back_inserter(trip), vertexDataToTriple);

  auto vertexDataNormalToTriple = [](vertexData& vd)
  {
    return triple(vd.normal[0], vd.normal[1], vd.normal[2]);
  };
  std::vector<triple> tripNorm;
  std::transform(vb.vertices.begin(), vb.vertices.end(), std::back_inserter(tripNorm), vertexDataNormalToTriple);

  assert(vb.indices.size() % 3 == 0);
  size_t vbSz = vb.indices.size() / 3;

  auto ptIdx=new(UseGC) uint32_t[vbSz][3];
  for (size_t i = 0; i < vbSz; ++i)
  {
    ptIdx[i][0] = vb.indices[3*i];
    ptIdx[i][1] = vb.indices[3*i+1];
    ptIdx[i][2] = vb.indices[3*i+2];
  }

  addTriangles(trip.size(), trip.data(),
               tripNorm.size(), tripNorm.data(),
               0, nullptr,
               vbSz, ptIdx, ptIdx, nullptr,
               Min, Max);

}

void v3dfile::addTriangles(size_t nP, triple const* P, size_t nN, triple const* N, size_t nC, prc::RGBAColour const* C,
                           size_t nI, uint32_t const (* PI)[3], uint32_t const (* NI)[3], uint32_t const (* CI)[3],
                           triple const& Min, triple const& Max)
{
  xdrfile << v3dTypes::triangles;
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

  xdrfile << materialIndex << Min << Max;
}

void v3dfile::addIndices(uint32_t const* v)
{
  xdrfile << v[0] << v[1] << v[2];
}


} //namespace camp
