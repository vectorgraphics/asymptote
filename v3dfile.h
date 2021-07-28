//
// Created by Supakorn on 7/24/2021.
//

#ifndef V3DFILE_H
#define V3DFILE_H

#include <prc/oPRCFile.h>

#include "common.h"
#include "xstream.h"
#include "triple.h"
#include "material.h"

namespace camp
{

using open_mode=xdr::xios::open_mode;

enum v3dTypes : uint32_t
{
  other=0,
  material_=1,
  transform_=2,
  element=3,

  line,
  triangle,
  quad,

  curve,
  bezierTriangle,
  bezierPatch,

  lineColor,
  triangleColor,
  quadColor,

  curveColor,
  bezierTriangleColor,
  bezierPatchColor,

  triangles, // specify nP,nN,nC

  //primitives
  disk,
  cylinder,
  tube,
  sphere,
  halfSphere,

  //other
//  element2D,
  animation,
};

enum v3dTriangleIndexType : uint32_t
{
  index_Pos=0,
  index_PosNorm=1,
  index_PosColor=2,
  index_PosNormColor=3,
};

class v3dfile
{
public:
  explicit v3dfile(string const& name, uint32_t const& version=-1, open_mode mode=xdr::xios::open_mode::out);
  ~v3dfile();
  void addPatch(triple const* controls, size_t n, triple const& Min, triple const& Max, prc::RGBAColour const* c,
                size_t nc);

  void addMaterial(Material const& mat);

  void addSphere(triple const& center, double radius);
  void addSphereHalf(triple const& center, double radius, double const& polar, double const& azimuth);

  void addCylinder(triple const& center, double radius, double height,
                   double const& polar, const double& azimuth,
                   bool core=false);
  void addDisk(triple const& center, double radius,
               double const& polar=0.0, const double& azimuth=0.0);
  void addTube(const triple *g, double width,
               const triple& Min, const triple& Max, bool core=false);

  void addTriangles(size_t nP, const triple* P, size_t nN,
                            const triple* N, size_t nC, const prc::RGBAColour* C,
                            size_t nI, const uint32_t (*PI)[3],
                            const uint32_t (*NI)[3], const uint32_t (*CI)[3],
                            const triple& Min, const triple& Max);

protected:
  void addvec4(glm::vec4 const& vec);
  void addCenterIndexMat();
  void addIndices(uint32_t const* trip);
  void addTriples(triple const* triples, size_t n);
  void addColors(prc::RGBAColour const* col, size_t nc);

  xdr::oxstream xdrfile;
};

} //namespace camp
#endif
