//
// Created by Supakorn on 7/24/2021.
//

#ifndef V3DFILE_H
#define V3DFILE_H

#include <prc/oPRCFile.h>
#include <glm/vec4.hpp>

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

  bezierPatch=100,
  bezierTriangle=101,
  line=102,
  curve=103,
  triangle=104,
  quad=105,

  bezierPatch_noColor=200,
  bezierTriangle_noColor=201,
  line_noColor=202,
  curve_noColor=203,
  triangle_noColor=204,
  quad_noColor=205,

  //primitives
  disk=300,
  cylinder=301,
  tube=302,
  sphere=303,
  halfSphere=304,

  //other
  animation=400,
  twodimElem=401,
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

protected:
  void addvec4(glm::vec4 const& vec);
  void addCenterIndexMat();
  xdr::oxstream xdrfile;
};

} //namespace camp
#endif