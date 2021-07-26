//
// Created by Supakorn on 7/24/2021.
//

#include "v3dfile.h"
#include "drawelement.h"

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

void v3dfile::addPatch(triple const* controls, size_t n, triple const& Min,
                       triple const& Max, prc::RGBAColour const* c,
                       size_t nc)
{
  std::vector<double> ctlPts;
  for (size_t i=0;i<n;++i)
  {
    auto arr = controls[i].array();
    ctlPts.insert(ctlPts.end(), arr.begin(), arr.end());
  }
  if (n == 4 || n == 16) // quad patches
  {
    xdrfile << (c == nullptr ? v3dTypes::bezierPatch_noColor : v3dTypes::bezierPatch);
  }
  else if (n == 3 || n == 10) // triangles
  {
    xdrfile << (c == nullptr ? v3dTypes::bezierTriangle_noColor : v3dTypes::bezierTriangle);
  }
  // xdr does not support 16 bit. Treated as int
  xdrfile << ctlPts << drawElement::centerIndex << materialIndex;

  if (c != nullptr)
  {
    std::vector<double> clrPts;
    for (size_t i=0;i<n;++i)
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

void v3dfile::addvec4(glm::vec4 const& vec)
{
  xdrfile << static_cast<float>(vec.x) << static_cast<float>(vec.y)
    << static_cast<float>(vec.z) << static_cast<float>(vec.w);
}


} //namespace camp