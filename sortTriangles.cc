/*****
 * sortTriangles.cc
 * Depth-based sorting of transparent triangles.
 *****/

#include <cstdlib>
#include <vector>
#include "rgba.h"
#include "bezierpatch.h"

namespace camp {

std::vector<double> zbuffer;

void transform(const std::vector<ColorVertex>& b)
{
  unsigned n=b.size();
  zbuffer.resize(n);

  const glm::dmat4& projView=getProjViewMat();
  double Tz0=projView[2][0];
  double Tz1=projView[2][1];
  double Tz2=projView[2][2];
  for(unsigned i=0; i < n; ++i) {
    const glm::vec3& v=b[i].position;
    zbuffer[i]=Tz0*v.x+Tz1*v.y+Tz2*v.z;
  }
}

// Sort nonintersecting triangles by depth.
int compare(const void *p, const void *P)
{
  unsigned Ia=((uint32_t *) p)[0];
  unsigned Ib=((uint32_t *) p)[1];
  unsigned Ic=((uint32_t *) p)[2];

  unsigned IA=((uint32_t *) P)[0];
  unsigned IB=((uint32_t *) P)[1];
  unsigned IC=((uint32_t *) P)[2];

  return zbuffer[Ia]+zbuffer[Ib]+zbuffer[Ic] <
    zbuffer[IA]+zbuffer[IB]+zbuffer[IC] ? -1 : 1;
}

// Sort nonintersecting triangles by depth.
void sortTriangles()
{
  if(!transparentData.indices.empty()) {
    transform(transparentData.colorVertices);
    qsort(&transparentData.indices[0],transparentData.indices.size()/3,
          3*sizeof(uint32_t),compare);
  }
}

} //namespace camp
