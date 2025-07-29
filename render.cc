// g++ -DOMIT_OPTIONAL render.cc bezierpatch.cc

#include <iostream>
#include "rgba.h"
#include "bezierpatch.h"

using namespace std;
using namespace camp;

namespace camp {
glm::dmat4 projViewMat;
glm::dmat4 normMat;
}

int main()
{
  int n=16;
  triple Controls[]={
    triple(39.68504,0,68.0315),triple(37.91339,0,71.75197),triple(40.74803,0,71.75197),triple(42.51969,0,68.0315),
    triple(39.68504,-22.22362,68.0315),triple(37.91339,-21.2315,71.75197),triple(40.74803,-22.8189,71.75197),triple(42.51969,-23.81102,68.0315),
    triple(22.22362,-39.68504,68.0315),triple(21.2315,-37.91339,71.75197),triple(22.8189,-40.74803,71.75197),triple(23.81102,-42.51969,68.0315),
    triple(0,-39.68504,68.0315),triple(0,-37.91339,71.75197),triple(0,-40.74803,71.75197),triple(0,-42.51969,68.0315)};

  BezierPatch S;

  double width=1920;
  double height=1080;

  bool orthographic=false;
  triple Min,Max;
  boundstriples(Min,Max,n,Controls);

  triple b=Min, B=Max; // cumulative scene bounds; for now use patch bounds
  double Zmax=B.getz();

  double perspective=orthographic ? 0.0 : 1.0/Zmax;
  double s=perspective ? Min.getz()*perspective : 1.0; // Move to glrender
  double size2=hypot(width,height);

  const camp::pair size3(s*(B.getx()-b.getx()),s*(B.gety()-b.gety()));
  bool transparent=false;
  bool straight=false;
  bool remesh=true;

  S.queue(Controls,straight,size3.length()/size2,transparent,NULL);
  cout << materialData.materialVertices.size() << endl;

  return 0;
}
