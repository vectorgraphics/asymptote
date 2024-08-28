/*****
 * drawclipbegin.cc
 * John Bowman
 *
 * Begin clip of picture to specified volume
 *****/

#include "drawclipbegin.h"

#include "drawpath3.h"
#include "drawsurface.h"
#include "material.h"
#include "glrender.h"

using namespace prc;

namespace camp {

void drawClip3Begin::render(double size2, const triple& Min, const triple& Max,
                            double perspective, bool remesh) {

  size_t offset=clipData.indices.size();
  int n=V->size();

  if(!remesh && clipPatch.Onscreen && false) { // Fully onscreen; no need to re-render
    clipPatch.append();
  } else {
    for(int i=0; i < n; ++i) {
      clipFace *face=(*V)[i];
      if(face->ncontrols == 16) {

//      if(bbox2(Min,Max).offscreen()) { // Fully offscreen
//      clipPatch.data.clear();
//      return;
//      }


//    double s=perspective ? Min.getz()*perspective : 1.0; // Move to glrender
//    const pair size3(s*(B.getx()-b.getx()),s*(B.gety()-b.gety()));
        const triple size3(100,100,100);

        clipPatch.queue(face->controls,face->Straight(),size3.length()/size2,false,
                        NULL,true);
      } else {
        const triple size3(100,100,100);
        clipTriangle.queue(face->controls,face->Straight(),size3.length()/size2,false,
                           NULL,true);
      }

    }
  }
  size_t size=clipData.indices.size()-offset;
  clipStack.push_back(new clipIndex(offset,size));
  clipStackStack.push_back(new clipStack_t(clipStack));
  clipStackStackIndex=clipStackStack.size();
}

}
