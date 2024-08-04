import three;
import inside3;

triple[][] tpoly =  // https://jive-manual.dynaflow.com/LinearTetrahedron.jpg
  { { (0,0,0), (0,0,1), (1,0,0) }
  , { (0,0,0), (1,0,0), (0,1,0) }
  , { (0,0,0), (0,1,0), (0,0,1) }
  , { (0,1,0), (1,0,0), (0,0,1) }
  };

triple[][] cube =
  { { (0,0,0), (1,0,0), (1,0,1) }
  , { (0,0,0), (1,0,1), (0,0,1) }
  , { (0,0,0), (0,0,1), (0,1,1) }
  , { (0,0,0), (0,1,1), (0,1,0) }
  , { (0,0,0), (0,1,0), (1,1,0) }
  , { (0,0,0), (1,1,0), (1,0,0) }
  , { (1,1,1), (0,1,1), (0,0,1) }
  , { (1,1,1), (0,0,1), (1,0,1) }
  , { (1,1,1), (1,0,1), (1,0,0) }
  , { (1,1,1), (1,0,0), (1,1,0) }
  , { (1,1,1), (0,1,0), (0,1,1) }
  , { (1,1,1), (1,1,0), (0,1,0) }
  };

void dshape3(triple[][] polyhedron) {
  for (int i=0; i<polyhedron.length; ++i) {
    triple[] face = polyhedron[i];
    draw(surface(face[0]--face[1]--face[2]--cycle), Pen(i)+opacity(0.5));
  }
}

bool insidePolyhedronManualOutside(triple[][] p, triple v, triple outside) {
  int winding() {
    var W=straightContribution3(outside,v);
    for(triple[] f : p) {
      if(W.onBoundary(f[0],f[1],f[2]))
        return undefined;
    }
    return W.count;
  }
  return winding() != 0;
}

void grouptest() {
  void testPoint(triple[][] polyhedron, triple point_, triple outside) {
    dot(point_, insidePolyhedronManualOutside(polyhedron, point_, outside) ? blue : red);
  }
  size(10cm);
  void cube() {
    dshape3(cube);
    testPoint(cube, (.3,.1,.7), (0,2,2));
    testPoint(cube, (1.2,-0.2,1.3),(0,2,2));
    testPoint(cube, (0.9,0.13,.7),(2,0,2));
    testPoint(cube, (1.1,1.2,1.3),(2,0,2));
    testPoint(cube, (.5,.3,.4),(.5,-.1,1));
    testPoint(cube, (1.1,1.2,-.2),(.5,-.1,1));
  }
  void polyhedron() {
    dshape3(tpoly);
    testPoint(tpoly, (.2,.2,.1), (2,0,0));
    testPoint(tpoly, (-.2,.2,.8), (2,0,0));
    testPoint(tpoly, (1,.5,.3), (2,2,0));
    testPoint(tpoly, (.4,1.2,.4), (2,2,0));
    testPoint(tpoly, (0,0,1.3),(0,.6,.6));
    testPoint(tpoly, (.1,.15,.6),(0,.6,.6));
    testPoint(tpoly, (0.05,.9,0),(-.3,-.3,0));
    testPoint(tpoly, (-0.05,.3,.1),(-.3,-.3,0));
    testPoint(tpoly, (.2,.3,.4), (1.2,-.3,.1));
    testPoint(tpoly, (1.3,-.2,.3), (1.2,-.3,.1));
  }
  cube();
  shipout();erase();
  polyhedron();
}
grouptest();
