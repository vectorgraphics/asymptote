import inside2;

pair[] square =
  { (0,0)
  , (1,0)
  , (1,1)
  , (0,1)
  };

pair[] selfintersecting =
  { (0,0)
  , (0,1)
  , (1,0)
  , (1,1)
  };

pair[] star =
  { (0,1.65)
  , (-.5,.8)
  , (-1.75,.9)
  , (-.8,-.3)
  , (-1.2,-1.5)
  , (0,-.5)
  , (1.2,-1.5)
  , (.8,-.3)
  , (1.75,.9)
  , (.5,.8)
  };

pair[] toothbox =
  { (0,0)
  , (.7, 0)
  , (.75,.65)
  , (.9,0)
  , (1,0)
  , (1,1)
  , (0,1)
  };

pair[] horseshoe =
  { (0,0)
  , (0,1)
  , (0.4,1)
  , (0.4,0.5)
  , (0.6,0.5)
  , (0.6,1)
  , (1,1)
  , (1,0)
  };

path toPath(pair[] shape) {
  path p = shape[0];
  for (int i=1; i<shape.length; ++i) {
    p = p--shape[i];
  }
  return p--cycle;
}

void dshape(pair[] shape) {
  guide g;
  for (int i=0; i<shape.length; ++i) {
    g=g--shape[i];
  }
  draw(g--cycle);
}

bool insidePolygonManualOutside(pair[] p, pair z, pair outside) {
  int winding() {
    var W=StraightContribution(outside);

    pair prevPoint = p[p.length - 1];
    for (int i=0; i < p.length; ++i) {
      pair currentPoint = p[i];
      if(W.onBoundary(prevPoint,currentPoint,z)) return undefined;
      prevPoint = currentPoint;
    }
    return W.count;
  }
  return winding() != 0;
}

/* Test the inside algorithm by manually specifying an outside point
 */
void test() {
  void testPoint(pair[] p, pair v, pair outside) {
    dot(v,insidePolygonManualOutside(p,v,outside) ? blue : red);
  }
  void square () {
    dshape(square);
    testPoint(square, (.2,.2), (2,0));
    testPoint(square, (-.2,-1), (0,2));
    testPoint(square, (-.1, 1.2), (-.1,0));
    testPoint(square, (1.2,1),(1.3,1));
    testPoint(square, (0.9,-.1), (0.1,-3));
  }
  void other() {
    dshape(horseshoe);
    testPoint(horseshoe, (.5,.9), (0,2));
    testPoint(horseshoe, (.1,.9), (0,2));
    testPoint(horseshoe, (.5,.1), (2,1));
  }

  square();
  //other();
}

size(10cm);
test();
