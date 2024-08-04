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

void test(triple[][] polyhedron, triple p, string msg) {
  write("other      "+msg+": ", insidePolyhedron(polyhedron,p));
}

void run() {
  test(tpoly, (.1,.1,.1), "inside");
  test(tpoly, (.2,.2,.2), "inside");
  test(tpoly, (1/3,1/3,1/3), "border");
  test(tpoly, (.5,.5,.5), "outside");
  test(tpoly, (1/2.9,1/2.9,1/2.9), "slighltly outside");
  test(tpoly, (1,1,1), "outside");
  test(doubledTpoly, (.1,.1,.1), "inside double");
  test(tpolyhollow, (.25,.25,.25), "middle of hollow");
  test(negativeTpoly, (.1,.1,.1), "inside negative");
  test(tpoly, (-.2,-.2,-.2), "outside*");
  test(negativeTpoly, (-.2,-.2,-.2), "outside negative*");
  test(tpoly, (0,0,-.2), "outside flat");
}

void drawrandom(triple[][] shape) {
  size(10cm);
  dshape3(shape);
  void testDot(triple p) {
    if(insidePolyhedron(shape, p))
      dot(p,blue);
    else
      dot(p,red);
  }

  triple[] manualPoints =
    { (-.1,-.1,-.1)
    , (-.2,-.2,-.2)
    , (-.2,-.2,-.21)
    //, (-.2,-.2,-0.02)
    //, (.25,.25,.25)
    //, (.1,.1,.1)
    //, (1/3,1/3,1/3)
    //, (1,1,1)
    };
  for (triple p : manualPoints)
    testDot(p);

  for (int i=0; i<1000; ++i) {
    real x = (unitrand() < .5 ? 1 : -1)*unitrand();
    real y = (unitrand() < .5 ? 1 : -1)*unitrand();
    real z = (unitrand() < .5 ? 1 : -1)*unitrand();

    testDot((x,y,z));
  }
}
run();
drawrandom(tpoly);
