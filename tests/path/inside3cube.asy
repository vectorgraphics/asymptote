import inside3;
import three;

triple[][] cube =  // made of triangles oriented positively outwards
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

bool trueInsideCubeFn(triple point) {
  return 0 <= point.x && point.x <= 1
    && 0 <= point.y && point.y <= 1
    && 0 <= point.z && point.z <= 1;
}

int rsign() {
  return unitrand() < .5 ? -1 : 1;
}

srand(round((cputime().parent.clock%1)*1e9));
int numOfTransforms = 1000;
int numOfTests = 10;
write("Testing cube");
write("Num of transforms: ", numOfTransforms);
write("Num of tests/transform: ", numOfTests);

int insideCount = 0;
int outsideCount = 0;
for (int i=0; i<numOfTransforms;++i) {
  real randAngle = 360*unitrand();
  triple randAxis = (rsign()*unitrand(),rsign()*unitrand(),rsign()*unitrand());
  real randScale = 1000*unitrand();
  real randXShift = 500*rsign()*unitrand();
  real randYShift = 500*rsign()*unitrand();
  real randZShift = 500*rsign()*unitrand();

  transform3 randTransform =
    shift(randXShift,randYShift,randZShift)
    *scale3(randScale)
    *rotate(randAngle, randAxis);
  triple[][] transformedCube = {};
  for (triple[] face : cube) {
    triple[] newFace = {};
    for (triple v : face) newFace.push(randTransform*v);
    transformedCube.push(newFace);
  }

  for (int j=0; j<numOfTests; ++j) {
    triple randomPoint = (2*rsign()*unitrand(), 2*rsign()*unitrand(), 2*rsign()*unitrand());
    triple transformedRandomPoint = randTransform*randomPoint;

    bool isActuallyInside = trueInsideCubeFn(randomPoint);
    bool isReportedInside = insidePolyhedron(transformedCube, transformedRandomPoint);
    if (isActuallyInside) ++insideCount;
    else ++outsideCount;
    assert(isActuallyInside == isReportedInside);
  }
}

write("Inside count: ", insideCount);
write("Outside count: ", outsideCount);
