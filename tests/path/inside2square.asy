import TestLib;
import inside2;

path square = (0,0)--(1,0)--(1,1)--(0,1)--cycle;

bool trueInsideSquareFn(pair p) {
  return 0 <= p.x && p.x <= 1
    && 0 <= p.y && p.y <= 1;
}

pair[] toPairArray(path p) {
  return new pair[] {point(p, 0), point(p,1), point(p,2), point(p,3)};
}

int rsign() {
  return unitrand() < .5 ? -1 : 1;
}

srand(round((cputime().parent.clock%1)*1e9));

int numOfTransforms = 1000;
int numOfTests = 10;
write("Testing square");
write("Num of transforms: ", numOfTransforms);
write("Num of tests/transform: ", numOfTests);

int insideCount = 0;
int outsideCount = 0;
for (int i=0;i<numOfTransforms;++i) {
  real randAngle = 360*unitrand();
  real randXScale = 100*unitrand();
  real randYScale = 100*unitrand();
  real randXShift = 500*rsign()*unitrand();
  real randYShift = 500*rsign()*unitrand();

  transform randTransform =
    rotate(randAngle)
    *scale(randXScale,randYScale)
    *shift(randXShift,randYShift);

  path transformedSquare = randTransform*square;
  pair[] transformedSquarePairs = toPairArray(transformedSquare);
  //write("Path: ", transformedSquare);

  for (int j=0;j<numOfTests;++j) {
    pair randPoint = (2*rsign()*unitrand(), 2*rsign()*unitrand());
    pair transformedRandPoint = randTransform*randPoint;
    //write("Point: ", transformedRandPoint);

    bool isActuallyInside = trueInsideSquareFn(randPoint);
    bool isReportedInside = insidePolygon(transformedSquarePairs, transformedRandPoint);

    if (isActuallyInside) ++insideCount;
      else ++outsideCount;
    assert(isActuallyInside==isReportedInside);
  }
}

write("Inside count: ", insideCount);
write("Outside count: ", outsideCount);
