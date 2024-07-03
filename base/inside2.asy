// given 3 colinear points, return true if point z lies between points
// z0 and z1.
bool insideSegment(pair z0, pair z1, pair z) {
  if(z == z1 || z == z0) return true;
  if(z0 == z1) return false;
  pair h = z0+(z1-z0)*I;
  int s1 = sgn(orient(z0,z,h));
  int s2 = sgn(orient(z1,z,h));
  return s1 != s2;
}

struct StraightContribution {
  pair outside;
  int count=0;

  void operator init(pair outside)  {
    this.outside=outside;
  }

  bool onBoundary(pair z0, pair z1, pair z) {
    int s1 = sgn(orient(z,z0,z1));
    int s2 = sgn(orient(outside,z0,z1));

    if (s1 == s2 && s1 != 0)
      return false;

    int s3 = sgn(orient(z,outside,z0));
    int s4 = sgn(orient(z,outside,z1));
    if (s3 != s4) {
      if (s1 == 0)
        return true;
      count += s3;
    } else if (s1 == 0)
      return insideSegment(z0,z1,z);
    return false;
  }
}

bool insidePolygon(pair[] p, pair z) {
  pair outside = 2*maxbound(p) - minbound(p);
  var W=StraightContribution(outside);
  pair prevPoint = p[p.length - 1];
  for (int i=0; i < p.length; ++i) {
    pair currentPoint = p[i];
    if(W.onBoundary(prevPoint,currentPoint,z)) return true;
    prevPoint = currentPoint;
  }
  return W.count != 0;
}
