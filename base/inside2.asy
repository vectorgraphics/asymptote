// given 3 colinear points, return true if point z lies between points
// z0 and z1.
bool insideSegment(pair z0, pair z1, pair z) {
  if(z == z1 || z == z0) return true;
  if(z0 == z1) return false;
  real norm = sqrt(max(abs2(z0), abs2(z1), abs2(z)));
  pair h = z0+unit(z1-z0)*I*norm;
  int s1 = sgn(orient(z0,z,h));
  int s2 = sgn(orient(z1,z,h));
  return s1 != s2;
}

struct StraightContribution {
  pair outside;
  pair z;
  real Epsilon;
  int count=0;
  bool redo;

  void operator init(pair outside, pair z, real Epsilon)  {
    this.outside=outside;
    this.z=z;
    this.Epsilon=Epsilon;
  }

  // Ensure that outside does not lie on the extension of the non-degenerate
  // line segment v--z
  void avoidColinear(pair v) {
    if (v != z) {
      pair normal=unit(z-v)*I;
      outside += normal*Epsilon;
    }
  }

  int onBoundary(pair z0, pair z1, pair z) {
    int s1 = sgn(orient(z,z0,z1));
    if (s1 == 0)
      return insideSegment(z0,z1,z) ? 1 : 0;

    int s2 = sgn(orient(outside,z0,z1));

    if (s1 == s2)
      return 0;

    redo=false;

    int s3 = sgn(orient(z,outside,z0));
    if(s3 == 0)
      avoidColinear(z0);

    int s4 = sgn(orient(z,outside,z1));
    if(s4 == 0)
      avoidColinear(z1);

    if(redo) return -1;

    if (s3 != s4)
      count += s3;

    return 0;
  }
}

// Return the winding number of polygon p relative to point z,
// or the largest odd integer if z lies on p.
int windingnumberPolygon(pair[] p, pair z) {
  pair M = maxbound(p);
  pair m = minbound(p);
  pair outside = 2*M-m;
  real epsilon = sqrt(realEpsilon);
  real Epsilon=abs(M-m)*epsilon;

  int onboundary=-1;
  StraightContribution W=StraightContribution(outside,z,Epsilon);
  while(onboundary == -1) {
    W.count=0;
    pair prevPoint = p[p.length - 1];
    for (int i=0; i < p.length; ++i) {
      pair currentPoint = p[i];
      onboundary=W.onBoundary(prevPoint,currentPoint,z);
      if(onboundary == -1) break;
      if(onboundary == 1) return undefined;
      prevPoint = currentPoint;
    }
  }

  return W.count;
}

// Return true if point z lies on or inside polygon p.
bool insidePolygon(pair[] p, pair z) {
  return windingnumberPolygon(p,z) != 0;
}
