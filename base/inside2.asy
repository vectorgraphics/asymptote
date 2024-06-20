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
    } else {
      if (s1 != 0)
        return false;
      // only do these checks if outside is not colinear with z0 z1
      //    if (s2 != 0) return false; // CHECK
      if (z0 == z1)
        return z == z0;
      if (s3 == 0 && s4 == 0) {
        real norm = sqrt(max(abs2(z0), abs2(z1), abs2(z)));
        do {
          pair ref = (unitrand()*norm, unitrand()*norm);
          s3=sgn(orient(z,ref,z0));
          s4=sgn(orient(z,ref,z1));
        } while (s3 == 0 && s4 == 0);
      }
      return s3 != s4;
    }
    return false;
  }
}

bool inside(pair[] polygon, pair p) {
  pair outside = 2*maxbound(polygon) - minbound(polygon);
  var W=StraightContribution(outside);
  pair prevPoint = polygon[polygon.length - 1];
  for (int i=0; i < polygon.length; ++i) {
    pair currentPoint = polygon[i];
    if(W.onBoundary(prevPoint,currentPoint,p)) return true;
    prevPoint = currentPoint;
  }
  return W.count != 0;
}
