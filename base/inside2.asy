
// line segment l1 l2
// edge t1 t2
int intersect(pair l1, pair l2, pair t1, pair t2) { //p outside prev curr
  // ensure that both points are not on the same side of the edge
  int s1 = sgn(orient(l1, t1, t2));
  int s2 = sgn(orient(l2, t1, t2));

  if (s1 == s2) {
    return 0;
  }


  int s3 = sgn(orient(l1, l2, t1));
  int s4 = sgn(orient(l1, l2, t2));

  return s3 == s4 ? 0 : s3;
}

bool inside(pair[] polygon, pair p) {
  pair outside = 2*maxbound(polygon) - minbound(polygon);
  int count = 0;
  pair prevPoint = polygon[polygon.length - 1];
  for (int i=0; i<polygon.length; ++i) {
    pair currentPoint = polygon[i];
    count += intersect(p, outside, prevPoint, currentPoint);
    prevPoint = currentPoint;
  }
  return count != 0;
}


bool insideOpt(pair[] polygon, pair p) {
  pair outside = 2*maxbound(polygon) - minbound(polygon);
  int count = 0;
  pair prevPoint = polygon[polygon.length - 1];
  bool canUseOldS4 = false;
  int s4;
  for (int i=0; i<polygon.length; ++i) {
    pair currentPoint = polygon[i];
    int s1 = sgn(orient(p, prevPoint, currentPoint));
    int s2 = sgn(orient(outside, prevPoint, currentPoint));

    if (s1 == s2) {
      prevPoint = currentPoint;
      canUseOldS4 = false;
      continue;
    }

    int s3 = canUseOldS4 ? s4 : sgn(orient(p, outside, prevPoint));
    s4 = sgn(orient(p, outside, currentPoint));
    canUseOldS4 = true;
    if (s3 != s4) count += s3;

    prevPoint = currentPoint;
  }
  return count != 0;
}
