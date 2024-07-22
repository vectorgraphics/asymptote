import three;

struct StraightContribution {
  triple outside,normal;
  triple H;
  int count=0;

  void operator init(triple outside, triple normal, triple H)  {
    this.outside=outside;
    this.normal=normal;
    this.H=H;
  }

  // given 3 colinear points, return true if point v lies between points v0 and v1.
  bool insideSegment(triple z0, triple z1, triple z) {
    if(z == z1 || z == z0) return true;
    if(z0 == z1) return false;

    triple h = cross(z1-z0,normal);
    int s1 = sgn(orient(z0,z,h,H));
    int s2 = sgn(orient(z1,z,h,H));
    assert(s1 != 0 && s2 != 0);
    return s1 != s2;
  }

  bool onBoundary(triple z0, triple z1, triple z) {
    int s1 = sgn(orient(z,z0,z1,H));
    if (s1 == 0)
      return insideSegment(z0,z1,z);

    int s2 = sgn(orient(outside,z0,z1,H));

    if (s1 == s2)
      return false;

    int s3 = sgn(orient(z,outside,z0,H));
    int s4 = sgn(orient(z,outside,z1,H));
    if (s3 != s4)
      count += s3;

    return false;
  }
}

// Return the winding number of planar polygon relative to point v
// lying in the same plane, or the largest odd integer if v lies on p.
int windingnumberPolygon(triple[] p, triple v) {
  triple prevPoint = p[p.length - 1];
  triple M = maxbound(p);
  triple m = minbound(p);
  triple outside = 2*M-m;
  real epsilon = sqrt(realEpsilon);
  real norm=abs(M-m);
  real Epsilon=norm*epsilon;

  triple n=normal(p);
  triple normal=norm*n;
  triple H=v+normal;

  outside=outside-dot(outside,n)*n;

  // Check that each vertex u distinct from v is not colinear w/ outside
  bool checkColinear(triple u) {
    if (u != v && orient(u,v,outside,H) == 0) {
      triple normal=unit(cross(v-u,H-u));
      assert(normal != O);
      outside += normal*Epsilon;
      outside=outside-dot(outside,n)*n;
      return true; // need to restart & recheck
    }
    return false;
  }

  bool check=true;
  while(check) {
    check = false;
    for (triple v : p)
      check = check || checkColinear(v);
  }

  var W=StraightContribution(outside,normal,H);

  for (int i=0; i < p.length; ++i) {
    triple currentPoint = p[i];
    if(W.onBoundary(prevPoint,currentPoint,v)) return undefined;
    prevPoint = currentPoint;
  }
  return W.count;
}

// Return true if point v lies on or inside planar polygon p.
bool insidePolygon(triple[] p, triple v) {
  return windingnumberPolygon(p,v) != 0;
}

triple minbound(triple[][] polyhedron) {
  triple m = polyhedron[0][0];
  for (triple[] face : polyhedron) {
    m = minbound(m, minbound(face));
  }
  return m;
}

triple maxbound(triple[][] polyhedron) {
  triple m = polyhedron[0][0];
  for (triple[] face : polyhedron) {
    m = maxbound(m, maxbound(face));
  }
  return m;
}

struct straightContribution3 {
  triple outside;
  triple v;
  int count=0;

  void operator init(triple outside, triple v) {
    this.outside=outside;
    this.v=v;
  }

  bool onBoundary(triple t1, triple t2, triple t3) {
    int s1 = sgn(orient(v, t1, t2, t3));
    if (s1 == 0) {
      triple[] tri = {t1,t2,t3};
      return insidePolygon(tri, v);
    }

    int s2 = sgn(orient(outside, t1, t2, t3));

    // Test whether the two extermities of the segment
    // are on the same side of the supporting plane of
    // the triangle
    if (s1 == s2)
      return false;

    // Now we know that the segment 'straddles' the supporing
    // plane. We need to test whether the three tetrahedra formed
    // by the segment and the three edges of the triangle have
    // the same orientation
    int s3 = sgn(orient(v, outside, t1, t2));
    int s4 = sgn(orient(v, outside, t2, t3));
    int s5 = sgn(orient(v, outside, t3, t1));

    if (s3 == s4 && s4 == s5)
      count += s3;

    return false;
  }
}

int windingnumberPolyhedron(triple[][] p, triple v) {
  triple M = maxbound(p);
  triple m = minbound(p);
  triple outside = 2*M-m;
  real epsilon = sqrt(realEpsilon);
  real norm=abs(M-m);
  real Epsilon=norm*epsilon;

  // Check that outside does not lie on the extension of the non-degenerate
  // triangle u--v--w
  bool checkCoplanar(triple u, triple w) {
    triple normal=unit(cross(v-u,w-u));
    triple H=v+normal;
    if (orient(u,v,w,H) != 0 &&
        orient(u,v,w,outside) == 0) {
      assert(normal != O);
      outside += normal*Epsilon;
      return true; // need to restart & recheck
    }
    return false;
  }

  bool check=true;
  while(check) {
    check = false;
    for(triple[] f : p) {
      check = check || checkCoplanar(f[0],f[1]);
      check = check || checkCoplanar(f[1],f[2]);
      check = check || checkCoplanar(f[2],f[0]);
    }
  }

  var W=straightContribution3(outside,v);

  for(triple[] f : p) {
    if(W.onBoundary(f[0],f[1],f[2]))
      return undefined;
  }
  return W.count;
}

// Return true if point v lies on or inside polyhedron p.
bool insidePolyhedron(triple[][] p, triple v) {
  return windingnumberPolyhedron(p,v) != 0;
}
