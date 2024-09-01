import three;

real orient(triple a, triple b, triple c, triple d) {
  return dot(cross(a-d,b-d),c-d);
}

struct StraightContribution {
  triple outside;
  triple z;
  real Epsilon;
  triple normal;
  triple H;
  int count=0;
  bool redo;

  void operator init(triple outside, triple z, real Epsilon, triple normal,
                     triple H)  {
    this.outside=outside;
    this.z=z;
    this.Epsilon=Epsilon;
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

  // Check that each vertex u distinct from v is not colinear w/ outside
  void avoidColinear(triple u) {
    if(u != z) {
      redo=true;
      triple n=unit(normal);
      write("adjust");
      triple Normal=unit(cross(z-u,H-u));
      assert(Normal != O);
      outside += Normal*Epsilon;
      outside=outside-dot(outside,n)*n;
    }
  }

  int onBoundary(triple z0, triple z1, triple z) {
    int s1 = sgn(orient(z,z0,z1,H));
    if (s1 == 0)
      return insideSegment(z0,z1,z) ? 1 : 0;

    int s2 = sgn(orient(outside,z0,z1,H));

    if (s1 == s2)
      return 0;

    redo=false;

    int s3 = sgn(orient(z,outside,z0,H));
    if(s3 == 0)
      avoidColinear(z0);

    int s4 = sgn(orient(z,outside,z1,H));
    if(s4 == 0)
      avoidColinear(z1);

    if(redo) return -1;

    if (s3 != s4)
      count += s3;

    return 0;
  }
}

// Return the winding number of planar polygon relative to point v
// lying in the same plane, or the largest odd integer if v lies on p.
int windingnumberPolygon(triple[] p, triple v) {
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

  int onboundary=-1;
  var W=StraightContribution(outside,v,Epsilon,normal,H);
  while(onboundary == -1) {
    triple prevPoint = p[p.length - 1];
    for (int i=0; i < p.length; ++i) {
      triple currentPoint = p[i];
      onboundary=W.onBoundary(prevPoint,currentPoint,v);
      if(onboundary == -1) break;
      if(onboundary == 1) return undefined;
      prevPoint = currentPoint;
    }
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

struct StraightContribution3 {
  triple outside;
  triple v;
  int count;
  real Epsilon;
  bool redo;

  void operator init(triple outside, triple v, real Epsilon) {
    this.outside=outside;
    this.v=v;
    this.Epsilon=Epsilon;
  }

  // Ensure that outside does not lie on the extension of the non-degenerate
  // triangle u--v--w
  void avoidCoplanar(triple u, triple w) {
    triple normal=unit(cross(v-u,w-u));
    triple H=v+normal;
    if(orient(u,v,w,H) != 0) {
      redo=true;
      outside += normal*Epsilon;
    }
  }

  int onBoundary(triple a, triple b, triple c) {
    int s1=sgn(orient(v,a,b,c));
    if (s1 == 0) {
      triple[] tri = {a,b,c};
      return insidePolygon(tri, v) ? 1 : 0;
    }

    int s2 = sgn(orient(outside, a, b, c));

    // Test whether the two extermities of the segment
    // are on the same side of the supporting plane of
    // the triangle
    if (s1 == s2)
      return 0;

    redo=false;

    // Now we know that the segment 'straddles' the supporting
    // plane. We need to test whether the three tetrahedra formed
    // by the segment and the three edges of the triangle have
    // the same orientation
    int s3 = sgn(orient(v, outside, a, b));
    if(s3 == 0)
      avoidCoplanar(a,b);

    int s4 = sgn(orient(v, outside, b, c));
    if(s4 == 0)
      avoidCoplanar(b,c);

    int s5 = sgn(orient(v, outside, c, a));
    if(s5 == 0)
      avoidCoplanar(c,a);

    if(redo) return -1;

    if (s3 == s4 && s4 == s5)
      count += s3;

    return 0;
  }
}

int windingnumberPolyhedron(triple[][] p, triple v) {
  triple M = maxbound(p);
  triple m = minbound(p);
  triple outside = 2*M-m;
  real epsilon = sqrt(realEpsilon);
  real norm=abs(M-m);
  real Epsilon=norm*epsilon;

  int onboundary=-1;
  while(onboundary == -1) {
    StraightContribution3 W=StraightContribution3(outside,v,Epsilon);
    for(triple[] f : p) {
      onboundary=W.onBoundary(f[0],f[1],f[2]);
      if(onboundary == -1) break;
      if(onboundary == 1) return undefined;
    }
  }

  return W.count;
}

// Return true if point v lies on or inside polyhedron p.
bool insidePolyhedron(triple[][] p, triple v) {
  return windingnumberPolyhedron(p,v) != 0;
}
