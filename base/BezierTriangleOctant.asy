/*
  Cubic Bezier Triangle Approximation to the First Octant of the Unit Sphere
  ==========================================================================

  Subdivision scheme:
  - The three exterior edges (great-circle arcs between axis points) are
  bisected to find midpoints mXY, mYZ, mZX.
  - These midpoints form an inner triangle; together with the three corner
  regions this yields 4 triangles

  Each triangle is approximated by a Bezier triangle with:
  - Corner control points on the sphere.
  - Edge control points are optimal cubic approximation of great-circle arcs
  - Interior control point tuned so that the barycentric centroid
    of the Bezier triangle lies exactly on the sphere.
*/

// ---------------------------------------------------------------------------
// Midpoint of great-circle arc from P to Q.
// ---------------------------------------------------------------------------
triple gcMidPoint(triple P, triple Q) {
  return (P+Q)/(sqrt(2)*sqrt(1+dot(P,Q)));
}

// ---------------------------------------------------------------------------
// Optimal cubic Bezier control points for a great-circle arc from P to Q.
// Returns the two interior edge control points {c1, c2}.
// Here k is chosen such that the Bezier midpoint lies on the sphere.
// ---------------------------------------------------------------------------
triple[] bezierEdge(triple P, triple Q){
  real x = dot(P,Q);
  real k = (4/3)*sqrt(1-x)/(sqrt(2)+sqrt(1+x));
  return new triple[] {P+k*unit(Q-x*P), Q+k*unit(P-x*Q)};
}

// ---------------------------------------------------------------------------
// Analytically compute the interior control point scale so that the Bezier
// triangle's barycentric centroid b(1/3,1/3,1/3) lies on the unit sphere.
//
// For a Bezier triangle with corner points A,B,C and edge control
// points from bezierEdge(), the interior point p9 is placed along the
// direction of (A+B+C)/3 at some radial distance R.  We solve for R such
// that |b(1/3,1/3,1/3)| = 1.
//
// The barycentric centroid evaluates to:
//   b = (1/27)*sum_of_all_10_control_points
// So we need |sum| = 27.  Since p9 = R * dir, and the other 9 points are
// known, we solve for R from |S9 + p9|^2 = 27^2 where S9 is the sum of the
// other 9 control points.
// ---------------------------------------------------------------------------
real findPullOut(triple A, triple B, triple C) {
  // Corner control points (weight 1 each in 27*b)
  triple p0 = A;       // P[0][0]
  triple p3 = B;       // P[3][0]
  triple p6 = C;       // P[3][3]

  // Edge control points (weight 3 each in 27*b)
  triple[] ab = bezierEdge(A, B);
  triple p1 = ab[0];   // P[1][0]
  triple p2 = ab[1];   // P[2][0]

  triple[] bc = bezierEdge(B, C);
  triple p4 = bc[0];   // P[3][1]
  triple p5 = bc[1];   // P[3][2]

  triple[] ca = bezierEdge(C, A);
  triple p7 = ca[0];   // P[2][2]
  triple p8 = ca[1];   // P[1][1]

  triple S9 = p0 + p3 + p6 + 3*(p1+p2+p4+p5+p7+p8);

  // Interior point P[2][1] has weight 6.
  // Direction: along (A+B+C)/3, normalized
  triple dir = unit(A + B + C);

  // We need |S9 + 6*R*dir|^2 = 27^2 = 729
  //   => |S9|^2 + 12*R*(S9.dir) + 36*R^2 = 729
  // Solve: 36*R^2 + 12*(S9.dir)*R + (|S9|^2 - 729) = 0
  real dotSD = dot(S9, dir);
  real S9sq = dot(S9, S9);
  real a = 36;
  real b = 12 * dotSD;
  real c = S9sq - 729;

  real disc = b*b - 4*a*c;
  if(disc < 0) disc = 0;
  real R = (-b + sqrt(disc)) / (2*a);

  return R;
}

// ---------------------------------------------------------------------------
// Build a Bezier triangle for vertices A, B, C with interior control
// chosen so that (A+B+C)/3 lies on the unit sphere.
// ---------------------------------------------------------------------------
surface makeBezierTriangle(triple A, triple B, triple C) {
  triple p0 = A;
  triple p3 = B;
  triple p6 = C;

  triple[] ab = bezierEdge(A, B);
  triple p1 = ab[0];
  triple p2 = ab[1];

  triple[] bc = bezierEdge(B, C);
  triple p4 = bc[0];
  triple p5 = bc[1];

  triple[] ca = bezierEdge(C, A);
  triple p7 = ca[0];
  triple p8 = ca[1];

  triple p9 = findPullOut(A, B, C) * unit(A + B + C);

  return surface(patch(new triple[][] {{p0},{p1,p8},{p2,p9,p7},{p3,p4,p5,p6}},
                       triangular=true));
}

// ---------------------------------------------------------------------------
// Recursively subdivide a triangle defined by vertices A, B, C.
// Returns an array of Bézier triangle surfaces.
// depth = number of subdivision levels (0 = no subdivision).
// ---------------------------------------------------------------------------
surface subdivideTriangle(triple A, triple B, triple C, int depth=2) {
  if(depth == 0)
    return makeBezierTriangle(A, B, C);

  // Find edge midpoints on the sphere via great-circle arcs
  triple midAB = gcMidPoint(A, B);
  triple midBC = gcMidPoint(B, C);
  triple midCA = gcMidPoint(C, A);

  // Subdivide into 4 sub-triangles
  surface result;
  result.append(subdivideTriangle(A, midAB, midCA, depth-1));
  result.append(subdivideTriangle(B, midBC, midAB, depth-1));
  result.append(subdivideTriangle(C, midCA, midBC, depth-1));
  result.append(subdivideTriangle(midAB, midBC, midCA, depth-1));

  return result;
}

restricted surface octant1=subdivideTriangle(X,Y,Z);
