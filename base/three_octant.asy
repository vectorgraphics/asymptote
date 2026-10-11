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
// Build a Bezier triangle for vertices A, B, C with interior control
// chosen so that the barycentric centroid b(1/3,1/3,1/3) lies on the unit sphere.
//
// The interior point p9 is placed along the direction of (A+B+C)/3 at a
// radial distance R solved from |S9 + 6*R*dir|^2 = 27^2 = 729, where S9
// is the weighted sum of the other 9 control points.
// ---------------------------------------------------------------------------
surface makeBezierTriangle(triple A, triple B, triple C) {
  triple[] ab = bezierEdge(A, B);
  triple[] bc = bezierEdge(B, C);
  triple[] ca = bezierEdge(C, A);

  triple S9 = A + B + C + 3*(ab[0]+ab[1]+bc[0]+bc[1]+ca[0]+ca[1]);
  triple dir = unit(A + B + C);

  // Solve: 36*R^2 + 12*(S9.dir)*R + (|S9|^2 - 729) = 0
  real dotSD = dot(S9, dir);
  real disc = 144*dotSD*dotSD - 144*(dot(S9,S9) - 729);
  if(disc < 0) disc = 0;
  triple p9 = (-12*dotSD + sqrt(disc)) / 72 * dir;

  return surface(patch(new triple[][] {{A},{ab[0],ca[1]},{ab[1],p9,ca[0]},{B,bc[0],bc[1],C}},
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
