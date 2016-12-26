// Copyright 2015 Charles Staats III
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// smoothcontour3
// An Asymptote module for drawing smooth implicitly defined surfaces
// author: Charles Staats III
// charles dot staats dot iii at gmail dot com

import graph_settings;  // for nmesh
import three;
import math;

/***********************************************/
/******** CREATING BEZIER PATCHES **************/
/******** WITH SPECIFIED NORMALS  **************/
/***********************************************/

// The weight given to minimizing the sum of squares of
// the mixed partials at the corners of the bezier patch.
// If this weight is zero, the result is undefined in
// places and can be rather wild even where it is
// defined.
// The struct is used to as a namespace.
struct pathwithnormals_settings {
  static real wildnessweight = 1e-3;
}
private from pathwithnormals_settings unravel wildnessweight;

// The Bernstein basis polynomials of degree 3:
real B03(real t) { return (1-t)^3; }
real B13(real t) { return 3*t*(1-t)^2; }
real B23(real t) { return 3*t^2*(1-t); }
real B33(real t) { return t^3; }

private typedef real function(real);
function[] bernstein = new function[] {B03, B13, B23, B33};

// This function attempts to produce a Bezier patch
// with the specified boundary path and normal directions.
// For instance, the patch should be normal to
// u0normals[0] at (0, 0.25),
// normal to u0normals[1] at (0, 0.5), and
// normal to u0normals[2] at (0, 0.75).
// The actual normal (as computed by the patch.normal() function)
// may be parallel to the specified normal, antiparallel, or
// even zero.
//
// A small amount of deviation is allowed in order to stabilize
// the algorithm (by keeping the mixed partials at the corners from
// growing too large).
//
// Note that the specified normals are projected to be orthogonal to
// the specified boundary path. However, the entries in the array
// remain intact.
patch patchwithnormals(path3 external, triple[] u0normals, triple[] u1normals,
                       triple[] v0normals, triple[] v1normals)
{
  assert(cyclic(external));
  assert(length(external) == 4);
  assert(u0normals.length == 3);
  assert(u1normals.length == 3);
  assert(v0normals.length == 3);
  assert(v1normals.length == 3);
  
  triple[][] controlpoints = new triple[4][4];
  controlpoints[0][0] = point(external,0);
  controlpoints[1][0] = postcontrol(external,0);
  controlpoints[2][0] = precontrol(external,1);
  controlpoints[3][0] = point(external,1);
  controlpoints[3][1] = postcontrol(external,1);
  controlpoints[3][2] = precontrol(external,2);
  controlpoints[3][3] = point(external,2);
  controlpoints[2][3] = postcontrol(external,2);
  controlpoints[1][3] = precontrol(external,3);
  controlpoints[0][3] = point(external,3);
  controlpoints[0][2] = postcontrol(external,3);
  controlpoints[0][1] = precontrol(external, 4);

  real[][] matrix = new real[24][12];
  for (int i = 0; i < matrix.length; ++i)
    for (int j = 0; j < matrix[i].length; ++j)
      matrix[i][j] = 0;
  real[] rightvector = new real[24];
  for (int i = 0; i < rightvector.length; ++i)
    rightvector[i] = 0;

  void addtocoeff(int i, int j, int count, triple coeffs) {
    if (1 <= i && i <= 2 && 1 <= j && j <= 2) {
      int position = 3 * (2 * (i-1) + (j-1));
      matrix[count][position] += coeffs.x;
      matrix[count][position+1] += coeffs.y;
      matrix[count][position+2] += coeffs.z;
    } else {
      rightvector[count] -= dot(controlpoints[i][j], coeffs);
    }
  }

  void addtocoeff(int i, int j, int count, real coeff) {
    if (1 <= i && i <= 2 && 1 <= j && j <= 2) {
      int position = 3 * (2 * (i-1) + (j-1));
      matrix[count][position] += coeff;
      matrix[count+1][position+1] += coeff;
      matrix[count+2][position+2] += coeff;
    } else {
      rightvector[count] -= controlpoints[i][j].x * coeff;
      rightvector[count+1] -= controlpoints[i][j].y * coeff;
      rightvector[count+2] -= controlpoints[i][j].z * coeff;
    }
  }

  int count = 0;

  void apply_u0(int j, real a, triple n) {
    real factor = 3 * bernstein[j](a);
    addtocoeff(0,j,count,-factor*n);
    addtocoeff(1,j,count,factor*n);
  }
  void apply_u0(real a, triple n) {
    triple tangent = dir(external, 4-a);
    n -= dot(n,tangent)*tangent;
    n = unit(n);
    for (int j = 0; j < 4; ++j) {
      apply_u0(j,a,n);
    }
    ++count;
  }
  apply_u0(0.25, u0normals[0]);
  apply_u0(0.5, u0normals[1]);
  apply_u0(0.75, u0normals[2]);

  void apply_u1(int j, real a, triple n) {
    real factor = 3 * bernstein[j](a);
    addtocoeff(3,j,count,factor*n);
    addtocoeff(2,j,count,-factor*n);
  }
  void apply_u1(real a, triple n) {
    triple tangent = dir(external, 1+a);
    n -= dot(n,tangent)*tangent;
    n = unit(n);
    for (int j = 0; j < 4; ++j)
      apply_u1(j,a,n);
    ++count;
  }
  apply_u1(0.25, u1normals[0]);
  apply_u1(0.5, u1normals[1]);
  apply_u1(0.75, u1normals[2]);

  void apply_v0(int i, real a, triple n) {
    real factor = 3 * bernstein[i](a);
    addtocoeff(i,0,count,-factor*n);
    addtocoeff(i,1,count,factor*n);
  }
  void apply_v0(real a, triple n) {
    triple tangent = dir(external, a);
    n -= dot(n,tangent) * tangent;
    n = unit(n);
    for (int i = 0; i < 4; ++i)
      apply_v0(i,a,n);
    ++count;
  }
  apply_v0(0.25, v0normals[0]);
  apply_v0(0.5, v0normals[1]);
  apply_v0(0.75, v0normals[2]);

  void apply_v1(int i, real a, triple n) {
    real factor = 3 * bernstein[i](a);
    addtocoeff(i,3,count,factor*n);
    addtocoeff(i,2,count,-factor*n);
  }
  void apply_v1(real a, triple n) {
    triple tangent = dir(external, 3-a);
    n -= dot(n,tangent)*tangent;
    n = unit(n);
    for (int i = 0; i < 4; ++i)
      apply_v1(i,a,n);
    ++count;
  }
  apply_v1(0.25, v1normals[0]);
  apply_v1(0.5, v1normals[1]);
  apply_v1(0.75, v1normals[2]);

  addtocoeff(0,0,count,9*wildnessweight);
  addtocoeff(1,1,count,9*wildnessweight);
  addtocoeff(0,1,count,-9*wildnessweight);
  addtocoeff(1,0,count,-9*wildnessweight);
  count+=3;
  addtocoeff(3,3,count,9*wildnessweight);
  addtocoeff(2,2,count,9*wildnessweight);
  addtocoeff(3,2,count,-9*wildnessweight);
  addtocoeff(2,3,count,-9*wildnessweight);
  count+=3;
  addtocoeff(0,3,count,9*wildnessweight);
  addtocoeff(1,2,count,9*wildnessweight);
  addtocoeff(1,3,count,-9*wildnessweight);
  addtocoeff(0,2,count,-9*wildnessweight);
  count += 3;
  addtocoeff(3,0,count,9*wildnessweight);
  addtocoeff(2,1,count,9*wildnessweight);
  addtocoeff(3,1,count,-9*wildnessweight);
  addtocoeff(2,0,count,-9*wildnessweight);
  count += 3;

  real[] solution = leastsquares(matrix, rightvector, warn=false);
  if (solution.length == 0) { // if the matrix was singular
    write("Warning: unable to solve matrix for specifying edge normals "
          + "on bezier patch. Using coons patch.");
    return patch(external);
  }
  
  for (int i = 1; i <= 2; ++i) {
    for (int j = 1; j <= 2; ++j) {
      int position = 3 * (2 * (i-1) + (j-1));
      controlpoints[i][j] = (solution[position],
                             solution[position+1],
                             solution[position+2]);
    }
  }

  return patch(controlpoints);
}

// This function attempts to produce a Bezier triangle
// with the specified boundary path and normal directions at the
// edge midpoints. The bezier triangle should be normal to
// n1 at point(external, 0.5),
// normal to n2 at point(external, 1.5), and
// normal to n3 at point(external, 2.5).
// The actual normal (as computed by the patch.normal() function)
// may be parallel to the specified normal, antiparallel, or
// even zero.
//
// A small amount of deviation is allowed in order to stabilize
// the algorithm (by keeping the mixed partials at the corners from
// growing too large).
patch trianglewithnormals(path3 external, triple n1,
                          triple n2, triple n3) {
  assert(cyclic(external));
  assert(length(external) == 3);
  // Use the formal symbols a3, a2b, abc, etc. to denote the control points,
  // following the Wikipedia article on Bezier triangles.
  triple a3 = point(external, 0), a2b = postcontrol(external, 0),
    ab2 = precontrol(external, 1), b3 = point(external, 1),
    b2c = postcontrol(external, 1), bc2 = precontrol(external, 2),
    c3 = point(external, 2), ac2 = postcontrol(external, 2),
    a2c = precontrol(external, 0);

  // Use orthogonal projection to ensure that the normal vectors are
  // actually normal to the boundary path.
  triple tangent = dir(external, 0.5);
  n1 -= dot(n1,tangent)*tangent;
  n1 = unit(n1);

  tangent = dir(external, 1.5);
  n2 -= dot(n2,tangent)*tangent;
  n2 = unit(n2);

  tangent = dir(external, 2.5);
  n3 -= dot(n3,tangent)*tangent;
  n3 = unit(n3);
  
  real wild = 2 * wildnessweight;
  real[][] matrix = { {n1.x, n1.y, n1.z},
                      {n2.x, n2.y, n2.z},
                      {n3.x, n3.y, n3.z},
                      {      wild,          0,          0},
                      {         0,       wild,          0},
                      {         0,          0,       wild} };
  real[] rightvector =
    { dot(n1, (a3 + 3a2b + 3ab2 + b3 - 2a2c - 2b2c)) / 4,
      dot(n2, (b3 + 3b2c + 3bc2 + c3 - 2ab2 - 2ac2)) / 4,
      dot(n3, (c3 + 3ac2 + 3a2c + a3 - 2bc2 - 2a2b)) / 4 };

  // The inner control point that minimizes the sum of squares of
  // the mixed partials on the corners.
  triple tameinnercontrol =
    ((a2b + a2c - a3) + (ab2 + b2c - b3) + (ac2 + bc2 - c3)) / 3;
  rightvector.append(wild * new real[]
                     {tameinnercontrol.x, tameinnercontrol.y, tameinnercontrol.z});
  real[] solution = leastsquares(matrix, rightvector, warn=false);
  if (solution.length == 0) { // if the matrix was singular
    write("Warning: unable to solve matrix for specifying edge normals "
          + "on bezier triangle. Using coons triangle.");
    return patch(external);
  }
  triple innercontrol = (solution[0], solution[1], solution[2]);
  return patch(external, innercontrol);
}

// A wrapper for the previous functions when the normal direction
// is given as a function of direction. The wrapper can also
// accommodate cyclic boundary paths of between one and four
// segments, although the results are best by far when there
// are three or four segments.
patch patchwithnormals(path3 external, triple normalat(triple)) {
  assert(cyclic(external));
  assert(1 <= length(external) && length(external) <= 4);
  if (length(external) == 3) {
    triple n1 = normalat(point(external, 0.5));
    triple n2 = normalat(point(external, 1.5));
    triple n3 = normalat(point(external, 2.5));
    return trianglewithnormals(external, n1, n2, n3);
  }
  while (length(external) < 4) external = external -- cycle;
  triple[] u0normals = new triple[3];
  triple[] u1normals = new triple[3];
  triple[] v0normals = new triple[3];
  triple[] v1normals = new triple[3];
  for (int i = 1; i <= 3; ++i) {
    v0normals[i-1] = unit(normalat(point(external, i/4)));
    u1normals[i-1] = unit(normalat(point(external, 1 + i/4)));
    v1normals[i-1] = unit(normalat(point(external, 3 - i/4)));
    u0normals[i-1] = unit(normalat(point(external, 4 - i/4)));
  }
  return patchwithnormals(external, u0normals, u1normals, v0normals, v1normals);
}

/***********************************************/
/********* DUAL CUBE GRAPH UTILITY *************/
/***********************************************/

// Suppose a plane intersects a (hollow) cube, and
// does not intersect any vertices. Then its intersection
// with cube forms a cycle. The goal of the code below
// is to reconstruct the order of the cycle
// given only an unordered list of which edges the plane
// intersects.
//
// Basically, the question is this: If we know the points
// in which a more-or-less planar surface intersects the
// edges of cube, how do we connect those points?
//
// When I wrote the code, I was thinking in terms of the
// dual graph of a cube, in which "vertices" are really
// faces of the cube and "edges" connect those "vertices."

// An enum for the different "vertices" (i.e. faces)
// available. NULL_VERTEX is primarily intended as a
// return value to indicate the absence of a desired
// vertex.
private int NULL_VERTEX = -1;
private int XHIGH = 0;
private int XLOW = 1;
private int YHIGH = 2;
private int YLOW = 3;
private int ZHIGH = 4;
private int ZLOW = 5;

// An unordered set of nonnegative integers.
// Since the intent is to use
// only the six values from the enum above, no effort
// was made to use scalable algorithms.
struct intset {
  private bool[] ints = new bool[0];
  private int size = 0;

  bool contains(int item) {
    assert(item >= 0);
    if (item >= ints.length) return false;
    return ints[item];
  }

  // Returns true if the item was added (i.e., was
  // not already present).
  bool add(int item) {
    assert(item >= 0);
    while (item >= ints.length) ints.push(false);
    if (ints[item]) return false;
    ints[item] = true;
    ++size;
    return true;
  }

  int[] elements() {
    int[] toreturn;
    for (int i = 0; i < ints.length; ++i) {
      if (ints[i]) toreturn.push(i);
    }
    return toreturn;
  }

  int size() { return size; }
}

// A map from integers to sets of integers. Again, no
// attempt is made to use scalable data structures.
struct int_to_intset {
  int[] keys = new int[0];
  intset[] values = new intset[0];

  void add(int key, int value) {
    for (int i = 0; i < keys.length; ++i) {
      if (keys[i] == key) {
        values[i].add(value);
        return;
      }
    }
    keys.push(key);
    intset newset;
    values.push(newset);
    newset.add(value);
  }

  private int indexOf(int key) {
    for (int i = 0; i < keys.length; ++i) {
      if (keys[i] == key) return i;
    }
    return -1;
  }

  int[] get(int key) {
    int i = indexOf(key);
    if (i < 0) return new int[0];
    else return values[i].elements();
  }

  int numvalues(int key) {
    int i = indexOf(key);
    if (i < 0) return 0;
    else return values[i].size();
  }

  int numkeys() {
    return keys.length;
  }
}

// A struct intended to represent an undirected edge between
// two "vertices."
struct edge {
  int start;
  int end;
  void operator init(int a, int b) {
    start = a;
    end = b;
  }
  bool bordersvertex(int v) { return start == v || end == v; }
}

string operator cast(edge e) {
  int a, b;
  if (e.start <= e.end) {a = e.start; b = e.end;}
  else {a = e.end; b = e.start; }
  return (string)a + " <-> " + (string)b;
}

bool operator == (edge a, edge b) {
  if (a.start == b.start && a.end == b.end) return true;
  if (a.start == b.end && a.end == b.start) return true;
  return false;
}

string operator cast(edge[] edges) {
  string toreturn = "{ ";
  for (int i = 0; i < edges.length; ++i) {
    toreturn += edges[i];
    if (i < edges.length-1) toreturn += ", ";
  }
  return toreturn + " }";
}

// Finally, the function that strings together a list of edges
// into a cycle. It makes assumptions that hold true if the
// list of edges did in fact come from a plane intersection
// containing no vertices of the cube. For instance, such a
// plane can contain at most two noncollinear points of any
// one face; consequently, no face can border more than two of
// the selected edges.
//
// If the underlying assumptions prove to be false, the function
// returns null.
int[] makecircle(edge[] edges) {
  if (edges.length == 0) return new int[0];
  int_to_intset graph;
  for (edge e : edges) {
    graph.add(e.start, e.end);
    graph.add(e.end, e.start);
  }
  int currentvertex = edges[0].start;
  int startvertex = currentvertex;
  int lastvertex = NULL_VERTEX;
  int[] toreturn = new int[0];
  do {
    toreturn.push(currentvertex);
    int[] adjacentvertices = graph.get(currentvertex);
    if (adjacentvertices.length != 2) return null;
    for (int v : adjacentvertices) {
      if (v != lastvertex) {
        lastvertex = currentvertex;
        currentvertex = v;
        break;
      }
    }
  } while (currentvertex != startvertex);
  if (toreturn.length != graph.numkeys()) return null;
  toreturn.cyclic = true;
  return toreturn;
}

/***********************************************/
/********** PATHS BETWEEN POINTS ***************/
/***********************************************/
// Construct paths between two points with additional
// constraints; for instance, the path must be orthogonal
// to a certain vector at each of the endpoints, must
// lie within a specified plane or a specified face
// of a rectangular solid,....

// A vector (typically a normal vector) at a specified position.
struct positionedvector {
  triple position;
  triple direction;
  void operator init(triple position, triple direction) {
    this.position = position;
    this.direction = direction;
  }
}

string operator cast(positionedvector vv) {
  return "position: " + (string)(vv.position) + " vector: " + (string)vv.direction;
}

// The angle, in degrees, between two vectors.
real angledegrees(triple a, triple b) {
  real dotprod = dot(a,b);
  real lengthprod = max(abs(a) * abs(b), abs(dotprod));
  if (lengthprod == 0) return 0;
  return aCos(dotprod / lengthprod);
}

// A path (single curved segment) between two points. At each point
// is specified a vector orthogonal to the path.
path3 pathbetween(positionedvector v1, positionedvector v2) {
  triple n1 = unit(v1.direction);
  triple n2 = unit(v2.direction);

  triple p1 = v1.position;
  triple p2 = v2.position;
  triple delta = p2-p1;

  triple dir1 = delta - dot(delta, n1)*n1;
  triple dir2 = delta - dot(delta, n2)*n2;
  return p1 {dir1} .. {dir2} p2;
}

// Assuming v1 and v2 are linearly independent, returns an array {a, b}
// such that a v1 + b v2 is the orthogonal projection of toproject onto
// the span of v1 and v2. If v1 and v2 are dependent, returns an empty array
// (if warn==false) or throws an error (if warn==true).
real[] projecttospan_findcoeffs(triple toproject, triple v1, triple v2,
                                bool warn=false) {
  real[][] matrix = {{v1.x, v2.x},
                     {v1.y, v2.y},
                     {v1.z, v2.z}};
  real[] desiredanswer = {toproject.x, toproject.y, toproject.z};
  return leastsquares(matrix, desiredanswer, warn=warn);
}

// Project the triple toproject into the span of a and b, but restrict
// to the quarter-plane of linear combinations a v1 + b v2 such that
// a >= mincoeff and b >= mincoeff. If v1 and v2 are linearly dependent,
// return a random (positive) linear combination.
triple projecttospan(triple toproject, triple v1, triple v2,
                     real mincoeff = 0.05) {
  real[] coeffs = projecttospan_findcoeffs(toproject, v1, v2, warn=false);
  real a, b;
  if (coeffs.length == 0) {
    a = mincoeff + unitrand();
    b = mincoeff + unitrand();
  } else {
    a = max(coeffs[0], mincoeff);
    b = max(coeffs[1], mincoeff);
  }
  return a*v1 + b*v2;
}

// A path between two specified vertices of a cyclic path. The
// path tangent at each endpoint is guaranteed to lie within the
// quarter-plane spanned by positive linear combinations of the
// tangents of the two outgoing paths at that endpoint.
path3 pathbetween(path3 edgecycle, int vertex1, int vertex2) {
  triple point1 = point(edgecycle, vertex1);
  triple point2 = point(edgecycle, vertex2);
  
  triple v1 = -dir(edgecycle, vertex1, sign=-1);
  triple v2 =  dir(edgecycle, vertex1, sign= 1);
  triple direction1 = projecttospan(unit(point2-point1), v1, v2);

  v1 = -dir(edgecycle, vertex2, sign=-1);
  v2 =  dir(edgecycle, vertex2, sign= 1);
  triple direction2 = projecttospan(unit(point1-point2), v1, v2);

  return point1 {direction1} .. {-direction2} point2;
}

// This function applies a heuristic to choose two "opposite"
// vertices (separated by three segments) of edgecycle, which
// is required to be a cyclic path consisting of 5 or 6 segments.
// The two chosen vertices are pushed to savevertices.
//
// The function returns a path between the two chosen vertices. The
// path tangent at each endpoint is guaranteed to lie within the
// quarter-plane spanned by positive linear combinations of the
// tangents of the two outgoing paths at that endpoint.
path3 bisector(path3 edgecycle, int[] savevertices) {
  real mincoeff = 0.05;
  assert(cyclic(edgecycle));
  int n = length(edgecycle);
  assert(n >= 5 && n <= 6);
  triple[] forwarddirections = sequence(new triple(int i) {
      return dir(edgecycle, i, sign=1);
    }, n);
  forwarddirections.cyclic = true;
  triple[] backwarddirections = sequence(new triple(int i) {
      return -dir(edgecycle, i, sign=-1);
    }, n);
  backwarddirections.cyclic = true;
  real[] angles = sequence(new real(int i) {
      return angledegrees(forwarddirections[i], backwarddirections[i]);
    }, n);
  angles.cyclic = true;
  int lastindex = (n == 5 ? 4 : 2);
  real maxgoodness = 0;
  int chosenindex = -1;
  triple directionout, directionin;
  for (int i = 0; i <= lastindex; ++i) {
    int opposite = i + 3;
    triple vec = unit(point(edgecycle, opposite) - point(edgecycle, i));
    real[] coeffsbegin = projecttospan_findcoeffs(vec, forwarddirections[i],
                                                  backwarddirections[i]);
    if (coeffsbegin.length == 0) continue;
    coeffsbegin[0] = max(coeffsbegin[0], mincoeff);
    coeffsbegin[1] = max(coeffsbegin[1], mincoeff);

    real[] coeffsend = projecttospan_findcoeffs(-vec, forwarddirections[opposite],
                                                backwarddirections[opposite]);
    if (coeffsend.length == 0) continue;
    coeffsend[0] = max(coeffsend[0], mincoeff);
    coeffsend[1] = max(coeffsend[1], mincoeff);

    real goodness = angles[i] * angles[opposite] * coeffsbegin[0] * coeffsend[0]
        * coeffsbegin[1] * coeffsend[1];
    if (goodness > maxgoodness) {
      maxgoodness = goodness;
      directionout = coeffsbegin[0] * forwarddirections[i] +
          coeffsbegin[1] * backwarddirections[i];
      directionin = -(coeffsend[0] * forwarddirections[opposite] +
                      coeffsend[1] * backwarddirections[opposite]);
      chosenindex = i;
    }
  }
  if (chosenindex == -1) {
    savevertices.push(0);
    savevertices.push(3);
    return pathbetween(edgecycle, 0, 3);
  } else {
    savevertices.push(chosenindex);
    savevertices.push(chosenindex+3);
    return point(edgecycle, chosenindex) {directionout} ..
      {directionin} point(edgecycle, chosenindex + 3);
  }
}

// A path between two specified points (with specified normals) that lies
// within a specified face of a rectangular solid.
path3 pathinface(positionedvector v1, positionedvector v2,
                 triple facenorm, triple edge1normout, triple edge2normout)
{
  triple dir1 = cross(v1.direction, facenorm);
  real dotprod = dot(dir1, edge1normout);
  if (dotprod > 0) dir1 = -dir1;
  // Believe it or not, this "tiebreaker" is actually relevant at times,
  // for instance, when graphing the cone x^2 + y^2 = z^2 over the region
  // -1 <= x,y,z <= 1.
  else if (dotprod == 0 && dot(dir1, v2.position - v1.position) < 0) dir1 = -dir1;

  triple dir2 = cross(v2.direction, facenorm);
  dotprod = dot(dir2, edge2normout);
  if (dotprod < 0) dir2 = -dir2;
  else if (dotprod == 0 && dot(dir2, v2.position - v1.position) < 0) dir2 = -dir2;

  return v1.position {dir1} .. {dir2} v2.position;
}

triple normalout(int face) {
  if (face == XHIGH) return X;
  else if (face == YHIGH) return Y;
  else if (face == ZHIGH) return Z;
  else if (face == XLOW) return -X;
  else if (face == YLOW) return -Y;
  else if (face == ZLOW) return -Z;
  else return O;
}

// A path between two specified points (with specified normals) that lies
// within a specified face of a rectangular solid.
path3 pathinface(positionedvector v1, positionedvector v2,
                 int face, int edge1face, int edge2face) {
  return pathinface(v1, v2, normalout(face), normalout(edge1face),
                    normalout(edge2face));
}

/***********************************************/
/******** DRAWING IMPLICIT SURFACES ************/
/***********************************************/

// DEPRECATED
// Quadrilateralization:
// Produce a surface (array of *nondegenerate* Bezier patches) with a
// specified three-segment boundary. The surface should approximate the
// zero locus of the specified f with its specified gradient.
//
// If it is not possible to produce the desired result without leaving the
// specified rectangular region, returns a length-zero array.
//
// Dividing a triangle into smaller quadrilaterals this way is opposite
// the usual trend in mathematics. However, *before the introduction of bezier
// triangles,* the pathwithnormals algorithm
// did a poor job of choosing a good surface when the boundary path did
// not consist of four positive-length segments.
patch[] triangletoquads(path3 external, real f(triple), triple grad(triple),
                        triple a, triple b) {
  static real epsilon = 1e-3;
  assert(length(external) == 3);
  assert(cyclic(external));

  triple c0 = point(external, 0);
  triple c1 = point(external, 1);
  triple c2 = point(external, 2);

  triple center = (c0 + c1 + c2) / 3;
  triple n = unit(cross(c1-c0, c2-c0));

  real g(real t) { return f(center + t*n); }

  real tmin = -realMax, tmax = realMax;
  void absorb(real t) {
    if (t < 0) tmin = max(t,tmin);
    else tmax = min(t,tmax);
  }
  if (n.x != 0) {
    absorb((a.x - center.x) / n.x);
    absorb((b.x - center.x) / n.x);
  }
  if (n.y != 0) {
    absorb((a.y - center.y) / n.y);
    absorb((b.y - center.y) / n.y);
  }
  if (n.z != 0) {
    absorb((a.z - center.z) / n.z);
    absorb((b.z - center.z) / n.z);
  }

  real fa = g(tmin);
  real fb = g(tmax);
  if ((fa > 0 && fb > 0) || (fa < 0 && fb < 0)) {
    return new patch[0];
  } else {
    real t = findroot(g, tmin, tmax, fa=fa, fb=fb);
    center += t * n;
  }

  n = unit(grad(center));

  triple m0 = point(external, 0.5);
  positionedvector m0 = positionedvector(m0, unit(grad(m0)));
  triple m1 = point(external, 1.5);
  positionedvector m1 = positionedvector(m1, unit(grad(m1)));
  triple m2 = point(external, 2.5);
  positionedvector m2 = positionedvector(m2, unit(grad(m2)));
  positionedvector c = positionedvector(center, unit(grad(center)));

  path3 pathto_m0 = pathbetween(c, m0);
  path3 pathto_m1 = pathbetween(c, m1);
  path3 pathto_m2 = pathbetween(c, m2);

  path3 quad0 = subpath(external, 0, 0.5)
    & reverse(pathto_m0)
    & pathto_m2
    & subpath(external, -0.5, 0)
    & cycle;
  path3 quad1 = subpath(external, 1, 1.5)
    & reverse(pathto_m1)
    & pathto_m0
    & subpath(external, 0.5, 1)
    & cycle;
  path3 quad2 = subpath(external, 2, 2.5)
    & reverse(pathto_m2)
    & pathto_m1
    & subpath(external, 1.5, 2)
    & cycle;

  return new patch[] {patchwithnormals(quad0, grad),
      patchwithnormals(quad1, grad),
      patchwithnormals(quad2, grad)};   
}

// Attempts to fill the path external (which should by a cyclic path consisting of
// three segments) with bezier triangle(s). Returns an empty array if it fails.
//
// In more detail: A single bezier triangle is computed using trianglewithnormals. The normals of
// the resulting triangle at the midpoint of each edge are computed. If any of these normals
// is in the negative f direction, the external triangle is subdivided into four external triangles
// and the same procedure is applied to each. If one or more of them has an incorrectly oriented
// edge normal, the function gives up and returns an empty array.
//
// Thus, the returned array consists of 0, 1, or 4 bezier triangles; no other array lengths
// are possible.
//
// This function assumes that the path orientation is consistent with f (and its gradient)
// -- i.e., that
// at a corner, (tangent in) x (tangent out) is in the positive f direction.
patch[] maketriangle(path3 external, real f(triple),
                     triple grad(triple), bool allowsubdivide = true) {
  assert(cyclic(external));
  assert(length(external) == 3);
  triple m1 = point(external, 0.5);
  triple n1 = unit(grad(m1));
  triple m2 = point(external, 1.5);
  triple n2 = unit(grad(m2));
  triple m3 = point(external, 2.5);
  triple n3 = unit(grad(m3));
  patch beziertriangle = trianglewithnormals(external, n1, n2, n3);
  if (dot(n1, beziertriangle.normal(0.5, 0)) >= 0 &&
      dot(n2, beziertriangle.normal(0.5, 0.5)) >= 0 &&
      dot(n3, beziertriangle.normal(0, 0.5)) >= 0)
    return new patch[] {beziertriangle};

  if (!allowsubdivide) return new patch[0];
  
  positionedvector m1 = positionedvector(m1, n1);
  positionedvector m2 = positionedvector(m2, n2);
  positionedvector m3 = positionedvector(m3, n3);
  path3 p12 = pathbetween(m1, m2);
  path3 p23 = pathbetween(m2, m3);
  path3 p31 = pathbetween(m3, m1);
  patch[] triangles = maketriangle(p12 & p23 & p31 & cycle, f, grad=grad,
                                   allowsubdivide=false);
  if (triangles.length < 1) return new patch[0];

  triangles.append(maketriangle(subpath(external, -0.5, 0.5) & reverse(p31) & cycle,
                                f, grad=grad, allowsubdivide=false));
  if (triangles.length < 2) return new patch[0];

  triangles.append(maketriangle(subpath(external, 0.5, 1.5) & reverse(p12) & cycle,
                                f, grad=grad, allowsubdivide=false));
  if (triangles.length < 3) return new patch[0];
  
  triangles.append(maketriangle(subpath(external, 1.5, 2.5) & reverse(p23) & cycle,
                                f, grad=grad, allowsubdivide=false));
  if (triangles.length < 4) return new patch[0];

  return triangles;
}


// Returns true if the point is "nonsingular" (in the sense that the magnitude
// of the gradient is not too small) AND very close to the zero locus of f
// (assuming f is locally linear).
bool check_fpt_zero(triple testpoint, real f(triple), triple grad(triple)) {
  real testval = f(testpoint);
  real slope = abs(grad(testpoint));
  static real tolerance = 2*rootfinder_settings.roottolerance;
  return !(slope > tolerance && abs(testval) / slope > tolerance);
}

// Returns true if pt lies within the rectangular solid with
// opposite corners at a and b.
bool checkptincube(triple pt, triple a, triple b) {
  real xmin = a.x;
  real xmax = b.x;
  real ymin = a.y;
  real ymax = b.y;
  real zmin = a.z;
  real zmax = b.z;
  if (xmin > xmax) { real t = xmax; xmax=xmin; xmin=t; }
  if (ymin > ymax) { real t = ymax; ymax=ymin; ymin=t; }
  if (zmin > zmax) { real t = zmax; zmax=zmin; zmin=t; }

  return ((xmin <= pt.x) && (pt.x <= xmax) &&
          (ymin <= pt.y) && (pt.y <= ymax) &&
          (zmin <= pt.z) && (pt.z <= zmax));

}

// A convenience function for combining the previous two tests.
bool checkpt(triple testpt, real f(triple), triple grad(triple),
             triple a, triple b) {
  return checkptincube(testpt, a, b) &&
    check_fpt_zero(testpt, f, grad);
}

// Attempts to fill in the boundary cycle with a collection of
// patches to approximate smoothly the zero locus of f. If unable to
// do so while satisfying certain checks, returns null.
// This is distinct from returning an empty
// array, which merely indicates that the boundary cycle is too small
// to be worth filling in.
patch[] quadpatches(path3 edgecycle, positionedvector[] corners,
                    real f(triple), triple grad(triple),
                    triple a, triple b, bool usetriangles) {
  assert(corners.cyclic);

  // The tolerance for considering two points "essentially identical."
  static real tolerance = 2.5 * rootfinder_settings.roottolerance;
  
  // If there are two neighboring vertices that are essentially identical,
  // unify them into one.
  for (int i = 0; i < corners.length; ++i) {
    if (abs(corners[i].position - corners[i+1].position) < tolerance) {
      if (corners.length == 2) return new patch[0];
      corners.delete(i);
      edgecycle = subpath(edgecycle, 0, i)
          & subpath(edgecycle, i+1, length(edgecycle))
          & cycle;
      --i;
      assert(length(edgecycle) == corners.length);
    }
  }

  static real areatolerance = tolerance^2;

  assert(corners.length >= 2);
  if (corners.length == 2) {
    // If the area is too small, just ignore it; otherwise, subdivide.
    real area0 = abs(cross(-dir(edgecycle, 0, sign=-1, normalize=false),
                           dir(edgecycle, 0, sign=1, normalize=false)));
    real area1 = abs(cross(-dir(edgecycle, 1, sign=-1, normalize=false),
                           dir(edgecycle, 1, sign=1, normalize=false)));
    if (area0 < areatolerance && area1 < areatolerance) return new patch[0];
    else return null;
  }
  if (length(edgecycle) > 6) abort("too many edges: not possible.");

  for (int i = 0; i < length(edgecycle); ++i) {
    if (angledegrees(dir(edgecycle,i,sign=1),
                     dir(edgecycle,i+1,sign=-1)) > 80) {
      return null;
    }
  }
  
  if (length(edgecycle) == 3) {
    patch[] toreturn = usetriangles ? maketriangle(edgecycle, f, grad)
        : triangletoquads(edgecycle, f, grad, a, b);
    if (toreturn.length == 0) return null;
    else return toreturn;
  }
  if (length(edgecycle) == 4) {
    return new patch[] {patchwithnormals(edgecycle, grad)};
  }

  int[] bisectorindices;
  path3 middleguide = bisector(edgecycle, bisectorindices);
  
  triple testpoint = point(middleguide, 0.5);
  if (!checkpt(testpoint, f, grad, a, b)) {
    return null;
  }
  
  patch[] toreturn = null;
  path3 firstpatch = subpath(edgecycle, bisectorindices[0], bisectorindices[1])
    & reverse(middleguide) & cycle;
  if (length(edgecycle) == 5) {
    path3 secondpatch = middleguide
        & subpath(edgecycle, bisectorindices[1], 5+bisectorindices[0]) & cycle;
    toreturn = usetriangles ? maketriangle(secondpatch, f, grad)
      : triangletoquads(secondpatch, f, grad, a, b);
    if (toreturn.length == 0) return null;
    toreturn.push(patchwithnormals(firstpatch, grad));
  } else {
    // now length(edgecycle) == 6
    path3 secondpatch = middleguide
      & subpath(edgecycle, bisectorindices[1], 6+bisectorindices[0])
      & cycle;
    toreturn = new patch[] {patchwithnormals(firstpatch, grad),
        patchwithnormals(secondpatch, grad)};
  }
  return toreturn;
}

// Numerical gradient of a function
typedef triple vectorfunction(triple);
vectorfunction nGrad(real f(triple)) {
  static real epsilon = 1e-3;
  return new triple(triple v) {
    return ( (f(v + epsilon*X) - f(v - epsilon*X)) / (2 epsilon),
             (f(v + epsilon*Y) - f(v - epsilon*Y)) / (2 epsilon),
             (f(v + epsilon*Z) - f(v - epsilon*Z)) / (2 epsilon) );
  };
}

// A point together with a value at that location.
struct evaluatedpoint {
  triple pt;
  real value;
  void operator init(triple pt, real value) {
    this.pt = pt;
    this.value = value;
  }
}

triple operator cast(evaluatedpoint p) { return p.pt; }

// Compute the values of a function at every vertex of an nx by ny by nz
// array of rectangular solids.
evaluatedpoint[][][] make3dgrid(triple a, triple b, int nx, int ny, int nz,
                                real f(triple), bool allowzero = false)
{
  evaluatedpoint[][][] toreturn = new evaluatedpoint[nx+1][ny+1][nz+1];
  for (int i = 0; i <= nx; ++i) {
    for (int j = 0; j <= ny; ++j) {
      for (int k = 0; k <= nz; ++k) {
        triple pt = (interp(a.x, b.x, i/nx),
                     interp(a.y, b.y, j/ny),
                     interp(a.z, b.z, k/nz));
        real value = f(pt);
        if (value == 0 && !allowzero) value = 1e-5;
        toreturn[i][j][k] = evaluatedpoint(pt, value);
      }
    }
  }
  return toreturn;
}

// The following utilities make, for instance, slice(A, i, j, k, l)
// equivalent to what A[i:j][k:l] ought to mean for two- and three-
// -dimensional arrays of evaluatedpoints and of positionedvectors.
typedef evaluatedpoint T;
T[][] slice(T[][] a, int start1, int end1, int start2, int end2) {
  T[][] toreturn = new T[end1-start1][];
  for (int i = start1; i < end1; ++i) {
    toreturn[i-start1] = a[i][start2:end2];
  }
  return toreturn;
}
T[][][] slice(T[][][] a, int start1, int end1,
              int start2, int end2,
              int start3, int end3) {
  T[][][] toreturn = new T[end1-start1][][];
  for (int i = start1; i < end1; ++i) {
    toreturn[i-start1] = slice(a[i], start2, end2, start3, end3);
  }
  return toreturn;
}
typedef positionedvector T;
T[][] slice(T[][] a, int start1, int end1, int start2, int end2) {
  T[][] toreturn = new T[end1-start1][];
  for (int i = start1; i < end1; ++i) {
    toreturn[i-start1] = a[i][start2:end2];
  }
  return toreturn;
}
T[][][] slice(T[][][] a, int start1, int end1,
              int start2, int end2,
              int start3, int end3) {
  T[][][] toreturn = new T[end1-start1][][];
  for (int i = start1; i < end1; ++i) {
    toreturn[i-start1] = slice(a[i], start2, end2, start3, end3);
  }
  return toreturn;
}

// An object of class gridwithzeros stores the values of a function at each vertex
// of a three-dimensional grid, together with zeros of the function along edges
// of the grid and the gradient of the function at each such zero.
struct gridwithzeros {
  int nx, ny, nz;
  evaluatedpoint[][][] corners;
  positionedvector[][][] xdirzeros;
  positionedvector[][][] ydirzeros;
  positionedvector[][][] zdirzeros;
  triple grad(triple);
  real f(triple);
  int maxdepth;
  bool usetriangles;

  // Populate the edges with zeros that have a sign change and are not already
  // populated.
  void fillzeros() {
    for (int j = 0; j < ny+1; ++j) {
      for (int k = 0; k < nz+1; ++k) {
        real y = corners[0][j][k].pt.y;
        real z = corners[0][j][k].pt.z;
        real f_along_x(real t) { return f((t, y, z)); }
        for (int i = 0; i < nx; ++i) {
          if (xdirzeros[i][j][k] != null) continue;
          evaluatedpoint start = corners[i][j][k];
          evaluatedpoint end = corners[i+1][j][k];
          if ((start.value > 0 && end.value > 0) || (start.value < 0 && end.value < 0))
            xdirzeros[i][j][k] = null;
          else {
            triple root = (0,y,z);
            root += X * findroot(f_along_x, start.pt.x, end.pt.x,
                                 fa=start.value, fb=end.value);
            triple normal = grad(root);
            xdirzeros[i][j][k] = positionedvector(root, normal);
          }
        }
      }
    }

    for (int i = 0; i < nx+1; ++i) {
      for (int k = 0; k < nz+1; ++k) {
        real x = corners[i][0][k].pt.x;
        real z = corners[i][0][k].pt.z;
        real f_along_y(real t) { return f((x, t, z)); }
        for (int j = 0; j < ny; ++j) {
          if (ydirzeros[i][j][k] != null) continue;
          evaluatedpoint start = corners[i][j][k];
          evaluatedpoint end = corners[i][j+1][k];
          if ((start.value > 0 && end.value > 0) || (start.value < 0 && end.value < 0))
            ydirzeros[i][j][k] = null;
          else {
            triple root = (x,0,z);
            root += Y * findroot(f_along_y, start.pt.y, end.pt.y,
                                 fa=start.value, fb=end.value);
            triple normal = grad(root);
            ydirzeros[i][j][k] = positionedvector(root, normal);
          }
        }
      }
    }

    for (int i = 0; i < nx+1; ++i) {
      for (int j = 0; j < ny+1; ++j) {
        real x = corners[i][j][0].pt.x;
        real y = corners[i][j][0].pt.y;
        real f_along_z(real t) { return f((x, y, t)); }
        for (int k = 0; k < nz; ++k) {
          if (zdirzeros[i][j][k] != null) continue;
          evaluatedpoint start = corners[i][j][k];
          evaluatedpoint end = corners[i][j][k+1];
          if ((start.value > 0 && end.value > 0) || (start.value < 0 && end.value < 0))
            zdirzeros[i][j][k] = null;
          else {
            triple root = (x,y,0);
            root += Z * findroot(f_along_z, start.pt.z, end.pt.z,
                                 fa=start.value, fb=end.value);
            triple normal = grad(root);
            zdirzeros[i][j][k] = positionedvector(root, normal);
          }
        }
      }
    }
  }

  // Fill in the grid vertices and the zeros along edges. Each cube starts at
  // depth one and the depth increases each time it subdivides; maxdepth is the
  // maximum subdivision depth. When a cube at maxdepth cannot be resolved to
  // patches, it is left empty.
  void operator init(int nx, int ny, int nz,
                     real f(triple), triple a, triple b,
                     int maxdepth = 6, bool usetriangles) {
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    grad = nGrad(f);
    this.f = f;
    this.maxdepth = maxdepth;
    this.usetriangles = usetriangles;
    corners = make3dgrid(a, b, nx, ny, nz, f);
    xdirzeros = new positionedvector[nx][ny+1][nz+1];
    ydirzeros = new positionedvector[nx+1][ny][nz+1];
    zdirzeros = new positionedvector[nx+1][ny+1][nz];

    for (int i = 0; i <= nx; ++i) {
      for (int j = 0; j <= ny; ++j) {
        for (int k = 0; k <= nz; ++k) {
          if (i < nx) xdirzeros[i][j][k] = null;
          if (j < ny) ydirzeros[i][j][k] = null;
          if (k < nz) zdirzeros[i][j][k] = null;
        }
      }
    }

    fillzeros();
  }

  // Doubles nx, ny, and nz by halving the sizes of the cubes along the x, y, and z
  // directions (resulting in 8 times as many cubes). Already existing data about
  // function values and zeros is copied; vertices and edges with no such pre-existing
  // data are populated.
  //
  // Returns true if subdivide succeeded, false if it failed (because maxdepth
  // was exceeded).
  bool subdivide() {
    if (maxdepth <= 1) {
      return false;
    }
    --maxdepth;
    triple a = corners[0][0][0];
    triple b = corners[nx][ny][nz];
    nx *= 2;
    ny *= 2;
    nz *= 2;
    evaluatedpoint[][][] oldcorners = corners;
    corners = new evaluatedpoint[nx+1][ny+1][nz+1];
    for (int i = 0; i <= nx; ++i) {
      for (int j = 0; j <= ny; ++j) {
        for (int k = 0; k <= nz; ++k) {
          if (i % 2 == 0 && j % 2 == 0 && k % 2 == 0) {
            corners[i][j][k] = oldcorners[quotient(i,2)][quotient(j,2)][quotient(k,2)];
          } else {
            triple pt = (interp(a.x, b.x, i/nx),
                         interp(a.y, b.y, j/ny),
                         interp(a.z, b.z, k/nz));
            real value = f(pt);
            if (value == 0) value = 1e-5;
            corners[i][j][k] = evaluatedpoint(pt, value);
          }
        }
      }
    }

    positionedvector[][][] oldxdir = xdirzeros;
    xdirzeros = new positionedvector[nx][ny+1][nz+1];
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny + 1; ++j) {
        for (int k = 0; k < nz + 1; ++k) {
          if (j % 2 != 0 || k % 2 != 0) {
            xdirzeros[i][j][k] = null;
          } else {
            positionedvector zero = oldxdir[quotient(i,2)][quotient(j,2)][quotient(k,2)];
            if (zero == null) {
              xdirzeros[i][j][k] = null;
              continue;
            }
            real x = zero.position.x;
            if (x > interp(a.x, b.x, i/nx) && x < interp(a.x, b.x, (i+1)/nx)) {
              xdirzeros[i][j][k] = zero;
            } else {
              xdirzeros[i][j][k] = null;
            }
          }
        }
      }
    }

    positionedvector[][][] oldydir = ydirzeros;
    ydirzeros = new positionedvector[nx+1][ny][nz+1];
    for (int i = 0; i < nx+1; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz + 1; ++k) {
          if (i % 2 != 0 || k % 2 != 0) {
            ydirzeros[i][j][k] = null;
          } else {
            positionedvector zero = oldydir[quotient(i,2)][quotient(j,2)][quotient(k,2)];
            if (zero == null) {
              ydirzeros[i][j][k] = null;
              continue;
            }
            real y = zero.position.y;
            if (y > interp(a.y, b.y, j/ny) && y < interp(a.y, b.y, (j+1)/ny)) {
              ydirzeros[i][j][k] = zero;
            } else {
              ydirzeros[i][j][k] = null;
            }
          }
        }
      }
    }

    positionedvector[][][] oldzdir = zdirzeros;
    zdirzeros = new positionedvector[nx+1][ny+1][nz];
    for (int i = 0; i < nx + 1; ++i) {
      for (int j = 0; j < ny + 1; ++j) {
        for (int k = 0; k < nz; ++k) {
          if (i % 2 != 0 || j % 2 != 0) {
            zdirzeros[i][j][k] = null;
          } else {
            positionedvector zero = oldzdir[quotient(i,2)][quotient(j,2)][quotient(k,2)];
            if (zero == null) {
              zdirzeros[i][j][k] = null;
              continue;
            }
            real z = zero.position.z;
            if (z > interp(a.z, b.z, k/nz) && z < interp(a.z, b.z, (k+1)/nz)) {
              zdirzeros[i][j][k] = zero;
            } else {
              zdirzeros[i][j][k] = null;
            }
          }
        }
      }
    }

    fillzeros();
    return true;
  }

  // Forward declaration of the draw method, which will be called by drawcube().
  patch[] draw(bool[] reportactive = null);

  // Construct the patches, assuming that we are working
  // with a single cube (nx = ny = nz = 1). This method will subdivide the
  // cube if necessary. The parameter reportactive should be an array of
  // length 6. Setting an entry to true indicates that the surface abuts the
  // corresponding face (according to the earlier enum), and thus that the
  // algorithm should be sure that something is drawn in the cube sharing
  // that face--even if all the vertices of that cube have the same sign.
  patch[] drawcube(bool[] reportactive = null) {
    // First, determine which edges (if any) actually have zeros on them.
    edge[] zeroedges = new edge[0];
    positionedvector[] zeros = new positionedvector[0];

    int currentface, nextface;

    void pushifnonnull(positionedvector v) {
      if (v != null) {
        zeroedges.push(edge(currentface, nextface));
        zeros.push(v);
      }
    }
    positionedvector findzero(int face1, int face2) {
      edge e = edge(face1, face2);
      for (int i = 0; i < zeroedges.length; ++i) {
        if (zeroedges[i] == e) return zeros[i];
      }
      return null;
    }
    
    currentface = XLOW;
    nextface = YHIGH;
    pushifnonnull(zdirzeros[0][1][0]);
    nextface = YLOW;
    pushifnonnull(zdirzeros[0][0][0]);
    nextface = ZHIGH;
    pushifnonnull(ydirzeros[0][0][1]);
    nextface = ZLOW;
    pushifnonnull(ydirzeros[0][0][0]);

    currentface = XHIGH;
    nextface = YHIGH;
    pushifnonnull(zdirzeros[1][1][0]);
    nextface = YLOW;
    pushifnonnull(zdirzeros[1][0][0]);
    nextface = ZHIGH;
    pushifnonnull(ydirzeros[1][0][1]);
    nextface = ZLOW;
    pushifnonnull(ydirzeros[1][0][0]);

    currentface = YHIGH;
    nextface = ZHIGH;
    pushifnonnull(xdirzeros[0][1][1]);
    currentface = ZHIGH;
    nextface = YLOW;
    pushifnonnull(xdirzeros[0][0][1]);
    currentface = YLOW;
    nextface = ZLOW;
    pushifnonnull(xdirzeros[0][0][0]);
    currentface = ZLOW;
    nextface = YHIGH;
    pushifnonnull(xdirzeros[0][1][0]);

    //Now, string those edges together to make a circle.

    patch[] subdividecube() {
      if (!subdivide()) {
        return new patch[0];
      }
      return draw(reportactive);
    }
    if (zeroedges.length < 3) {
      return subdividecube();
    }
    int[] faceorder = makecircle(zeroedges);
    if (alias(faceorder,null)) {
      return subdividecube();
    }
    positionedvector[] patchcorners = new positionedvector[0];
    for (int i = 0; i < faceorder.length; ++i) {
      patchcorners.push(findzero(faceorder[i], faceorder[i+1]));
    }
    patchcorners.cyclic = true;

    //Now, produce the cyclic path around the edges.
    path3 edgecycle;
    for (int i = 0; i < faceorder.length; ++i) {
      path3 currentpath = pathinface(patchcorners[i], patchcorners[i+1],
                                     faceorder[i+1], faceorder[i],
                                     faceorder[i+2]);
      triple testpoint = point(currentpath, 0.5);
      if (!checkpt(testpoint, f, grad, corners[0][0][0], corners[1][1][1])) {
        return subdividecube();
      }
       
      edgecycle = edgecycle & currentpath;
    }
    edgecycle = edgecycle & cycle;

    
    {  // Ensure the outward normals are pointing in the same direction as the gradient.
      triple tangentin = patchcorners[0].position - precontrol(edgecycle, 0);
      triple tangentout = postcontrol(edgecycle, 0) - patchcorners[0].position;
      triple normal = cross(tangentin, tangentout);
      if (dot(normal, patchcorners[0].direction) < 0) {
        edgecycle = reverse(edgecycle);
        patchcorners = patchcorners[-sequence(patchcorners.length)];
        patchcorners.cyclic = true;
      }
    }

    patch[] toreturn = quadpatches(edgecycle, patchcorners, f, grad,
                                   corners[0][0][0], corners[1][1][1], usetriangles);
    if (alias(toreturn, null)) return subdividecube();
    return toreturn;
  }

  // Extracts the specified cube as a gridwithzeros object with
  // nx = ny = nz = 1.
  gridwithzeros getcube(int i, int j, int k) {
    gridwithzeros cube = new gridwithzeros;
    cube.grad = grad;
    cube.f = f;
    cube.nx = 1;
    cube.ny = 1;
    cube.nz = 1;
    cube.maxdepth = maxdepth;
    cube.usetriangles = usetriangles;
    cube.corners = slice(corners,i,i+2,j,j+2,k,k+2);
    cube.xdirzeros = slice(xdirzeros,i,i+1,j,j+2,k,k+2);
    cube.ydirzeros = slice(ydirzeros,i,i+2,j,j+1,k,k+2);
    cube.zdirzeros = slice(zdirzeros,i,i+2,j,j+2,k,k+1);
    return cube;
  }

  // Returns an array of patches representing the surface.
  // The parameter reportactive should be an array of
  // length 6. Setting an entry to true indicates that the surface abuts the
  // corresponding face of the cube that bounds the entire grid.
  //
  // If reportactive == null, it is assumed that this is a top-level call;
  // a dot is printed to stdout for each cube drawn as a very rough
  // progress indicator.
  //
  // If reportactive != null, then it is assumed that the caller had a strong
  // reason to believe that this grid contains a part of the surface; the
  // grid will subdivide all the way to maxdepth if necessary to find points
  // on the surface.
  draw = new patch[](bool[] reportactive = null) {
    if (alias(reportactive, null)) progress(true);
    // A list of all the patches not already drawn but known
    // to contain part of the surface. This "queue" is
    // actually implemented as stack for simplicity, since
    // it does not make any difference. In a multi-threaded
    // version of the algorithm, a queue (shared across all threads)
    // would make more sense than a stack.
    triple[] queue = new triple[0];
    bool[][][] enqueued = new bool[nx][ny][nz];
    for (int i = 0; i < enqueued.length; ++i) {
      for (int j = 0; j < enqueued[i].length; ++j) {
        for (int k = 0; k < enqueued[i][j].length; ++k) {
          enqueued[i][j][k] = false;
        }
      }
    }

    void enqueue(int i, int j, int k) {
      if (i >= 0 && i < nx
          && j >= 0 && j < ny
          && k >= 0 && k < nz
          && !enqueued[i][j][k]) {
        queue.push((i,j,k));
        enqueued[i][j][k] = true;
      }
      if (!alias(reportactive, null)) {
        if (i < 0) reportactive[XLOW] = true;
        if (i >= nx) reportactive[XHIGH] = true;
        if (j < 0) reportactive[YLOW] = true;
        if (j >= ny) reportactive[YHIGH] = true;
        if (k < 0) reportactive[ZLOW] = true;
        if (k >= nz) reportactive[ZHIGH] = true;
      }
    }
    
    for (int i = 0; i < nx+1; ++i) {
      for (int j = 0; j < ny+1; ++j) {
        for (int k = 0; k < nz+1; ++k) {
          if (i < nx && xdirzeros[i][j][k] != null) {
            for (int jj = j-1; jj <= j; ++jj)
              for (int kk = k-1; kk <= k; ++kk)
                enqueue(i, jj, kk);
          }
          if (j < ny && ydirzeros[i][j][k] != null) {
            for (int ii = i-1; ii <= i; ++ii)
              for (int kk = k-1; kk <= k; ++kk)
                enqueue(ii, j, kk);
          }
          if (k < nz && zdirzeros[i][j][k] != null) {
            for (int ii = i-1; ii <= i; ++ii)
              for (int jj = j-1; jj <= j; ++jj)
                enqueue(ii, jj, k);
          }
        }
      }
    }

    if (!alias(reportactive, null) && queue.length == 0) {
      if (subdivide()) return draw(reportactive);
    }

    patch[] surface = new patch[0];

    while (queue.length > 0) {
      triple coord = queue.pop();
      int i = floor(coord.x);
      int j = floor(coord.y);
      int k = floor(coord.z);
      bool[] reportface = array(6, false);
      patch[] toappend = getcube(i,j,k).drawcube(reportface);
      if (reportface[XLOW]) enqueue(i-1,j,k);
      if (reportface[XHIGH]) enqueue(i+1,j,k);
      if (reportface[YLOW]) enqueue(i,j-1,k);
      if (reportface[YHIGH]) enqueue(i,j+1,k);
      if (reportface[ZLOW]) enqueue(i,j,k-1);
      if (reportface[ZHIGH]) enqueue(i,j,k+1);
      surface.append(toappend);
      if (alias(reportactive, null)) progress();
    }
    if (alias(reportactive, null)) progress(false);
    return surface;
  };
}

// The external interface of this whole module. Accepts exactly one 
// function (throws an error if two or zero functions are specified).
// The function should be differentiable. (Whatever you do, do not
// pass in an indicator function!) Ideally, the zero locus of the
// function should be smooth; singularities will significantly slow
// down the algorithm and potentially give bad results.
//
// Returns a plot of the zero locus of the function within the
// rectangular solid with opposite corners at a and b.
//
// Additional parameters:
// n - the number of initial segments in each of the x, y, z directions.
// overlapedges - if true, the patches of the surface are slightly enlarged
//     to compensate for an artifact in which the viewer can see through the
//     boundary between patches. (Some of this may actually be a result of
//     edges not lining up perfectly, but I'm fairly sure a lot of it arises
//     purely as a rendering artifact.)
// nx - override n in the x direction
// ny - override n in the y direction
// nz - override n in the z direction
// maxdepth - the maximum depth to which the algorithm will subdivide in
//     an effort to find patches that closely approximate the true surface.
surface implicitsurface(real f(triple) = null, real ff(real,real,real) = null,
                        triple a, triple b,
                        int n = nmesh,
                        bool keyword overlapedges = false,
                        int keyword nx=n, int keyword ny=n,
                        int keyword nz=n,
                        int keyword maxdepth = 8,
                        bool keyword usetriangles=true) {
  if (f == null && ff == null)
    abort("implicitsurface called without specifying a function.");
  if (f != null && ff != null)
    abort("Only specify one function when calling implicitsurface.");
  if (f == null) f = new real(triple w) { return ff(w.x, w.y, w.z); };
  gridwithzeros grid = gridwithzeros(nx, ny, nz, f, a, b, maxdepth=maxdepth,
                                     usetriangles=usetriangles);
  patch[] patches = grid.draw();
  if (overlapedges) {
    for (int i = 0; i < patches.length; ++i) {
      triple center = (patches[i].triangular ?
                       patches[i].point(1/3, 1/3) : patches[i].point(1/2,1/2));
      transform3 T=shift(center) * scale3(1.03) * shift(-center);
      patches[i] = T * patches[i];
    }
  }
  return surface(...patches);
}
