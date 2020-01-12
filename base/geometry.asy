// geometry.asy

// Copyright (C) 2007
// Author: Philippe IVALDI 2007/09/01
// http://www.piprime.fr/

// This program is free software ; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation ; either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY ; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with this program ; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

// COMMENTARY:
// An Asymptote geometry module.

// THANKS:
// Special thanks to Olivier Guibe for his help in mathematical issues.

// BUGS:

// CODE:

import math;
import markers;

real Infinity=1.0/(1000*realEpsilon);

// A rotation in the direction dir limited to [-90,90]
// This is useful for rotating text along a line in the direction dir.
private transform rotate(explicit pair dir)
{
  real angle=degrees(dir);
  if(angle > 90 && angle < 270) angle -= 180;
  return rotate(angle);
} 

// *=======================================================*
// *........................HEADER.........................*
/*<asyxml><variable type="real" signature="epsgeo"><code></asyxml>*/
real epsgeo = 10 * sqrt(realEpsilon);/*<asyxml></code><documentation>Variable used in the approximate calculations.</documentation></variable></asyxml>*/

/*<asyxml><function type="void" signature="addMargins(picture,real,real,real,real)"><code></asyxml>*/
void addMargins(picture pic = currentpicture,
                real lmargin = 0, real bmargin = 0,
                real rmargin = lmargin, real tmargin = bmargin,
                bool rigid = true, bool allObject = true)
{/*<asyxml></code><documentation>Add margins to 'pic' with respect to
   the current bounding box of 'pic'.
   If 'rigid' is false, margins are added iff an infinite curve will
   be prolonged on the margin.
   If 'allObject' is false, fixed - size objects (such as labels and
   arrowheads) will be ignored.</documentation></function></asyxml>*/
  pair m = allObject ? truepoint(pic, SW) : point(pic, SW);
  pair M = allObject ? truepoint(pic, NE) : point(pic, NE);
  if(rigid) {
    draw(m - inverse(pic.calculateTransform()) * (lmargin, bmargin), invisible);
    draw(M + inverse(pic.calculateTransform()) * (rmargin, tmargin), invisible);
  } else pic.addBox(m, M, -(lmargin, bmargin), (rmargin, tmargin));
}

real approximate(real t)
{
  real ot = t;
  if(abs(t - ceil(t)) < epsgeo) ot = ceil(t);
  else if(abs(t - floor(t)) < epsgeo) ot = floor(t);
  return ot;
}

real[] approximate(real[] T)
{
  return map(approximate, T);
}

/*<asyxml><function type="real" signature="binomial(real,real)"><code></asyxml>*/
real binomial(real n, real k)
{/*<asyxml></code><documentation>Return n!/((n - k)!*k!)</documentation></function></asyxml>*/
  return gamma(n + 1)/(gamma(n - k + 1) * gamma(k + 1));
}

/*<asyxml><function type="real" signature="rf(real,real,real)"><code></asyxml>*/
real rf(real x, real y, real z)
{/*<asyxml></code><documentation>Computes Carlson's elliptic integral of the first kind.
   x, y, and z must be non negative, and at most one can be zero.</documentation></function></asyxml>*/
  real ERRTOL = 0.0025,
    TINY = 1.5e-38,
    BIG = 3e37,
    THIRD = 1/3,
    C1 = 1/24,
    C2 = 0.1,
    C3 = 3/44,
    C4 = 1/14;
  real alamb, ave, delx, dely, delz, e2, e3, sqrtx, sqrty, sqrtz, xt, yt, zt;
  if(min(x, y, z) < 0 || min(x + y, x + z, y + z) < TINY ||
     max(x, y, z) > BIG) abort("rf: invalid arguments.");
  xt = x;
  yt = y;
  zt = z;
  do {
    sqrtx = sqrt(xt);
    sqrty = sqrt(yt);
    sqrtz = sqrt(zt);
    alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
    xt = 0.25 * (xt + alamb);
    yt = 0.25 * (yt + alamb);
    zt = 0.25 * (zt + alamb);
    ave = THIRD * (xt + yt + zt);
    delx = (ave - xt)/ave;
    dely = (ave - yt)/ave;
    delz = (ave - zt)/ave;
  } while(max(fabs(delx), fabs(dely), fabs(delz)) > ERRTOL);
  e2 = delx * dely - delz * delz;
  e3 = delx * dely * delz;
  return (1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3)/sqrt(ave);
}

/*<asyxml><function type="real" signature="rd(real,real,real)"><code></asyxml>*/
real rd(real x, real y, real z)
{/*<asyxml></code><documentation>Computes Carlson's elliptic integral of the second kind.
   x and y must be positive, and at most one can be zero.
   z must be non negative.</documentation></function></asyxml>*/
  real ERRTOL = 0.0015,
    TINY = 1e-25,
    BIG = 4.5 * 10.0^21,
    C1 = (3/14),
    C2 = (1/6),
    C3 = (9/22),
    C4 = (3/26),
    C5 = (0.25 * C3),
    C6 = (1.5 * C4);
  real alamb, ave, delx, dely, delz, ea, eb, ec, ed, ee, fac, sqrtx, sqrty,
    sqrtz, sum, xt, yt, zt;
  if (min(x, y) < 0 || min(x + y, z) < TINY || max(x, y, z) > BIG)
    abort("rd: invalid arguments");
  xt = x;
  yt = y;
  zt = z;
  sum = 0;
  fac = 1;
  do {
    sqrtx = sqrt(xt);
    sqrty = sqrt(yt);
    sqrtz = sqrt(zt);
    alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
    sum += fac/(sqrtz * (zt + alamb));
    fac = 0.25 * fac;
    xt = 0.25 * (xt + alamb);
    yt = 0.25 * (yt + alamb);
    zt = 0.25 * (zt + alamb);
    ave = 0.2 * (xt + yt + 3.0 * zt);
    delx = (ave - xt)/ave;
    dely = (ave - yt)/ave;
    delz = (ave - zt)/ave;
  } while (max(fabs(delx), fabs(dely), fabs(delz)) > ERRTOL);
  ea = delx * dely;
  eb = delz * delz;
  ec = ea - eb;
  ed = ea - 6 * eb;
  ee = ed + ec + ec;
  return 3 * sum + fac * (1.0 + ed * (-C1 + C5 * ed - C6 * delz * ee)
                    +delz * (C2 * ee + delz * (-C3 * ec + delz * C4 * ea)))/(ave * sqrt(ave));
}

/*<asyxml><function type="real" signature="elle(real,real)"><code></asyxml>*/
real elle(real phi, real k)
{/*<asyxml></code><documentation>Legendre elliptic integral of the 2nd kind,
   evaluated using Carlson's functions RD and RF.
   The argument ranges are -infinity < phi < +infinity, 0 <= k * sin(phi) <= 1.</documentation></function></asyxml>*/
  real result;
  if (phi >= 0 && phi <= pi/2) {
    real cc, q, s;
    s = sin(phi);
    cc = cos(phi)^2;
    q = (1 - s * k) * (1 + s * k);
    result = s * (rf(cc, q, 1) - (s * k)^2 * rd(cc, q, 1)/3);
  } else
    if (phi <= pi && phi >= 0) {
      result = 2 * elle(pi/2, k) - elle(pi - phi, k);
    } else
      if (phi <= 3 * pi/2 && phi >= 0) {
        result = 2 * elle(pi/2, k) + elle(phi - pi, k);
      } else
        if (phi <= 2 * pi && phi >= 0) {
          result = 4 * elle(pi/2, k) - elle(2 * pi - phi, k);
        } else
          if (phi >= 0) {
            int nb = floor(0.5 * phi/pi);
            result = nb * elle(2 * pi, k) + elle(phi%(2 * pi), k);
          } else result = -elle(-phi, k);
  return result;
}

/*<asyxml><function type="pair[]" signature="intersectionpoints(pair,pair,real,real,real,real,real,real)"><code></asyxml>*/
pair[] intersectionpoints(pair A, pair B,
                          real a, real b, real c, real d, real f, real g)
{/*<asyxml></code><documentation>Intersection points with the line (AB) and the quadric curve
   a * x^2 + b * x * y + c * y^2 + d * x + f * y + g = 0 given in the default coordinate system</documentation></function></asyxml>*/
  pair[] op;
  real ap = B.y - A.y,
    bpp = A.x - B.x,
    cp = A.y * B.x - A.x * B.y;
  real sol[];
  if (abs(ap) > epsgeo) {
    real aa = ap * c + a * bpp^2/ap - b * bpp,
      bb = ap * f - bpp * d + 2 * a * bpp * cp/ap - b * cp,
      cc = ap * g - cp * d + a * cp^2/ap;
    sol = quadraticroots(aa, bb, cc);
    for (int i = 0; i < sol.length; ++i) {
      op.push((-bpp * sol[i]/ap - cp/ap, sol[i]));
    }
  } else {
    real aa = a * bpp,
      bb = d * bpp - b * cp,
      cc = g * bpp - cp * f + c * cp^2/bpp;
    sol = quadraticroots(aa, bb, cc);
    for (int i = 0; i < sol.length; ++i) {
      op.push((sol[i], -cp/bpp));
    }
  }
  return op;
}

/*<asyxml><function type="pair[]" signature="intersectionpoints(pair,pair,real[])"><code></asyxml>*/
pair[] intersectionpoints(pair A, pair B, real[] equation)
{/*<asyxml></code><documentation>Return the intersection points of the line AB with
   the conic whose an equation is
   equation[0] * x^2 + equation[1] * x * y + equation[2] * y^2 + equation[3] * x + equation[4] * y + equation[5] = 0</documentation></function></asyxml>*/
  if(equation.length != 6) abort("intersectionpoints: bad length of array for a conic equation.");
  return intersectionpoints(A, B, equation[0], equation[1], equation[2],
                            equation[3], equation[4], equation[5]);
}
// *........................HEADER.........................*
// *=======================================================*

// *=======================================================*
// *......................COORDINATES......................*

real EPS = sqrt(realEpsilon);

/*<asyxml><typedef type = "convert" return = "pair" params = "pair"><code></asyxml>*/
typedef pair convert(pair);/*<asyxml></code><documentation>Function type to convert pair in an other coordinate system.</documentation></typedef></asyxml>*/
/*<asyxml><typedef type = "abs" return = "real" params = "pair"><code></asyxml>*/
typedef real abs(pair);/*<asyxml></code><documentation>Function type to calculate modulus of pair.</documentation></typedef></asyxml>*/
/*<asyxml><typedef type = "dot" return = "real" params = "pair, pair"><code></asyxml>*/
typedef real dot(pair, pair);/*<asyxml></code><documentation>Function type to calculate dot product.</documentation></typedef></asyxml>*/
/*<asyxml><typedef type = "polar" return = "pair" params = "real, real"><code></asyxml>*/
typedef pair polar(real, real);/*<asyxml></code><documentation>Function type to calculate the coordinates from the polar coordinates.</documentation></typedef></asyxml>*/

/*<asyxml><struct signature="coordsys"><code></asyxml>*/
struct coordsys
{/*<asyxml></code><documentation>This structure represents a coordinate system in the plane.</documentation></asyxml>*/
  /*<asyxml><method type = "pair" signature="relativetodefault(pair)"><code></asyxml>*/
  restricted convert relativetodefault = new pair(pair m){return m;};/*<asyxml></code><documentation>Convert a pair given relatively to this coordinate system to
                                                                     the pair relatively to the default coordinate system.</documentation></method></asyxml>*/
  /*<asyxml><method type = "pair" signature="defaulttorelativet(pair)"><code></asyxml>*/
  restricted convert defaulttorelative = new pair(pair m){return m;};/*<asyxml></code><documentation>Convert a pair given relatively to the default coordinate system to
                                                                     the pair relatively to this coordinate system.</documentation></method></asyxml>*/
  /*<asyxml><method type = "real" signature="dot(pair,pair)"><code></asyxml>*/
  restricted dot dot = new real(pair m, pair n){return dot(m, n);};/*<asyxml></code><documentation>Return the dot product of this coordinate system.</documentation></method></asyxml>*/
  /*<asyxml><method type = "real" signature="abs(pair)"><code></asyxml>*/
  restricted abs abs = new real(pair m){return abs(m);};/*<asyxml></code><documentation>Return the modulus of a pair in this coordinate system.</documentation></method></asyxml>*/
  /*<asyxml><method type = "pair" signature="polar(real,real)"><code></asyxml>*/
  restricted polar polar = new pair(real r, real a){return (r * cos(a), r * sin(a));};/*<asyxml></code><documentation>Polar coordinates routine of this coordinate system.</documentation></method></asyxml>*/
  /*<asyxml><property type = "pair" signature="O,i,j"><code></asyxml>*/
  restricted pair O = (0, 0), i = (1, 0), j = (0, 1);/*<asyxml></code><documentation>Origin and units vector.</documentation></property></asyxml>*/
  /*<asyxml><method type = "void" signature="init(convert,convert,polar,dot)"><code></asyxml>*/
  void init(convert rtd, convert dtr,
            polar polar, dot dot)
  {/*<asyxml></code><documentation>The default constructor of the coordinate system.</documentation></method></asyxml>*/
    this.relativetodefault = rtd;
    this.defaulttorelative = dtr;
    this.polar = polar;
    this.dot = dot;
    this.abs = new real(pair m){return sqrt(dot(m, m));};;
    this.O = rtd((0, 0));
    this.i = rtd((1, 0)) - O;
    this.j = rtd((0, 1)) - O;
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><operator type = "bool" signature="==(coordsys,coordsys)"><code></asyxml>*/
bool operator ==(coordsys c1, coordsys c2)
{/*<asyxml></code><documentation>Return true iff the coordinate system have the same origin and units vector.</documentation></operator></asyxml>*/
  return c1.O == c2.O && c1.i == c2.i && c1.j == c2.j;
}

/*<asyxml><function type="coordsys" signature="cartesiansystem(pair,pair,pair)"><code></asyxml>*/
coordsys cartesiansystem(pair O = (0, 0), pair i, pair j)
{/*<asyxml></code><documentation>Return the Cartesian coordinate system (O, i, j).</documentation></function></asyxml>*/
  coordsys R;
  real[][] P = {{0, 0}, {0, 0}};
  real[][] iP;
  P[0][0] = i.x;
  P[0][1] = j.x;
  P[1][0] = i.y;
  P[1][1] = j.y;
  iP = inverse(P);
  real ni = abs(i);
  real nj = abs(j);
  real ij = angle(j) - angle(i);

  pair rtd(pair m)
  {
    return O + (P[0][0] * m.x + P[0][1] * m.y, P[1][0] * m.x + P[1][1] * m.y);
  }

  pair dtr(pair m)
  {
    m-=O;
    return (iP[0][0] * m.x + iP[0][1] * m.y, iP[1][0] * m.x + iP[1][1] * m.y);
  }

  pair polar(real r, real a)
  {
    real ca = sin(ij - a)/(ni * sin(ij));
    real sa = sin(a)/(nj * sin(ij));
    return r * (ca, sa);
  }

  real tdot(pair m, pair n)
  {
    return m.x * n.x * ni^2 + m.y * n.y * nj^2 + (m.x * n.y + n.x * m.y) * dot(i, j);
  }

  R.init(rtd, dtr, polar, tdot);
  return R;
}


/*<asyxml><function type="void" signature="show(picture,Label,Label,Label,coordsys,pen,pen,pen,pen,pen)"><code></asyxml>*/
void show(picture pic = currentpicture, Label lo = "$O$",
          Label li = "$\vec{\imath}$",
          Label lj = "$\vec{\jmath}$",
          coordsys R,
          pen dotpen = currentpen, pen xpen = currentpen, pen ypen = xpen,
          pen ipen = red,
          pen jpen = ipen,
          arrowbar arrow = Arrow)
{/*<asyxml></code><documentation>Draw the components (O, i, j, x - axis, y - axis) of 'R'.</documentation></function></asyxml>*/
  unravel R;
  dot(pic, O, dotpen);
  drawline(pic, O, O + i, xpen);
  drawline(pic, O, O + j, ypen);
  draw(pic, li, O--(O + i), ipen, arrow);
  Label lj = lj.copy();
  lj.align(lj.align, unit(I * j));
  draw(pic, lj, O--(O + j), jpen, arrow);
  draw(pic, lj, O--(O + j), jpen, arrow);
  Label lo = lo.copy();
  lo.align(lo.align, -2 * dir(O--O + i, O--O + j));
  lo.p(dotpen);
  label(pic, lo, O);
}

/*<asyxml><operator type = "pair" signature="/(pair,coordsys)"><code></asyxml>*/
pair operator /(pair p, coordsys R)
{/*<asyxml></code><documentation>Return the xy - coordinates of 'p' relatively to
   the coordinate system 'R'.
   For example, if R = cartesiansystem((1, 2), (1, 0), (0, 1)), (0, 0)/R is (-1, -2).</documentation></operator></asyxml>*/
  return R.defaulttorelative(p);
}

/*<asyxml><operator type = "pair" signature="*(coordsys,pair)"><code></asyxml>*/
pair operator *(coordsys R, pair p)
{/*<asyxml></code><documentation>Return the coordinates of 'p' given in the
   xy - coordinates 'R'.
   For example, if R = cartesiansystem((1, 2), (1, 0), (0, 1)), R * (0, 0) is (1, 2).</documentation></operator></asyxml>*/
  return R.relativetodefault(p);
}

/*<asyxml><operator type = "path" signature="*(coordsys,path)"><code></asyxml>*/
path operator *(coordsys R, path g)
{/*<asyxml></code><documentation>Return the reconstructed path applying R * pair to each node, pre and post control point of 'g'.</documentation></operator></asyxml>*/
  guide og = R * point(g, 0);
  real l = length(g);
  for(int i = 1; i <= l; ++i)
    {
      pair P = R * point(g, i);
      pair post = R * postcontrol(g, i - 1);
      pair pre = R * precontrol(g, i);
      if(i == l && (cyclic(g)))
        og = og..controls post and pre..cycle;
      else
        og = og..controls post and pre..P;
    }
  return og;
}

/*<asyxml><operator type = "coordsys" signature="*(transform,coordsys)"><code></asyxml>*/
coordsys operator *(transform t,coordsys R)
{/*<asyxml></code><documentation>Provide transform * coordsys.
   Note that shiftless(t) is applied to R.i and R.j.</documentation></operator></asyxml>*/
  coordsys oc;
  oc = cartesiansystem(t * R.O, shiftless(t) * R.i, shiftless(t) * R.j);
  return oc;
}

/*<asyxml><constant type = "coordsys" signature="defaultcoordsys"><code></asyxml>*/
restricted coordsys defaultcoordsys = cartesiansystem(0, (1, 0), (0, 1));/*<asyxml></code><documentation>One can always refer to the default coordinate system using this constant.</documentation></constant></asyxml>*/
/*<asyxml><variable type="coordsys" signature="currentcoordsys"><code></asyxml>*/
coordsys currentcoordsys = defaultcoordsys;/*<asyxml></code><documentation>The coordinate system used by default.</documentation></variable></asyxml>*/

/*<asyxml><struct signature="point"><code></asyxml>*/
struct point
{/*<asyxml></code><documentation>This structure replaces the pair to embed its coordinate system.
   For example, if 'P = point(cartesiansystem((1, 2), i, j), (0, 0))',
   P is equal to the pair (1, 2).</documentation></asyxml>*/
  /*<asyxml><property type = "coordsys" signature="coordsys"><code></asyxml>*/
  coordsys coordsys;/*<asyxml></code><documentation>The coordinate system of this point.</documentation></property><property type = "pair" signature="coordinates"><code></asyxml>*/
  restricted pair coordinates;/*<asyxml></code><documentation>The coordinates of this point relatively to the coordinate system 'coordsys'.</documentation></property><property type = "real" signature="x, y"><code></asyxml>*/
  restricted real x, y;/*<asyxml></code><documentation>The xpart and the ypart of 'coordinates'.</documentation></property></asyxml>*/
  /*<asyxml><method type = "" signature="init(coordsys,pair)"><code><property type = "real" signature="m"><code></asyxml>*/
  real m = 1;/*<asyxml></code><documentation>Used to cast mass<->point.</documentation></property></asyxml>*/
  void init(coordsys R, pair coordinates, real mass)
  {/*<asyxml></code><documentation>The constructor.</documentation></method></asyxml>*/
    this.coordsys = R;
    this.coordinates = coordinates;
    this.x = coordinates.x;
    this.y = coordinates.y;
    this.m = mass;
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="point" signature="point(coordsys,pair,real)"><code></asyxml>*/
point point(coordsys R, pair p, real m = 1)
{/*<asyxml></code><documentation>Return the point which has the coodinates 'p' in the
   coordinate system 'R' and the mass 'm'.</documentation></function></asyxml>*/
  point op;
  op.init(R, p, m);
  return op;
}

/*<asyxml><function type="point" signature="point(explicit pair,real)"><code></asyxml>*/
point point(explicit pair p, real m)
{/*<asyxml></code><documentation>Return the point which has the coodinates 'p' in the current
   coordinate system and the mass 'm'.</documentation></function></asyxml>*/
  point op;
  op.init(currentcoordsys, p, m);
  return op;
}

/*<asyxml><function type="point" signature="point(coordsys,explicit point,real)"><code></asyxml>*/
point point(coordsys R, explicit point M, real m = M.m)
{/*<asyxml></code><documentation>Return the point of 'R' which has the coordinates of 'M' and the mass 'm'.
   Do not confuse this routine with the further routine 'changecoordsys'.</documentation></function></asyxml>*/
  point op;
  op.init(R, M.coordinates, M.m);
  return op;
}

/*<asyxml><function type="point" signature="changecoordsys(coordsys,point)"><code></asyxml>*/
point changecoordsys(coordsys R, point M)
{/*<asyxml></code><documentation>Return the point 'M' in the coordinate system 'coordsys'.
   In other words, the returned point marks the same plot as 'M' does.</documentation></function></asyxml>*/
  point op;
  coordsys mco = M.coordsys;
  op.init(R, R.defaulttorelative(mco.relativetodefault(M.coordinates)), M.m);
  return op;
}

/*<asyxml><function type="pair" signature="pair coordinates(point)"><code></asyxml>*/
pair coordinates(point M)
{/*<asyxml></code><documentation>Return the coordinates of 'M' in its coordinate system.</documentation></function></asyxml>*/
  return M.coordinates;
}

/*<asyxml><function type="bool" signature="bool samecoordsys(bool...point[])"><code></asyxml>*/
bool samecoordsys(bool warn = true ... point[] M)
{/*<asyxml></code><documentation>Return true iff all the points have the same coordinate system.
   If 'warn' is true and the coordinate systems are different, a warning is sent.</documentation></function></asyxml>*/
  bool ret = true;
  coordsys t = M[0].coordsys;
  for (int i = 1; i < M.length; ++i) {
    ret = (t == M[i].coordsys);
    if(!ret) break;
    t = M[i].coordsys;
  }
  if(warn && !ret)
    warning("coodinatesystem",
            "the coordinate system of two objects are not the same.
The operation will be done relative to the default coordinate system.");
  return ret;
}

/*<asyxml><function type="point[]" signature="standardizecoordsys(coordsys,bool...point[])"><code></asyxml>*/
point[] standardizecoordsys(coordsys R = currentcoordsys,
                            bool warn = true ... point[] M)
{/*<asyxml></code><documentation>Return the points with the same coordinate system 'R'.
   If 'warn' is true and the coordinate systems are different, a warning is sent.</documentation></function></asyxml>*/
  point[] op = new point[];
  op = M;
  if(!samecoordsys(warn ... M))
    for (int i = 1; i < M.length; ++i)
      op[i] = changecoordsys(R, M[i]);
  return op;
}

/*<asyxml><operator type = "pair" signature="cast(point)"><code></asyxml>*/
pair operator cast(point P)
{/*<asyxml></code><documentation>Cast point to pair.</documentation></operator></asyxml>*/
  return P.coordsys.relativetodefault(P.coordinates);
}

/*<asyxml><operator type = "pair[]" signature="cast(point[])"><code></asyxml>*/
pair[] operator cast(point[] P)
{/*<asyxml></code><documentation>Cast point[] to pair[].</documentation></operator></asyxml>*/
  pair[] op;
  for (int i = 0; i < P.length; ++i) {
    op.push((pair)P[i]);
  }
  return op;
}

/*<asyxml><operator type = "point" signature="cast(pair)"><code></asyxml>*/
point operator cast(pair p)
{/*<asyxml></code><documentation>Cast pair to point relatively to the current coordinate
   system 'currentcoordsys'.</documentation></operator></asyxml>*/
  return point(currentcoordsys, p);
}

/*<asyxml><operator type = "point[]" signature="cast(pair[])"><code></asyxml>*/
point[] operator cast(pair[] p)
{/*<asyxml></code><documentation>Cast pair[] to point[] relatively to the current coordinate
   system 'currentcoordsys'.</documentation></operator></asyxml>*/
  pair[] op;
  for (int i = 0; i < p.length; ++i) {
    op.push((point)p[i]);
  }
  return op;
}

/*<asyxml><function type="pair" signature="locate(point)"><code></asyxml>*/
pair locate(point P)
{/*<asyxml></code><documentation>Return the coordinates of 'P' in the default coordinate system.</documentation></function></asyxml>*/
  return P.coordsys * P.coordinates;
}

/*<asyxml><function type="point" signature="locate(pair)"><code></asyxml>*/
point locate(pair p)
{/*<asyxml></code><documentation>Return the point in the current coordinate system 'currentcoordsys'.</documentation></function></asyxml>*/
  return p; //automatic casting 'pair to point'.
}

/*<asyxml><operator type = "point" signature="*(real,explicit point)"><code></asyxml>*/
point operator *(real x, explicit point P)
{/*<asyxml></code><documentation>Multiply the coordinates (not the mass) of 'P' by 'x'.</documentation></operator></asyxml>*/
  return point(P.coordsys, x * P.coordinates, P.m);
}

/*<asyxml><operator type = "point" signature="/(explicit point,real)"><code></asyxml>*/
point operator /(explicit point P, real x)
{/*<asyxml></code><documentation>Divide the coordinates (not the mass) of 'P' by 'x'.</documentation></operator></asyxml>*/
  return point(P.coordsys, P.coordinates/x, P.m);
}

/*<asyxml><operator type = "point" signature="/(real,explicit point)"><code></asyxml>*/
point operator /(real x, explicit point P)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  return point(P.coordsys, x/P.coordinates, P.m);
}

/*<asyxml><operator type = "point" signature="-(explicit point)"><code></asyxml>*/
point operator -(explicit point P)
{/*<asyxml></code><documentation>-P. The mass is inchanged.</documentation></operator></asyxml>*/
  return point(P.coordsys, -P.coordinates, P.m);
}

/*<asyxml><operator type = "point" signature="+(explicit point,explicit point)"><code></asyxml>*/
point operator +(explicit point P1, explicit point P2)
{/*<asyxml></code><documentation>Provide 'point + point'.
   If the two points haven't the same coordinate system, a warning is sent and the
   returned point has the default coordinate system 'defaultcoordsys'.
   The masses are added.</documentation></operator></asyxml>*/
  point[] P = standardizecoordsys(P1, P2);
  coordsys R = P[0].coordsys;
  return point(R, P[0].coordinates + P[1].coordinates, P1.m + P2.m);
}

/*<asyxml><operator type = "point" signature="+(explicit point,explicit pair)"><code></asyxml>*/
point operator +(explicit point P1, explicit pair p2)
{/*<asyxml></code><documentation>Provide 'point + pair'.
   The pair 'p2' is supposed to be coordinates relatively to the coordinates system of 'P1'.
   The mass is not changed.</documentation></operator></asyxml>*/
  coordsys R = currentcoordsys;
  return point(R, P1.coordinates + point(R, p2).coordinates, P1.m);
}
point operator +(explicit pair p1, explicit point p2)
{
  return p2 + p1;
}

/*<asyxml><operator type = "point" signature="-(explicit point,explicit point)"><code></asyxml>*/
point operator -(explicit point P1, explicit point P2)
{/*<asyxml></code><documentation>Provide 'point - point'.</documentation></operator></asyxml>*/
  return P1 + (-P2);
}

/*<asyxml><operator type = "point" signature="-(explicit point,explicit pair)"><code></asyxml>*/
point operator -(explicit point P1, explicit pair p2)
{/*<asyxml></code><documentation>Provide 'point - pair'.
   The pair 'p2' is supposed to be coordinates relatively to the coordinates system of 'P1'.</documentation></operator></asyxml>*/
  return P1 + (-p2);
}
point operator -(explicit pair p1, explicit point P2)
{
  return p1 + (-P2);
}

/*<asyxml><operator type = "point" signature="*(transform,explicit point)"><code></asyxml>*/
point operator *(transform t, explicit point P)
{/*<asyxml></code><documentation>Provide 'transform * point'.
   Note that the transforms scale, xscale, yscale and rotate are carried out relatively
   the default coordinate system 'defaultcoordsys' which is not desired for point
   defined in an other coordinate system.
   On can use scale(real, point), xscale(real, point), yscale(real, point), rotate(real, point),
   scaleO(real), xscaleO(real), yscaleO(real) and rotateO(real) (described further)
   to change the coordinate system of reference.</documentation></operator></asyxml>*/
  coordsys R = P.coordsys;
  return point(R, (t * locate(P))/R, P.m);
}

/*<asyxml><operator type = "point" signature="*(explicit point,explicit point)"><code></asyxml>*/
point operator *(explicit point P1, explicit point P2)
{/*<asyxml></code><documentation>Provide 'point * point'.
   The resulted mass is the mass of P2</documentation></operator></asyxml>*/
  point[] P = standardizecoordsys(P1, P2);
  coordsys R = P[0].coordsys;
  return point(R, P[0].coordinates * P[1].coordinates, P2.m);
}

/*<asyxml><operator type = "point" signature="*(explicit point,explicit pair)"><code></asyxml>*/
point operator *(explicit point P1, explicit pair p2)
{/*<asyxml></code><documentation>Provide 'point * pair'.
   The pair 'p2' is supposed to be the coordinates of
   the point in the coordinates system of 'P1'.
   'pair * point' is also defined.</documentation></operator></asyxml>*/
  point P = point(P1.coordsys, p2, P1.m);
  return P1 * P;
}
point operator *(explicit pair p1, explicit point p2)
{
  return p2 * p1;
}

/*<asyxml><operator type = "bool" signature="==(explicit point,explicit point)"><code></asyxml>*/
bool operator ==(explicit point M, explicit point N)
{/*<asyxml></code><documentation>Provide the test 'M == N' wish returns true iff MN < EPS</documentation></operator></asyxml>*/
  return abs(locate(M) - locate(N)) < EPS;
}

/*<asyxml><operator type = "bool" signature="!=(explicit point,explicit point)"><code></asyxml>*/
bool operator !=(explicit point M, explicit point N)
{/*<asyxml></code><documentation>Provide the test 'M != N' wish return true iff MN >= EPS</documentation></operator></asyxml>*/
  return !(M == N);
}

/*<asyxml><operator type = "guide" signature="cast(point)"><code></asyxml>*/
guide operator cast(point p)
{/*<asyxml></code><documentation>Cast point to guide.</documentation></operator></asyxml>*/
  return locate(p);
}

/*<asyxml><operator type = "path" signature="cast(point)"><code></asyxml>*/
path operator cast(point p)
{/*<asyxml></code><documentation>Cast point to path.</documentation></operator></asyxml>*/
  return locate(p);
}

/*<asyxml><function type="void" signature="dot(picture,Label,explicit point,align,string,pen)"><code></asyxml>*/
void dot(picture pic = currentpicture, Label L, explicit point Z,
         align align = NoAlign,
         string format = defaultformat, pen p = currentpen)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  Label L = L.copy();
  L.position(locate(Z));
  if(L.s == "") {
    if(format == "") format = defaultformat;
    L.s = "("+format(format, Z.x)+", "+format(format, Z.y)+")";
  }
  L.align(align, E);
  L.p(p);
  dot(pic, locate(Z), p);
  add(pic, L);
}

/*<asyxml><function type="real" signature="abs(coordsys,pair)"><code></asyxml>*/
real abs(coordsys R, pair m)
{/*<asyxml></code><documentation>Return the modulus |m| in the coordinate system 'R'.</documentation></function></asyxml>*/
  return R.abs(m);
}

/*<asyxml><function type="real" signature="abs(explicit point)"><code></asyxml>*/
real abs(explicit point M)
{/*<asyxml></code><documentation>Return the modulus |M| in its coordinate system.</documentation></function></asyxml>*/
  return M.coordsys.abs(M.coordinates);
}

/*<asyxml><function type="real" signature="length(explicit point)"><code></asyxml>*/
real length(explicit point M)
{/*<asyxml></code><documentation>Return the modulus |M| in its coordinate system (same as 'abs').</documentation></function></asyxml>*/
  return M.coordsys.abs(M.coordinates);
}

/*<asyxml><function type="point" signature="conj(explicit point)"><code></asyxml>*/
point conj(explicit point M)
{/*<asyxml></code><documentation>Conjugate.</documentation></function></asyxml>*/
  return point(M.coordsys, conj(M.coordinates), M.m);
}

/*<asyxml><function type="real" signature="degrees(explicit point,coordsys,bool)"><code></asyxml>*/
real degrees(explicit point M, coordsys R = M.coordsys, bool warn = true)
{/*<asyxml></code><documentation>Return the angle of M (in degrees) relatively to 'R'.</documentation></function></asyxml>*/
  return (degrees(locate(M) - R.O, warn) - degrees(R.i))%360;
}

/*<asyxml><function type="real" signature="angle(explicit point,coordsys,bool)"><code></asyxml>*/
real angle(explicit point M, coordsys R = M.coordsys, bool warn = true)
{/*<asyxml></code><documentation>Return the angle of M (in radians) relatively to 'R'.</documentation></function></asyxml>*/
  return radians(degrees(M, R, warn));
}

bool Finite(explicit point z)
{
  return abs(z.x) < Infinity && abs(z.y) < Infinity;
}

/*<asyxml><function type="bool" signature="finite(explicit point)"><code></asyxml>*/
bool finite(explicit point p)
{/*<asyxml></code><documentation>Avoid to compute 'finite((pair)(infinite_point))'.</documentation></function></asyxml>*/
  return finite(p.coordinates);
}

/*<asyxml><function type="real" signature="dot(point,point)"><code></asyxml>*/
real dot(point A, point B)
{/*<asyxml></code><documentation>Return the dot product in the coordinate system of 'A'.</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A.coordsys, A, B);
  return P[0].coordsys.dot(P[0].coordinates, P[1].coordinates);
}

/*<asyxml><function type="real" signature="dot(point,explicit pair)"><code></asyxml>*/
real dot(point A, explicit pair B)
{/*<asyxml></code><documentation>Return the dot product in the default coordinate system.
   dot(explicit pair, point) is also defined.</documentation></function></asyxml>*/
  return dot(locate(A), B);
}
real dot(explicit pair A, point B)
{
  return dot(A, locate(B));
}

/*<asyxml><function type="transforms" signature="rotateO(real)"><code></asyxml>*/
transform rotateO(real a)
{/*<asyxml></code><documentation>Rotation around the origin of the current coordinate system.</documentation></function></asyxml>*/
  return rotate(a, currentcoordsys.O);
}

/*<asyxml><function type="transform" signature="projection(point,point)"><code></asyxml>*/
transform projection(point A, point B)
{/*<asyxml></code><documentation>Return the orthogonal projection on the line (AB).</documentation></function></asyxml>*/
  pair dir = unit(locate(A) - locate(B));
  pair a = locate(A);
  real cof = dir.x * a.x + dir.y * a.y;
  real tx = a.x - dir.x * cof;
  real txx = dir.x^2;
  real txy = dir.x * dir.y;
  real ty = a.y - dir.y * cof;
  real tyx = txy;
  real tyy = dir.y^2;
  transform t = (tx, ty, txx, txy, tyx, tyy);
  return t;
}

/*<asyxml><function type="transform" signature="projection(point,point,point,point,bool)"><code></asyxml>*/
transform projection(point A, point B, point C, point D, bool safe = false)
{/*<asyxml></code><documentation>Return the (CD) parallel projection on (AB).
   If 'safe = true' and (AB)//(CD) return the identity.
   If 'safe = false' and (AB)//(CD) return an infinity scaling.</documentation></function></asyxml>*/
  pair a = locate(A);
  pair u = unit(locate(B) - locate(A));
  pair v = unit(locate(D) - locate(C));
  real c = u.x * a.y - u.y * a.x;
  real d = (conj(u) * v).y;
  if (abs(d) < epsgeo) {
    return safe ? identity() : scale(infinity);
  }
  real tx = c * v.x/d;
  real ty = c * v.y/d;
  real txx = u.x * v.y/d;
  real txy = -u.x * v.x/d;
  real tyx = u.y * v.y/d;
  real tyy = -u.y * v.x/d;
  transform t = (tx, ty, txx, txy, tyx, tyy);
  return t;
}

/*<asyxml><function type="transform" signature="scale(real,point)"><code></asyxml>*/
transform scale(real k, point M)
{/*<asyxml></code><documentation>Homothety.</documentation></function></asyxml>*/
  pair P = locate(M);
  return shift(P) * scale(k) * shift(-P);
}

/*<asyxml><function type="transform" signature="xscale(real,point)"><code></asyxml>*/
transform xscale(real k, point M)
{/*<asyxml></code><documentation>xscale from 'M' relatively to the x - axis of the coordinate system of 'M'.</documentation></function></asyxml>*/
  pair P = locate(M);
  real a = degrees(M.coordsys.i);
  return (shift(P) * rotate(a)) * xscale(k) * (rotate(-a) * shift(-P));
}

/*<asyxml><function type="transform" signature="yscale(real,point)"><code></asyxml>*/
transform yscale(real k, point M)
{/*<asyxml></code><documentation>yscale from 'M' relatively to the y - axis of the coordinate system of 'M'.</documentation></function></asyxml>*/
  pair P = locate(M);
  real a = degrees(M.coordsys.j) - 90;
  return (shift(P) * rotate(a)) * yscale(k) * (rotate(-a) * shift(-P));
}

/*<asyxml><function type="transform" signature="scale(real,point,point,point,point,bool)"><code></asyxml>*/
transform scale(real k, point A, point B, point C, point D, bool safe = false)
{/*<asyxml></code><documentation><url href = "http://fr.wikipedia.org/wiki/Affinit%C3%A9_%28math%C3%A9matiques%29"/>
   (help me for English translation...)
   If 'safe = true' and (AB)//(CD) return the identity.
   If 'safe = false' and (AB)//(CD) return a infinity scaling.</documentation></function></asyxml>*/
  pair a = locate(A);
  pair u = unit(locate(B) - locate(A));
  pair v = unit(locate(D) - locate(C));
  real c = u.x * a.y - u.y * a.x;
  real d = (conj(u) * v).y;
  real d = (conj(u) * v).y;
  if (abs(d) < epsgeo) {
    return safe ? identity() : scale(infinity);
  }
  real tx = (1 - k) * c * v.x/d;
  real ty = (1 - k) * c * v.y/d;
  real txx = (1 - k) * u.x * v.y/d + k;
  real txy = (k - 1) * u.x * v.x/d;
  real tyx = (1 - k) * u.y * v.y/d;
  real tyy = (k - 1) * u.y * v.x/d + k;
  transform t = (tx, ty, txx, txy, tyx, tyy);
  return t;
}

/*<asyxml><function type="transform" signature="scaleO(real)"><code></asyxml>*/
transform scaleO(real x)
{/*<asyxml></code><documentation>Homothety from the origin of the current coordinate system.</documentation></function></asyxml>*/
  return scale(x, (0, 0));
}

/*<asyxml><function type="transform" signature="xscaleO(real)"><code></asyxml>*/
transform xscaleO(real x)
{/*<asyxml></code><documentation>xscale from the origin and relatively to the current coordinate system.</documentation></function></asyxml>*/
  return scale(x, (0, 0), (0, 1), (0, 0), (1, 0));
}

/*<asyxml><function type="transform" signature="yscaleO(real)"><code></asyxml>*/
transform yscaleO(real x)
{/*<asyxml></code><documentation>yscale from the origin and relatively to the current coordinate system.</documentation></function></asyxml>*/
  return scale(x, (0, 0), (1, 0), (0, 0), (0, 1));
}

/*<asyxml><struct signature="vector"><code></asyxml>*/
struct vector
{/*<asyxml></code><documentation>Like a point but casting to pair, adding etc does not take account
   of the origin of the coordinate system.</documentation><property type = "point" signature="v"><code></asyxml>*/
  point v;/*<asyxml></code><documentation>Coordinates as a point (embed coordinate system and pair).</documentation></property></asyxml>*/
}/*<asyxml></struct></asyxml>*/

/*<asyxml><operator type = "point" signature="cast(vector)"><code></asyxml>*/
point operator cast(vector v)
{/*<asyxml></code><documentation>Cast vector 'v' to point 'M' so that OM = v.</documentation></operator></asyxml>*/
  return v.v;
}

/*<asyxml><operator type = "vector" signature="cast(pair)"><code></asyxml>*/
vector operator cast(pair v)
{/*<asyxml></code><documentation>Cast pair to vector relatively to the current coordinate
   system 'currentcoordsys'.</documentation></operator></asyxml>*/
  vector ov;
  ov.v = point(currentcoordsys, v);
  return ov;
}

/*<asyxml><operator type = "vector" signature="cast(explicit point)"><code></asyxml>*/
vector operator cast(explicit point v)
{/*<asyxml></code><documentation>A point can be interpreted like a vector using the code
   '(vector)a_point'.</documentation></operator></asyxml>*/
  vector ov;
  ov.v = v;
  return ov;
}

/*<asyxml><operator type = "pair" signature="cast(explicit vector)"><code></asyxml>*/
pair operator cast(explicit vector v)
{/*<asyxml></code><documentation>Cast vector to pair (the coordinates of 'v' in the default coordinate system).</documentation></operator></asyxml>*/
  return locate(v.v) - v.v.coordsys.O;
}

/*<asyxml><operator type = "align" signature="cast(vector)"><code></asyxml>*/
align operator cast(vector v)
{/*<asyxml></code><documentation>Cast vector to align.</documentation></operator></asyxml>*/
  return (pair)v;
}

/*<asyxml><function type="vector" signature="vector(coordsys, pair)"><code></asyxml>*/
vector vector(coordsys R = currentcoordsys, pair v)
{/*<asyxml></code><documentation>Return the vector of 'R' which has the coordinates 'v'.</documentation></function></asyxml>*/
  vector ov;
  ov.v = point(R, v);
  return ov;
}

/*<asyxml><function type="vector" signature="vector(point)"><code></asyxml>*/
vector vector(point M)
{/*<asyxml></code><documentation>Return the vector OM, where O is the origin of the coordinate system of 'M'.
   Useful to write 'vector(P - M);' instead of '(vector)(P - M)'.</documentation></function></asyxml>*/
  return M;
}

/*<asyxml><function type="point" signature="point(explicit vector)"><code></asyxml>*/
point point(explicit vector u)
{/*<asyxml></code><documentation>Return the point M so that OM = u, where O is the origin of the coordinate system of 'u'.</documentation></function></asyxml>*/
  return u.v;
}

/*<asyxml><function type="pair" signature="locate(explicit vector)"><code></asyxml>*/
pair locate(explicit vector v)
{/*<asyxml></code><documentation>Return the coordinates of 'v' in the default coordinate system (like casting vector to pair).</documentation></function></asyxml>*/
  return (pair)v;
}

/*<asyxml><function type="void" signature="show(Label,pen,arrowbar)"><code></asyxml>*/
void show(Label L, vector v, pen p = currentpen, arrowbar arrow = Arrow)
{/*<asyxml></code><documentation>Draw the vector v (from the origin of its coordinate system).</documentation></function></asyxml>*/
  coordsys R = v.v.coordsys;
  draw(L, R.O--v.v, p, arrow);
}

/*<asyxml><function type="vector" signature="changecoordsys(coordsys,vector)"><code></asyxml>*/
vector changecoordsys(coordsys R, vector v)
{/*<asyxml></code><documentation>Return the vector 'v' relatively to coordinate system 'R'.</documentation></function></asyxml>*/
  vector ov;
  ov.v = point(R, (locate(v) + R.O)/R);
  return ov;
}

/*<asyxml><operator type = "vector" signature="*(real,explicit vector)"><code></asyxml>*/
vector operator *(real x, explicit vector v)
{/*<asyxml></code><documentation>Provide real * vector.</documentation></operator></asyxml>*/
  return x * v.v;
}

/*<asyxml><operator type = "vector" signature="/(explicit vector,real)"><code></asyxml>*/
vector operator /(explicit vector v, real x)
{/*<asyxml></code><documentation>Provide vector/real</documentation></operator></asyxml>*/
  return v.v/x;
}

/*<asyxml><operator type = "vector" signature="*(transform t,explicit vector)"><code></asyxml>*/
vector operator *(transform t, explicit vector v)
{/*<asyxml></code><documentation>Provide transform * vector.</documentation></operator></asyxml>*/
  return t * v.v;
}

/*<asyxml><operator type = "vector" signature="*(explicit point,explicit vector)"><code></asyxml>*/
vector operator *(explicit point M, explicit vector v)
{/*<asyxml></code><documentation>Provide point * vector</documentation></operator></asyxml>*/
  return M * v.v;
}

/*<asyxml><operator type = "point" signature="+(explicit point,explicit vector)"><code></asyxml>*/
point operator +(point M, explicit vector v)
{/*<asyxml></code><documentation>Return 'M' shifted by 'v'.</documentation></operator></asyxml>*/
  return shift(locate(v)) * M;
}

/*<asyxml><operator type = "point" signature="-(explicit point,explicit vector)"><code></asyxml>*/
point operator -(point M, explicit vector v)
{/*<asyxml></code><documentation>Return 'M' shifted by '-v'.</documentation></operator></asyxml>*/
  return shift(-locate(v)) * M;
}

/*<asyxml><operator type = "vector" signature="-(explicit vector)"><code></asyxml>*/
vector operator -(explicit vector v)
{/*<asyxml></code><documentation>Provide -v.</documentation></operator></asyxml>*/
  return -v.v;
}

/*<asyxml><operator type = "point" signature="+(explicit pair,explicit vector)"><code></asyxml>*/
point operator +(explicit pair m, explicit vector v)
{/*<asyxml></code><documentation>The pair 'm' is supposed to be the coordinates of
   a point in the current coordinates system 'currentcoordsys'.
   Return this point shifted by the vector 'v'.</documentation></operator></asyxml>*/
  return locate(m) + v;
}

/*<asyxml><operator type = "point" signature="-(explicit pair,explicit vector)"><code></asyxml>*/
point operator -(explicit pair m, explicit vector v)
{/*<asyxml></code><documentation>The pair 'm' is supposed to be the coordinates of
   a point in the current coordinates system 'currentcoordsys'.
   Return this point shifted by the vector '-v'.</documentation></operator></asyxml>*/
  return m + (-v);
}

/*<asyxml><operator type = "vector" signature="+(explicit vector,explicit vector)"><code></asyxml>*/
vector operator +(explicit vector v1, explicit vector v2)
{/*<asyxml></code><documentation>Provide vector + vector.
   If the two vector haven't the same coordinate system, the returned
   vector is relative to the default coordinate system (without warning).</documentation></operator></asyxml>*/
  coordsys R = v1.v.coordsys;
  if(samecoordsys(false, v1, v2)){R = defaultcoordsys;}
  return vector(R, (locate(v1) + locate(v2))/R);
}

/*<asyxml><operator type = "vector" signature="-(explicit vector, explicit vector)"><code></asyxml>*/
vector operator -(explicit vector v1, explicit vector v2)
{/*<asyxml></code><documentation>Provide vector - vector.
   If the two vector haven't the same coordinate system, the returned
   vector is relative to the default coordinate system (without warning).</documentation></operator></asyxml>*/
  return v1 + (-v2);
}

/*<asyxml><operator type = "bool" signature="==(explicit vector,explicit vector)"><code></asyxml>*/
bool operator ==(explicit vector u, explicit vector v)
{/*<asyxml></code><documentation>Return true iff |u - v|<EPS.</documentation></operator></asyxml>*/
  return abs(u - v) < EPS;
}

/*<asyxml><function type="bool" signature="collinear(vector,vector)"><code></asyxml>*/
bool collinear(vector u, vector v)
{/*<asyxml></code><documentation>Return 'true' iff the vectors 'u' and 'v' are collinear.</documentation></function></asyxml>*/
  return abs(ypart((conj((pair)u) * (pair)v))) < EPS;
}

/*<asyxml><function type="vector" signature="unit(point)"><code></asyxml>*/
vector unit(point M)
{/*<asyxml></code><documentation>Return the unit vector according to the modulus of its coordinate system.</documentation></function></asyxml>*/
  return M/abs(M);
}

/*<asyxml><function type="vector" signature="unit(vector)"><code></asyxml>*/
vector unit(vector u)
{/*<asyxml></code><documentation>Return the unit vector according to the modulus of its coordinate system.</documentation></function></asyxml>*/
  return u.v/abs(u.v);
}

/*<asyxml><function type="real" signature="degrees(vector,coordsys,bool)"><code></asyxml>*/
real degrees(vector v,
             coordsys R = v.v.coordsys,
             bool warn = true)
{/*<asyxml></code><documentation>Return the angle of 'v' (in degrees) relatively to 'R'.</documentation></function></asyxml>*/
  return (degrees(locate(v), warn) - degrees(R.i))%360;
}

/*<asyxml><function type="real" signature="angle(vector,coordsys,bool)"><code></asyxml>*/
real angle(explicit vector v,
           coordsys R = v.v.coordsys,
           bool warn = true)
{/*<asyxml></code><documentation>Return the angle of 'v' (in radians) relatively to 'R'.</documentation></function></asyxml>*/
  return radians(degrees(v, R, warn));
}

/*<asyxml><function type="vector" signature="conj(explicit vector)"><code></asyxml>*/
vector conj(explicit vector u)
{/*<asyxml></code><documentation>Conjugate.</documentation></function></asyxml>*/
  return conj(u.v);
}

/*<asyxml><function type="transform" signature="rotate(explicit vector)"><code></asyxml>*/
transform rotate(explicit vector dir)
{/*<asyxml></code><documentation>A rotation in the direction 'dir' limited to [-90, 90]
   This is useful for rotating text along a line in the direction dir.
   rotate(explicit point dir) is also defined.
   </documentation></function></asyxml>*/
  return rotate(locate(dir));
}
transform rotate(explicit point dir){return rotate(locate(vector(dir)));}
// *......................COORDINATES......................*
// *=======================================================*

// *=======================================================*
// *.........................BASES.........................*
/*<asyxml><variable type="point" signature="origin"><code></asyxml>*/
point origin = point(defaultcoordsys, (0, 0));/*<asyxml></code><documentation>The origin of the current coordinate system.</documentation></variable></asyxml>*/

/*<asyxml><function type="point" signature="origin(coordsys)"><code></asyxml>*/
point origin(coordsys R = currentcoordsys)
{/*<asyxml></code><documentation>Return the origin of the coordinate system 'R'.</documentation></function></asyxml>*/
  return point(R, (0, 0)); //use automatic casting;
}

/*<asyxml><variable type="real" signature="linemargin"><code></asyxml>*/
real linemargin = 0;/*<asyxml></code><documentation>Margin used to draw lines.</documentation></variable></asyxml>*/
/*<asyxml><function type="real" signature="linemargin()"><code></asyxml>*/
real linemargin()
{/*<asyxml></code><documentation>Return the margin used to draw lines.</documentation></function></asyxml>*/
  return linemargin;
}

/*<asyxml><variable type="pen" signature="addpenline"><code></asyxml>*/
pen addpenline = squarecap;/*<asyxml></code><documentation>Add this property to the drawing pen of "finish" lines.</documentation></variable></asyxml>*/
pen addpenline(pen p) {
  return addpenline + p;
}

/*<asyxml><variable type="pen" signature="addpenarc"><code></asyxml>*/
pen addpenarc = squarecap;/*<asyxml></code><documentation>Add this property to the drawing pen of arcs.</documentation></variable></asyxml>*/
pen addpenarc(pen p) {return addpenarc + p;}

/*<asyxml><variable type="string" signature="defaultmassformat"><code></asyxml>*/
string defaultmassformat = "$\left(%L;%.4g\right)$";/*<asyxml></code><documentation>Format used to construct the default label of masses.</documentation></variable></asyxml>*/

/*<asyxml><function type="int" signature="sgnd(real)"><code></asyxml>*/
int sgnd(real x)
{/*<asyxml></code><documentation>Return the -1 if x < 0, 1 if x >= 0.</documentation></function></asyxml>*/
  return (x == 0) ? 1 : sgn(x);
}
int sgnd(int x)
{
  return (x == 0) ? 1 : sgn(x);
}

/*<asyxml><function type="bool" signature="defined(pair)"><code></asyxml>*/
bool defined(point P)
{/*<asyxml></code><documentation>Return true iff the coordinates of 'P' are finite.</documentation></function></asyxml>*/
  return finite(P.coordinates);
}

/*<asyxml><function type="bool" signature="onpath(picture,path,point,pen)"><code></asyxml>*/
bool onpath(picture pic = currentpicture, path g, point M, pen p = currentpen)
{/*<asyxml></code><documentation>Return true iff 'M' is on the path drawn with the pen 'p' in 'pic'.</documentation></function></asyxml>*/
  transform t = inverse(pic.calculateTransform());
  return intersect(g, shift(locate(M)) * scale(linewidth(p)/2) * t * unitcircle).length > 0;
}

/*<asyxml><function type="bool" signature="sameside(point,point,point)"><code></asyxml>*/
bool sameside(point M, point N, point O)
{/*<asyxml></code><documentation>Return 'true' iff 'M' and 'N' are same side of the point 'O'.</documentation></function></asyxml>*/
  pair m = M, n = N, o = O;
  return dot(m - o, n - o) >= -epsgeo;
}

/*<asyxml><function type="bool" signature="between(point,point,point)"><code></asyxml>*/
bool between(point M, point O, point N)
{/*<asyxml></code><documentation>Return 'true' iff 'O' is between 'M' and 'N'.</documentation></function></asyxml>*/
  return (!sameside(N, M, O) || M == O || N == O);
}


typedef path pathModifier(path);
pathModifier NoModifier = new path(path g){return g;};

private void Drawline(picture pic = currentpicture, Label L = "", pair P, bool dirP = true, pair Q, bool dirQ = true,
                      align align = NoAlign, pen p = currentpen,
                      arrowbar arrow = None,
                      Label legend = "", marker marker = nomarker,
                      pathModifier pathModifier = NoModifier)
{/* Add the two parameters 'dirP' and 'dirQ' to the native routine
    'drawline' of the module 'math'.
    Segment [PQ] will be prolonged in direction of P if 'dirP = true', in
    direction of Q if 'dirQ = true'.
    If 'dirP = dirQ = true', the behavior is that of the native 'drawline'.
    Add all the other parameters of 'Draw'.*/
  pic.add(new void (frame f, transform t, transform T, pair m, pair M) {
      picture opic;
      // Reduce the bounds by the size of the pen.
      m -= min(p) - (linemargin(), linemargin()); M -= max(p) + (linemargin(), linemargin());

      // Calculate the points and direction vector in the transformed space.
      t = t * T;
      pair z = t * P;
      pair q = t * Q;
      pair v = q - z;
      // path g;
      pair ptp, ptq;
      real cp = dirP ? 1:0;
      real cq = dirQ ? 1:0;
      // Handle horizontal and vertical lines.
      if(v.x == 0) {
        if(m.x <= z.x && z.x <= M.x)
          if (dot(v, m - z) < 0) {
            ptp = (z.x, z.y + cp * (m.y - z.y));
            ptq = (z.x, q.y + cq * (M.y - q.y));
          } else {
            ptq = (z.x, q.y + cq * (m.y - q.y));
            ptp = (z.x, z.y + cp * (M.y - z.y));
          }
      } else if(v.y == 0) {
        if (dot(v, m - z) < 0) {
          ptp = (z.x + cp * (m.x - z.x), z.y);
          ptq = (q.x + cq * (M.x - q.x), z.y);
        } else {
          ptq = (q.x + cq * (m.x - q.x), z.y);
          ptp = (z.x + cp * (M.x - z.x), z.y);
        }
      } else {
        // Calculate the maximum and minimum t values allowed for the
        // parametric equation z + t * v
        real mx = (m.x - z.x)/v.x, Mx = (M.x - z.x)/v.x;
        real my = (m.y - z.y)/v.y, My = (M.y - z.y)/v.y;
        real tmin = max(v.x > 0 ? mx : Mx, v.y > 0 ? my : My);
        real tmax = min(v.x > 0 ? Mx : mx, v.y > 0 ? My : my);
        pair pmin = z + tmin * v;
        pair pmax = z + tmax * v;
        if(tmin <= tmax) {
          ptp = z + cp * tmin * v;
          ptq = z + (cq == 0 ? v:tmax * v);
        }
      }
      path g = ptp--ptq;
      if (length(g)>0)
        {
          if(L.s != "") {
            Label lL = L.copy();
            if(L.defaultposition) lL.position(Relative(.9));
            lL.p(p);
            lL.out(opic, g);
          }
          g = pathModifier(g);
          if(linetype(p).length == 0){
            pair m = midpoint(g);
            pen tp;
            tp = dirP ? p : addpenline(p);
            draw(opic, pathModifier(m--ptp), tp);
            tp = dirQ ? p : addpenline(p);
            draw(opic, pathModifier(m--ptq), tp);
          } else {
            draw(opic, g, p);
          }
          marker.markroutine(opic, marker.f, g);
          arrow(opic, g, p, NoMargin);
          add(f, opic.fit());
        }
    });
}

/*<asyxml><function type="void" signature="clipdraw(picture,Label,path,align,pen,arrowbar,arrowbar,real,real,Label,marker)"><code></asyxml>*/
void clipdraw(picture pic = currentpicture, Label L = "", path g,
              align align = NoAlign, pen p = currentpen,
              arrowbar arrow = None, arrowbar bar = None,
              real xmargin = 0, real ymargin = xmargin,
              Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation>Draw the path 'g' on 'pic' clipped to the bounding box of 'pic'.</documentation></function></asyxml>*/
  if(L.s != "") {
    picture tmp;
    label(tmp, L, g, p);
    add(pic, tmp);
  }
  pic.add(new void (frame f, transform t, transform T, pair m, pair M) {
      // Reduce the bounds by the size of the pen and the margins.
      m += min(p) + (xmargin, ymargin); M -= max(p) + (xmargin, ymargin);
      path bound = box(m, M);
      picture tmp;
      draw(tmp, "", t * T * g, align, p, arrow, bar, NoMargin, legend, marker);
      clip(tmp, bound);
      add(f, tmp.fit());
    });
}

/*<asyxml><function type="void" signature="distance(picture pic,Label,point,point,bool,real,pen,pen,arrow)"><code></asyxml>*/
void distance(picture pic = currentpicture, Label L = "", point A, point B,
              bool rotated = true, real offset = 3mm,
              pen p = currentpen, pen joinpen = invisible,
              arrowbar arrow = Arrows(NoFill))
{/*<asyxml></code><documentation>Draw arrow between A and B (from FAQ).</documentation></function></asyxml>*/
  pair A = A, B = B;
  path g = A--B;
  transform Tp = shift(-offset * unit(B - A) * I);
  pic.add(new void(frame f, transform t) {
      picture opic;
      path G = Tp * t * g;
      transform id = identity();
      transform T = rotated ? rotate(B - A) : id;
      Label L = L.copy();
      L.align(L.align, Center);
      if(abs(ypart((conj(A - B) * L.align.dir))) < epsgeo && L.filltype == NoFill)
        L.filltype = UnFill(1);
      draw(opic, T * L, G, p, arrow, Bars, PenMargins);
      pair Ap = t * A, Bp = t * B;
      draw(opic, (Ap--Tp * Ap)^^(Bp--Tp * Bp), joinpen);
      add(f, opic.fit());
    }, true);
  pic.addBox(min(g), max(g), Tp * min(p), Tp * max(p));
}

/*<asyxml><variable type="real" signature="perpfactor"><code></asyxml>*/
real perpfactor = 1;/*<asyxml></code><documentation>Factor for drawing perpendicular symbol.</documentation></variable></asyxml>*/
/*<asyxml><function type="void" signature="perpendicularmark(picture,point,explicit pair,explicit pair,real,pen,margin,filltype)"><code></asyxml>*/
void perpendicularmark(picture pic = currentpicture, point z,
                       explicit pair align,
                       explicit pair dir = E, real size = 0,
                       pen p = currentpen,
                       margin margin = NoMargin,
                       filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw a perpendicular symbol at z aligned in the direction align
   relative to the path z--z + dir.
   dir(45 + n * 90), where n in N*, are common values for 'align'.</documentation></function></asyxml>*/
  p = squarecap + miterjoin + p;
  if(size == 0) size = perpfactor * 3mm + linewidth(p) / 2;
  frame apic;
  pair d1 = size * align * unit(dir) * dir(-45);
  pair d2 = I * d1;
  path g = d1--d1 + d2--d2;
  g = margin(g, p).g;
  draw(apic, g, p);
  if(filltype != NoFill) filltype.fill(apic, (relpoint(g, 0) - relpoint(g, 0.5)+
                                             relpoint(g, 1))--g--cycle, p + solid);
  add(pic, apic, locate(z));
}

/*<asyxml><function type="void" signature="perpendicularmark(picture,point,vector,vector,real,pen,margin,filltype)"><code></asyxml>*/
void perpendicularmark(picture pic = currentpicture, point z,
                       vector align,
                       vector dir = E, real size = 0,
                       pen p = currentpen,
                       margin margin = NoMargin,
                       filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw a perpendicular symbol at z aligned in the direction align
   relative to the path z--z + dir.
   dir(45 + n * 90), where n in N, are common values for 'align'.</documentation></function></asyxml>*/
  perpendicularmark(pic, z, (pair)align, (pair)dir, size,
                    p, margin, filltype);
}

/*<asyxml><function type="void" signature="perpendicularmark(picture,point,explicit pair,path,real,pen,margin,filltype)"><code></asyxml>*/
void perpendicularmark(picture pic = currentpicture, point z, explicit pair align, path g,
                       real size = 0, pen p = currentpen,
                       margin margin = NoMargin,
                       filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw a perpendicular symbol at z aligned in the direction align
   relative to the path z--z + dir(g, 0).
   dir(45 + n * 90), where n in N, are common values for 'align'.</documentation></function></asyxml>*/
  perpendicularmark(pic, z, align, dir(g, 0), size, p, margin, filltype);
}

/*<asyxml><function type="void" signature="perpendicularmark(picture,point,vector,path,real,pen,margin,filltype)"><code></asyxml>*/
void perpendicularmark(picture pic = currentpicture, point z, vector align, path g,
                       real size = 0, pen p = currentpen,
                       margin margin = NoMargin,
                       filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw a perpendicular symbol at z aligned in the direction align
   relative to the path z--z + dir(g, 0).
   dir(45 + n * 90), where n in N, are common values for 'align'.</documentation></function></asyxml>*/
  perpendicularmark(pic, z, (pair)align, dir(g, 0), size, p, margin, filltype);
}

/*<asyxml><function type="void" signature="markrightangle(picture,point,point,point,real,pen,margin,filltype)"><code></asyxml>*/
void markrightangle(picture pic = currentpicture, point A, point O,
                    point B, real size = 0, pen p = currentpen,
                    margin margin = NoMargin,
                    filltype filltype = NoFill)
{/*<asyxml></code><documentation>Mark the angle AOB with a perpendicular symbol.</documentation></function></asyxml>*/
  pair Ap = A, Bp = B, Op = O;
  pair dir = Ap - Op;
  real a1 = degrees(dir);
  pair align = rotate(-a1) * unit(dir(Op--Ap, Op--Bp));
  perpendicularmark(pic = pic, z = O, align = align,
                    dir = dir, size = size, p = p,
                    margin = margin, filltype = filltype);
}

/*<asyxml><function type="bool" signature="simeq(point,point,real)"><code></asyxml>*/
bool simeq(point A, point B, real fuzz = epsgeo)
{/*<asyxml></code><documentation>Return true iff abs(A - B) < fuzz.
   This routine is used internally to know if two points are equal, in particular by the operator == in 'point == point'.</documentation></function></asyxml>*/
  return (abs(A - B) < fuzz);
}
bool simeq(point a, real b, real fuzz = epsgeo)
{
  coordsys R = a.coordsys;
  return (abs(a - point(R, ((pair)b)/R)) < fuzz);
}

/*<asyxml><function type="pair" signature="attract(pair,path,real)"><code></asyxml>*/
pair attract(pair m, path g, real fuzz = 0)
{/*<asyxml></code><documentation>Return the nearest point (A PAIR) of 'm' which is on the path g.
   'fuzz' is the argument 'fuzz' of 'intersect'.</documentation></function></asyxml>*/
  if(intersect(m, g, fuzz).length > 0) return m;
  pair p;
  real step = 1, r = 0;
  real[] t;
  static real eps = sqrt(realEpsilon);
  do {// Find a radius for intersection
    r += step;
    t = intersect(shift(m) * scale(r) * unitcircle, g);
  } while(t.length <= 0);
  p = point(g, t[1]);
  real rm = 0, rM = r;
  while(rM - rm > eps) {
    r = (rm + rM)/2;
    t = intersect(shift(m) * scale(r) * unitcircle, g, fuzz);
    if(t.length <= 0) {
      rm = r;
    } else {
      rM = r;
      p = point(g, t[1]);
    }
  }
  return p;
}

/*<asyxml><function type="point" signature="attract(point,path,real)"><code></asyxml>*/
point attract(point M, path g, real fuzz = 0)
{/*<asyxml></code><documentation>Return the nearest point (A POINT) of 'M' which is on the path g.
   'fuzz' is the argument 'fuzz' of 'intersect'.</documentation></function></asyxml>*/
  return point(M.coordsys, attract(locate(M), g)/M.coordsys);
}

/*<asyxml><function type="real[]" signature="intersect(path,explicit pair)"><code></asyxml>*/
real[] intersect(path g, explicit pair p, real fuzz = 0)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  fuzz = fuzz <= 0 ? sqrt(realEpsilon) : fuzz;
  real[] or;
  real r = realEpsilon;
  do{
    or = intersect(g, shift(p) * scale(r) * unitcircle, fuzz);
    r *= 2;
  } while(or.length == 0);
  return or;
}

/*<asyxml><function type="real[]" signature="intersect(path,explicit point)"><code></asyxml>*/
real[] intersect(path g, explicit point P, real fuzz = epsgeo)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersect(g, locate(P), fuzz);
}
// *.........................BASES.........................*
// *=======================================================*

// *=======================================================*
// *.........................LINES.........................*
/*<asyxml><struct signature="line"><code></asyxml>*/
struct line
{/*<asyxml></code><documentation>This structure provides the objects line, semi - line and segment oriented from A to B.
   All the calculus with this structure will be as exact as Asymptote can do.
   For a full precision, you must not cast 'line' to 'path' excepted for drawing routines.</documentation></asyxml>*/
  /*<asyxml><property type = "point" signature="A,B"><code></asyxml>*/
  restricted point A,B;/*<asyxml></code><documentation>Two line's points with same coordinate system.</documentation></property><property type = "bool" signature="extendA,extendB"><code></asyxml>*/
  bool extendA,extendB;/*<asyxml></code><documentation>If true,extend 'l' in direction of A (resp. B).</documentation></property><property type = "vector" signature="u,v"><code></asyxml>*/
  restricted vector u,v;/*<asyxml></code><documentation>u = unit(AB) = direction vector,v = normal vector.</documentation></property><property type = "real" signature="a,b,c"><code></asyxml>*/
  restricted real a,b,c;/*<asyxml></code><documentation>Coefficients of the equation ax + by + c = 0 in the coordinate system of 'A'.</documentation></property><property type = "real" signature="slope,origin"><code></asyxml>*/
  restricted real slope, origin;/*<asyxml></code><documentation>Slope and ordinate at the origin.</documentation></property></asyxml>*/
  /*<asyxml><method type = "line" signature="copy()"><code></asyxml>*/
  line copy()
  {/*<asyxml></code><documentation>Copy a line in a new instance.</documentation></method></asyxml>*/
    line l = new line;
    l.A = A;
    l.B = B;
    l.a = a;
    l.b = b;
    l.c = c;
    l.slope = slope;
    l.origin = origin;
    l.u = u;
    l.v = v;
    l.extendA = extendA;
    l.extendB = extendB;
    return l;
  }

  /*<asyxml><method type = "void" signature="init(point,bool,point,bool)"><code></asyxml>*/
  void init(point A, bool extendA = true, point B, bool extendB = true)
  {/*<asyxml></code><documentation>Initialize line.
     If 'extendA' is true, the "line" is infinite in the direction of A.</documentation></method></asyxml>*/
    point[] P = standardizecoordsys(A, B);
    this.A = P[0];
    this.B = P[1];
    this.a = B.y - A.y;
    this.b = A.x - B.x;
    this.c = A.y * B.x - A.x * B.y;
    this.slope= (this.b == 0) ? infinity : -this.a/this.b;
    this.origin = (this.b == 0) ? (this.c == 0) ? 0:infinity : -this.c/this.b;
    this.u = unit(P[1]-P[0]);
    //     int tmp = sgnd(this.slope);
    //     this.u = (dot((pair)this.u, N) >= 0) ? tmp * this.u : -tmp * this.u;
    this.v = rotate(90, point(P[0].coordsys, (0, 0))) * this.u;
    this.extendA = extendA;
    this.extendB = extendB;
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="line" signature="line(point,bool,point,bool)"><code></asyxml>*/
line line(point A, bool extendA = true, point B, bool extendB = true)
{/*<asyxml></code><documentation>Return the line passing through 'A' and 'B'.
   If 'extendA' is true, the "line" is infinite in the direction of A.
   A "line" can be half-line or segment.</documentation></function></asyxml>*/
  if (A == B) abort("line: the points must be distinct.");
  line l;
  l.init(A, extendA, B, extendB);
  return l;
}

/*<asyxml><struct signature="segment"><code></asyxml>*/
struct segment
{/*<asyxml></code><documentation><look href = "struct line"/>.</documentation></asyxml>*/
  restricted point A, B;// Extremity.
  restricted vector u, v;// u = direction vector, v = normal vector.
  restricted real a, b, c;// Coefficients of the equation ax + by + c = 0
  restricted real slope, origin;
  segment copy()
  {
    segment s = new segment;
    s.A = A;
    s.B = B;
    s.a = a;
    s.b = b;
    s.c = c;
    s.slope = slope;
    s.origin = origin;
    s.u = u;
    s.v = v;
    return s;
  }

  void init(point A, point B)
  {
    line l;
    l.init(A, B);
    this.A = l.A; this.B = l.B;
    this.a = l.a; this.b = l.b; this.c = l.c;
    this.slope = l.slope; this.origin = l.origin;
    this.u = l.u; this.v = l.v;
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="segment" signature="segment(point,point)"><code></asyxml>*/
segment segment(point A, point B)
{/*<asyxml></code><documentation>Return the segment whose the extremities are A and B.</documentation></function></asyxml>*/
  segment s;
  s.init(A, B);
  return s;
}

/*<asyxml><function type="real" signature="length(segment)"><code></asyxml>*/
real length(segment s)
{/*<asyxml></code><documentation>Return the length of 's'.</documentation></function></asyxml>*/
  return abs(s.A - s.B);
}

/*<asyxml><operator type = "line" signature="cast(segment)"><code></asyxml>*/
line operator cast(segment s)
{/*<asyxml></code><documentation>A segment is casted to a "finite line".</documentation></operator></asyxml>*/
  return line(s.A, false, s.B, false);
}

/*<asyxml><operator type = "segment" signature="cast(line)"><code></asyxml>*/
segment operator cast(line l)
{/*<asyxml></code><documentation>Cast line 'l' to segment [l.A l.B].</documentation></operator></asyxml>*/
  return segment(l.A, l.B);
}

/*<asyxml><operator type = "line" signature="*(transform,line)"><code></asyxml>*/
line operator *(transform t, line l)
{/*<asyxml></code><documentation>Provide transform * line</documentation></operator></asyxml>*/
  return line(t * l.A, l.extendA, t * l.B, l.extendB);
}
/*<asyxml><operator type = "line" signature="/(line,real)"><code></asyxml>*/
line operator /(line l, real x)
{/*<asyxml></code><documentation>Provide l/x.
   Return the line passing through l.A/x and l.B/x.</documentation></operator></asyxml>*/
  return line(l.A/x, l.extendA, l.B/x, l.extendB);
}
line operator /(line l, int x){return line(l.A/x, l.B/x);}
/*<asyxml><operator type = "line" signature="*(real,line)"><code></asyxml>*/
line operator *(real x, line l)
{/*<asyxml></code><documentation>Provide x * l.
   Return the line passing through x * l.A and x * l.B.</documentation></operator></asyxml>*/
  return line(x * l.A, l.extendA, x * l.B, l.extendB);
}
line operator *(int x, line l){return line(x * l.A, l.extendA, x * l.B, l.extendB);}

/*<asyxml><operator type = "line" signature="*(point,line)"><code></asyxml>*/
line operator *(point M, line l)
{/*<asyxml></code><documentation>Provide point * line.
   Return the line passing through unit(M) * l.A and unit(M) * l.B.</documentation></operator></asyxml>*/
  return line(unit(M) * l.A, l.extendA, unit(M) * l.B, l.extendB);
}
/*<asyxml><operator type = "line" signature="+(line,point)"><code></asyxml>*/
line operator +(line l, vector u)
{/*<asyxml></code><documentation>Provide line + vector (and so line + point).
   Return the line 'l' shifted by 'u'.</documentation></operator></asyxml>*/
  return line(l.A + u, l.extendA, l.B + u, l.extendB);
}
/*<asyxml><operator type = "line" signature="-(line,vector)"><code></asyxml>*/
line operator -(line l, vector u)
{/*<asyxml></code><documentation>Provide line - vector (and so line - point).
   Return the line 'l' shifted by '-u'.</documentation></operator></asyxml>*/
  return line(l.A - u, l.extendA, l.B - u, l.extendB);
}

/*<asyxml><operator type = "line[]" signature="^^(line,line)"><code></asyxml>*/
line[] operator ^^(line l1, line l2)
{/*<asyxml></code><documentation>Provide line^^line.
   Return the line array {l1, l2}.</documentation></operator></asyxml>*/
  line[] ol;
  ol.push(l1); ol.push(l2);
  return ol;
}

/*<asyxml><operator type = "line[]" signature="^^(line,line[])"><code></asyxml>*/
line[] operator ^^(line l1, line[] l2)
{/*<asyxml></code><documentation>Provide line^^line[].
   Return the line array {l1, l2[0], l2[1]...}.
   line[]^^line is also defined.</documentation></operator></asyxml>*/
  line[] ol;
  ol.push(l1);
  for (int i = 0; i < l2.length; ++i) {
    ol.push(l2[i]);
  }
  return ol;
}
line[] operator ^^(line[] l2, line l1)
{
  line[] ol = l2;
  ol.push(l1);
  return ol;
}

/*<asyxml><operator type = "line[]" signature="^^(line,line[])"><code></asyxml>*/
line[] operator ^^(line l1[], line[] l2)
{/*<asyxml></code><documentation>Provide line[]^^line[].
   Return the line array {l1[0], l1[1], ..., l2[0], l2[1], ...}.</documentation></operator></asyxml>*/
  line[] ol = l1;
  for (int i = 0; i < l2.length; ++i) {
    ol.push(l2[i]);
  }
  return ol;
}

/*<asyxml><function type="bool" signature="sameside(point,point,line)"><code></asyxml>*/
bool sameside(point M, point P, line l)
{/*<asyxml></code><documentation>Return 'true' iff 'M' and 'N' are same side of the line (or on the line) 'l'.</documentation></function></asyxml>*/
  pair A = l.A, B = l.B, m = M, p = P;
  pair mil = (A + B)/2;
  pair mA = rotate(90, mil) * A;
  pair mB = rotate(-90, mil) * A;
  return (abs(m - mA) <= abs(m - mB)) == (abs(p - mA) <= abs(p - mB));
  // transform proj = projection(l.A, l.B);
  // point Mp = proj * M;
  // point Pp = proj * P;
  // dot(Mp);dot(Pp);
  // return dot(locate(Mp - M), locate(Pp - P)) >= 0;
}

/*<asyxml><function type="line" signature="line(segment)"><code></asyxml>*/
line line(segment s)
{/*<asyxml></code><documentation>Return the line passing through 's.A'
   and 's.B'.</documentation></function></asyxml>*/
  return line(s.A, s.B);
}
/*<asyxml><function type="segment" signature="segment(line)"><code></asyxml>*/
segment segment(line l)
{/*<asyxml></code><documentation>Return the segment whose extremities
   are 'l.A' and 'l.B'.</documentation></function></asyxml>*/
  return segment(l.A, l.B);
}

/*<asyxml><function type="point" signature="midpoint(segment)"><code></asyxml>*/
point midpoint(segment s)
{/*<asyxml></code><documentation>Return the midpoint of 's'.</documentation></function></asyxml>*/
  return 0.5 * (s.A + s.B);
}

/*<asyxml><function type="void" signature="write(line)"><code></asyxml>*/
void write(explicit line l)
{/*<asyxml></code><documentation>Write some informations about 'l'.</documentation></function></asyxml>*/
  write("A = "+(string)((pair)l.A));
  write("Extend A = "+(l.extendA ? "true" : "false"));
  write("B = "+(string)((pair)l.B));
  write("Extend B = "+(l.extendB ? "true" : "false"));
  write("u = "+(string)((pair)l.u));
  write("v = "+(string)((pair)l.v));
  write("a = "+(string) l.a);
  write("b = "+(string) l.b);
  write("c = "+(string) l.c);
  write("slope = "+(string) l.slope);
  write("origin = "+(string) l.origin);
}

/*<asyxml><function type="void" signature="write(explicit segment)"><code></asyxml>*/
void write(explicit segment s)
{/*<asyxml></code><documentation>Write some informations about 's'.</documentation></function></asyxml>*/
  write("A = "+(string)((pair)s.A));
  write("B = "+(string)((pair)s.B));
  write("u = "+(string)((pair)s.u));
  write("v = "+(string)((pair)s.v));
  write("a = "+(string) s.a);
  write("b = "+(string) s.b);
  write("c = "+(string) s.c);
  write("slope = "+(string) s.slope);
  write("origin = "+(string) s.origin);
}

/*<asyxml><operator type = "bool" signature="==(line,line)"><code></asyxml>*/
bool operator ==(line l1, line l2)
{/*<asyxml></code><documentation>Provide the test 'line == line'.</documentation></operator></asyxml>*/
  return (collinear(l1.u, l2.u) &&
          abs(ypart((locate(l1.A) - locate(l1.B))/(locate(l1.A) - locate(l2.B)))) < epsgeo &&
          l1.extendA == l2.extendA && l1.extendB == l2.extendB);
}

/*<asyxml><operator type = "bool" signature="!=(line,line)"><code></asyxml>*/
bool operator !=(line l1, line l2)
{/*<asyxml></code><documentation>Provide the test 'line != line'.</documentation></operator></asyxml>*/
  return !(l1 == l2);
}

/*<asyxml><operator type = "bool" signature="@(point,line)"><code></asyxml>*/
bool operator @(point m, line l)
{/*<asyxml></code><documentation>Provide the test 'point @ line'.
   Return true iff 'm' is on the 'l'.</documentation></operator></asyxml>*/
  point M = changecoordsys(l.A.coordsys, m);
  if (abs(l.a * M.x + l.b * M.y + l.c) >= epsgeo) return false;
  if (l.extendA && l.extendB) return true;
  if (!l.extendA && !l.extendB) return between(l.A, M, l.B);
  if (l.extendA) return sameside(M, l.A, l.B);
  return sameside(M, l.B, l.A);
}

/*<asyxml><function type="coordsys" signature="coordsys(line)"><code></asyxml>*/
coordsys coordsys(line l)
{/*<asyxml></code><documentation>Return the coordinate system in which 'l' is defined.</documentation></function></asyxml>*/
  return l.A.coordsys;
}

/*<asyxml><function type="line" signature="reverse(line)"><code></asyxml>*/
line reverse(line l)
{/*<asyxml></code><documentation>Permute the points 'A' and 'B' of 'l' and so its orientation.</documentation></function></asyxml>*/
  return line(l.B, l.extendB, l.A, l.extendA);
}

/*<asyxml><function type="line" signature="extend(line)"><code></asyxml>*/
line extend(line l)
{/*<asyxml></code><documentation>Return the infinite line passing through 'l.A' and 'l.B'.</documentation></function></asyxml>*/
  line ol = l.copy();
  ol.extendA = true;
  ol.extendB = true;
  return ol;
}

/*<asyxml><function type="line" signature="complementary(explicit line)"><code></asyxml>*/
line complementary(explicit line l)
{/*<asyxml></code><documentation>Return the complementary of a half-line with respect of
   the full line 'l'.</documentation></function></asyxml>*/
  if (l.extendA && l.extendB)
    abort("complementary: the parameter is not a half-line.");
  point origin = l.extendA ? l.B : l.A;
  point ptdir = l.extendA ?
    rotate(180, l.B) * l.A : rotate(180, l.A) * l.B;
  return line(origin, false, ptdir);
}

/*<asyxml><function type="line[]" signature="complementary(explicit segment)"><code></asyxml>*/
line[] complementary(explicit segment s)
{/*<asyxml></code><documentation>Return the two half-lines of origin 's.A' and 's.B' respectively.</documentation></function></asyxml>*/
  line[] ol = new line[2];
  ol[0] = complementary(line(s.A, false, s.B));
  ol[1] = complementary(line(s.A, s.B, false));
  return ol;
}

/*<asyxml><function type="line" signature="Ox(coordsys)"><code></asyxml>*/
line Ox(coordsys R = currentcoordsys)
{/*<asyxml></code><documentation>Return the x-axis of 'R'.</documentation></function></asyxml>*/
  return line(point(R, (0, 0)), point(R, E));
}
/*<asyxml><constant type = "line" signature="Ox"><code></asyxml>*/
restricted line Ox = Ox();/*<asyxml></code><documentation>the x-axis of
                          the default coordinate system.</documentation></constant></asyxml>*/

/*<asyxml><function type="line" signature="Oy(coordsys)"><code></asyxml>*/
line Oy(coordsys R = currentcoordsys)
{/*<asyxml></code><documentation>Return the y-axis of 'R'.</documentation></function></asyxml>*/
  return line(point(R, (0, 0)), point(R, N));
}
/*<asyxml><constant type = "line" signature="Oy"><code></asyxml>*/
restricted line Oy = Oy();/*<asyxml></code><documentation>the y-axis of
                          the default coordinate system.</documentation></constant></asyxml>*/

/*<asyxml><function type="line" signature="line(real,point)"><code></asyxml>*/
line line(real a, point A = point(currentcoordsys, (0, 0)))
{/*<asyxml></code><documentation>Return the line passing through 'A' with an
   angle (in the coordinate system of A) 'a' in degrees.
   line(point, real) is also defined.</documentation></function></asyxml>*/
  return line(A, A + point(A.coordsys, A.coordsys.polar(1, radians(a))));
}
line line(point A = point(currentcoordsys, (0, 0)), real a)
{
  return line(a, A);
}
line line(int a, point A = point(currentcoordsys, (0, 0)))
{
  return line((real)a, A);
}

/*<asyxml><function type="line" signature="line(coordsys,real,real)"><code></asyxml>*/
line line(coordsys R = currentcoordsys, real slope, real origin)
{/*<asyxml></code><documentation>Return the line defined by slope and y-intercept relative to 'R'.</documentation></function></asyxml>*/
  if (slope == infinity || slope == -infinity)
    abort("The slope is infinite. Please, use the routine 'vline'.");
  return line(point(R, (0, origin)), point(R, (1, origin + slope)));
}

/*<asyxml><function type="line" signature="line(coordsys,real,real,real)"><code></asyxml>*/
line line(coordsys R = currentcoordsys, real a, real b, real c)
{/*<asyxml></code><documentation>Retrun the line defined by equation relative to 'R'.</documentation></function></asyxml>*/
  if (a == 0 && b == 0) abort("line: inconsistent equation...");
  pair M;
  M = (a == 0) ? (0, -c/b) : (-c/a, 0);
  return line(point(R, M), point(R, M + (-b, a)));
}

/*<asyxml><function type="line" signature="vline(coordsys)"><code></asyxml>*/
line vline(coordsys R = currentcoordsys)
{/*<asyxml></code><documentation>Return a vertical line in 'R' passing through the origin of 'R'.</documentation></function></asyxml>*/
  point P = point(R, (0, 0));
  point PP = point(R, (R.O + N)/R);
  return line(P, PP);
}
/*<asyxml><constant type = "line" signature="vline"><code></asyxml>*/
restricted line vline = vline();/*<asyxml></code><documentation>The vertical line in the current coordinate system passing
                                through the origin of this system.</documentation></constant></asyxml>*/

/*<asyxml><function type="line" signature="hline(coordsys)"><code></asyxml>*/
line hline(coordsys R = currentcoordsys)
{/*<asyxml></code><documentation>Return a horizontal line in 'R' passing through the origin of 'R'.</documentation></function></asyxml>*/
  point P = point(R, (0, 0));
  point PP = point(R, (R.O + E)/R);
  return line(P, PP);
}
/*<asyxml><constant type = "line" signature="hline"><code></asyxml>*/
line hline = hline();/*<asyxml></code><documentation>The horizontal line in the current coordinate system passing
                     through the origin of this system.</documentation></constant></asyxml>*/

/*<asyxml><function type="line" signature="changecoordsys(coordsys,line)"><code></asyxml>*/
line changecoordsys(coordsys R, line l)
{/*<asyxml></code><documentation>Return the line 'l' in the coordinate system 'R'.</documentation></function></asyxml>*/
  point A = changecoordsys(R, l.A);
  point B = changecoordsys(R, l.B);
  return line(A, B);
}

/*<asyxml><function type="transform" signature="scale(real,line,line,bool)"><code></asyxml>*/
transform scale(real k, line l1, line l2, bool safe = false)
{/*<asyxml></code><documentation>Return the dilatation with respect to
   'l1' in the direction of 'l2'.</documentation></function></asyxml>*/
  return scale(k, l1.A, l1.B, l2.A, l2.B, safe);
}

/*<asyxml><function type="transform" signature="reflect(line)"><code></asyxml>*/
transform reflect(line l)
{/*<asyxml></code><documentation>Return the reflect about the line 'l'.</documentation></function></asyxml>*/
  return reflect((pair)l.A, (pair)l.B);
}

/*<asyxml><function type="transform" signature="reflect(line,line)"><code></asyxml>*/
transform reflect(line l1, line l2, bool safe = false)
{/*<asyxml></code><documentation>Return the reflect about the line
   'l1' in the direction of 'l2'.</documentation></function></asyxml>*/
  return scale(-1.0, l1, l2, safe);
}


/*<asyxml><function type="point[]" signature="intersectionpoints(line,path)"><code></asyxml>*/
point[] intersectionpoints(line l, path g)
{/*<asyxml></code><documentation>Return all points of intersection of the line 'l' with the path 'g'.</documentation></function></asyxml>*/
  // TODO utiliser la version 1.44 de intersections(path g, pair p, pair q)
  // real [] t = intersections(g, l.A, l.B);
  // coordsys R = coordsys(l);
  // return sequence(new point(int n){return point(R, point(g, t[n])/R);}, t.length);
  real [] t;
  pair[] op;
  pair A = l.A;
  pair B = l.B;
  real dy = B.y - A.y,
    dx = A.x - B.x,
    lg = length(g);

  for (int i = 0; i < lg; ++i)
    {
      pair z0 = point(g, i),
        z1 = point(g, i + 1),
        c0 = postcontrol(g, i),
        c1 = precontrol(g, i + 1),
        t3 = z1 - z0 - 3 * c1 + 3 * c0,
        t2 = 3 * z0 + 3 * c1 - 6 * c0,
        t1 = 3 * c0 - 3z0;
      real a = dy * t3.x + dx * t3.y,
        b = dy * t2.x + dx * t2.y,
        c = dy * t1.x + dx * t1.y,
        d = dy * z0.x + dx * z0.y + A.y * B.x - A.x * B.y;

      t = cubicroots(a, b, c, d);
      for (int j = 0; j < t.length; ++j)
        if (
            t[j]>=0
            && (
                t[j]<1
                || (
                    t[j] == 1
                    && (i == lg - 1)
                    && !cyclic(g)
                    )
                )
            ) {
          op.push(point(g, i + t[j]));
        }
    }

  point[] opp;
  for (int i = 0; i < op.length; ++i)
    opp.push(point(coordsys(l), op[i]/coordsys(l)));
  return opp;
}

/*<asyxml><function type="point" signature="intersectionpoint(line,line)"><code></asyxml>*/
point intersectionpoint(line l1, line l2)
{/*<asyxml></code><documentation>Return the point of intersection of line 'l1' with 'l2'.
   If 'l1' and 'l2' have an infinity or none point of intersection,
   this routine return (infinity, infinity).</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(l1.A, l1.B, l2.A, l2.B);
  coordsys R = P[0].coordsys;
  pair p = extension(P[0], P[1], P[2], P[3]);
  if(finite(p)){
    point p = point(R, p/R);
    if (p @ l1 && p @ l2) return p;
  }
  return point(R, (infinity, infinity));
}

/*<asyxml><function type="line" signature="parallel(point,line)"><code></asyxml>*/
line parallel(point M, line l)
{/*<asyxml></code><documentation>Return the line parallel to 'l' passing through 'M'.</documentation></function></asyxml>*/
  point A, B;
  if (M.coordsys != coordsys(l))
    {
      A = changecoordsys(M.coordsys, l.A);
      B = changecoordsys(M.coordsys, l.B);
    } else {A = l.A;B = l.B;}
  return line(M, M - A + B);
}

/*<asyxml><function type="line" signature="parallel(point,explicit vector)"><code></asyxml>*/
line parallel(point M, explicit vector dir)
{/*<asyxml></code><documentation>Return the line of direction 'dir' and passing through 'M'.</documentation></function></asyxml>*/
  return line(M, M + locate(dir));
}

/*<asyxml><function type="line" signature="parallel(point,explicit pair)"><code></asyxml>*/
line parallel(point M, explicit pair dir)
{/*<asyxml></code><documentation>Return the line of direction 'dir' and passing through 'M'.</documentation></function></asyxml>*/
  return line(M, M + vector(currentcoordsys, dir));
}

/*<asyxml><function type="bool" signature="parallel(line,line)"><code></asyxml>*/
bool parallel(line l1, line l2, bool strictly = false)
{/*<asyxml></code><documentation>Return 'true' if 'l1' and 'l2' are (strictly ?) parallel.</documentation></function></asyxml>*/
  bool coll = collinear(l1.u, l2.u);
  return strictly ? coll && (l1 != l2) : coll;
}

/*<asyxml><function type="bool" signature="concurrent(...line[])"><code></asyxml>*/
bool concurrent(... line[] l)
{/*<asyxml></code><documentation>Returns true if all the lines 'l' are concurrent.</documentation></function></asyxml>*/
  if (l.length < 3) abort("'concurrent' needs at least for three lines ...");
  pair point = intersectionpoint(l[0], l[1]);
  bool conc;
  for (int i = 2; i < l.length; ++i) {
    pair pt = intersectionpoint(l[i - 1], l[i]);
    conc = simeq(pt, point);
    if (!conc) break;
  }
  return conc;
}

/*<asyxml><function type="transform" signature="projection(line)"><code></asyxml>*/
transform projection(line l)
{/*<asyxml></code><documentation>Return the orthogonal projection on 'l'.</documentation></function></asyxml>*/
  return projection(l.A, l.B);
}

/*<asyxml><function type="transform" signature="projection(line,line,bool)"><code></asyxml>*/
transform projection(line l1, line l2, bool safe = false)
{/*<asyxml></code><documentation>Return the projection on (AB) in parallel of (CD).
   If 'safe = true' and (l1)//(l2) return the identity.
   If 'safe = false' and (l1)//(l2) return a infinity scaling.</documentation></function></asyxml>*/
  return projection(l1.A, l1.B, l2.A, l2.B, safe);
}

/*<asyxml><function type="transform" signature="vprojection(line,bool)"><code></asyxml>*/
transform vprojection(line l, bool safe = false)
{/*<asyxml></code><documentation>Return the projection on 'l' in parallel of N--S.
   If 'safe' is 'true' the projected point keeps the same place if 'l'
   is vertical.</documentation></function></asyxml>*/
  coordsys R = defaultcoordsys;
  return projection(l, line(point(R, N), point(R, S)), safe);
}

/*<asyxml><function type="transform" signature="hprojection(line,bool)"><code></asyxml>*/
transform hprojection(line l, bool safe = false)
{/*<asyxml></code><documentation>Return the projection on 'l' in parallel of E--W.
   If 'safe' is 'true' the projected point keeps the same place if 'l'
   is horizontal.</documentation></function></asyxml>*/
  coordsys R = defaultcoordsys;
  return projection(l, line(point(R, E), point(R, W)), safe);
}

/*<asyxml><function type="line" signature="perpendicular(point,line)"><code></asyxml>*/
line perpendicular(point M, line l)
{/*<asyxml></code><documentation>Return the perpendicular line of 'l' passing through 'M'.</documentation></function></asyxml>*/
  point Mp = projection(l) * M;
  point A = Mp == l.A ? l.B : l.A;
  return line(Mp, rotate(90, Mp) * A);
}

/*<asyxml><function type="line" signature="perpendicular(point,explicit vector)"><code></asyxml>*/
line perpendicular(point M, explicit vector normal)
{/*<asyxml></code><documentation>Return the line passing through 'M'
   whose normal is \param{normal}.</documentation></function></asyxml>*/
  return perpendicular(M, line(M, M + locate(normal)));
}

/*<asyxml><function type="line" signature="perpendicular(point,explicit pair)"><code></asyxml>*/
line perpendicular(point M, explicit pair normal)
{/*<asyxml></code><documentation>Return the line passing through 'M'
   whose normal is \param{normal} (given in the currentcoordsys).</documentation></function></asyxml>*/
  return perpendicular(M, line(M, M + vector(currentcoordsys, normal)));
}

/*<asyxml><function type="bool" signature="perpendicular(line,line)"><code></asyxml>*/
bool perpendicular(line l1, line l2)
{/*<asyxml></code><documentation>Return 'true' if 'l1' and 'l2' are perpendicular.</documentation></function></asyxml>*/
  return abs(dot(locate(l1.u), locate(l2.u))) < epsgeo ;
}

/*<asyxml><function type="real" signature="angle(line,coordsys)"><code></asyxml>*/
real angle(line l, coordsys R = coordsys(l))
{/*<asyxml></code><documentation>Return the angle of the oriented line 'l',
   in radian, in the interval ]-pi, pi] and relatively to 'R'.</documentation></function></asyxml>*/
  return angle(l.u, R, false);
}

/*<asyxml><function type="real" signature="degrees(line,coordsys,bool)"><code></asyxml>*/
real degrees(line l, coordsys R = coordsys(l))
{/*<asyxml></code><documentation>Returns the angle of the oriented line 'l' in degrees,
   in the interval [0, 360[ and relatively to 'R'.</documentation></function></asyxml>*/
  return degrees(angle(l, R));
}

/*<asyxml><function type="real" signature="sharpangle(line,line)"><code></asyxml>*/
real sharpangle(line l1, line l2)
{/*<asyxml></code><documentation>Return the measure in radians of the sharp angle formed by 'l1' and 'l2'.</documentation></function></asyxml>*/
  vector u1 = l1.u;
  vector u2 = (dot(l1.u, l2.u) < 0) ? -l2.u : l2.u;
  real a12 = angle(locate(u2)) - angle(locate(u1));
  a12 = a12%(sgnd(a12) * pi);
  if (a12 <= -pi/2) {
    a12 += pi;
  } else if (a12 > pi/2) {
    a12 -= pi;
  }
  return a12;
}

/*<asyxml><function type="real" signature="angle(line,line)"><code></asyxml>*/
real angle(line l1, line l2)
{/*<asyxml></code><documentation>Return the measure in radians of oriented angle (l1.u, l2.u).</documentation></function></asyxml>*/
  return angle(locate(l2.u)) - angle(locate(l1.u));
}

/*<asyxml><function type="real" signature="degrees(line,line)"><code></asyxml>*/
real degrees(line l1, line l2)
{/*<asyxml></code><documentation>Return the measure in degrees of the
   angle formed by the oriented lines 'l1' and 'l2'.</documentation></function></asyxml>*/
  return degrees(angle(l1, l2));
}

/*<asyxml><function type="real" signature="sharpdegrees(line,line)"><code></asyxml>*/
real sharpdegrees(line l1, line l2)
{/*<asyxml></code><documentation>Return the measure in degrees of the sharp angle formed by 'l1' and 'l2'.</documentation></function></asyxml>*/
  return degrees(sharpangle(l1, l2));
}

/*<asyxml><function type="line" signature="bisector(line,line,real,bool)"><code></asyxml>*/
line bisector(line l1, line l2, real angle = 0, bool sharp = true)
{/*<asyxml></code><documentation>Return the bisector of the angle formed by 'l1' and 'l2'
   rotated by the angle 'angle' (in degrees) around intersection point of 'l1' with 'l2'.
   If 'sharp' is true (the default), this routine returns the bisector of the sharp angle.
   Note that the returned line inherit of coordinate system of 'l1'.</documentation></function></asyxml>*/
  line ol;
  if (l1 == l2) return l1;
  point A = intersectionpoint(l1, l2);
  if (finite(A)) {
    if(sharp) ol = rotate(sharpdegrees(l1, l2)/2 + angle, A) * l1;
    else {
      coordsys R = coordsys(l1);
      pair a = A, b = A + l1.u, c = A + l2.u;
      pair pp = extension(a, a + dir(a--b, a--c), b, b + dir(b--a, b--c));
      return rotate(angle, A) * line(A, point(R, pp/R));
    }
  } else {
    ol = l1;
  }
  return ol;
}

/*<asyxml><function type="line" signature="sector(int,int,line,line,real,bool)"><code></asyxml>*/
line sector(int n = 2, int p = 1, line l1, line l2, real angle = 0, bool sharp = true)
{/*<asyxml></code><documentation>Return the p-th nth-sector of the angle
   formed by the oriented line 'l1' and 'l2'
   rotated by the angle 'angle' (in degrees) around the intersection point of 'l1' with 'l2'.
   If 'sharp' is true (the default), this routine returns the bisector of the sharp angle.
   Note that the returned line inherit of coordinate system of 'l1'.</documentation></function></asyxml>*/
  line ol;
  if (l1 == l2) return l1;
  point A = intersectionpoint(l1, l2);
  if (finite(A)) {
    if(sharp) ol = rotate(p * sharpdegrees(l1, l2)/n + angle, A) * l1;
    else {
      ol = rotate(p * degrees(l1, l2)/n + angle, A) * l1;
    }
  } else {
    ol = l1;
  }
  return ol;
}

/*<asyxml><function type="line" signature="bisector(point,point,point,point,real)"><code></asyxml>*/
line bisector(point A, point B, point C, point D, real angle = 0, bool sharp = true)
{/*<asyxml></code><documentation>Return the bisector of the angle formed by the lines (AB) and (CD).
   <look href = "#bisector(line, line, real, bool)"/>.</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B, C, D);
  return bisector(line(P[0], P[1]), line(P[2], P[3]), angle, sharp);
}

/*<asyxml><function type="line" signature="bisector(segment,real)"><code></asyxml>*/
line bisector(segment s, real angle = 0)
{/*<asyxml></code><documentation>Return the bisector of the segment line 's' rotated by 'angle' (in degrees) around the
   midpoint of 's'.</documentation></function></asyxml>*/
  coordsys R = coordsys(s);
  point m = midpoint(s);
  vector dir = rotateO(90) * unit(s.A - m);
  return rotate(angle, m) * line(m + dir, m - dir);
}

/*<asyxml><function type="line" signature="bisector(point,point,real)"><code></asyxml>*/
line bisector(point A, point B, real angle = 0)
{/*<asyxml></code><documentation>Return the bisector of the segment line [AB] rotated by 'angle' (in degrees) around the
   midpoint of [AB].</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B);
  return bisector(segment(P[0], P[1]), angle);
}

/*<asyxml><function type="real" signature="distance(point,line)"><code></asyxml>*/
real distance(point M, line l)
{/*<asyxml></code><documentation>Return the distance from 'M' to 'l'.
   distance(line, point) is also defined.</documentation></function></asyxml>*/
  point A = changecoordsys(defaultcoordsys, l.A);
  point B = changecoordsys(defaultcoordsys, l.B);
  line ll = line(A, B);
  pair m = locate(M);
  return abs(ll.a * m.x + ll.b * m.y + ll.c)/sqrt(ll.a^2 + ll.b^2);
}

real distance(line l, point M)
{
  return distance(M, l);
}

/*<asyxml><function type="void" signature="draw(picture,Label,line,bool,bool,align,pen,arrowbar,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "",
          line l, bool dirA = l.extendA, bool dirB = l.extendB,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None,
          Label legend = "", marker marker = nomarker,
          pathModifier pathModifier = NoModifier)
{/*<asyxml></code><documentation>Draw the line 'l' without altering the size of picture pic.
   The boolean parameters control the infinite section.
   The global variable 'linemargin' (default value is 0) allows to modify
   the bounding box in which the line must be drawn.</documentation></function></asyxml>*/
  if(!(dirA || dirB)) draw(l.A--l.B, invisible);// l is a segment.
  Drawline(pic, L, l.A, dirP = dirA, l.B, dirQ = dirB,
           align, p, arrow,
           legend, marker, pathModifier);
}

/*<asyxml><function type="void" signature="draw(picture,Label[], line[], align,pen[], arrowbar,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label[] L = new Label[], line[] l,
          align align = NoAlign, pen[] p = new pen[],
          arrowbar arrow = None,
          Label[] legend = new Label[], marker marker = nomarker,
          pathModifier pathModifier = NoModifier)
{/*<asyxml></code><documentation>Draw each lines with the corresponding pen.</documentation></function></asyxml>*/
  for (int i = 0; i < l.length; ++i) {
    draw(pic, L.length>0 ? L[i] : "", l[i],
         align, p = p.length>0 ? p[i] : currentpen,
         arrow, legend.length>0 ? legend[i] : "", marker,
         pathModifier);
  }
}

/*<asyxml><function type="void" signature="draw(picture,Label[], line[], align,pen,arrowbar,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label[] L = new Label[], line[] l,
          align align = NoAlign, pen p,
          arrowbar arrow = None,
          Label[] legend = new Label[], marker marker = nomarker,
          pathModifier pathModifier = NoModifier)
{/*<asyxml></code><documentation>Draw each lines with the same pen 'p'.</documentation></function></asyxml>*/
  pen[] tp = sequence(new pen(int i){return p;}, l.length);
  draw(pic, L, l, align, tp, arrow, legend, marker, pathModifier);
}

/*<asyxml><function type="void" signature="show(picture,line,pen)"><code></asyxml>*/
void show(picture pic = currentpicture, line l, pen p = red)
{/*<asyxml></code><documentation>Draw some informations of 'l'.</documentation></function></asyxml>*/
  dot("$A$", (pair)l.A, align = -locate(l.v), p);
  dot("$B$", (pair)l.B, align = -locate(l.v), p);
  draw(l, dotted);
  draw("$\vec{u}$", locate(l.A)--locate(l.A + l.u), p, Arrow);
  draw("$\vec{v}$", locate(l.A)--locate(l.A + l.v), p, Arrow);
}

/*<asyxml><function type="point[]" signature="sameside(point,line,line)"><code></asyxml>*/
point[] sameside(point M, line l1, line l2)
{/*<asyxml></code><documentation>Return two points on 'l1' and 'l2' respectively.
   The first point is from the same side of M relatively to 'l2',
   the second point is from the same side of M relatively to 'l1'.</documentation></function></asyxml>*/
  point[] op;
  coordsys R1 = coordsys(l1);
  coordsys R2 = coordsys(l2);
  if (parallel(l1, l2)) {
    op.push(projection(l1) * M);
    op.push(projection(l2) * M);
  } else {
    point O = intersectionpoint(l1, l2);
    if (M @ l2) op.push((sameside(M, O + l1.u, l2)) ? O + l1.u : rotate(180, O) * (O + l1.u));
    else op.push(projection(l1, l2) * M);
    if (M @ l1) op.push((sameside(M, O + l2.u, l1)) ? O + l2.u : rotate(180, O) * (O + l2.u));
    else {op.push(projection(l2, l1) * M);}
  }
  return op;
}

/*<asyxml><function type="void" signature="markangle(picture,Label,int,real,real,explicit line,explicit line,explicit pair,arrowbar,pen,filltype,margin,marker)"><code></asyxml>*/
void markangle(picture pic = currentpicture,
               Label L = "", int n = 1, real radius = 0, real space = 0,
               explicit line l1, explicit line l2, explicit pair align = dir(1),
               arrowbar arrow = None, pen p = currentpen,
               filltype filltype = NoFill,
               margin margin = NoMargin, marker marker = nomarker)
{/*<asyxml></code><documentation>Mark the angle (l1, l2) aligned in the direction 'align' relative to 'l1'.
   Commune values for 'align' are dir(real).</documentation></function></asyxml>*/
  if (parallel(l1, l2, true)) return;
  real al = degrees(l1, defaultcoordsys);
  pair O, A, B;
  if (radius == 0) radius = markangleradius(p);
  real d = degrees(locate(l1.u));
  align = rotate(d) * align;
  if (l1 == l2) {
    O = midpoint(segment(l1.A, l1.B));
    A = l1.A;B = l1.B;
    if (sameside(rotate(sgn(angle(B-A)) * 45, O) * A, O + align, l1)) {radius = -radius;}
  } else {
    O = intersectionpoint(extend(l1), extend(l2));
    pair R = O + align;
    point [] ss = sameside(point(coordsys(l1), R/coordsys(l1)), l1, l2);
    A = ss[0];
    B = ss[1];
  }
  markangle(pic = pic, L = L, n = n, radius = radius, space = space,
            O = O, A = A, B = B,
            arrow = arrow, p = p, filltype = filltype,
            margin = margin, marker = marker);
}

/*<asyxml><function type="void" signature="markangle(picture,Label,int,real,real,explicit line,explicit line,explicit vector,arrowbar,pen,filltype,margin,marker)"><code></asyxml>*/
void markangle(picture pic = currentpicture,
               Label L = "", int n = 1, real radius = 0, real space = 0,
               explicit line l1, explicit line l2, explicit vector align,
               arrowbar arrow = None, pen p = currentpen,
               filltype filltype = NoFill,
               margin margin = NoMargin, marker marker = nomarker)
{/*<asyxml></code><documentation>Mark the angle (l1, l2) in the direction 'dir' given relatively to 'l1'.</documentation></function></asyxml>*/
  markangle(pic, L, n, radius, space, l1, l2, (pair)align, arrow,
            p, filltype, margin, marker);
}

/*<asyxml><function type="void" signature="markangle(picture,Label,int,real,real,line,line,arrowbar,pen,filltype,margin,marker)"><code></asyxml>*/
// void markangle(picture pic = currentpicture,
//                Label L = "", int n = 1, real radius = 0, real space = 0,
//                explicit line l1, explicit line l2,
//                arrowbar arrow = None, pen p = currentpen,
//                filltype filltype = NoFill,
//                margin margin = NoMargin, marker marker = nomarker)
// {/*<asyxml></code><documentation>Mark the oriented angle (l1, l2).</documentation></function></asyxml>*/
//   if (parallel(l1, l2, true)) return;
//   real al = degrees(l1, defaultcoordsys);
//   pair O, A, B;
//   if (radius == 0) radius = markangleradius(p);
//   real d = degrees(locate(l1.u));
//   if (l1 == l2) {
//     O = midpoint(segment(l1.A, l1.B));
//   } else {
//     O = intersectionpoint(extend(l1), extend(l2));
//   }
//   A = O + locate(l1.u);
//   B = O + locate(l2.u);
//   markangle(pic = pic, L = L, n = n, radius = radius, space = space,
//             O = O, A = A, B = B,
//             arrow = arrow, p = p, filltype = filltype,
//             margin = margin, marker = marker);
// }

/*<asyxml><function type="void" signature="perpendicularmark(picture,line,line,real,pen,int,margin,filltype)"><code></asyxml>*/
void perpendicularmark(picture pic = currentpicture, line l1, line l2,
                       real size = 0, pen p = currentpen, int quarter = 1,
                       margin margin = NoMargin, filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw a right angle at the intersection point of lines and
   aligned in the 'quarter' nth quarter of circle formed by 'l1.u' and
   'l2.u'.</documentation></function></asyxml>*/
  point P = intersectionpoint(l1, l2);
  pair align = rotate(90 * (quarter - 1)) * dir(45);
  perpendicularmark(P, align, locate(l1.u), size, p, margin, filltype);
}
// *.........................LINES.........................*
// *=======================================================*

// *=======================================================*
// *........................CONICS.........................*
/*<asyxml><struct signature="bqe"><code></asyxml>*/
struct bqe
{/*<asyxml></code><documentation>Bivariate Quadratic Equation.</documentation></asyxml>*/
  /*<asyxml><property type = "real[]" signature="a"><code></asyxml>*/
  real[] a;/*<asyxml></code><documentation>a[0] * x^2 + a[1] * x * y + a[2] * y^2 + a[3] * x + a[4] * y + a[5] = 0</documentation></property><property type = "coordsys" signature="coordsys"><code></asyxml>*/
  coordsys coordsys;/*<asyxml></code></property></asyxml>*/
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="bqe" signature="bqe(coordsys,real,real,real,real,real,real)"><code></asyxml>*/
bqe bqe(coordsys R = currentcoordsys,
        real a, real b, real c, real d, real e, real f)
{/*<asyxml></code><documentation>Return the bivariate quadratic equation
   a[0] * x^2 + a[1] * x * y + a[2] * y^2 + a[3] * x + a[4] * y + a[5] = 0
   relatively to the coordinate system R.</documentation></function></asyxml>*/
  bqe obqe;
  obqe.coordsys = R;
  obqe.a = new real[] {a, b, c, d, e, f};
  return obqe;
}

/*<asyxml><function type="bqe" signature="changecoordsys(coordsys,bqe)"><code></asyxml>*/
bqe changecoordsys(coordsys R, bqe bqe)
{/*<asyxml></code><documentation>Returns the bivariate quadratic equation relatively to 'R'.</documentation></function></asyxml>*/
  pair i = coordinates(changecoordsys(R, vector(defaultcoordsys,
                                             bqe.coordsys.i)));
  pair j = coordinates(changecoordsys(R, vector(defaultcoordsys,
                                             bqe.coordsys.j)));
  pair O = coordinates(changecoordsys(R, point(defaultcoordsys,
                                            bqe.coordsys.O)));
  real a = bqe.a[0], b = bqe.a[1], c = bqe.a[2], d = bqe.a[3], f = bqe.a[4], g = bqe.a[5];
  real ux = i.x, uy = i.y;
  real vx = j.x, vy = j.y;
  real ox = O.x, oy = O.y;
  real D = ux * vy - uy * vx;
  real ap = (a * vy^2 - b * uy * vy + c * uy^2)/D^2;
  real bpp = (-2 * a * vx * vy + b * ux * vy + b * uy * vx - 2 * c * ux * uy)/D^2;
  real cp = (a * vx^2 - b * ux * vx + c * ux^2)/D^2;
  real dp = (-2a * ox * vy^2 + 2a * oy * vx * vy + 2b * ox * uy * vy-
           b * oy * ux * vy - b * oy * uy * vx - 2c * ox * uy^2 + 2c * oy * uy * ux)/D^2+
    (d * vy - f * uy)/D;
  real fp = (2a * ox * vx * vy - b * ox * ux * vy - 2a * oy * vx^2-
           b * ox * uy * vx + 2 * b * oy * ux * vx + 2c * ox * ux * uy - 2c * oy * ux^2)/D^2+
    (f * ux - d * vx)/D;
  g = (a * ox^2 * vy^2 - 2a * ox * oy * vx * vy - b * ox^2 * uy * vy + b * ox * oy * ux * vy+
     a * oy^2 * vx^2 + b * ox * oy * uy * vx - b * oy^2 * ux * vx + c * ox^2 * uy^2-
     2 * c * ox * oy * ux * uy + c * oy^2 * ux^2)/D^2+
    (d * oy * vx + f * ox * uy - d * ox * vy - f * oy * ux)/D + g;
  bqe obqe;
  obqe.a = approximate(new real[] {ap, bpp, cp, dp, fp, g});
  obqe.coordsys = R;
  return obqe;
}

/*<asyxml><function type="bqe" signature="bqe(point,point,point,point,point)"><code></asyxml>*/
bqe bqe(point M1, point M2, point M3, point M4, point M5)
{/*<asyxml></code><documentation>Return the bqe of conic passing through the five points (if possible).</documentation></function></asyxml>*/
  coordsys R;
  pair[] pts;
  if (samecoordsys(M1, M2, M3, M4, M5)) {
    R = M1.coordsys;
    pts= new pair[] {M1.coordinates, M2.coordinates, M3.coordinates, M4.coordinates, M5.coordinates};
  } else {
    R = defaultcoordsys;
    pts= new pair[] {M1, M2, M3, M4, M5};
  }
  real[][] M;
  real[] x;
  bqe bqe;
  bqe.coordsys = R;
  for (int i = 0; i < 5; ++i) {// Try a = -1
    M[i] = new real[] {pts[i].x * pts[i].y, pts[i].y^2, pts[i].x, pts[i].y, 1};
    x[i] = pts[i].x^2;
  }
  if(abs(determinant(M)) < 1e-5) {// Try c = -1
    for (int i = 0; i < 5; ++i) {
      M[i] = new real[] {pts[i].x^2, pts[i].x * pts[i].y, pts[i].x, pts[i].y, 1};
      x[i] = pts[i].y^2;
    }
    real[] coef = solve(M, x);
    bqe.a = new real[] {coef[0], coef[1], -1, coef[2], coef[3], coef[4]};
  } else {
    real[] coef = solve(M, x);
    bqe.a = new real[] {-1, coef[0], coef[1], coef[2], coef[3], coef[4]};
  }
  bqe.a = approximate(bqe.a);
  return bqe;
}

/*<asyxml><function type="bool" signature="samecoordsys(bool...bqe[])"><code></asyxml>*/
bool samecoordsys(bool warn = true ... bqe[] bqes)
{/*<asyxml></code><documentation>Return true if all the bivariate quadratic equations have the same coordinate system.</documentation></function></asyxml>*/
  bool ret = true;
  coordsys t = bqes[0].coordsys;
  for (int i = 1; i < bqes.length; ++i) {
    ret = (t == bqes[i].coordsys);
    if(!ret) break;
    t = bqes[i].coordsys;
  }
  if(warn && !ret)
    warning("coodinatesystem",
            "the coordinate system of two bivariate quadratic equations are not
the same. The operation will be done relatively to the default coordinate
system.");
  return ret;
}

/*<asyxml><function type="real[]" signature="realquarticroots(real,real,real,real,real)"><code></asyxml>*/
real[] realquarticroots(real a, real b, real c, real d, real e)
{/*<asyxml></code><documentation>Return the real roots of the quartic equation ax^4 + b^x3 + cx^2 + dx = 0.</documentation></function></asyxml>*/
  static real Fuzz = sqrt(realEpsilon);
  pair[] zroots = quarticroots(a, b, c, d, e);
  real[] roots;
  real p(real x){return a * x^4 + b * x^3 + c * x^2 + d * x + e;}
  real prime(real x){return 4 * a * x^3 + 3 * b * x^2 + 2 * c * x + d;}
  real x;
  bool search = true;
  int n;
  void addroot(real x)
  {
    bool exist = false;
    for (int i = 0; i < roots.length; ++i) {
      if(abs(roots[i]-x) < 1e-5) {exist = true; break;}
    }
    if(!exist) roots.push(x);
  }
  for(int i = 0; i < zroots.length; ++i) {
    if(zroots[i].y == 0 || abs(p(zroots[i].x)) < Fuzz) addroot(zroots[i].x);
    else {
      if(abs(zroots[i].y) < 1e-3) {
        x = zroots[i].x;
        search = true;
        n = 200;
        while(search) {
          real tx = abs(p(x)) < Fuzz ? x : newton(iterations = n, p, prime, x);
          if(tx < realMax) {
            if(abs(p(tx)) < Fuzz) {
              addroot(tx);
              search = false;
            } else if(n < 200) n *=2;
            else {
              search = false;
            }
          } else search = false; //It's not a real root.
        }
      }
    }
  }
  return roots;
}

/*<asyxml><struct signature="conic"><code></asyxml>*/
struct conic
{/*<asyxml></code><documentation></documentation><property type = "real" signature="e,p,h"><code></asyxml>*/
  real e, p, h;/*<asyxml></code><documentation>BE CAREFUL: h = distance(F, D) and p = h * e (http://en.wikipedia.org/wiki/Ellipse)
                 While http://mathworld.wolfram.com/ takes p = distance(F,D).</documentation></property><property type = "point" signature="F"><code></asyxml>*/
  point F;/*<asyxml></code><documentation>Focus.</documentation></property><property type = "line" signature="D"><code></asyxml>*/
  line D;/*<asyxml></code><documentation>Directrix.</documentation></property><property type = "line" signature="l"><code></asyxml>*/
  line[] l;/*<asyxml></code><documentation>Case of degenerated conic (not yet implemented !).</documentation></property></asyxml>*/
}/*<asyxml></struct></asyxml>*/

bool degenerate(conic c)
{
  return !finite(c.p) || !finite(c.h);
}

/*ANCconic conic(point, line, real)ANC*/
conic conic(point F, line l, real e)
{/*DOC
   The conic section define by the eccentricity 'e', the focus 'F'
   and the directrix 'l'.
   Note that an eccentricity equal to 0 defines a circle centered at F,
   with a radius equal at the distance from 'F' to 'l'.
   If the coordinate system of 'F' and 'l' are not identical, the conic is
   attached to 'defaultcoordsys'.
   DOC*/
  if(e < 0) abort("conic: 'e' can't be negative.");
  conic oc;
  point[] P = standardizecoordsys(F, l.A, l.B);
  line ll;
  ll = line(P[1], P[2]);
  oc.e = e < epsgeo ? 0 : e; // Handle case of circle.
  oc.F = P[0];
  oc.D = ll;
  oc.h = distance(P[0], ll);
  oc.p = abs(e) < epsgeo ? oc.h : e * oc.h;
  return oc;
}

/*<asyxml><struct signature="circle"><code></asyxml>*/
struct circle
{/*<asyxml></code><documentation>All the calculus with this structure will be as exact as Asymptote can do.
   For a full precision, you must not cast 'circle' to 'path' excepted for drawing routines.</documentation></asyxml>*/
  /*<asyxml><property type = "point" signature="C"><code></asyxml>*/
  point C;/*<asyxml></code><documentation>Center</documentation></property><property><code></asyxml>*/
  real r;/*<asyxml></code><documentation>Radius</documentation></property><property><code></asyxml>*/
  line l;/*<asyxml></code><documentation>If the radius is infinite, this line is used instead of circle.</documentation></property></asyxml>*/
}/*<asyxml></struct></asyxml>*/

bool degenerate(circle c)
{
  return !finite(c.r);
}

line line(circle c){
  if(finite(c.r)) abort("Circle can not be casted to line here.");
  return c.l;
}

/*<asyxml><struct signature="ellipse"><code></asyxml>*/
struct ellipse
{/*<asyxml></code><documentation>Look at <html><a href = "http://mathworld.wolfram.com/Ellipse.html">http://mathworld.wolfram.com/Ellipse.html</a></html></documentation></asyxml>*/
  /*<asyxml><property type = "point" signature="F1,F2,C"><code></asyxml>*/
  restricted point F1,F2,C;/*<asyxml></code><documentation>Foci and center.</documentation></property><property type = "real" signature="a,b,c,e,p"><code></asyxml>*/
  restricted real a,b,c,e,p;/*<asyxml></code></property><property type = "real" signature="angle"><code></asyxml>*/
  restricted real angle;/*<asyxml></code><documentation>Value is degrees(F2 - F1).</documentation></property><property type = "line" signature="D1,D2"><code></asyxml>*/
  restricted line D1,D2;/*<asyxml></code><documentation>Directrices.</documentation></property><property type = "line" signature="l"><code></asyxml>*/
  line l;/*<asyxml></code><documentation>If one axis is infinite, this line is used instead of ellipse.</documentation></property></asyxml>*/

  /*<asyxml><method type = "void" signature="init(point,point,real)"><code></asyxml>*/
  void init(point f1, point f2, real a)
  {/*<asyxml></code><documentation>Ellipse given by foci and semimajor axis.</documentation></method></asyxml>*/
    point[] P = standardizecoordsys(f1, f2);
    this.F1 = P[0];
    this.F2 = P[1];
    this.C = (P[0] + P[1])/2;
    this.angle = degrees(F2 - F1, warn=false);
    this.a = a;
    if(!finite(a)) {
      this.l = line(P[0], P[1]);
      this.b = infinity;
      this.e = 0;
      this.c = 0;
    } else {
      this.c = abs(C - P[0]);
      this.b = this.c < epsgeo ? a : sqrt(a^2 - c^2); // Handle case of circle.
      this.e = this.c < epsgeo ? 0 : this.c/a; // Handle case of circle.
      if(this.e >= 1) abort("ellipse.init: wrong parameter: e >= 1.");
      this.p = a * (1 - this.e^2);
      if (this.c != 0) {// directrix is not set for a circle.
        point A = this.C + (a^2/this.c) * unit(P[0]-this.C);
        this.D1 = line(A, A + rotateO(90) * unit(A - this.C));
        this.D2 = reverse(rotate(180, C) * D1);
      }
    }
  }
}/*<asyxml></struct></asyxml>*/

bool degenerate(ellipse el)
{
  return !finite(el.a) || !finite(el.b);
}

/*<asyxml><struct signature="parabola"><code></asyxml>*/
struct parabola
{/*<asyxml></code><documentation>Look at <html><a href = "http://mathworld.wolfram.com/Parabola.html">http://mathworld.wolfram.com/Parabola.html</a></html></documentation><property type = "point" signature="F,V"><code></asyxml>*/
  restricted point F,V;/*<asyxml></code><documentation>Focus and vertex</documentation></property><property type = "real" signature="a,p,e = 1"><code></asyxml>*/
  restricted real a,p,e = 1;/*<asyxml></code></property><property type = "real" signature="angle"><code></asyxml>*/
  restricted real angle;/*<asyxml></code><documentation>Value is degrees(F - V).</documentation></property><property type = "line" signature="D"><code></asyxml>*/
  restricted line D;/*<asyxml></code><documentation>Directrix</documentation></property><property type = "pair" signature="bmin,bmax"><code></asyxml>*/
  pair bmin, bmax;/*<asyxml></code><documentation>The (left, bottom) and (right, top) coordinates of region bounding box for drawing the parabola.
                    If unset the current picture bounding box is used instead.</documentation></property></asyxml>*/

  /*<asyxml><method type = "void" signature="init(point,line)"><code></asyxml>*/
  void init(point F, line directrix)
  {/*<asyxml></code><documentation>Parabola given by focus and directrix.</documentation></method></asyxml>*/
    point[] P = standardizecoordsys(F, directrix.A, directrix.B);
    this.F = P[0];
    line l = line(P[1], P[2]);
    this.D = l;
    this.a = distance(P[0], l)/2;
    this.p = 2 * a;
    this.V = 0.5 * (F + projection(D) * P[0]);
    this.angle = degrees(F - V, warn=false);
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><struct signature="hyperbola"><code></asyxml>*/
struct hyperbola
{/*<asyxml></code><documentation><html>Look at <a href = "http://mathworld.wolfram.com/Hyperbola.html">http://mathworld.wolfram.com/Hyperbola.html</a></html></documentation><property type = "point" signature="F1,F2"><code></asyxml>*/
  restricted point F1,F2;/*<asyxml></code><documentation>Foci.</documentation></property><property type = "point" signature="C,V1,V2"><code></asyxml>*/
  restricted point C,V1,V2;/*<asyxml></code><documentation>Center and vertices.</documentation></property><property type = "real" signature="a,b,c,e,p"><code></asyxml>*/
  restricted real a,b,c,e,p;/*<asyxml></code><documentation></documentation></property><property type = "real" signature="angle"><code></asyxml>*/
  restricted real angle;/*<asyxml></code><documentation>Value is degrees(F2 - F1).</documentation></property><property type = "line" signature="D1,D2,A1,A2"><code></asyxml>*/
  restricted line D1,D2,A1,A2;/*<asyxml></code><documentation>Directrices and asymptotes.</documentation></property><property type = "pair" signature="bmin,bmax"><code></asyxml>*/
  pair bmin, bmax; /*<asyxml></code><documentation>The (left, bottom) and (right, top) coordinates of region bounding box for drawing the hyperbola.
                     If unset the current picture bounding box is used instead.</documentation></property></asyxml>*/

  /*<asyxml><method type = "void" signature="init(point,point,real)"><code></asyxml>*/
  void init(point f1, point f2, real a)
  {/*<asyxml></code><documentation>Hyperbola given by foci and semimajor axis.</documentation></method></asyxml>*/
    point[] P = standardizecoordsys(f1, f2);
    this.F1 = P[0];
    this.F2 = P[1];
    this.C = (P[0] + P[1])/2;
    this.angle = degrees(F2 - F1, warn=false);
    this.a = a;
    this.c = abs(C - P[0]);
    this.e = this.c/a;
    if(this.e <= 1) abort("hyperbola.init: wrong parameter: e <= 1.");
    this.b = a * sqrt(this.e^2 - 1);
    this.p = a * (this.e^2 - 1);
    point A = this.C + (a^2/this.c) * unit(P[0]-this.C);
    this.D1 = line(A, A + rotateO(90) * unit(A - this.C));
    this.D2 = reverse(rotate(180, C) * D1);
    this.V1 = C + a * unit(F1 - C);
    this.V2 = C + a * unit(F2 - C);
    this.A1 = line(C, V1 + b * unit(rotateO(-90) * (C - V1)));
    this.A2 = line(C, V1 + b * unit(rotateO(90) * (C - V1)));
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><variable type="int" signature="conicnodesfactor"><code></asyxml>*/
int conicnodesfactor = 1;/*<asyxml></code><documentation>Factor for the node number of all conics.</documentation></variable></asyxml>*/

/*<asyxml><variable type="int" signature="circlenodesnumberfactor"><code></asyxml>*/
int circlenodesnumberfactor = 100;/*<asyxml></code><documentation>Factor for the node number of circles.</documentation></variable></asyxml>*/
/*<asyxml><function type="int" signature="circlenodesnumber(real)"><code></asyxml>*/
int circlenodesnumber(real r)
{/*<asyxml></code><documentation>Return the number of nodes for drawing a circle of radius 'r'.</documentation></function></asyxml>*/
  if (circlenodesnumberfactor < 100)
    warning("circlenodesnumberfactor",
            "variable 'circlenodesnumberfactor' may be too small.");
  int oi = ceil(circlenodesnumberfactor * abs(r)^0.1);
  oi = 45 * floor(oi/45);
  return oi == 0 ? 4 : conicnodesfactor * oi;
}

/*<asyxml><function type="int" signature="circlenodesnumber(real,real,real)"><code></asyxml>*/
int circlenodesnumber(real r, real angle1, real angle2)
{/*<asyxml></code><documentation>Return the number of nodes to draw a circle arc.</documentation></function></asyxml>*/
  return (r > 0) ?
    ceil(circlenodesnumber(r) * abs(angle1 - angle2)/360) :
    ceil(circlenodesnumber(r) * abs((1 - abs(angle1 - angle2)/360)));
}

/*<asyxml><variable type="int" signature="ellispenodesnumberfactor"><code></asyxml>*/
int ellipsenodesnumberfactor = 250;/*<asyxml></code><documentation>Factor for the node number of ellispe (non-circle).</documentation></variable></asyxml>*/
/*<asyxml><function type="int" signature="ellipsenodesnumber(real,real)"><code></asyxml>*/
int ellipsenodesnumber(real a, real b)
{/*<asyxml></code><documentation>Return the number of nodes to draw a ellipse of axis 'a' and 'b'.</documentation></function></asyxml>*/
  if (ellipsenodesnumberfactor < 250)
    write("ellipsenodesnumberfactor",
          "variable 'ellipsenodesnumberfactor' maybe too small.");
  int tmp = circlenodesnumberfactor;
  circlenodesnumberfactor = ellipsenodesnumberfactor;
  int oi = circlenodesnumber(max(abs(a), abs(b))/min(abs(a), abs(b)));
  circlenodesnumberfactor = tmp;
  return conicnodesfactor * oi;
}

/*<asyxml><function type="int" signature="ellipsenodesnumber(real,real,real)"><code></asyxml>*/
int ellipsenodesnumber(real a, real b, real angle1, real angle2, bool dir)
{/*<asyxml></code><documentation>Return the number of nodes to draw an ellipse arc.</documentation></function></asyxml>*/
  real d;
  real da = angle2 - angle1;
  if(dir) {
    d = angle1 < angle2 ? da : 360 + da;
  } else {
    d = angle1 < angle2 ? -360 + da : da;
  }
  int n = floor(ellipsenodesnumber(a, b) * abs(d)/360);
  return n < 5 ? 5 : n;
}

/*<asyxml><variable type="int" signature="parabolanodesnumberfactor"><code></asyxml>*/
int parabolanodesnumberfactor = 100;/*<asyxml></code><documentation>Factor for the number of nodes of parabolas.</documentation></variable></asyxml>*/
/*<asyxml><function type="int" signature="parabolanodesnumber(parabola,real,real)"><code></asyxml>*/
int parabolanodesnumber(parabola p, real angle1, real angle2)
{/*<asyxml></code><documentation>Return the number of nodes for drawing a parabola.</documentation></function></asyxml>*/
  return conicnodesfactor * floor(0.01 * parabolanodesnumberfactor * abs(angle1 - angle2));
}

/*<asyxml><variable type="int" signature="hyperbolanodesnumberfactor"><code></asyxml>*/
int hyperbolanodesnumberfactor = 100;/*<asyxml></code><documentation>Factor for the number of nodes of hyperbolas.</documentation></variable></asyxml>*/
/*<asyxml><function type="int" signature="hyperbolanodesnumber(hyperbola,real,real)"><code></asyxml>*/
int hyperbolanodesnumber(hyperbola h, real angle1, real angle2)
{/*<asyxml></code><documentation>Return the number of nodes for drawing an hyperbola.</documentation></function></asyxml>*/
  return conicnodesfactor * floor(0.01 * hyperbolanodesnumberfactor * abs(angle1 - angle2)/h.e);
}

/*<asyxml><operator type = "conic" signature="+(conic,explicit point)"><code></asyxml>*/
conic operator +(conic c, explicit point M)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  return conic(c.F + M, c.D + M, c.e);
}
/*<asyxml><operator type = "conic" signature="-(conic,explicit point)"><code></asyxml>*/
conic operator -(conic c, explicit point M)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  return conic(c.F - M, c.D - M, c.e);
}
/*<asyxml><operator type = "conic" signature="+(conic,explicit pair)"><code></asyxml>*/
conic operator +(conic c, explicit pair m)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  point M = point(c.F.coordsys, m);
  return conic(c.F + M, c.D + M, c.e);
}
/*<asyxml><operator type = "conic" signature="-(conic,explicit pair)"><code></asyxml>*/
conic operator -(conic c, explicit pair m)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  point M = point(c.F.coordsys, m);
  return conic(c.F - M, c.D - M, c.e);
}
/*<asyxml><operator type = "conic" signature="+(conic,vector)"><code></asyxml>*/
conic operator +(conic c, vector v)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  return conic(c.F + v, c.D + v, c.e);
}
/*<asyxml><operator type = "conic" signature="-(conic,vector)"><code></asyxml>*/
conic operator -(conic c, vector v)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  return conic(c.F - v, c.D - v, c.e);
}

/*<asyxml><function type="coordsys" signature="coordsys(conic)"><code></asyxml>*/
coordsys coordsys(conic co)
{/*<asyxml></code><documentation>Return the coordinate system of 'co'.</documentation></function></asyxml>*/
  return co.F.coordsys;
}

/*<asyxml><function type="conic" signature="changecoordsys(coordsys,conic)"><code></asyxml>*/
conic changecoordsys(coordsys R, conic co)
{/*<asyxml></code><documentation>Change the coordinate system of 'co' to 'R'</documentation></function></asyxml>*/
  line l = changecoordsys(R, co.D);
  point F = changecoordsys(R, co.F);
  return conic(F, l, co.e);
}

/*<asyxml><typedef type = "polarconicroutine" return = "path" params = "conic, real, real, int, bool"><code></asyxml>*/
typedef path polarconicroutine(conic co, real angle1, real angle2, int n, bool direction);/*<asyxml></code><documentation>Routine type used to draw conics from 'angle1' to 'angle2'</documentation></typedef></asyxml>*/

/*<asyxml><function type="path" signature="arcfromfocus(conic,real,real,int,bool)"><code></asyxml>*/
path arcfromfocus(conic co, real angle1, real angle2, int n = 400, bool direction = CCW)
{/*<asyxml></code><documentation>Return the path of the conic section 'co' from angle1 to angle2 in degrees,
   drawing in the given direction, with n nodes.</documentation></function></asyxml>*/
  guide op;
  if (n < 1) return op;
  if (angle1 > angle2) {
    path g = arcfromfocus(co, angle2, angle1, n, !direction);
    return g == nullpath ? g : reverse(g);
  }
  point O = projection(co.D) * co.F;
  pair i = unit(locate(co.F) - locate(O));
  pair j = rotate(90) * i;
  coordsys Rp = cartesiansystem(co.F, i, j);
  real a1 = direction ? radians(angle1) : radians(angle2);
  real a2 = direction ? radians(angle2) : radians(angle1) + 2 * pi;
  real step = n == 1 ? 0 : (a2 - a1)/(n - 1);
  real a, r;
  for (int i = 0; i < n; ++i) {
    a = a1 + i * step;
    if(co.e >= 1) {
      r = 1 - co.e * cos(a);
      if(r > epsgeo) {
        r = co.p/r;
        op = op--Rp * Rp.polar(r, a);
      }
    } else {
      r = co.p/(1 - co.e * cos(a));
      op = op..Rp * Rp.polar(r, a);
    }
  }
  if(co.e < 1 && abs(abs(a2 - a1) - 2 * pi) < epsgeo) op = (path)op..cycle;

  return (direction ? op : op == nullpath ? op :reverse(op));
}

/*<asyxml><variable type="polarconicroutine" signature="currentpolarconicroutine"><code></asyxml>*/
polarconicroutine currentpolarconicroutine = arcfromfocus;/*<asyxml></code><documentation>Default routine used to cast conic section to path.</documentation></variable></asyxml>*/

/*<asyxml><function type="point" signature="angpoint(conic,real)"><code></asyxml>*/
point angpoint(conic co, real angle)
{/*<asyxml></code><documentation>Return the point of 'co' whose the angular (in degrees)
   coordinate is 'angle' (mesured from the focus of 'co', relatively
   to its 'natural coordinate system').</documentation></function></asyxml>*/
  coordsys R = coordsys(co);
  return point(R, point(arcfromfocus(co, angle, angle, 1, CCW), 0)/R);
}

/*<asyxml><operator type = "bool" signature="@(point,conic)"><code></asyxml>*/
bool operator @(point M, conic co)
{/*<asyxml></code><documentation>Return true iff 'M' on 'co'.</documentation></operator></asyxml>*/
  if(co.e == 0) return abs(abs(co.F - M) - co.p) < 10 * epsgeo;
  return abs(co.e * distance(M, co.D) - abs(co.F - M)) < 10 * epsgeo;
}

/*<asyxml><function type="coordsys" signature="coordsys(ellipse)"><code></asyxml>*/
coordsys coordsys(ellipse el)
{/*<asyxml></code><documentation>Return the coordinate system of 'el'.</documentation></function></asyxml>*/
  return el.F1.coordsys;
}

/*<asyxml><function type="coordsys" signature="canonicalcartesiansystem(ellipse)"><code></asyxml>*/
coordsys canonicalcartesiansystem(ellipse el)
{/*<asyxml></code><documentation>Return the canonical cartesian system of the ellipse 'el'.</documentation></function></asyxml>*/
  if(degenerate(el)) return cartesiansystem(el.l.A, el.l.u, el.l.v);
  pair O = locate(el.C);
  pair i = el.e == 0 ? el.C.coordsys.i : unit(locate(el.F1) - O);
  pair j = rotate(90) * i;
  return cartesiansystem(O, i, j);
}

/*<asyxml><function type="coordsys" signature="canonicalcartesiansystem(parabola)"><code></asyxml>*/
coordsys canonicalcartesiansystem(parabola p)
{/*<asyxml></code><documentation>Return the canonical cartesian system of a parabola,
   so that Origin = vertex of 'p' and directrix: x = -a.</documentation></function></asyxml>*/
  point A = projection(p.D) * p.F;
  pair O = locate((A + p.F)/2);
  pair i = unit(locate(p.F) - O);
  pair j = rotate(90) * i;
  return cartesiansystem(O, i, j);
}

/*<asyxml><function type="coordsys" signature="canonicalcartesiansystem(hyperbola)"><code></asyxml>*/
coordsys canonicalcartesiansystem(hyperbola h)
{/*<asyxml></code><documentation>Return the canonical cartesian system of an hyperbola.</documentation></function></asyxml>*/
  pair O = locate(h.C);
  pair i = unit(locate(h.F2) - O);
  pair j = rotate(90) * i;
  return cartesiansystem(O, i, j);
}

/*<asyxml><function type="ellipse" signature="ellipse(point,point,real)"><code></asyxml>*/
ellipse ellipse(point F1, point F2, real a)
{/*<asyxml></code><documentation>Return the ellipse whose the foci are 'F1' and 'F2'
   and the semimajor axis is 'a'.</documentation></function></asyxml>*/
  ellipse oe;
  oe.init(F1, F2, a);
  return oe;
}

/*<asyxml><constant type = "bool" signature="byfoci,byvertices"><code></asyxml>*/
restricted bool byfoci = true, byvertices = false;/*<asyxml></code><documentation>Constants useful for the routine 'hyperbola(point P1, point P2, real ae, bool byfoci = byfoci)'</documentation></constant></asyxml>*/

/*<asyxml><function type="hyperbola" signature="hyperbola(point,point,real,bool)"><code></asyxml>*/
hyperbola hyperbola(point P1, point P2, real ae, bool byfoci = byfoci)
{/*<asyxml></code><documentation>if 'byfoci = true':
   return the hyperbola whose the foci are 'P1' and 'P2'
   and the semimajor axis is 'ae'.
   else return the hyperbola whose vertexes are 'P1' and 'P2' with eccentricity 'ae'.</documentation></function></asyxml>*/
  hyperbola oh;
  point[] P = standardizecoordsys(P1, P2);
  if(byfoci) {
    oh.init(P[0], P[1], ae);
  } else {
    real a = abs(P[0]-P[1])/2;
    vector V = unit(P[0]-P[1]);
    point F1 = P[0] + a * (ae - 1) * V;
    point F2 = P[1]-a * (ae - 1) * V;
    oh.init(F1, F2, a);
  }
  return oh;
}

/*<asyxml><function type="ellipse" signature="ellipse(point,point,point)"><code></asyxml>*/
ellipse ellipse(point F1, point F2, point M)
{/*<asyxml></code><documentation>Return the ellipse passing through 'M' whose the foci are 'F1' and 'F2'.</documentation></function></asyxml>*/
  real a = abs(F1 - M) + abs(F2 - M);
  return ellipse(F1, F2, finite(a) ? a/2 : a);
}

/*<asyxml><function type="ellipse" signature="ellipse(point,real,real,real)"><code></asyxml>*/
ellipse ellipse(point C, real a, real b, real angle = 0)
{/*<asyxml></code><documentation>Return the ellipse centered at 'C' with semimajor axis 'a' along C--C + dir(angle),
   semiminor axis 'b' along the perpendicular.</documentation></function></asyxml>*/
  ellipse oe;
  coordsys R = C.coordsys;
  angle += degrees(R.i);
  if(a < b) {angle += 90; real tmp = a; a = b; b = tmp;}
  if(finite(a) && finite(b)) {
    real c = sqrt(abs(a^2 - b^2));
    point f1, f2;
    if(abs(a - b) < epsgeo) {
      f1 = C; f2 = C;
    } else {
      f1 = point(R, (locate(C) + rotate(angle) * (-c, 0))/R);
      f2 = point(R, (locate(C) + rotate(angle) * (c, 0))/R);
    }
    oe.init(f1, f2, a);
  } else {
    if(finite(b) || !finite(a)) oe.init(C, C + R.polar(1, angle), infinity);
    else oe.init(C, C + R.polar(1, 90 + angle), infinity);
  }
  return oe;
}

/*<asyxml><function type="ellipse" signature="ellipse(bqe)"><code></asyxml>*/
ellipse ellipse(bqe bqe)
{/*<asyxml></code><documentation>Return the ellipse a[0] * x^2 + a[1] * xy + a[2] * y^2 + a[3] * x + a[4] * y + a[5] = 0
   given in the coordinate system of 'bqe' with a[i] = bque.a[i].
   <url href = "http://mathworld.wolfram.com/QuadraticCurve.html"/>
   <url href = "http://mathworld.wolfram.com/Ellipse.html"/>.</documentation></function></asyxml>*/
  bqe lbqe = changecoordsys(defaultcoordsys, bqe);
  real a = lbqe.a[0], b = lbqe.a[1]/2, c = lbqe.a[2], d = lbqe.a[3]/2, f = lbqe.a[4]/2, g = lbqe.a[5];
  coordsys R = bqe.coordsys;
  string message = "ellipse: the given equation is not an equation of an ellipse.";
  real u = b^2 * g + d^2 * c + f^2 * a;
  real delta = a * c * g + b * f * d + d * b * f - u;
  if(abs(delta) < epsgeo) abort(message);
  real j = b^2 - a * c;
  real i = a + c;
  real dd = j * (sgnd(c - a) * sqrt((a - c)^2 + 4 * (b^2)) - c-a);
  real ddd = j * (-sgnd(c - a) * sqrt((a - c)^2 + 4 * (b^2)) - c-a);

  if(abs(ddd) < epsgeo || abs(dd) < epsgeo ||
     j >= -epsgeo || delta/sgnd(i) > 0) abort(message);

  real x = (c * d - b * f)/j, y = (a * f - b * d)/j;
  // real dir = abs(b) < epsgeo ? 0 : pi/2-0.5 * acot(0.5 * (c-a)/b);
  real dir = abs(b) < epsgeo ? 0 : 0.5 * acot(0.5 * (c - a)/b);
  if(dir * (c - a) * b < 0) dir = dir < 0 ? dir + pi/2 : dir - pi/2;
  real cd = cos(dir), sd = sin(dir);
  real t = a * cd^2 - 2 * b * cd * sd + c * sd^2;
  real tt = a * sd^2 + 2 * b * cd * sd + c * cd^2;
  real gg = -g + ((d * cd - f * sd)^2)/t + ((d * sd + f * cd)^2)/tt;
  t = t/gg; tt = tt/gg;
  // The equation of the ellipse is t * (x - center.x)^2 + tt * (y - center.y)^2 = 1;
  real aa, bb;
  aa = sqrt(2 * (u - 2 * b * d * f - a * c * g)/dd);
  bb = sqrt(2 * (u - 2 * b * d * f - a * c * g)/ddd);
  a = t > tt ? max(aa, bb) : min(aa, bb);
  b = t > tt ? min(aa, bb) : max(aa, bb);
  return ellipse(point(R, (x, y)/R),
                 a, b, degrees(pi/2 - dir - angle(R.i)));
}

/*<asyxml><function type="ellipse" signature="ellipse(point,point,point,point,point)"><code></asyxml>*/
ellipse ellipse(point M1, point M2, point M3, point M4, point M5)
{/*<asyxml></code><documentation>Return the ellipse passing through the five points (if possible)</documentation></function></asyxml>*/
  return ellipse(bqe(M1, M2, M3, M4, M5));
}

/*<asyxml><function type="bool" signature="inside(ellipse,point)"><code></asyxml>*/
bool inside(ellipse el, point M)
{/*<asyxml></code><documentation>Return 'true' iff 'M' is inside 'el'.</documentation></function></asyxml>*/
  return abs(el.F1 - M) + abs(el.F2 - M) - 2 * el.a < -epsgeo;
}

/*<asyxml><function type="bool" signature="inside(parabola,point)"><code></asyxml>*/
bool inside(parabola p, point M)
{/*<asyxml></code><documentation>Return 'true' if 'M' is inside 'p'.</documentation></function></asyxml>*/
  return distance(p.D, M) - abs(p.F - M) > epsgeo;
}

/*<asyxml><function type="parabola" signature="parabola(point,line)"><code></asyxml>*/
parabola parabola(point F, line l)
{/*<asyxml></code><documentation>Return the parabola whose focus is 'F' and directrix is 'l'.</documentation></function></asyxml>*/
  parabola op;
  op.init(F, l);
  return op;
}

/*<asyxml><function type="parabola" signature="parabola(point,point)"><code></asyxml>*/
parabola parabola(point F, point vertex)
{/*<asyxml></code><documentation>Return the parabola whose focus is 'F' and vertex is 'vertex'.</documentation></function></asyxml>*/
  parabola op;
  point[] P = standardizecoordsys(F, vertex);
  point A = rotate(180, P[1]) * P[0];
  point B = A + rotateO(90) * unit(P[1]-A);
  op.init(P[0], line(A, B));
  return op;
}

/*<asyxml><function type="parabola" signature="parabola(point,real,real)"><code></asyxml>*/
parabola parabola(point F, real a, real angle)
{/*<asyxml></code><documentation>Return the parabola whose focus is F, latus rectum is 4a and
   the angle of the axis of symmetry (in the coordinate system of F) is 'angle'.</documentation></function></asyxml>*/
  parabola op;
  coordsys R = F.coordsys;
  point A = F - point(R, R.polar(2a, radians(angle)));
  point B = A + point(R, R.polar(1, radians(90 + angle)));
  op.init(F, line(A, B));
  return op;
}

/*<asyxml><function type="bool" signature="isparabola(bqe)"><code></asyxml>*/
bool isparabola(bqe bqe)
{/*<asyxml></code><documentation>Return true iff 'bqe' is the equation of a parabola.</documentation></function></asyxml>*/
  bqe lbqe = changecoordsys(defaultcoordsys, bqe);
  real a = lbqe.a[0], b = lbqe.a[1]/2, c = lbqe.a[2], d = lbqe.a[3]/2, f = lbqe.a[4]/2, g = lbqe.a[5];
  real delta = a * c * g + b * f * d + d * b * f - (b^2 * g + d^2 * c + f^2 * a);
  return (abs(delta) > epsgeo && abs(b^2 - a * c) < epsgeo);
}

/*<asyxml><function type="parabola" signature="parabola(bqe)"><code></asyxml>*/
parabola parabola(bqe bqe)
{/*<asyxml></code><documentation>Return the parabola a[0]x^2 + a[1]xy + a[2]y^2 + a[3]x + a[4]y + a[5]] = 0 (a[n] means bqe.a[n]).
   <url href = "http://mathworld.wolfram.com/QuadraticCurve.html"/>
   <url href = "http://mathworld.wolfram.com/Parabola.html"/></documentation></function></asyxml>*/
  bqe lbqe = changecoordsys(defaultcoordsys, bqe);
  real a = lbqe.a[0], b = lbqe.a[1]/2, c = lbqe.a[2], d = lbqe.a[3]/2, f = lbqe.a[4]/2, g = lbqe.a[5];
  string message = "parabola: the given equation is not an equation of a parabola.";
  real delta = a * c * g + b * f * d + d * b * f - (b^2 * g + d^2 * c + f^2 * a);
  if(abs(delta) < 10 * epsgeo || abs(b^2 - a * c) > 10 * epsgeo) abort(message);
  real dir = abs(b) < epsgeo ? 0 : 0.5 * acot(0.5 * (c - a)/b);
  if(dir * (c - a) * b < 0) dir = dir < 0 ? dir + pi/2 : dir - pi/2;
  real cd = cos(dir), sd = sin(dir);
  real ap = a * cd^2 - 2 * b * cd * sd + c * sd^2;
  real cp = a * sd^2 + 2 * b * cd * sd + c * cd^2;
  real dp = d * cd - f * sd;
  real fp = d * sd + f * cd;
  real gp = g;
  parabola op;
  coordsys R = bqe.coordsys;
  // The equation of the parabola is ap * x'^2 + cp * y'^2 + 2dp * x'+2fp * y'+gp = 0
  if (abs(ap) < epsgeo) {/* directrix parallel to the rotated(dir) y-axis
                            equation: (y-vertex.y)^2 = 4 * a * (x-vertex)
                         */
    pair pvertex = rotate(degrees(-dir)) * (0.5(-gp + fp^2/cp)/dp, -fp/cp);
    real a = -0.5 * dp/cp;
    point vertex = point(R, pvertex/R);
    point focus = point(R, (pvertex + a * expi(-dir))/R);
    op = parabola(focus, vertex);

  } else {/* directrix parallel to the rotated(dir) x-axis
             equation: (x-vertex)^2 = 4 * a * (y-vertex.y)
          */
    pair pvertex = rotate(degrees(-dir)) * (-dp/ap, 0.5 * (-gp + dp^2/ap)/fp);
    real a = -0.5 * fp/ap;
    point vertex = point(R, pvertex/R);
    point focus = point(R, (pvertex + a * expi(pi/2 - dir))/R);
    op = parabola(focus, vertex);
  }
  return op;
}

/*<asyxml><function type="parabola" signature="parabola(point,point,point,line)"><code></asyxml>*/
parabola parabola(point M1, point M2, point M3, line l)
{/*<asyxml></code><documentation>Return the parabola passing through the three points with its directix
   parallel to the line 'l'.</documentation></function></asyxml>*/
  coordsys R;
  pair[] pts;
  if (samecoordsys(M1, M2, M3)) {
    R = M1.coordsys;
  } else {
    R = defaultcoordsys;
  }
  real gle = degrees(l);
  coordsys Rp = cartesiansystem(R.O, rotate(gle) * R.i, rotate(gle) * R.j);
  pts = new pair[] {coordinates(changecoordsys(Rp, M1)),
                  coordinates(changecoordsys(Rp, M2)),
                  coordinates(changecoordsys(Rp, M3))};
  real[][] M;
  real[] x;
  for (int i = 0; i < 3; ++i) {
    M[i] = new real[] {pts[i].x, pts[i].y, 1};
    x[i] = -pts[i].x^2;
  }
  real[] coef = solve(M, x);
  return parabola(changecoordsys(R, bqe(Rp, 1, 0, 0, coef[0], coef[1], coef[2])));
}

/*<asyxml><function type="parabola" signature="parabola(point,point,point,point,point)"><code></asyxml>*/
parabola parabola(point M1, point M2, point M3, point M4, point M5)
{/*<asyxml></code><documentation>Return the parabola passing through the five points.</documentation></function></asyxml>*/
  return parabola(bqe(M1, M2, M3, M4, M5));
}

/*<asyxml><function type="hyperbola" signature="hyperbola(point,point,point)"><code></asyxml>*/
hyperbola hyperbola(point F1, point F2, point M)
{/*<asyxml></code><documentation>Return the hyperbola passing through 'M' whose the foci are 'F1' and 'F2'.</documentation></function></asyxml>*/
  real a = abs(abs(F1 - M) - abs(F2 - M));
  return hyperbola(F1, F2, finite(a) ? a/2 : a);
}

/*<asyxml><function type="hyperbola" signature="hyperbola(point,real,real,real)"><code></asyxml>*/
hyperbola hyperbola(point C, real a, real b, real angle = 0)
{/*<asyxml></code><documentation>Return the hyperbola centered at 'C' with semimajor axis 'a' along C--C + dir(angle),
   semiminor axis 'b' along the perpendicular.</documentation></function></asyxml>*/
  hyperbola oh;
  coordsys R = C.coordsys;
  angle += degrees(R.i);
  real c = sqrt(a^2 + b^2);
  point f1 = point(R, (locate(C) + rotate(angle) * (-c, 0))/R);
  point f2 = point(R, (locate(C) + rotate(angle) * (c, 0))/R);
  oh.init(f1, f2, a);
  return oh;
}

/*<asyxml><function type="hyperbola" signature="hyperbola(bqe)"><code></asyxml>*/
hyperbola hyperbola(bqe bqe)
{/*<asyxml></code><documentation>Return the hyperbola a[0]x^2 + a[1]xy + a[2]y^2 + a[3]x + a[4]y + a[5]] = 0 (a[n] means bqe.a[n]).
   <url href = "http://mathworld.wolfram.com/QuadraticCurve.html"/>
   <url href = "http://mathworld.wolfram.com/Hyperbola.html"/></documentation></function></asyxml>*/
  bqe lbqe = changecoordsys(defaultcoordsys, bqe);
  real a = lbqe.a[0], b = lbqe.a[1]/2, c = lbqe.a[2], d = lbqe.a[3]/2, f = lbqe.a[4]/2, g = lbqe.a[5];
  string message = "hyperbola: the given equation is not an equation of a hyperbola.";
  real delta = a * c * g + b * f * d + d * b * f - (b^2 * g + d^2 * c + f^2 * a);
  if(abs(delta) < 10 * epsgeo || abs(b^2 - a * c) < 0) abort(message);
  real dir = abs(b) < epsgeo ? 0 : 0.5 * acot(0.5 * (c - a)/b);
  real cd = cos(dir), sd = sin(dir);
  real ap = a * cd^2 - 2 * b * cd * sd + c * sd^2;
  real cp = a * sd^2 + 2 * b * cd * sd + c * cd^2;
  real dp = d * cd - f * sd;
  real fp = d * sd + f * cd;
  real gp = -g + dp^2/ap + fp^2/cp;
  hyperbola op;
  coordsys R = bqe.coordsys;
  real j = b^2 - a * c;
  point C = point(R, ((c * d - b * f)/j, (a * f - b * d)/j)/R);
  real aa = gp/ap, bb = gp/cp;
  real a = sqrt(abs(aa)), b = sqrt(abs(bb));
  if(aa < 0) {dir -= pi/2; aa = a; a = b; b = aa;}
  return hyperbola(C, a, b, degrees(-dir - angle(R.i)));
}

/*<asyxml><function type="hyperbola" signature="hyperbola(point,point,point,point,point)"><code></asyxml>*/
hyperbola hyperbola(point M1, point M2, point M3, point M4, point M5)
{/*<asyxml></code><documentation>Return the hyperbola passing through the five points (if possible).</documentation></function></asyxml>*/
  return hyperbola(bqe(M1, M2, M3, M4, M5));
}

/*<asyxml><function type="hyperbola" signature="conj(hyperbola)"><code></asyxml>*/
hyperbola conj(hyperbola h)
{/*<asyxml></code><documentation>Conjugate.</documentation></function></asyxml>*/
  return hyperbola(h.C, h.b, h.a, 90 + h.angle);
}

/*<asyxml><function type="circle" signature="circle(explicit point,real)"><code></asyxml>*/
circle circle(explicit point C, real r)
{/*<asyxml></code><documentation>Circle given by center and radius.</documentation></function></asyxml>*/
  circle oc = new circle;
  oc.C = C;
  oc.r = r;
  if(!finite(r)) oc.l = line(C, C + vector(C.coordsys, (1, 0)));
  return oc;
}

/*<asyxml><function type="circle" signature="circle(point,point)"><code></asyxml>*/
circle circle(point A, point B)
{/*<asyxml></code><documentation>Return the circle of diameter AB.</documentation></function></asyxml>*/
  real r;
  circle oc;
  real a = abs(A), b = abs(B);
  if(finite(a) && finite(b)) {
    oc = circle((A + B)/2, abs(A - B)/2);
  } else {
    oc.r = infinity;
    if(finite(abs(A))) oc.l = line(A, A + unit(B));
    else {
      if(finite(abs(B))) oc.l = line(B, B + unit(A));
      else if(finite(abs(A - B)/2)) oc = circle((A + B)/2, abs(A - B)/2); else
        oc.l = line(A, B);
    }
  }
  return oc;
}

/*<asyxml><function type="circle" signature="circle(segment)"><code></asyxml>*/
circle circle(segment s)
{/*<asyxml></code><documentation>Return the circle of diameter 's'.</documentation></function></asyxml>*/
  return circle(s.A, s.B);
}

/*<asyxml><function type="point" signature="circumcenter(point,point,point)"><code></asyxml>*/
point circumcenter(point A, point B, point C)
{/*<asyxml></code><documentation>Return the circumcenter of triangle ABC.</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B, C);
  coordsys R = P[0].coordsys;
  pair a = A, b = B, c = C;
  pair mAB = (a + b)/2;
  pair mAC = (a + c)/2;
  pair pp = extension(mAB, rotate(90, mAB) * a, mAC, rotate(90, mAC) * c);
  return point(R, pp/R);
}

/*<asyxml><function type="circle" signature="circle(point,point,point)"><code></asyxml>*/
circle circle(point A, point B, point C)
{/*<asyxml></code><documentation>Return the circumcircle of the triangle ABC.</documentation></function></asyxml>*/
  if(collinear(A - B, A - C)) {
    circle oc;
    oc.r = infinity;
    oc.C = (A + B + C)/3;
    oc.l = line(oc.C, oc.C == A ? B : A);
    return oc;
  }
  point c = circumcenter(A, B, C);
  return circle(c, abs(c - A));
}

/*<asyxml><function type="circle" signature="circumcircle(point,point,point)"><code></asyxml>*/
circle circumcircle(point A, point B, point C)
{/*<asyxml></code><documentation>Return the circumcircle of the triangle ABC.</documentation></function></asyxml>*/
  return circle(A, B, C);
}

/*<asyxml><operator type = "circle" signature="*(real,explicit circle)"><code></asyxml>*/
circle operator *(real x, explicit circle c)
{/*<asyxml></code><documentation>Multiply the radius of 'c'.</documentation></operator></asyxml>*/
  return finite(c.r) ? circle(c.C, x * c.r) : c;
}
circle operator *(int x, explicit circle c)
{
  return finite(c.r) ? circle(c.C, x * c.r) : c;
}
/*<asyxml><operator type = "circle" signature="/(explicit circle,real)"><code></asyxml>*/
circle operator /(explicit circle c, real x)
{/*<asyxml></code><documentation>Divide the radius of 'c'</documentation></operator></asyxml>*/
  return finite(c.r) ? circle(c.C, c.r/x) : c;
}
circle operator /(explicit circle c, int x)
{
  return finite(c.r) ? circle(c.C, c.r/x) : c;
}
/*<asyxml><operator type = "circle" signature="+(explicit circle,explicit point)"><code></asyxml>*/
circle operator +(explicit circle c, explicit point M)
{/*<asyxml></code><documentation>Translation of 'c'.</documentation></operator></asyxml>*/
  return circle(c.C + M, c.r);
}
/*<asyxml><operator type = "circle" signature="-(explicit circle,explicit point)"><code></asyxml>*/
circle operator -(explicit circle c, explicit point M)
{/*<asyxml></code><documentation>Translation of 'c'.</documentation></operator></asyxml>*/
  return circle(c.C - M, c.r);
}
/*<asyxml><operator type = "circle" signature="+(explicit circle,pair)"><code></asyxml>*/
circle operator +(explicit circle c, pair m)
{/*<asyxml></code><documentation>Translation of 'c'.
   'm' represent coordinates in the coordinate system where 'c' is defined.</documentation></operator></asyxml>*/
  return circle(c.C + m, c.r);
}
/*<asyxml><operator type = "circle" signature="-(explicit circle,pair)"><code></asyxml>*/
circle operator -(explicit circle c, pair m)
{/*<asyxml></code><documentation>Translation of 'c'.
   'm' represent coordinates in the coordinate system where 'c' is defined.</documentation></operator></asyxml>*/
  return circle(c.C - m, c.r);
}
/*<asyxml><operator type = "circle" signature="+(explicit circle,vector)"><code></asyxml>*/
circle operator +(explicit circle c, vector m)
{/*<asyxml></code><documentation>Translation of 'c'.</documentation></operator></asyxml>*/
  return circle(c.C + m, c.r);
}
/*<asyxml><operator type = "circle" signature="-(explicit circle,vector)"><code></asyxml>*/
circle operator -(explicit circle c, vector m)
{/*<asyxml></code><documentation>Translation of 'c'.</documentation></operator></asyxml>*/
  return circle(c.C - m, c.r);
}
/*<asyxml><operator type = "real" signature="^(point,explicit circle)"><code></asyxml>*/
real operator ^(point M, explicit circle c)
{/*<asyxml></code><documentation>The power of 'M' with respect to the circle 'c'</documentation></operator></asyxml>*/
  return xpart((abs(locate(M) - locate(c.C)), c.r)^2);
}
/*<asyxml><operator type = "bool" signature="@(point,explicit circle)"><code></asyxml>*/
bool operator @(point M, explicit circle c)
{/*<asyxml></code><documentation>Return true iff 'M' is on the circle 'c'.</documentation></operator></asyxml>*/
  return finite(c.r) ?
    abs(abs(locate(M) - locate(c.C)) - abs(c.r)) <= 10 * epsgeo :
    M @ c.l;
}

/*<asyxml><operator type = "ellipse" signature="cast(circle)"><code></asyxml>*/
ellipse operator cast(circle c)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  return finite(c.r) ? ellipse(c.C, c.r, c.r, 0) : ellipse(c.l.A, c.l.B, infinity);
}

/*<asyxml><operator type = "circle" signature="cast(ellipse)"><code></asyxml>*/
circle operator ecast(ellipse el)
{/*<asyxml></code><documentation></documentation></operator></asyxml>*/
  circle oc;
  bool infb = (!finite(el.a) || !finite(el.b));
  if(!infb && abs(el.a - el.b) > epsgeo)
    abort("Can not cast ellipse with different axis values to circle");
  oc = circle(el.C, infb ? infinity : el.a);
  oc.l = el.l.copy();
  return oc;
}

/*<asyxml><operator type = "ellipse" signature="cast(conic)"><code></asyxml>*/
ellipse operator ecast(conic co)
{/*<asyxml></code><documentation>Cast a conic to an ellipse (can be a circle).</documentation></operator></asyxml>*/
  if(degenerate(co) && co.e < 1) return ellipse(co.l[0].A, co.l[0].B, infinity);
  ellipse oe;
  if(co.e < 1) {
    real a = co.p/(1 - co.e^2);
    real c = co.e * a;
    vector v = co.D.v;
    if(!sameside(co.D.A + v, co.F, co.D)) v = -v;
    point f2 = co.F + 2 * c * v;
    f2 = changecoordsys(co.F.coordsys, f2);
    oe = a == 0 ? ellipse(co.F, co.p, co.p, 0) : ellipse(co.F, f2, a);
  } else
    abort("casting: The conic section is not an ellipse.");
  return oe;
}

/*<asyxml><operator type = "parabola" signature="cast(conic)"><code></asyxml>*/
parabola operator ecast(conic co)
{/*<asyxml></code><documentation>Cast a conic to a parabola.</documentation></operator></asyxml>*/
  parabola op;
  if(abs(co.e - 1) > epsgeo) abort("casting: The conic section is not a parabola.");
  op.init(co.F, co.D);
  return op;
}

/*<asyxml><operator type = "conic" signature="cast(parabola)"><code></asyxml>*/
conic operator cast(parabola p)
{/*<asyxml></code><documentation>Cast a parabola to a conic section.</documentation></operator></asyxml>*/
  return conic(p.F, p.D, 1);
}

/*<asyxml><operator type = "hyperbola" signature="cast(conic)"><code></asyxml>*/
hyperbola operator ecast(conic co)
{/*<asyxml></code><documentation>Cast a conic section to an hyperbola.</documentation></operator></asyxml>*/
  hyperbola oh;
  if(co.e > 1) {
    real a = co.p/(co.e^2 - 1);
    real c = co.e * a;
    vector v = co.D.v;
    if(sameside(co.D.A + v, co.F, co.D)) v = -v;
    point f2 = co.F + 2 * c * v;
    f2 = changecoordsys(co.F.coordsys, f2);
    oh = hyperbola(co.F, f2, a);
  } else
    abort("casting: The conic section is not an hyperbola.");
  return oh;
}

/*<asyxml><operator type = "conic" signature="cast(hyperbola)"><code></asyxml>*/
conic operator cast(hyperbola h)
{/*<asyxml></code><documentation>Hyperbola to conic section.</documentation></operator></asyxml>*/
  return conic(h.F1, h.D1, h.e);
}

/*<asyxml><operator type = "conic" signature="cast(ellipse)"><code></asyxml>*/
conic operator cast(ellipse el)
{/*<asyxml></code><documentation>Ellipse to conic section.</documentation></operator></asyxml>*/
  conic oc;
  if(abs(el.c) > epsgeo) {
    real x = el.a^2/el.c;
    point O = (el.F1 + el.F2)/2;
    point A = O + x * unit(el.F1 - el.F2);
    oc = conic(el.F1, perpendicular(A, line(el.F1, el.F2)), el.e);
  } else {//The ellipse is a circle
    coordsys R = coordsys(el);
    point M = el.F1 + point(R, R.polar(el.a, 0));
    line l = line(rotate(90, M) * el.F1, M);
    oc = conic(el.F1, l, 0);
  }
  if(degenerate(el)) {
    oc.p = infinity;
    oc.h = infinity;
    oc.l = new line[]{el.l};
  }
  return oc;
}

/*<asyxml><operator type = "conic" signature="cast(circle)"><code></asyxml>*/
conic operator cast(circle c)
{/*<asyxml></code><documentation>Circle to conic section.</documentation></operator></asyxml>*/
  return (conic)((ellipse)c);
}

/*<asyxml><operator type = "circle" signature="cast(conic)"><code></asyxml>*/
circle operator ecast(conic c)
{/*<asyxml></code><documentation>Conic section to circle.</documentation></operator></asyxml>*/
  ellipse el = (ellipse)c;
  circle oc;
  if(abs(el.a - el.b) < epsgeo) {
    oc = circle(el.C, el.a);
    if(degenerate(c)) oc.l = c.l[0];
  }
  else abort("Can not cast this conic to a circle");
  return oc;
}

/*<asyxml><operator type = "ellipse" signature="*(transform,ellipse)"><code></asyxml>*/
ellipse operator *(transform t, ellipse el)
{/*<asyxml></code><documentation>Provide transform * ellipse.</documentation></operator></asyxml>*/
  if(!degenerate(el)) {
    point[] ep;
    for (int i = 0; i < 360; i += 72) {
      ep.push(t * angpoint(el, i));
    }
    ellipse oe = ellipse(ep[0], ep[1], ep[2], ep[3], ep[4]);
    if(angpoint(oe, 0) != ep[0]) return ellipse(oe.F2, oe.F1, oe.a);
    return oe;
  }
  return ellipse(t * el.l.A, t * el.l.B, infinity);
}

/*<asyxml><operator type = "parabola" signature="*(transform,parabola)"><code></asyxml>*/
parabola operator *(transform t, parabola p)
{/*<asyxml></code><documentation>Provide transform * parabola.</documentation></operator></asyxml>*/
  point[] P;
  P.push(t * angpoint(p, 45));
  P.push(t * angpoint(p, -45));
  P.push(t * angpoint(p, 180));
  parabola op = parabola(P[0], P[1], P[2], t * p.D);
  op.bmin = p.bmin;
  op.bmax = p.bmax;

  return op;
}

/*<asyxml><operator type = "ellipse" signature="*(transform,circle)"><code></asyxml>*/
ellipse operator *(transform t, circle c)
{/*<asyxml></code><documentation>Provide transform * circle.
   For example, 'circle C = scale(2) * circle' and 'ellipse E = xscale(2) * circle' are valid
   but 'circle C = xscale(2) * circle' is invalid.</documentation></operator></asyxml>*/
  return t * ((ellipse)c);
}

/*<asyxml><operator type = "hyperbola" signature="*(transform,hyperbola)"><code></asyxml>*/
hyperbola operator *(transform t, hyperbola h)
{/*<asyxml></code><documentation>Provide transform * hyperbola.</documentation></operator></asyxml>*/
  if (t == identity()) {
    return h;
  }

  point[] ep;
  for (int i = 90; i <= 270; i += 45) {
    ep.push(t * angpoint(h, i));
  }

  hyperbola oe = hyperbola(ep[0], ep[1], ep[2], ep[3], ep[4]);
  if(angpoint(oe, 90) != ep[0]) {
    oe = hyperbola(oe.F2, oe.F1, oe.a);
  }

  oe.bmin = h.bmin;
  oe.bmax = h.bmax;

  return oe;
}

/*<asyxml><operator type = "conic" signature="*(transform,conic)"><code></asyxml>*/
conic operator *(transform t, conic co)
{/*<asyxml></code><documentation>Provide transform * conic.</documentation></operator></asyxml>*/
  if(co.e < 1) return (t * ((ellipse)co));
  if(co.e == 1) return (t * ((parabola)co));
  return (t * ((hyperbola)co));
}

/*<asyxml><operator type = "ellipse" signature="*(real,ellipse)"><code></asyxml>*/
ellipse operator *(real x, ellipse el)
{/*<asyxml></code><documentation>Identical but more efficient (rapid) than 'scale(x, el.C) * el'.</documentation></operator></asyxml>*/
  return degenerate(el) ? el : ellipse(el.C, x * el.a, x * el.b, el.angle);
}

/*<asyxml><operator type = "ellipse" signature="/(ellipse,real)"><code></asyxml>*/
ellipse operator /(ellipse el, real x)
{/*<asyxml></code><documentation>Identical but more efficient (rapid) than 'scale(1/x, el.C) * el'.</documentation></operator></asyxml>*/
  return degenerate(el) ? el : ellipse(el.C, el.a/x, el.b/x, el.angle);
}

/*<asyxml><function type="path" signature="arcfromcenter(ellipse,real,real,int,bool)"><code></asyxml>*/
path arcfromcenter(ellipse el, real angle1, real angle2,
                   bool direction=CCW,
                   int n=ellipsenodesnumber(el.a,el.b,angle1,angle2,direction))
{/*<asyxml></code><documentation>Return the path of the ellipse 'el' from angle1 to angle2 in degrees,
   drawing in the given direction, with n nodes.
   The angles are mesured relatively to the  axis (C,x-axis) where C is
   the center of the ellipse.</documentation></function></asyxml>*/
  if(degenerate(el)) abort("arcfromcenter: can not convert degenerated ellipse to path.");
  if (angle1 > angle2)
    return reverse(arcfromcenter(el, angle2, angle1, !direction, n));

  guide op;
  coordsys Rp=coordsys(el);
  if (n < 1) return op;

  interpolate join = operator ..;
  real stretch = max(el.a/el.b, el.b/el.a);

  if (stretch > 10) {
    n *= floor(stretch/5);
    join = operator --;
  }

  real a1 = direction ? radians(angle1) : radians(angle2);
  real a2 = direction ? radians(angle2) : radians(angle1) + 2 * pi;
  real step=(a2 - a1)/(n != 1 ? n-1 : 1);
  real a, r;
  real da = radians(el.angle);

  for (int i=0; i < n; ++i) {
    a = a1 + i * step;
    r = el.b/sqrt(1 - (el.e * cos(a))^2);
    op = join(op, Rp*Rp.polar(r, da + a));
  }

  return shift(el.C.x*Rp.i + el.C.y*Rp.j) * (direction ? op : reverse(op));
}

/*<asyxml><function type="path" signature="arcfromcenter(hyperbola,real,real,int,bool)"><code></asyxml>*/
path arcfromcenter(hyperbola h, real angle1, real angle2,
                   int n = hyperbolanodesnumber(h, angle1, angle2),
                   bool direction = CCW)
{/*<asyxml></code><documentation>Return the path of the hyperbola 'h' from angle1 to angle2 in degrees,
   drawing in the given direction, with n nodes.
   The angles are mesured relatively to the axis (C, x-axis) where C is
   the center of the hyperbola.</documentation></function></asyxml>*/
  guide op;
  coordsys Rp = coordsys(h);
  if (n < 1) return op;
  if (angle1 > angle2) {
    path g = reverse(arcfromcenter(h, angle2, angle1, n, !direction));
    return g == nullpath ? g : reverse(g);
  }
  real a1 = direction ? radians(angle1) : radians(angle2);
  real a2 = direction ? radians(angle2) : radians(angle1) + 2 * pi;
  real step = (a2 - a1)/(n != 1 ? n - 1 : 1);
  real a, r;
  typedef guide interpolate(... guide[]);
  interpolate join = operator ..;
  real da = radians(h.angle);
  for (int i = 0; i < n; ++i) {
    a = a1 + i * step;
    r = (h.b * cos(a))^2 - (h.a * sin(a))^2;
    if(r > epsgeo) {
      r = sqrt(h.a^2 * h.b^2/r);
      op = join(op, Rp * Rp.polar(r, a + da));
      join = operator ..;
    } else join = operator --;
  }
  return shift(h.C.x * Rp.i + h.C.y * Rp.j)*
    (direction ? op : op == nullpath ? op : reverse(op));
}

/*<asyxml><function type="path" signature="arcfromcenter(explicit conic,real,real,int,bool)"><code></asyxml>*/
path arcfromcenter(explicit conic co, real angle1, real angle2,
                   int n, bool direction = CCW)
{/*<asyxml></code><documentation>Use arcfromcenter(ellipse, ...) or arcfromcenter(hyperbola, ...) depending of
   the eccentricity of 'co'.</documentation></function></asyxml>*/
  path g;
  if(co.e < 1)
    g = arcfromcenter((ellipse)co, angle1,
                    angle2, direction, n);
  else if(co.e > 1)
    g = arcfromcenter((hyperbola)co, angle1,
                    angle2, n, direction);
  else abort("arcfromcenter: does not exist for a parabola.");
  return g;
}

/*<asyxml><constant type = "polarconicroutine" signature="fromCenter"><code></asyxml>*/
restricted polarconicroutine fromCenter = arcfromcenter;/*<asyxml></code><documentation></documentation></constant></asyxml>*/
/*<asyxml><constant type = "polarconicroutine" signature="fromFocus"><code></asyxml>*/
restricted polarconicroutine fromFocus = arcfromfocus;/*<asyxml></code><documentation></documentation></constant></asyxml>*/

/*<asyxml><function type="bqe" signature="equation(ellipse)"><code></asyxml>*/
bqe equation(ellipse el)
{/*<asyxml></code><documentation>Return the coefficients of the equation of the ellipse in its coordinate system:
   bqe.a[0] * x^2 + bqe.a[1] * x * y + bqe.a[2] * y^2 + bqe.a[3] * x + bqe.a[4] * y + bqe.a[5] = 0.
   One can change the coordinate system of 'bqe' using the routine 'changecoordsys'.</documentation></function></asyxml>*/
  pair[] pts;
  for (int i = 0; i < 360; i += 72)
    pts.push(locate(angpoint(el, i)));

  real[][] M;
  real[] x;
  for (int i = 0; i < 5; ++i) {
    M[i] = new real[] {pts[i].x * pts[i].y, pts[i].y^2, pts[i].x, pts[i].y, 1};
    x[i] = -pts[i].x^2;
  }
  real[] coef = solve(M, x);
  bqe bqe = changecoordsys(coordsys(el),
                         bqe(defaultcoordsys,
                             1, coef[0], coef[1], coef[2], coef[3], coef[4]));
  bqe.a = approximate(bqe.a);
  return bqe;
}

/*<asyxml><function type="bqe" signature="equation(parabola)"><code></asyxml>*/
bqe equation(parabola p)
{/*<asyxml></code><documentation>Return the coefficients of the equation of the parabola in its coordinate system.
   bqe.a[0] * x^2 + bqe.a[1] * x * y + bqe.a[2] * y^2 + bqe.a[3] * x + bqe.a[4] * y + bqe.a[5] = 0
   One can change the coordinate system of 'bqe' using the routine 'changecoordsys'.</documentation></function></asyxml>*/
  coordsys R = canonicalcartesiansystem(p);
  parabola tp = (parabola) changecoordsys(R, p);
  point A = projection(tp.D) * point(R, (0, 0));
  real a = abs(A);
  return changecoordsys(coordsys(p),
                        bqe(R, 0, 0, 1, -4 * a, 0, 0));
}

/*<asyxml><function type="bqe" signature="equation(hyperbola)"><code></asyxml>*/
bqe equation(hyperbola h)
{/*<asyxml></code><documentation>Return the coefficients of the equation of the hyperbola in its coordinate system.
   bqe.a[0] * x^2 + bqe.a[1] * x * y + bqe.a[2] * y^2 + bqe.a[3] * x + bqe.a[4] * y + bqe.a[5] = 0
   One can change the coordinate system of 'bqe' using the routine 'changecoordsys'.</documentation></function></asyxml>*/
  coordsys R = canonicalcartesiansystem(h);
  return changecoordsys(coordsys(h),
                        bqe(R, 1/h.a^2, 0, -1/h.b^2, 0, 0, -1));
}

/*<asyxml><operator type = "path" signature="cast(ellipse)"><code></asyxml>*/
path operator cast(ellipse el)
{/*<asyxml></code><documentation>Cast ellipse to path.</documentation></operator></asyxml>*/
  if(degenerate(el))
    abort("Casting degenerated ellipse to path is not possible.");
  int n = el.e == 0 ? circlenodesnumber(el.a) : ellipsenodesnumber(el.a, el.b);
  return arcfromcenter(el, 0.0, 360, CCW, n)&cycle;
}

/*<asyxml><operator type = "path" signature="cast(circle)"><code></asyxml>*/
path operator cast(circle c)
{/*<asyxml></code><documentation>Cast circle to path.</documentation></operator></asyxml>*/
  return (path)((ellipse)c);
}

/*<asyxml><function type="real[]" signature="bangles(picture,parabola)"><code></asyxml>*/
real[] bangles(picture pic = currentpicture, parabola p)
{/*<asyxml></code><documentation>Return the array {ma, Ma} where 'ma' and 'Ma' are respectively
   the smaller and the larger angles for which the parabola 'p' is included
   in the bounding box of the picture 'pic'.</documentation></function></asyxml>*/
  pair bmin, bmax;
  pair[] b;
  if (p.bmin == p.bmax) {
    bmin = pic.userMin();
    bmax = pic.userMax();
  } else {
    bmin = p.bmin;bmax = p.bmax;
  }
  if(bmin.x == bmax.x || bmin.y == bmax.y || !finite(abs(bmin)) || !finite(abs(bmax)))
    return new real[] {0, 0};
  b[0] = bmin;
  b[1] = (bmax.x, bmin.y);
  b[2] = bmax;
  b[3] = (bmin.x, bmax.y);
  real[] eq = changecoordsys(defaultcoordsys, equation(p)).a;
  pair[] inter;
  for (int i = 0; i < 4; ++i) {
    pair[] tmp = intersectionpoints(b[i], b[(i + 1)%4], eq);
    for (int j = 0; j < tmp.length; ++j) {
      if(dot(b[i]-tmp[j], b[(i + 1)%4]-tmp[j]) <= epsgeo)
        inter.push(tmp[j]);
    }
  }
  pair F = p.F, V = p.V;
  real d = degrees(F - V);
  real[] a = sequence(new real(int n){
      return (360 - d + degrees(inter[n]-F))%360;
    }, inter.length);
  real ma = a.length != 0 ? min(a) : 0, Ma= a.length != 0 ? max(a) : 0;
  return new real[] {ma, Ma};
}

/*<asyxml><function type="real[][]" signature="bangles(picture,hyperbola)"><code></asyxml>*/
real[][] bangles(picture pic = currentpicture, hyperbola h)
{/*<asyxml></code><documentation>Return the array {{ma1, Ma1}, {ma2, Ma2}} where 'maX' and 'MaX' are respectively
   the smaller and the bigger angles (from h.FX) for which the hyperbola 'h' is included
   in the bounding box of the picture 'pic'.</documentation></function></asyxml>*/
  pair bmin, bmax;
  pair[] b;
  if (h.bmin == h.bmax) {
    bmin = pic.userMin();
    bmax = pic.userMax();
  } else {
    bmin = h.bmin;bmax = h.bmax;
  }
  if(bmin.x == bmax.x || bmin.y == bmax.y || !finite(abs(bmin)) || !finite(abs(bmax)))
    return new real[][] {{0, 0}, {0, 0}};
  b[0] = bmin;
  b[1] = (bmax.x, bmin.y);
  b[2] = bmax;
  b[3] = (bmin.x, bmax.y);
  real[] eq = changecoordsys(defaultcoordsys, equation(h)).a;
  pair[] inter0, inter1;
  pair C = locate(h.C);
  pair F1 = h.F1;
  for (int i = 0; i < 4; ++i) {
    pair[] tmp = intersectionpoints(b[i], b[(i + 1)%4], eq);
    for (int j = 0; j < tmp.length; ++j) {
      if(dot(b[i]-tmp[j], b[(i + 1)%4]-tmp[j]) <= epsgeo) {
        if(dot(F1 - C, tmp[j]-C) > 0) inter0.push(tmp[j]);
        else inter1.push(tmp[j]);
      }
    }
  }
  real d = degrees(F1 - C);
  real[] ma, Ma;
  pair[][] inter = new pair[][] {inter0, inter1};
  for (int i = 0; i < 2; ++i) {
    real[] a = sequence(new real(int n){
        return (360 - d + degrees(inter[i][n]-F1))%360;
      }, inter[i].length);
    ma[i] = a.length != 0 ? min(a) : 0;
    Ma[i] = a.length != 0 ? max(a) : 0;
  }
  return new real[][] {{ma[0], Ma[0]}, {ma[1], Ma[1]}};
}

/*<asyxml><operator type = "path" signature="cast(parabola)"><code></asyxml>*/
path operator cast(parabola p)
{/*<asyxml></code><documentation>Cast parabola to path.
   If possible, the returned path is restricted to the actual bounding box
   of the current picture if the variables 'p.bmin' and 'p.bmax' are not set else
   the bounding box of box(p.bmin, p.bmax) is used instead.</documentation></operator></asyxml>*/
  real[] bangles = bangles(p);
  int n = parabolanodesnumber(p, bangles[0], bangles[1]);
  return arcfromfocus(p, bangles[0], bangles[1], n, CCW);
}


/*<asyxml><function type="void" signature="draw(picture,Label,circle,align,pen,arrowbar,arrowbar,margin,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "", circle c,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None, arrowbar bar = None,
          margin margin = NoMargin, Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  if(degenerate(c)) draw(pic, L, c.l, align, p, arrow, legend, marker);
  else draw(pic, L, (path)c, align, p, arrow, bar, margin, legend, marker);
}

/*<asyxml><function type="void" signature="draw(picture,Label,ellipse,align,pen,arrowbar,arrowbar,margin,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "", ellipse el,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None, arrowbar bar = None,
          margin margin = NoMargin, Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation></documentation>Draw the ellipse 'el' if it is not degenerated else draw 'el.l'.</function></asyxml>*/
  if(degenerate(el)) draw(pic, L, el.l, align, p, arrow, legend, marker);
  else draw(pic, L, (path)el, align, p, arrow, bar, margin, legend, marker);
}

/*<asyxml><function type="void" signature="draw(picture,Label,parabola,align,pen,arrowbar,arrowbar,margin,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "", parabola parabola,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None, arrowbar bar = None,
          margin margin = NoMargin, Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation>Draw the parabola 'p' on 'pic' without (if possible) altering the
   size of picture pic.</documentation></function></asyxml>*/
  pic.add(new void (frame f, transform t, transform T, pair m, pair M) {
      // Reduce the bounds by the size of the pen and the margins.
      m -= min(p); M -= max(p);
      parabola.bmin = inverse(t) * m;
      parabola.bmax = inverse(t) * M;
      picture tmp;
      path pp = t * ((path) (T * parabola));

      if (pp != nullpath) {
        draw(tmp, L, pp, align, p, arrow, bar, NoMargin, legend, marker);
        add(f, tmp.fit());
      }
    }, true);

  pair m = pic.userMin(), M = pic.userMax();
  if(m != M) {
    pic.addBox(truepoint(SW), truepoint(NE));
  }
}

/*<asyxml><operator type = "path" signature="cast(hyperbola)"><code></asyxml>*/
path operator cast(hyperbola h)
{/*<asyxml></code><documentation>Cast hyperbola to path.
   If possible, the returned path is restricted to the actual bounding box
   of the current picture unless the variables 'h.bmin' and 'h.bmax'
   are set; in this case the bounding box of box(h.bmin, h.bmax) is used instead.
   Only the branch on the side of 'h.F1' is considered.</documentation></operator></asyxml>*/
  real[][] bangles = bangles(h);
  int n = hyperbolanodesnumber(h, bangles[0][0], bangles[0][1]);
  return arcfromfocus(h, bangles[0][0], bangles[0][1], n, CCW);
}

/*<asyxml><function type="void" signature="draw(picture,Label,hyperbola,align,pen,arrowbar,arrowbar,margin,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "", hyperbola h,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None, arrowbar bar = None,
          margin margin = NoMargin, Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation>Draw the hyperbola 'h' on 'pic' without (if possible) altering the
   size of the picture pic.</documentation></function></asyxml>*/
  pic.add(new void (frame f, transform t, transform T, pair m, pair M) {
      // Reduce the bounds by the size of the pen and the margins.
      m -= min(p); M -= max(p);
      h.bmin = inverse(t) * m;
      h.bmax = inverse(t) * M;
      path hp;

      picture tmp;
      hp = t * ((path) (T * h));
      if (hp != nullpath) {
        draw(tmp, L, hp, align, p, arrow, bar, NoMargin, legend, marker);
      }

      hyperbola ht = hyperbola(h.F2, h.F1, h.a);
      ht.bmin = h.bmin;
      ht.bmax = h.bmax;

      hp = t * ((path) (T * ht));
      if (hp != nullpath) {
        draw(tmp, "", hp, align, p, arrow, bar, NoMargin, marker);
      }

      add(f, tmp.fit());
    }, true);

  pair m = pic.userMin(), M = pic.userMax();
  if(m != M)
    pic.addBox(truepoint(SW), truepoint(NE));
}

/*<asyxml><function type="void" signature="draw(picture,Label,explicit conic,align,pen,arrowbar,arrowbar,margin,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "", explicit conic co,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None, arrowbar bar = None,
          margin margin = NoMargin, Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation>Use one of the routine 'draw(ellipse, ...)',
   'draw(parabola, ...)' or 'draw(hyperbola, ...)' depending of the value of eccentricity of 'co'.</documentation></function></asyxml>*/
  if(co.e == 0)
    draw(pic, L, (circle)co, align, p, arrow, bar, margin, legend, marker);
  else
    if(co.e < 1) draw(pic, L, (ellipse)co, align, p, arrow, bar, margin, legend, marker);
    else
      if(co.e == 1) draw(pic, L, (parabola)co, align, p, arrow, bar, margin, legend, marker);
      else
        if(co.e > 1) draw(pic, L, (hyperbola)co, align, p, arrow, bar, margin, legend, marker);
        else abort("draw: unknown conic.");
}

/*<asyxml><function type="int" signature="conicnodesnumber(conic,real,real)"><code></asyxml>*/
int conicnodesnumber(conic co, real angle1, real angle2, bool dir = CCW)
{/*<asyxml></code><documentation>Return the number of node to draw a conic arc.</documentation></function></asyxml>*/
  int oi;
  if(co.e == 0) {
    circle c = (circle)co;
    oi = circlenodesnumber(c.r, angle1, angle2);
  } else if(co.e < 1) {
    ellipse el = (ellipse)co;
    oi = ellipsenodesnumber(el.a, el.b, angle1, angle2, dir);
  } else if(co.e == 1) {
    parabola p = (parabola)co;
    oi = parabolanodesnumber(p, angle1, angle2);
  } else {
    hyperbola h = (hyperbola)co;
    oi = hyperbolanodesnumber(h, angle1, angle2);
  }
  return oi;
}

/*<asyxml><operator type = "path" signature="cast(conic)"><code></asyxml>*/
path operator cast(conic co)
{/*<asyxml></code><documentation>Cast conic section to path.</documentation></operator></asyxml>*/
  if(co.e < 1) return (path)((ellipse)co);
  if(co.e == 1) return (path)((parabola)co);
  return (path)((hyperbola)co);
}

/*<asyxml><function type="bqe" signature="equation(explicit conic)"><code></asyxml>*/
bqe equation(explicit conic co)
{/*<asyxml></code><documentation>Return the coefficients of the equation of conic section in its coordinate system:
   bqe.a[0] * x^2 + bqe.a[1] * x * y + bqe.a[2] * y^2 + bqe.a[3] * x + bqe.a[4] * y + bqe.a[5] = 0.
   One can change the coordinate system of 'bqe' using the routine 'changecoordsys'.</documentation></function></asyxml>*/
  bqe obqe;
  if(co.e == 0)
    obqe = equation((circle)co);
  else
    if(co.e < 1) obqe = equation((ellipse)co);
    else
      if(co.e == 1) obqe = equation((parabola)co);
      else
        if(co.e > 1) obqe = equation((hyperbola)co);
        else abort("draw: unknown conic.");
  return obqe;
}

/*<asyxml><function type="string" signature="conictype(bqe)"><code></asyxml>*/
string conictype(bqe bqe)
{/*<asyxml></code><documentation>Returned values are "ellipse" or "parabola" or "hyperbola"
   depending of the conic section represented by 'bqe'.</documentation></function></asyxml>*/
  bqe lbqe = changecoordsys(defaultcoordsys, bqe);
  string os = "degenerated";
  real a = lbqe.a[0], b = lbqe.a[1]/2, c = lbqe.a[2], d = lbqe.a[3]/2, f = lbqe.a[4]/2, g = lbqe.a[5];
  real delta = a * c * g + b * f * d + d * b * f - (b^2 * g + d^2 * c + f^2 * a);
  if(abs(delta) < 10 * epsgeo) return os;
  real J = a * c - b^2;
  real I = a + c;
  if(J > epsgeo) {
    if(delta/I < -epsgeo);
    os = "ellipse";
  } else {
    if(abs(J) < epsgeo) os = "parabola"; else os = "hyperbola";
  }
  return os;
}

/*<asyxml><function type="conic" signature="conic(point,point,point,point,point)"><code></asyxml>*/
conic conic(point M1, point M2, point M3, point M4, point M5)
{/*<asyxml></code><documentation>Return the conic passing through 'M1', 'M2', 'M3', 'M4' and 'M5' if the conic is not degenerated.</documentation></function></asyxml>*/
  bqe bqe = bqe(M1, M2, M3, M4, M5);
  string ct = conictype(bqe);
  if(ct == "degenerated") abort("conic: degenerated conic passing through five points.");
  if(ct == "ellipse") return ellipse(bqe);
  if(ct == "parabola") return parabola(bqe);
  return hyperbola(bqe);
}

/*<asyxml><function type="coordsys" signature="canonicalcartesiansystem(hyperbola)"><code></asyxml>*/
coordsys canonicalcartesiansystem(explicit conic co)
{/*<asyxml></code><documentation>Return the canonical cartesian system of the conic 'co'.</documentation></function></asyxml>*/
  if(co.e < 1) return canonicalcartesiansystem((ellipse)co);
  else if(co.e == 1) return canonicalcartesiansystem((parabola)co);
  return canonicalcartesiansystem((hyperbola)co);
}

/*<asyxml><function type="bqe" signature="canonical(bqe)"><code></asyxml>*/
bqe canonical(bqe bqe)
{/*<asyxml></code><documentation>Return the bivariate quadratic equation relative to the
   canonical coordinate system of the conic section represented by 'bqe'.</documentation></function></asyxml>*/
  string type = conictype(bqe);
  if(type == "") abort("canonical: the equation can not be performed.");
  bqe obqe;
  if(type == "ellipse") {
    ellipse el = ellipse(bqe);
    obqe = changecoordsys(canonicalcartesiansystem(el), equation(el));
  } else {
    if(type == "parabola") {
      parabola p = parabola(bqe);
      obqe = changecoordsys(canonicalcartesiansystem(p), equation(p));
    } else {
      hyperbola h = hyperbola(bqe);
      obqe = changecoordsys(canonicalcartesiansystem(h), equation(h));
    }
  }
  return obqe;
}

/*<asyxml><function type="conic" signature="conic(bqe)"><code></asyxml>*/
conic conic(bqe bqe)
{/*<asyxml></code><documentation>Return the conic section represented by the bivariate quartic equation 'bqe'.</documentation></function></asyxml>*/
  string type = conictype(bqe);
  if(type == "") abort("canonical: the equation can not be performed.");
  conic oc;
  if(type == "ellipse") {
    oc = ellipse(bqe);
  } else {
    if(type == "parabola") oc = parabola(bqe); else oc = hyperbola(bqe);
  }
  return oc;
}

/*<asyxml><function type="real" signature="arclength(circle)"><code></asyxml>*/
real arclength(circle c)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return c.r * 2 * pi;
}

/*<asyxml><function type="real" signature="focusToCenter(ellipse,real)"><code></asyxml>*/
real focusToCenter(ellipse el, real a)
{/*<asyxml></code><documentation>Return the angle relatively to the center of 'el' for the angle 'a'
   given relatively to the focus of 'el'.</documentation></function></asyxml>*/
  pair p = point(fromFocus(el, a, a, 1, CCW), 0);
  pair c = locate(el.C);
  real d = degrees(p - c) - el.angle;
  d = abs(d) < epsgeo ? 0 : d; // Avoid -1e-15
  return d%(sgnd(a) * 360);
}

/*<asyxml><function type="real" signature="centerToFocus(ellipse,real)"><code></asyxml>*/
real centerToFocus(ellipse el, real a)
{/*<asyxml></code><documentation>Return the angle relatively to the focus of 'el' for the angle 'a'
   given relatively to the center of 'el'.</documentation></function></asyxml>*/
  pair P = point(fromCenter(el, a, a, 1, CCW), 0);
  pair F1 = locate(el.F1);
  pair F2 = locate(el.F2);
  real d = degrees(P - F1) - degrees(F2 - F1);
  d = abs(d) < epsgeo ? 0 : d; // Avoid -1e-15
  return d%(sgnd(a) * 360);
}

/*<asyxml><function type="real" signature="arclength(ellipse)"><code></asyxml>*/
real arclength(ellipse el)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return degenerate(el) ? infinity : 4 * el.a * elle(pi/2, el.e);
}

/*<asyxml><function type="real" signature="arclength(ellipse,real,real,bool,polarconicroutine)"><code></asyxml>*/
real arclength(ellipse el, real angle1, real angle2,
               bool direction = CCW,
               polarconicroutine polarconicroutine = currentpolarconicroutine)
{/*<asyxml></code><documentation>Return the length of the arc of the ellipse between 'angle1'
   and 'angle2'.
   'angle1' and 'angle2' must be in the interval ]-360;+oo[ if polarconicroutine = fromFocus,
   ]-oo;+oo[ if polarconicroutine = fromCenter.</documentation></function></asyxml>*/
  if(degenerate(el)) return infinity;
  if(angle1 > angle2) return arclength(el, angle2, angle1, !direction, polarconicroutine);
  //   path g;int n = 1000;
  //   if(el.e == 0) g = arcfromcenter(el, angle1, angle2, n, direction);
  //   if(el.e != 1) g = polarconicroutine(el, angle1, angle2, n, direction);
  //   write("with path = ", arclength(g));
  if(polarconicroutine == fromFocus) {
    //   dot(point(fromFocus(el, angle1, angle1, 1, CCW), 0), 2mm + blue);
    //   dot(point(fromFocus(el, angle2, angle2, 1, CCW), 0), 2mm + blue);
    //   write("fromfocus1 = ", angle1);
    //   write("fromfocus2 = ", angle2);
    real gle1 = focusToCenter(el, angle1);
    real gle2 = focusToCenter(el, angle2);
    if((gle1 - gle2) * (angle1 - angle2) > 0) {
      angle1 = gle1; angle2 = gle2;
    } else {
      angle1 = gle2; angle2 = gle1;
    }
    //   dot(point(fromCenter(el, angle1, angle1, 1, CCW), 0), 1mm + red);
    //   dot(point(fromCenter(el, angle2, angle2, 1, CCW), 0), 1mm + red);
    //   write("fromcenter1 = ", angle1);
    //   write("fromcenter2 = ", angle2);
  }
  if(angle1 < 0 || angle2 < 0) return arclength(el, 180 + angle1, 180 + angle2, direction, fromCenter);
  real a1 = direction ? angle1 : angle2;
  real a2 = direction ? angle2 : angle1 + 360;
  real elleq = el.a * elle(pi/2, el.e);
  real S(real a)
  {//Return the arclength from 0 to the angle 'a' (in degrees)
    // given form the center of the ellipse.
    real gle = atan(el.a * tan(radians(a))/el.b)+
      pi * (((a%90 == 0 && a != 0) ? floor(a/90) - 1 : floor(a/90)) -
          ((a%180 == 0) ? 0 : floor(a/180)) -
          (a%360 == 0 ? floor(a/(360)) : 0));
    /* // Uncomment to visualize the used branches
       unitsize(2cm, 1cm);
       import graph;

       real xmin = 0, xmax = 3pi;

       xlimits( xmin, xmax);
       ylimits( 0, 10);
       yaxis( "y" , LeftRight(), RightTicks(pTick=.8red, ptick = lightgrey, extend = true));
       xaxis( "x - value", BottomTop(), Ticks(Label("$%.2f$", red), Step = pi/2, step = pi/4, pTick=.8red, ptick = lightgrey, extend = true));

       real p2 = pi/2;
       real f(real t)
       {
       return atan(0.6 * tan(t))+
       pi * ((t%p2 == 0 && t != 0) ? floor(t/p2) - 1 : floor(t/p2)) -
       ((t%pi == 0) ? 0 : pi * floor(t/pi)) - (t%(2pi) == 0 ? pi * floor(t/(2 * pi)) : 0);
       }

       draw(graph(f, xmin, xmax, 100));
       write(degrees(f(pi/2)));
       write(degrees(f(pi)));
       write(degrees(f(3pi/2)));
       write(degrees(f(2pi)));
       draw(graph(new real(real t){return t;}, xmin, xmax, 3));
    */
    return elleq - el.a * elle(pi/2 - gle, el.e);
  }
  return S(a2) - S(a1);
}

/*<asyxml><function type="real" signature="arclength(parabola,real)"><code></asyxml>*/
real arclength(parabola p, real angle)
{/*<asyxml></code><documentation>Return the arclength from 180 to 'angle' given from focus in the
   canonical coordinate system of 'p'.</documentation></function></asyxml>*/
  real a = p.a; /* In canonicalcartesiansystem(p) the equation of p
                 is x = y^2/(4a) */
  // integrate(sqrt(1 + (x/(2 * a))^2), x);
  real S(real t){return 0.5 * t * sqrt(1 + t^2/(4 * a^2)) + a * asinh(t/(2 * a));}
  real R(real gle){return 2 * a/(1 - Cos(gle));}
  real t = Sin(angle) * R(angle);
  return S(t);
}

/*<asyxml><function type="real" signature="arclength(parabola,real,real)"><code></asyxml>*/
real arclength(parabola p, real angle1, real angle2)
{/*<asyxml></code><documentation>Return the arclength from 'angle1' to 'angle2' given from
   focus in the canonical coordinate system of 'p'</documentation></function></asyxml>*/
  return arclength(p, angle1) - arclength(p, angle2);
}

/*<asyxml><function type="real" signature="arclength(parabola p)"><code></asyxml>*/
real arclength(parabola p)
{/*<asyxml></code><documentation>Return the length of the arc of the parabola bounded to the bounding
   box of the current picture.</documentation></function></asyxml>*/
  real[] b = bangles(p);
  return arclength(p, b[0], b[1]);
}
// *........................CONICS.........................*
// *=======================================================*

// *=======================================================*
// *.......................ABSCISSA........................*
/*<asyxml><struct signature="abscissa"><code></asyxml>*/
struct abscissa
{/*<asyxml></code><documentation>Provide abscissa structure on a curve used in the routine-like 'point(object, abscissa)'
   where object can be 'line','segment','ellipse','circle','conic'...</documentation><property type = "real" signature="x"><code></asyxml>*/
  real x;/*<asyxml></code><documentation>The abscissa value.</documentation></property><property type = "int" signature="system"><code></asyxml>*/
  int system;/*<asyxml></code><documentation>0 = relativesystem; 1 = curvilinearsystem; 2 = angularsystem; 3 = nodesystem</documentation></property><property type = "polarconicroutine" signature="polarconicroutine"><code></asyxml>*/
  polarconicroutine polarconicroutine = fromCenter;/*<asyxml></code><documentation>The routine used with angular system and two foci conic section.
                                                   Possible values are 'formCenter' and 'formFocus'.</documentation></property></asyxml>*/
  /*<asyxml><method type = "abscissa" signature="copy()"><code></asyxml>*/
  abscissa copy()
  {/*<asyxml></code><documentation>Return a copy of this abscissa.</documentation></method></asyxml>*/
    abscissa oa = new abscissa;
    oa.x = this.x;
    oa.system = this.system;
    oa.polarconicroutine = this.polarconicroutine;
    return oa;
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><constant type = "int" signature="relativesystem,curvilinearsystem,angularsystem,nodesystem"><code></asyxml>*/
restricted int relativesystem = 0, curvilinearsystem = 1, angularsystem = 2, nodesystem = 3;/*<asyxml></code><documentation>Constant used to set the abscissa system.</documentation></constant></asyxml>*/

/*<asyxml><operator type = "abscissa" signature="cast(explicit position)"><code></asyxml>*/
abscissa operator cast(explicit position position)
{/*<asyxml></code><documentation>Cast position to abscissa.
   If 'position' is relative, the abscissa is relative else it's a curvilinear abscissa.</documentation></operator></asyxml>*/
  abscissa oarcc;
  oarcc.x = position.position.x;
  oarcc.system = position.relative ? relativesystem : curvilinearsystem;
  return oarcc;
}

/*<asyxml><operator type = "abscissa" signature="+(real,explicit abscissa)"><code></asyxml>*/
abscissa operator +(real x, explicit abscissa a)
{/*<asyxml></code><documentation>Provide 'real + abscissa'.
   Return abscissa b so that b.x = a.x + x.
   +(explicit abscissa, real), -(real, explicit abscissa) and -(explicit abscissa, real) are also defined.</documentation></operator></asyxml>*/
  abscissa oa = a.copy();
  oa.x = a.x + x;
  return oa;
}

abscissa operator +(explicit abscissa a, real x)
{
  return x + a;
}
abscissa operator +(int x, explicit abscissa a)
{
  return ((real)x) + a;
}

/*<asyxml><operator type = "abscissa" signature="-(explicit abscissa a)"><code></asyxml>*/
abscissa operator -(explicit abscissa a)
{/*<asyxml></code><documentation>Return the abscissa b so that b.x = -a.x.</documentation></operator></asyxml>*/
  abscissa oa;
  oa.system = a.system;
  oa.x = -a.x;
  return oa;
}

abscissa operator -(real x, explicit abscissa a)
{
  abscissa oa;
  oa.system = a.system;
  oa.x = x - a.x;
  return oa;
}
abscissa operator -(explicit abscissa a, real x)
{
  abscissa oa;
  oa.system = a.system;
  oa.x = a.x - x;
  return oa;
}
abscissa operator -(int x, explicit abscissa a)
{
  return ((real)x) - a;
}

/*<asyxml><operator type = "abscissa" signature="*(real,abscissa)"><code></asyxml>*/
abscissa operator *(real x, explicit abscissa a)
{/*<asyxml></code><documentation>Provide 'real * abscissa'.
   Return abscissa b so that b.x = x * a.x.
   *(explicit abscissa, real), /(real, explicit abscissa) and /(explicit abscissa, real) are also defined.</documentation></operator></asyxml>*/
  abscissa oa;
  oa.system = a.system;
  oa.x = a.x * x;
  return oa;
}
abscissa operator *(explicit abscissa a, real x)
{
  return x * a;
}

abscissa operator /(real x, explicit abscissa a)
{
  abscissa oa;
  oa.system = a.system;
  oa.x = x/a.x;
  return oa;
}
abscissa operator /(explicit abscissa a, real x)
{
  abscissa oa;
  oa.system = a.system;
  oa.x = a.x/x;
  return oa;
}

abscissa operator /(int x, explicit abscissa a)
{
  return ((real)x)/a;
}

/*<asyxml><function type="abscissa" signature="relabscissa(real)"><code></asyxml>*/
abscissa relabscissa(real x)
{/*<asyxml></code><documentation>Return a relative abscissa.</documentation></function></asyxml>*/
  return (abscissa)(Relative(x));
}
abscissa relabscissa(int x)
{
  return (abscissa)(Relative(x));
}

/*<asyxml><function type="abscissa" signature="curabscissa(real)"><code></asyxml>*/
abscissa curabscissa(real x)
{/*<asyxml></code><documentation>Return a curvilinear abscissa.</documentation></function></asyxml>*/
  return (abscissa)((position)x);
}
abscissa curabscissa(int x)
{
  return (abscissa)((position)x);
}

/*<asyxml><function type="abscissa" signature="angabscissa(real,polarconicroutine)"><code></asyxml>*/
abscissa angabscissa(real x, polarconicroutine polarconicroutine = currentpolarconicroutine)
{/*<asyxml></code><documentation>Return a angular abscissa.</documentation></function></asyxml>*/
  abscissa oarcc;
  oarcc.x = x;
  oarcc.polarconicroutine = polarconicroutine;
  oarcc.system = angularsystem;
  return oarcc;
}
abscissa angabscissa(int x, polarconicroutine polarconicroutine = currentpolarconicroutine)
{
  return angabscissa((real)x, polarconicroutine);
}

/*<asyxml><function type="abscissa" signature="nodabscissa(real)"><code></asyxml>*/
abscissa nodabscissa(real x)
{/*<asyxml></code><documentation>Return an abscissa as time on the path.</documentation></function></asyxml>*/
  abscissa oarcc;
  oarcc.x = x;
  oarcc.system = nodesystem;
  return oarcc;
}
abscissa nodabscissa(int x)
{
  return nodabscissa((real)x);
}

/*<asyxml><operator type = "abscissa" signature="cast(real)"><code></asyxml>*/
abscissa operator cast(real x)
{/*<asyxml></code><documentation>Cast real to abscissa, precisely 'nodabscissa'.</documentation></operator></asyxml>*/
  return nodabscissa(x);
}
abscissa operator cast(int x)
{
  return nodabscissa((real)x);
}

/*<asyxml><function type="point" signature="point(circle,abscissa)"><code></asyxml>*/
point point(circle c, abscissa l)
{/*<asyxml></code><documentation>Return the point of 'c' which has the abscissa 'l.x'
   according to the abscissa system 'l.system'.</documentation></function></asyxml>*/
  coordsys R = c.C.coordsys;
  if (l.system == nodesystem)
    return point(R, point((path)c, l.x)/R);
  if (l.system == relativesystem)
    return c.C + point(R, R.polar(c.r, 2 * pi * l.x));
  if (l.system == curvilinearsystem)
    return c.C + point(R, R.polar(c.r, l.x/c.r));
  if (l.system == angularsystem)
    return c.C + point(R, R.polar(c.r, radians(l.x)));
  abort("point: bad abscissa system.");
  return (0, 0);
}

/*<asyxml><function type="point" signature="point(ellipse,abscissa)"><code></asyxml>*/
point point(ellipse el, abscissa l)
{/*<asyxml></code><documentation>Return the point of 'el' which has the abscissa 'l.x'
   according to the abscissa system 'l.system'.</documentation></function></asyxml>*/
  if(el.e == 0) return point((circle)el, l);
  coordsys R = coordsys(el);
  if (l.system == nodesystem)
    return point(R, point((path)el, l.x)/R);
  if (l.system == relativesystem) {
    return point(el, curabscissa((l.x%1) * arclength(el)));
  }
  if (l.system == curvilinearsystem) {
    real a1 = 0, a2 = 360, cx = 0;
    real aout = a1;
    real x = abs(l.x)%arclength(el);
    while (abs(cx - x) > epsgeo) {
      aout = (a1 + a2)/2;
      cx = arclength(el, 0, aout, CCW, fromCenter); //fromCenter is speeder
      if(cx > x) a2 = (a1 + a2)/2; else a1 = (a1 + a2)/2;
    }
    path pel = fromCenter(el, sgn(l.x) * aout, sgn(l.x) * aout, 1, CCW);
    return point(R, point(pel, 0)/R);
  }
  if (l.system == angularsystem) {
    return point(R, point(l.polarconicroutine(el, l.x, l.x, 1, CCW), 0)/R);
  }
  abort("point: bad abscissa system.");
  return (0, 0);
}

/*<asyxml><function type="point" signature="point(parabola,abscissa)"><code></asyxml>*/
point point(parabola p, abscissa l)
{/*<asyxml></code><documentation>Return the point of 'p' which has the abscissa 'l.x'
   according to the abscissa system 'l.system'.</documentation></function></asyxml>*/
  coordsys R = coordsys(p);
  if (l.system == nodesystem)
    return point(R, point((path)p, l.x)/R);
  if (l.system == relativesystem) {
    real[] b = bangles(p);
    real al = sgn(l.x) > 0 ? arclength(p, 180, b[1]) : arclength(p, 180, b[0]);
    return point(p, curabscissa(abs(l.x) * al));
  }
  if (l.system == curvilinearsystem) {
    real a1 = 1e-3, a2 = 360 - 1e-3, cx = infinity;
    while (abs(cx - l.x) > epsgeo) {
      cx = arclength(p, 180, (a1 + a2)/2);
      if(cx > l.x) a2 = (a1 + a2)/2; else a1 = (a1 + a2)/2;
    }
    path pp = fromFocus(p, a1, a1, 1, CCW);
    return point(R, point(pp, 0)/R);
  }
  if (l.system == angularsystem) {
    return point(R, point(fromFocus(p, l.x, l.x, 1, CCW), 0)/R);
  }
  abort("point: bad abscissa system.");
  return (0, 0);
}

/*<asyxml><function type="point" signature="point(hyperbola,abscissa)"><code></asyxml>*/
point point(hyperbola h, abscissa l)
{/*<asyxml></code><documentation>Return the point of 'h' which has the abscissa 'l.x'
   according to the abscissa system 'l.system'.</documentation></function></asyxml>*/
  coordsys R = coordsys(h);
  if (l.system == nodesystem)
    return point(R, point((path)h, l.x)/R);
  if (l.system == relativesystem) {
    abort("point(hyperbola, relativeSystem) is not implemented...
Try relpoint((path)your_hyperbola, x);");
  }
  if (l.system == curvilinearsystem) {
    abort("point(hyperbola, curvilinearSystem) is not implemented...");
  }
  if (l.system == angularsystem) {
    return point(R, point(l.polarconicroutine(h, l.x, l.x, 1, CCW), 0)/R);
  }
  abort("point: bad abscissa system.");
  return (0, 0);
}

/*<asyxml><function type="abscissa" signature="point(conic,point)"><code></asyxml>*/
point point(explicit conic co, abscissa l)
{/*<asyxml></code><documentation>Return the curvilinear abscissa of 'M' on the conic 'co'.</documentation></function></asyxml>*/
  if(co.e == 0) return point((circle)co, l);
  if(co.e < 1) return point((ellipse)co, l);
  if(co.e == 1) return point((parabola)co, l);
  return point((hyperbola)co, l);
}


/*<asyxml><function type="point" signature="point(line,abscissa)"><code></asyxml>*/
point point(line l, abscissa x)
{/*<asyxml></code><documentation>Return the point of 'l' which has the abscissa 'l.x' according to the abscissa system 'l.system'.
   Note that the origin is l.A, and point(l, relabscissa(x)) returns l.A + x.x * vector(l.B - l.A).</documentation></function></asyxml>*/
  coordsys R = l.A.coordsys;
  if (x.system == nodesystem)
    return l.A + (x.x < 0 ? 0 : x.x > 1 ? 1 : x.x) * vector(l.B - l.A);
  if (x.system == relativesystem)
    return l.A + x.x * vector(l.B - l.A);
  if (x.system == curvilinearsystem)
    return l.A + x.x * l.u;
  if (x.system == angularsystem)
    abort("point: what the meaning of angular abscissa on line ?.");
  abort("point: bad abscissa system.");
  return (0, 0);
}

/*<asyxml><function type="point" signature="point(line,real)"><code></asyxml>*/
point point(line l, explicit real x)
{/*<asyxml></code><documentation>Return the point between node l.A and l.B (x <= 0 means l.A, x >=1 means l.B).</documentation></function></asyxml>*/
  return point(l, nodabscissa(x));
}
point point(line l, explicit int x)
{
  return point(l, nodabscissa(x));
}

/*<asyxml><function type="circle" signature="point(explicit circle,explicit real)"><code></asyxml>*/
point point(explicit circle c, explicit real x)
{/*<asyxml></code><documentation>Return the point between node floor(x) and floor(x) + 1.</documentation></function></asyxml>*/
  return point(c, nodabscissa(x));
}
point point(explicit circle c, explicit int x)
{
  return point(c, nodabscissa(x));
}

/*<asyxml><function type="point" signature="point(explicit ellipse,explicit real)"><code></asyxml>*/
point point(explicit ellipse el, explicit real x)
{/*<asyxml></code><documentation>Return the point between node floor(x) and floor(x) + 1.</documentation></function></asyxml>*/
  return point(el, nodabscissa(x));
}
point point(explicit ellipse el, explicit int x)
{
  return point(el, nodabscissa(x));
}

/*<asyxml><function type="point" signature="point(explicit parabola,explicit real)"><code></asyxml>*/
point point(explicit parabola p, explicit real x)
{/*<asyxml></code><documentation>Return the point between node floor(x) and floor(x) + 1.</documentation></function></asyxml>*/
  return point(p, nodabscissa(x));
}
point point(explicit parabola p, explicit int x)
{
  return point(p, nodabscissa(x));
}

/*<asyxml><function type="point" signature="point(explicit hyperbola,explicit real)"><code></asyxml>*/
point point(explicit hyperbola h, explicit real x)
{/*<asyxml></code><documentation>Return the point between node floor(x) and floor(x) + 1.</documentation></function></asyxml>*/
  return point(h, nodabscissa(x));
}
point point(explicit hyperbola h, explicit int x)
{
  return point(h, nodabscissa(x));
}

/*<asyxml><function type="point" signature="point(explicit conic,explicit real)"><code></asyxml>*/
point point(explicit conic co, explicit real x)
{/*<asyxml></code><documentation>Return the point between node floor(x) and floor(x) + 1.</documentation></function></asyxml>*/
  point op;
  if(co.e == 0) op = point((circle)co, nodabscissa(x));
  else if(co.e < 1) op = point((ellipse)co, nodabscissa(x));
  else if(co.e == 1) op = point((parabola)co, nodabscissa(x));
  else op = point((hyperbola)co, nodabscissa(x));
  return op;
}
point point(explicit conic co, explicit int x)
{
  return point(co, (real)x);
}

/*<asyxml><function type="point" signature="relpoint(line,real)"><code></asyxml>*/
point relpoint(line l, real x)
{/*<asyxml></code><documentation>Return the relative point of 'l' (0 means l.A,
   1 means l.B, x means l.A + x * vector(l.B - l.A) ).</documentation></function></asyxml>*/
  return point(l, Relative(x));
}

/*<asyxml><function type="point" signature="relpoint(explicit circle,real)"><code></asyxml>*/
point relpoint(explicit circle c, real x)
{/*<asyxml></code><documentation>Return the relative point of 'c' (0 means origin, 1 means end).
   Origin is c.center + c.r * (1, 0).</documentation></function></asyxml>*/
  return point(c, Relative(x));
}

/*<asyxml><function type="point" signature="relpoint(explicit ellipse,real)"><code></asyxml>*/
point relpoint(explicit ellipse el, real x)
{/*<asyxml></code><documentation>Return the relative point of 'el' (0 means origin, 1 means end).</documentation></function></asyxml>*/
  return point(el, Relative(x));
}

/*<asyxml><function type="point" signature="relpoint(explicit parabola,real)"><code></asyxml>*/
point relpoint(explicit parabola p, real x)
{/*<asyxml></code><documentation>Return the relative point of the path of the parabola
   bounded by the bounding box of the current picture.
   0 means origin, 1 means end, where the origin is the vertex of 'p'.</documentation></function></asyxml>*/
  return point(p, Relative(x));
}

/*<asyxml><function type="point" signature="relpoint(explicit hyperbola,real)"><code></asyxml>*/
point relpoint(explicit hyperbola h, real x)
{/*<asyxml></code><documentation>Not yet implemented... <look href = "point(hyperbola, abscissa)"/></documentation></function></asyxml>*/
  return point(h, Relative(x));
}

/*<asyxml><function type="point" signature="relpoint(explicit conic,explicit real)"><code></asyxml>*/
point relpoint(explicit conic co, explicit real x)
{/*<asyxml></code><documentation>Return the relative point of 'co' (0 means origin, 1 means end).</documentation></function></asyxml>*/
  point op;
  if(co.e == 0) op = point((circle)co, Relative(x));
  else if(co.e < 1) op = point((ellipse)co, Relative(x));
  else if(co.e == 1) op = point((parabola)co, Relative(x));
  else op = point((hyperbola)co, Relative(x));
  return op;
}
point relpoint(explicit conic co, explicit int x)
{
  return relpoint(co, (real)x);
}

/*<asyxml><function type="point" signature="angpoint(explicit circle,real)"><code></asyxml>*/
point angpoint(explicit circle c, real x)
{/*<asyxml></code><documentation>Return the point of 'c' in the direction 'x' measured in degrees.</documentation></function></asyxml>*/
  return point(c, angabscissa(x));
}

/*<asyxml><function type="point" signature="angpoint(explicit ellipse,real,polarconicroutine)"><code></asyxml>*/
point angpoint(explicit ellipse el, real x,
               polarconicroutine polarconicroutine = currentpolarconicroutine)
{/*<asyxml></code><documentation>Return the point of 'el' in the direction 'x'
   measured in degrees according to 'polarconicroutine'.</documentation></function></asyxml>*/
  return el.e == 0 ? angpoint((circle) el, x) : point(el, angabscissa(x, polarconicroutine));
}

/*<asyxml><function type="point" signature="angpoint(explicit parabola,real)"><code></asyxml>*/
point angpoint(explicit parabola p, real x)
{/*<asyxml></code><documentation>Return the point of 'p' in the direction 'x' measured in degrees.</documentation></function></asyxml>*/
  return point(p, angabscissa(x));
}

/*<asyxml><function type="point" signature="angpoint(explicit hyperbola,real,polarconicroutine)"><code></asyxml>*/
point angpoint(explicit hyperbola h, real x,
               polarconicroutine polarconicroutine = currentpolarconicroutine)
{/*<asyxml></code><documentation>Return the point of 'h' in the direction 'x'
   measured in degrees according to 'polarconicroutine'.</documentation></function></asyxml>*/
  return point(h, angabscissa(x, polarconicroutine));
}

/*<asyxml><function type="point" signature="curpoint(line,real)"><code></asyxml>*/
point curpoint(line l, real x)
{/*<asyxml></code><documentation>Return the point of 'l' which has the curvilinear abscissa 'x'.
   Origin is l.A.</documentation></function></asyxml>*/
  return point(l, curabscissa(x));
}

/*<asyxml><function type="point" signature="curpoint(explicit circle,real)"><code></asyxml>*/
point curpoint(explicit circle c, real x)
{/*<asyxml></code><documentation>Return the point of 'c' which has the curvilinear abscissa 'x'.
   Origin is c.center + c.r * (1, 0).</documentation></function></asyxml>*/
  return point(c, curabscissa(x));
}

/*<asyxml><function type="point" signature="curpoint(explicit ellipse,real)"><code></asyxml>*/
point curpoint(explicit ellipse el, real x)
{/*<asyxml></code><documentation>Return the point of 'el' which has the curvilinear abscissa 'el'.</documentation></function></asyxml>*/
  return point(el, curabscissa(x));
}

/*<asyxml><function type="point" signature="curpoint(explicit parabola,real)"><code></asyxml>*/
point curpoint(explicit parabola p, real x)
{/*<asyxml></code><documentation>Return the point of 'p' which has the curvilinear abscissa 'x'.
   Origin is the vertex of 'p'.</documentation></function></asyxml>*/
  return point(p, curabscissa(x));
}

/*<asyxml><function type="point" signature="curpoint(conic,real)"><code></asyxml>*/
point curpoint(conic co, real x)
{/*<asyxml></code><documentation>Return the point of 'co' which has the curvilinear abscissa 'x'.</documentation></function></asyxml>*/
  point op;
  if(co.e == 0) op = point((circle)co, curabscissa(x));
  else if(co.e < 1) op = point((ellipse)co, curabscissa(x));
  else if(co.e == 1) op = point((parabola)co, curabscissa(x));
  else op = point((hyperbola)co, curabscissa(x));
  return op;
}

/*<asyxml><function type="abscissa" signature="angabscissa(circle,point)"><code></asyxml>*/
abscissa angabscissa(circle c, point M)
{/*<asyxml></code><documentation>Return the angular abscissa of 'M' on the circle 'c'.</documentation></function></asyxml>*/
  if(!(M @ c)) abort("angabscissa: the point is not on the circle.");
  abscissa oa;
  oa.system = angularsystem;
  oa.x = degrees(M - c.C);
  if(oa.x < 0) oa.x+=360;
  return oa;
}

/*<asyxml><function type="abscissa" signature="angabscissa(ellipse,point,polarconicroutine)"><code></asyxml>*/
abscissa angabscissa(ellipse el, point M,
                     polarconicroutine polarconicroutine = currentpolarconicroutine)
{/*<asyxml></code><documentation>Return the angular abscissa of 'M' on the ellipse 'el' according to 'polarconicroutine'.</documentation></function></asyxml>*/
  if(!(M @ el)) abort("angabscissa: the point is not on the ellipse.");
  abscissa oa;
  oa.system = angularsystem;
  oa.polarconicroutine = polarconicroutine;
  oa.x = polarconicroutine == fromCenter ? degrees(M - el.C) : degrees(M - el.F1);
  oa.x -= el.angle;
  if(oa.x < 0) oa.x += 360;
  return oa;
}

/*<asyxml><function type="abscissa" signature="angabscissa(hyperbola,point,polarconicroutine)"><code></asyxml>*/
abscissa angabscissa(hyperbola h, point M,
                     polarconicroutine polarconicroutine = currentpolarconicroutine)
{/*<asyxml></code><documentation>Return the angular abscissa of 'M' on the hyperbola 'h' according to 'polarconicroutine'.</documentation></function></asyxml>*/
  if(!(M @ h)) abort("angabscissa: the point is not on the hyperbola.");
  abscissa oa;
  oa.system = angularsystem;
  oa.polarconicroutine = polarconicroutine;
  oa.x = polarconicroutine == fromCenter ? degrees(M - h.C) : degrees(M - h.F1) + 180;
  oa.x -= h.angle;
  if(oa.x < 0) oa.x += 360;
  return oa;
}

/*<asyxml><function type="abscissa" signature="angabscissa(parabola,point)"><code></asyxml>*/
abscissa angabscissa(parabola p, point M)
{/*<asyxml></code><documentation>Return the angular abscissa of 'M' on the parabola 'p'.</documentation></function></asyxml>*/
  if(!(M @ p)) abort("angabscissa: the point is not on the parabola.");
  abscissa oa;
  oa.system = angularsystem;
  oa.polarconicroutine = fromFocus;// Not used
  oa.x = degrees(M - p.F);
  oa.x -= p.angle;
  if(oa.x < 0) oa.x += 360;
  return oa;
}

/*<asyxml><function type="abscissa" signature="angabscissa(conic,point)"><code></asyxml>*/
abscissa angabscissa(explicit conic co, point M)
{/*<asyxml></code><documentation>Return the angular abscissa of 'M' on the conic 'co'.</documentation></function></asyxml>*/
  if(co.e == 0) return angabscissa((circle)co, M);
  if(co.e < 1) return angabscissa((ellipse)co, M);
  if(co.e == 1) return angabscissa((parabola)co, M);
  return angabscissa((hyperbola)co, M);
}

/*<asyxml><function type="abscissa" signature="curabscissa(line,point)"><code></asyxml>*/
abscissa curabscissa(line l, point M)
{/*<asyxml></code><documentation>Return the curvilinear abscissa of 'M' on the line 'l'.</documentation></function></asyxml>*/
  if(!(M @ extend(l))) abort("curabscissa: the point is not on the line.");
  abscissa oa;
  oa.system = curvilinearsystem;
  oa.x = sgn(dot(M - l.A, l.B - l.A)) * abs(M - l.A);
  return oa;
}

/*<asyxml><function type="abscissa" signature="curabscissa(circle,point)"><code></asyxml>*/
abscissa curabscissa(circle c, point M)
{/*<asyxml></code><documentation>Return the curvilinear abscissa of 'M' on the circle 'c'.</documentation></function></asyxml>*/
  if(!(M @ c)) abort("curabscissa: the point is not on the circle.");
  abscissa oa;
  oa.system = curvilinearsystem;
  oa.x = pi * angabscissa(c, M).x * c.r/180;
  return oa;
}

/*<asyxml><function type="abscissa" signature="curabscissa(ellipse,point)"><code></asyxml>*/
abscissa curabscissa(ellipse el, point M)
{/*<asyxml></code><documentation>Return the curvilinear abscissa of 'M' on the ellipse 'el'.</documentation></function></asyxml>*/
  if(!(M @ el)) abort("curabscissa: the point is not on the ellipse.");
  abscissa oa;
  oa.system = curvilinearsystem;
  real a = angabscissa(el, M, fromCenter).x;
  oa.x = arclength(el, 0, a, fromCenter);
  oa.polarconicroutine = fromCenter;
  return oa;
}

/*<asyxml><function type="abscissa" signature="curabscissa(parabola,point)"><code></asyxml>*/
abscissa curabscissa(parabola p, point M)
{/*<asyxml></code><documentation>Return the curvilinear abscissa of 'M' on the parabola 'p'.</documentation></function></asyxml>*/
  if(!(M @ p)) abort("curabscissa: the point is not on the parabola.");
  abscissa oa;
  oa.system = curvilinearsystem;
  real a = angabscissa(p, M).x;
  oa.x = arclength(p, 180, a);
  oa.polarconicroutine = fromFocus; // Not used.
  return oa;
}

/*<asyxml><function type="abscissa" signature="curabscissa(conic,point)"><code></asyxml>*/
abscissa curabscissa(conic co, point M)
{/*<asyxml></code><documentation>Return the curvilinear abscissa of 'M' on the conic 'co'.</documentation></function></asyxml>*/
  if(co.e > 1) abort("curabscissa: not implemented for this hyperbola.");
  if(co.e == 0) return curabscissa((circle)co, M);
  if(co.e < 1) return curabscissa((ellipse)co, M);
  return curabscissa((parabola)co, M);
}

/*<asyxml><function type="abscissa" signature="nodabscissa(line,point)"><code></asyxml>*/
abscissa nodabscissa(line l, point M)
{/*<asyxml></code><documentation>Return the node abscissa of 'M' on the line 'l'.</documentation></function></asyxml>*/
  if(!(M @ (segment)l)) abort("nodabscissa: the point is not on the segment.");
  abscissa oa;
  oa.system = nodesystem;
  oa.x = abs(M - l.A)/abs(l.A - l.B);
  return oa;
}

/*<asyxml><function type="abscissa" signature="nodabscissa(circle,point)"><code></asyxml>*/
abscissa nodabscissa(circle c, point M)
{/*<asyxml></code><documentation>Return the node abscissa of 'M' on the circle 'c'.</documentation></function></asyxml>*/
  if(!(M @ c)) abort("nodabscissa: the point is not on the circle.");
  abscissa oa;
  oa.system = nodesystem;
  oa.x = intersect((path)c, locate(M))[0];
  return oa;
}

/*<asyxml><function type="abscissa" signature="nodabscissa(ellipse,point)"><code></asyxml>*/
abscissa nodabscissa(ellipse el, point M)
{/*<asyxml></code><documentation>Return the node abscissa of 'M' on the ellipse 'el'.</documentation></function></asyxml>*/
  if(!(M @ el)) abort("nodabscissa: the point is not on the ellipse.");
  abscissa oa;
  oa.system = nodesystem;
  oa.x = intersect((path)el, M)[0];
  return oa;
}

/*<asyxml><function type="abscissa" signature="nodabscissa(parabola,point)"><code></asyxml>*/
abscissa nodabscissa(parabola p, point M)
{/*<asyxml></code><documentation>Return the node abscissa OF 'M' on the parabola 'p'.</documentation></function></asyxml>*/
  if(!(M @ p)) abort("nodabscissa: the point is not on the parabola.");
  abscissa oa;
  oa.system = nodesystem;
  path pg = p;
  real[] t = intersect(pg, M, 1e-5);
  if(t.length == 0) abort("nodabscissa: the point is not on the path of the parabola.");
  oa.x = t[0];
  return oa;
}

/*<asyxml><function type="abscissa" signature="nodabscissa(conic,point)"><code></asyxml>*/
abscissa nodabscissa(conic co, point M)
{/*<asyxml></code><documentation>Return the node abscissa of 'M' on the conic 'co'.</documentation></function></asyxml>*/
  if(co.e > 1) abort("nodabscissa: not implemented for hyperbola.");
  if(co.e == 0) return nodabscissa((circle)co, M);
  if(co.e < 1) return nodabscissa((ellipse)co, M);
  return nodabscissa((parabola)co, M);
}


/*<asyxml><function type="abscissa" signature="relabscissa(line,point)"><code></asyxml>*/
abscissa relabscissa(line l, point M)
{/*<asyxml></code><documentation>Return the relative abscissa of 'M' on the line 'l'.</documentation></function></asyxml>*/
  if(!(M @ extend(l))) abort("relabscissa: the point is not on the line.");
  abscissa oa;
  oa.system = relativesystem;
  oa.x = sgn(dot(M - l.A, l.B - l.A)) * abs(M - l.A)/abs(l.A - l.B);
  return oa;
}

/*<asyxml><function type="abscissa" signature="relabscissa(circle,point)"><code></asyxml>*/
abscissa relabscissa(circle c, point M)
{/*<asyxml></code><documentation>Return the relative abscissa of 'M' on the circle 'c'.</documentation></function></asyxml>*/
  if(!(M @ c)) abort("relabscissa: the point is not on the circle.");
  abscissa oa;
  oa.system = relativesystem;
  oa.x = angabscissa(c, M).x/360;
  return oa;
}

/*<asyxml><function type="abscissa" signature="relabscissa(ellipse,point)"><code></asyxml>*/
abscissa relabscissa(ellipse el, point M)
{/*<asyxml></code><documentation>Return the relative abscissa of 'M' on the ellipse 'el'.</documentation></function></asyxml>*/
  if(!(M @ el)) abort("relabscissa: the point is not on the ellipse.");
  abscissa oa;
  oa.system = relativesystem;
  oa.x = curabscissa(el, M).x/arclength(el);
  oa.polarconicroutine = fromFocus;
  return oa;
}

/*<asyxml><function type="abscissa" signature="relabscissa(conic,point)"><code></asyxml>*/
abscissa relabscissa(conic co, point M)
{/*<asyxml></code><documentation>Return the relative abscissa of 'M'
   on the conic 'co'.</documentation></function></asyxml>*/
  if(co.e > 1) abort("relabscissa: not implemented for hyperbola and parabola.");
  if(co.e == 1) return relabscissa((parabola)co, M);
  if(co.e == 0) return relabscissa((circle)co, M);
  return relabscissa((ellipse)co, M);
}
// *.......................ABSCISSA........................*
// *=======================================================*

// *=======================================================*
// *.........................ARCS..........................*
/*<asyxml><struct signature="arc"><code></asyxml>*/
struct arc {
  /*<asyxml></code><documentation>Implement oriented ellipse (included circle) arcs.
    All the calculus with this structure will be as exact as Asymptote can do.
    For a full precision, you must not cast 'arc' to 'path' excepted for drawing routines.
    </documentation><property type = "ellipse" signature="el"><code></asyxml>*/
  ellipse el;/*<asyxml></code><documentation>The support of the arc.</documentation></property><property type = "real" signature="angle0"><code></asyxml>*/
  restricted real angle0 = 0;/*<asyxml></code><documentation>Internal use: rotating a circle does not modify the origin point,this variable stocks the eventual angle rotation. This value is not used for ellipses which are not circles.</documentation></property><property type = "real" signature="angle1,angle2"><code></asyxml>*/
  restricted  real angle1, angle2;/*<asyxml></code><documentation>Values (in degrees) in ]-360, 360[.</documentation></property><property type = "bool" signature="direction"><code></asyxml>*/
  bool direction = CCW;/*<asyxml></code><documentation>The arc will be drawn from 'angle1' to 'angle2' rotating in the direction 'direction'.</documentation></property><property type = "polarconicroutine" signature="polarconicroutine"><code></asyxml>*/
  polarconicroutine polarconicroutine = currentpolarconicroutine;/*<asyxml></code><documentation>The routine to which the angles refer.
                                                                 If 'el' is a circle 'fromCenter' is always used.</documentation></property></asyxml>*/

  /*<asyxml><method type = "void" signature="setangles(real,real,real)"><code></asyxml>*/
  void setangles(real a0, real a1, real a2)
  {/*<asyxml></code><documentation>Set the angles.</documentation></method></asyxml>*/
    if (a1 < 0 && a2 < 0) {
      a1 += 360;
      a2 += 360;
    }
    this.angle0 = a0%(sgnd(a0) * 360);
    this.angle1 = a1%(sgnd(a1) * 360);
    this.angle2 = a2%(sgnd(2) * 360);
  }

  /*<asyxml><method type = "void" signature="init(ellipse,real,real,real,polarconicroutine,bool)"><code></asyxml>*/
  void init(ellipse el, real angle0 = 0, real angle1, real angle2,
            polarconicroutine polarconicroutine,
            bool direction = CCW)
  {/*<asyxml></code><documentation>Constructor.</documentation></method></asyxml>*/
    if(abs(angle1 - angle2) > 360) abort("arc: |angle1 - angle2| > 360.");
    this.el = el;
    this.setangles(angle0, angle1, angle2);
    this.polarconicroutine = polarconicroutine;
    this.direction = direction;
  }

  /*<asyxml><method type = "arc" signature="copy()"><code></asyxml>*/
  arc copy()
  {/*<asyxml></code><documentation>Copy the arc.</documentation></method></asyxml>*/
    arc oa = new arc;
    oa.el = this.el;
    oa.direction = this.direction;
    oa.polarconicroutine = this.polarconicroutine;
    oa.angle1 = this.angle1;
    oa.angle2 = this.angle2;
    oa.angle0 = this.angle0;
    return oa;
  }
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="polarconicroutine" signature="polarconicroutine(ellipse)"><code></asyxml>*/
polarconicroutine polarconicroutine(conic co)
{/*<asyxml></code><documentation>Return the default routine used to draw a conic.</documentation></function></asyxml>*/
  if(co.e == 0) return fromCenter;
  if(co.e == 1) return fromFocus;
  return currentpolarconicroutine;
}

/*<asyxml><function type="arc" signature="arc(ellipse,real,real,polarconicroutine,bool)"><code></asyxml>*/
arc arc(ellipse el, real angle1, real angle2,
        polarconicroutine polarconicroutine = polarconicroutine(el),
        bool direction = CCW)
{/*<asyxml></code><documentation>Return the ellipse arc from 'angle1' to 'angle2' with respect to 'polarconicroutine' and rotating in the direction 'direction'.</documentation></function></asyxml>*/
  arc oa;
  oa.init(el, 0, angle1, angle2, polarconicroutine, direction);
  return oa;
}

/*<asyxml><function type="arc" signature="complementary(arc)"><code></asyxml>*/
arc complementary(arc a)
{/*<asyxml></code><documentation>Return the complementary of 'a'.</documentation></function></asyxml>*/
  arc oa;
  oa.init(a.el, a.angle0, a.angle2, a.angle1, a.polarconicroutine, a.direction);
  return oa;
}

/*<asyxml><function type="arc" signature="reverse(arc)"><code></asyxml>*/
arc reverse(arc a)
{/*<asyxml></code><documentation>Return arc 'a' oriented in reverse direction.</documentation></function></asyxml>*/
  arc oa;
  oa.init(a.el, a.angle0, a.angle2, a.angle1, a.polarconicroutine, !a.direction);
  return oa;
}

/*<asyxml><function type="real" signature="degrees(arc)"><code></asyxml>*/
real degrees(arc a)
{/*<asyxml></code><documentation>Return the measure in degrees of the oriented arc 'a'.</documentation></function></asyxml>*/
  real or;
  real da = a.angle2 - a.angle1;
  if(a.direction) {
    or = a.angle1 < a.angle2 ? da : 360 + da;
  } else {
    or = a.angle1 < a.angle2 ? -360 + da : da;
  }
  return or;
}

/*<asyxml><function type="real" signature="angle(a)"><code></asyxml>*/
real angle(arc a)
{/*<asyxml></code><documentation>Return the measure in radians of the oriented arc 'a'.</documentation></function></asyxml>*/
  return radians(degrees(a));
}

/*<asyxml><function type="int" signature="arcnodesnumber(explicit arc)"><code></asyxml>*/
int arcnodesnumber(explicit arc a)
{/*<asyxml></code><documentation>Return the number of nodes to draw the arc 'a'.</documentation></function></asyxml>*/
  return ellipsenodesnumber(a.el.a, a.el.b, a.angle1, a.angle2, a.direction);
}

private path arctopath(arc a, int n)
{
  if(a.el.e == 0) return arcfromcenter(a.el, a.angle0 + a.angle1, a.angle0 + a.angle2, a.direction, n);
  if(a.el.e != 1) return a.polarconicroutine(a.el, a.angle1, a.angle2, n, a.direction);
  return arcfromfocus(a.el, a.angle1, a.angle2, n, a.direction);
}

/*<asyxml><function type="point" signature="angpoint(arc,real)"><code></asyxml>*/
point angpoint(arc a, real angle)
{/*<asyxml></code><documentation>Return the point given by its angular position (in degrees) relative to the arc 'a'.
   If 'angle > degrees(a)' or 'angle < 0' the returned point is on the extended arc.</documentation></function></asyxml>*/
  pair p;
  if(a.el.e == 0) {
    real gle = a.angle0 + a.angle1 + (a.direction ? angle : -angle);
    p = point(arcfromcenter(a.el, gle, gle, CCW, 1), 0);
  }
  else {
    real gle = a.angle1 + (a.direction ? angle : -angle);
    p = point(a.polarconicroutine(a.el, gle, gle, 1, CCW), 0);
  }
  return point(coordsys(a.el), p/coordsys(a.el));
}

/*<asyxml><operator type = "path" signature="cast(explicit arc)"><code></asyxml>*/
path operator cast(explicit arc a)
{/*<asyxml></code><documentation>Cast arc to path.</documentation></operator></asyxml>*/
  return arctopath(a, arcnodesnumber(a));
}

/*<asyxml><operator type = "guide" signature="cast(explicit arc)"><code></asyxml>*/
guide operator cast(explicit arc a)
{/*<asyxml></code><documentation>Cast arc to guide.</documentation></operator></asyxml>*/
  return arctopath(a, arcnodesnumber(a));
}

/*<asyxml><operator type = "arc" signature="*(transform,explicit arc)"><code></asyxml>*/
arc operator *(transform t, explicit arc a)
{/*<asyxml></code><documentation>Provide transform * arc.</documentation></operator></asyxml>*/
  pair[] P, PP;
  path g = arctopath(a, 3);
  real a0, a1 = a.angle1, a2 = a.angle2, ap1, ap2;
  bool dir = a.direction;
  P[0] = t * point(g, 0);
  P[1] = t * point(g, 2);
  ellipse el = t * a.el;
  arc oa;
  a0 = (a.angle0 + angle(shiftless(t)))%360;
  pair C;
  if(a.polarconicroutine == fromCenter) C = el.C; else C = el.F1;
  real d = abs(locate(el.F2 - el.F1)) > epsgeo ?
    degrees(locate(el.F2 - el.F1)) : a0 + degrees(el.C.coordsys.i);
  ap1 = (degrees(P[0]-C, false) - d)%360;
  ap2 = (degrees(P[1]-C, false) - d)%360;
  oa.init(el, a0, ap1, ap2, a.polarconicroutine, dir);
  g = arctopath(oa, 3);
  PP[0] = point(g, 0);
  PP[1] = point(g, 2);
  if((a1 - a2) * (ap1 - ap2) < 0) {// Handle reflection.
    dir=!a.direction;
    oa.init(el, a0, ap1, ap2, a.polarconicroutine, dir);
  }
  return oa;
}

/*<asyxml><operator type = "arc" signature="*(real,explicit arc)"><code></asyxml>*/
arc operator *(real x, explicit arc a)
{/*<asyxml></code><documentation>Provide real * arc.
   Return the arc subtracting and adding '(x - 1) * degrees(a)/2' to 'a.angle1' and 'a.angle2' respectively.</documentation></operator></asyxml>*/
  real a1, a2, gle;
  gle = (x - 1) * degrees(a)/2;
  a1 = a.angle1 - gle;
  a2 = a.angle2 + gle;
  arc oa;
  oa.init(a.el, a.angle0, a1, a2, a.polarconicroutine, a.direction);
  return oa;
}
arc operator *(int x, explicit arc a){return (real)x * a;}
/*<asyxml><operator type = "arc" signature="/(real,explicit arc)"><code></asyxml>*/
arc operator /(explicit arc a, real x)
{/*<asyxml></code><documentation>Provide arc/real.
   Return the arc subtracting and adding '(1/x - 1) * degrees(a)/2' to 'a.angle1' and 'a.angle2' respectively.</documentation></operator></asyxml>*/
  return (1/x) * a;
}
/*<asyxml><operator type = "arc" signature="+(explicit arc,point)"><code></asyxml>*/
arc operator +(explicit arc a, point M)
{/*<asyxml></code><documentation>Provide arc + point.
   Return shifted arc.
   'operator +(explicit arc, point)', 'operator +(explicit arc, vector)' and 'operator -(explicit arc, vector)' are also defined.</documentation></operator></asyxml>*/
  return shift(M) * a;
}
arc operator -(explicit arc a, point M){return a + (-M);}
arc operator +(explicit arc a, vector v){return shift(locate(v)) * a;}
arc operator -(explicit arc a, vector v){return a + (-v);}


/*<asyxml><operator type = "bool" signature="@(point,arc)"><code></asyxml>*/
bool operator @(point M, arc a)
{/*<asyxml></code><documentation>Return true iff 'M' is on the arc 'a'.</documentation></operator></asyxml>*/
  if (!(M @ a.el)) return false;
  coordsys R = defaultcoordsys;
  path ap = arctopath(a, 3);
  line l = line(point(R, point(ap, 0)), point(R, point(ap, 2)));
  return sameside(M, point(R, point(ap, 1)), l);
}

/*<asyxml><function type="void" signature="draw(picture,Label,arc,align,pen,arrowbar,arrowbar,margin,Label,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, Label L = "", arc a,
          align align = NoAlign, pen p = currentpen,
          arrowbar arrow = None, arrowbar bar = None, margin margin = NoMargin,
          Label legend = "", marker marker = nomarker)
{/*<asyxml></code><documentation>Draw 'arc' adding the pen returned by 'addpenarc(p)' to the pen 'p'.
   <look href = "#addpenarc"/></documentation></function></asyxml>*/
  draw(pic, L, (path)a, align, addpenarc(p), arrow, bar, margin, legend, marker);
}

/*<asyxml><function type="real" signature="arclength(arc)"><code></asyxml>*/
real arclength(arc a)
{/*<asyxml></code><documentation>The arc length of 'a'.</documentation></function></asyxml>*/
  return arclength(a.el, a.angle1, a.angle2, a.direction, a.polarconicroutine);
}

private point ppoint(arc a, real x)
{// Return the point of the arc proportionally to its length.
  point oP;
  if(a.el.e == 0) { // Case of circle.
    oP = angpoint(a, x * abs(degrees(a)));
  } else { // Ellipse and not circle.
    if(!a.direction) {
      transform t = reflect(line(a.el.F1, a.el.F2));
      return t * ppoint(t * a, x);
    }

    real angle1 = a.angle1, angle2 = a.angle2;
    if(a.polarconicroutine == fromFocus) {
      //       dot(point(fromFocus(a.el, angle1, angle1, 1, CCW), 0), 2mm + blue);
      //       dot(point(fromFocus(a.el, angle2, angle2, 1, CCW), 0), 2mm + blue);
      //       write("fromfocus1 = ", angle1);
      //       write("fromfocus2 = ", angle2);
      real gle1 = focusToCenter(a.el, angle1);
      real gle2 = focusToCenter(a.el, angle2);
      if((gle1 - gle2) * (angle1 - angle2) > 0) {
        angle1 = gle1; angle2 = gle2;
      } else {
        angle1 = gle2; angle2 = gle1;
      }
      //       write("fromcenter1 = ", angle1);
      //       write("fromcenter2 = ", angle2);
      //       dot(point(fromCenter(a.el, angle1, angle1, 1, CCW), 0), 1mm + red);
      //       dot(point(fromCenter(a.el, angle2, angle2, 1, CCW), 0), 1mm + red);
    }

    if(angle1 > angle2) {
      arc ta = a.copy();
      ta.polarconicroutine = fromCenter;
      ta.setangles(a0 = a.angle0, a1 = angle1 - 360, a2 = angle2);
      return ppoint(ta, x);
    }
    ellipse co = a.el;
    real gle, a1, a2, cx = 0;
    bool direction;
    if(x >= 0) {
      a1 = angle1;
      a2 = a1 + 360;
      direction = CCW;
    } else {
      a1 = angle1 - 360;
      a2 = a1 - 360;
      direction = CW;
    }
    gle = a1;
    real L = arclength(co, angle1, angle2, a.direction, fromCenter);
    real tx = L * abs(x)%arclength(co);
    real aout = a1;
    while(abs(cx - tx) > epsgeo) {
      aout = (a1 + a2)/2;
      cx = abs(arclength(co, gle, aout, direction, fromCenter));
      if(cx > tx) a2 = (a1 + a2)/2 ; else a1 = (a1 + a2)/2;
    }
    pair p = point(arcfromcenter(co, aout, aout, CCW, 1), 0);
    oP = point(coordsys(co), p/coordsys(co));
  }
  return oP;
}

/*<asyxml><function type="point" signature="point(arc,abscissa)"><code></asyxml>*/
point point(arc a, abscissa l)
{/*<asyxml></code><documentation>Return the point of 'a' which has the abscissa 'l.x'
   according to the abscissa system 'l.system'.
   Note that 'a.polarconicroutine' is used instead of 'l.polarconicroutine'.
   <look href = "#struct abscissa"/></documentation></function></asyxml>*/
  real posx;
  arc ta = a.copy();
  ellipse co = a.el;
  if (l.system == relativesystem) {
    posx = l.x;
  } else
    if (l.system == curvilinearsystem) {
      real tl;
      if(co.e == 0) {
        tl = curabscissa(a.el, angpoint(a.el, a.angle0 + a.angle1)).x;
        return curpoint(a.el, tl + (a.direction ? l.x : -l.x));
      } else {
        tl = curabscissa(a.el, angpoint(a.el, a.angle1, a.polarconicroutine)).x;
        return curpoint(a.el, tl + (a.direction ? l.x : -l.x));
      }
    } else
      if (l.system == nodesystem) {
        coordsys R = coordsys(co);
        return point(R, point((path)a, l.x)/R);
      } else
        if (l.system == angularsystem) {
          return angpoint(a, l.x);
        } else abort("point: bad abscissa system.");
  return ppoint(ta, posx);
}


/*<asyxml><function type="point" signature="point(arc,real)"><code></asyxml>*/
point point(arc a, real x)
{/*<asyxml></code><documentation>Return the point between node floor(t) and floor(t) + 1.</documentation></function></asyxml>*/
  return point(a, nodabscissa(x));
}
pair point(explicit arc a, int x)
{
  return point(a, nodabscissa(x));
}

/*<asyxml><function type="point" signature="relpoint(arc,real)"><code></asyxml>*/
point relpoint(arc a, real x)
{/*<asyxml></code><documentation>Return the relative point of 'a'.
   If x > 1 or x < 0, the returned point is on the extended arc.</documentation></function></asyxml>*/
  return point(a, relabscissa(x));
}

/*<asyxml><function type="point" signature="curpoint(arc,real)"><code></asyxml>*/
point curpoint(arc a, real x)
{/*<asyxml></code><documentation>Return the point of 'a' which has the curvilinear abscissa 'x'.
   If x < 0 or x > arclength(a), the returned point is on the extended arc.</documentation></function></asyxml>*/
  return point(a, curabscissa(x));
}

/*<asyxml><function type="abscissa" signature="angabscissa(arc,point)"><code></asyxml>*/
abscissa angabscissa(arc a, point M)
{/*<asyxml></code><documentation>Return the angular abscissa of 'M' according to the arc 'a'.</documentation></function></asyxml>*/
  if(!(M @ a.el))
    abort("angabscissa: the point is not on the extended arc.");
  abscissa oa;
  oa.system = angularsystem;
  oa.polarconicroutine = a.polarconicroutine;
  real am = angabscissa(a.el, M, a.polarconicroutine).x;
  oa.x = (am - a.angle1 - (a.el.e == 0 ? a.angle0 : 0))%360;
  oa.x = a.direction ? oa.x : 360 - oa.x;
  return oa;
}

/*<asyxml><function type="abscissa" signature="curabscissa(arc,point)"><code></asyxml>*/
abscissa curabscissa(arc a, point M)
{/*<asyxml></code><documentation>Return the curvilinear abscissa according to the arc 'a'.</documentation></function></asyxml>*/
  ellipse el = a.el;
  if(!(M @ el))
    abort("angabscissa: the point is not on the extended arc.");
  abscissa oa;
  oa.system = curvilinearsystem;
  real xm = curabscissa(el, M).x;
  real a0 = el.e == 0 ? a.angle0 : 0;
  real am = curabscissa(el, angpoint(el, a.angle1 + a0, a.polarconicroutine)).x;
  real l = arclength(el);
  oa.x = (xm - am)%l;
  oa.x = a.direction ? oa.x : l - oa.x;
  return oa;
}

/*<asyxml><function type="abscissa" signature="nodabscissa(arc,point)"><code></asyxml>*/
abscissa nodabscissa(arc a, point M)
{/*<asyxml></code><documentation>Return the node abscissa according to the arc 'a'.</documentation></function></asyxml>*/
  if(!(M @ a))
    abort("nodabscissa: the point is not on the arc.");
  abscissa oa;
  oa.system = nodesystem;
  oa.x = intersect((path)a, M)[0];
  return oa;
}

/*<asyxml><function type="abscissa" signature="relabscissa(arc,point)"><code></asyxml>*/
abscissa relabscissa(arc a, point M)
{/*<asyxml></code><documentation>Return the relative abscissa according to the arc 'a'.</documentation></function></asyxml>*/
  ellipse el = a.el;
  if(!( M @ el))
    abort("relabscissa: the point is not on the prolonged arc.");
  abscissa oa;
  oa.system = relativesystem;
  oa.x = curabscissa(a, M).x/arclength(a);
  return oa;
}

/*<asyxml><function type="void" signature="markarc(picture,Label,int,real,real,arc,arrowbar,pen,pen,margin,marker)"><code></asyxml>*/
void markarc(picture pic = currentpicture,
             Label L = "",
             int n = 1, real radius = 0, real space = 0,
             arc a,
             pen sectorpen = currentpen,
             pen markpen = sectorpen,
             margin margin = NoMargin,
             arrowbar arrow = None,
             marker marker = nomarker)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  real Da = degrees(a);
  pair p1 = point(a, 0);
  pair p2 = relpoint(a, 1);
  pair c = a.polarconicroutine == fromCenter ? locate(a.el.C) : locate(a.el.F1);
  if(radius == 0) radius = markangleradius(markpen);
  if(abs(Da) > 180) radius = -radius;
  radius = (a.direction ? 1 : -1) * sgnd(Da) * radius;
  draw(c--p1^^c--p2, sectorpen);
  markangle(pic = pic, L = L, n = n, radius = radius, space = space,
            A = p1, O = c, B = p2,
            arrow = arrow, p = markpen, margin = margin,
            marker = marker);
}
// *.........................ARCS..........................*
// *=======================================================*

// *=======================================================*
// *........................MASSES.........................*
/*<asyxml><struct signature="mass"><code></asyxml>*/
struct mass {/*<asyxml></code><documentation></documentation><property type = "point" signature="M"><code></asyxml>*/
  point M;/*<asyxml></code><documentation></documentation></property><property type = "real" signature="m"><code></asyxml>*/
  real m;/*<asyxml></code><documentation></documentation></property></asyxml>*/
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="mass" signature="mass(point,real)"><code></asyxml>*/
mass mass(point M, real m)
{/*<asyxml></code><documentation>Constructor of mass point.</documentation></function></asyxml>*/
  mass om;
  om.M = M;
  om.m = m;
  return om;
}

/*<asyxml><operator type = "point" signature="cast(mass)"><code></asyxml>*/
point operator cast(mass m)
{/*<asyxml></code><documentation>Cast mass point to point.</documentation></operator></asyxml>*/
  point op;
  op = m.M;
  op.m = m.m;
  return op;
}
/*<asyxml><function type="point" signature="point(explicit mass)"><code></asyxml>*/
point point(explicit mass m){return m;}/*<asyxml></code><documentation>Cast
                                         'm' to point</documentation></function></asyxml>*/

/*<asyxml><operator type = "mass" signature="cast(point)"><code></asyxml>*/
mass operator cast(point M)
{/*<asyxml></code><documentation>Cast point to mass point.</documentation></operator></asyxml>*/
  mass om;
  om.M = M;
  om.m = M.m;
  return om;
}
/*<asyxml><function type="mass" signature="mass(explicit point)"><code></asyxml>*/
mass mass(explicit point P)
{/*<asyxml></code><documentation>Cast 'P' to mass.</documentation></function></asyxml>*/
  return mass(P, P.m);
}

/*<asyxml><operator type = "point[]" signature="cast(mass[])"><code></asyxml>*/
point[] operator cast(mass[] m)
{/*<asyxml></code><documentation>Cast mass[] to point[].</documentation></operator></asyxml>*/
  point[] op;
  for(mass am : m) op.push(point(am));
  return op;
}

/*<asyxml><operator type = "mass[]" signature="cast(point[])"><code></asyxml>*/
mass[] operator cast(point[] P)
{/*<asyxml></code><documentation>Cast point[] to mass[].</documentation></operator></asyxml>*/
  mass[] om;
  for(point op : P) om.push(mass(op));
  return om;
}

/*<asyxml><function type="mass" signature="mass(coordsys,explicit pair,real)"><code></asyxml>*/
mass mass(coordsys R, explicit pair p, real m)
{/*<asyxml></code><documentation>Return the mass which has coordinates
   'p' with respect to 'R' and weight 'm'.</documentation></function></asyxml>*/
  return point(R, p, m);// Using casting.
}

/*<asyxml><operator type = "mass" signature="cast(pair)"><code></asyxml>*/
mass operator cast(pair m){return mass((point)m, 1);}/*<asyxml></code><documentation>Cast pair to mass point.</documentation></operator></asyxml>*/
/*<asyxml><operator type = "path" signature="cast(mass)"><code></asyxml>*/
path operator cast(mass M){return M.M;}/*<asyxml></code><documentation>Cast mass point to path.</documentation></operator></asyxml>*/
/*<asyxml><operator type = "guide" signature="cast(mass)"><code></asyxml>*/
guide operator cast(mass M){return M.M;}/*<asyxml></code><documentation>Cast mass to guide.</documentation></operator></asyxml>*/

/*<asyxml><operator type = "mass" signature="+(mass,mass)"><code></asyxml>*/
mass operator +(mass M1, mass M2)
{/*<asyxml></code><documentation>Provide mass + mass.
   mass - mass is also defined.</documentation></operator></asyxml>*/
  return mass(M1.M + M2.M, M1.m + M2.m);
}
mass operator -(mass M1, mass M2)
{
  return mass(M1.M - M2.M, M1.m - M2.m);
}

/*<asyxml><operator type = "mass" signature="*(real,mass)"><code></asyxml>*/
mass operator *(real x, explicit mass M)
{/*<asyxml></code><documentation>Provide real * mass.
   The resulted mass is the mass of 'M' multiplied by 'x' .
   mass/real, mass + real and mass - real are also defined.</documentation></operator></asyxml>*/
  return mass(M.M, x * M.m);
}
mass operator *(int x, explicit mass M){return mass(M.M, x * M.m);}
mass operator /(explicit mass M, real x){return mass(M.M, M.m/x);}
mass operator /(explicit mass M, int x){return mass(M.M, M.m/x);}
mass operator +(explicit mass M, real x){return mass(M.M, M.m + x);}
mass operator +(explicit mass M, int x){return mass(M.M, M.m + x);}
mass operator -(explicit mass M, real x){return mass(M.M, M.m - x);}
mass operator -(explicit mass M, int x){return mass(M.M, M.m - x);}
/*<asyxml><operator type = "mass" signature="*(transform,mass)"><code></asyxml>*/
mass operator *(transform t, mass M)
{/*<asyxml></code><documentation>Provide transform * mass.</documentation></operator></asyxml>*/
  return mass(t * M.M, M.m);
}

/*<asyxml><function type="mass" signature="masscenter(... mass[])"><code></asyxml>*/
mass masscenter(... mass[] M)
{/*<asyxml></code><documentation>Return the center of the masses 'M'.</documentation></function></asyxml>*/
  point[] P;
  for (int i = 0; i < M.length; ++i)
    P.push(M[i].M);
  P = standardizecoordsys(currentcoordsys, true ... P);
  real m = M[0].m;
  point oM = M[0].m * P[0];
  for (int i = 1; i < M.length; ++i) {
    oM += M[i].m * P[i];
    m += M[i].m;
  }
  if (m == 0) abort("masscenter: the sum of masses is null.");
  return mass(oM/m, m);
}

/*<asyxml><function type="string" signature="massformat(string,string,mass)"><code></asyxml>*/
string massformat(string format = defaultmassformat,
                  string s, mass M)
{/*<asyxml></code><documentation>Return the string formated by 'format' with the mass value.
   In the parameter 'format', %L will be replaced by 's'.
   <look href = "#defaultmassformat"/>.</documentation></function></asyxml>*/
  return format == "" ? s :
    format(replace(format, "%L", replace(s, "$", "")), M.m);
}

/*<asyxml><function type="void" signature="label(picture,Label,explicit mass,align,string,pen,filltype)"><code></asyxml>*/
void label(picture pic = currentpicture, Label L, explicit mass M,
           align align = NoAlign, string format = defaultmassformat,
           pen p = nullpen, filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw label returned by massformat(format, L, M) at coordinates of M.
   <look href = "#massformat(string, string, mass)"/>.</documentation></function></asyxml>*/
  Label lL = L.copy();
  lL.s = massformat(format, lL.s, M);
  Label L = Label(lL, M.M, align, p, filltype);
  add(pic, L);
}

/*<asyxml><function type="void" signature="dot(picture,Label,explicit mass,align,string,pen)"><code></asyxml>*/
void dot(picture pic = currentpicture, Label L, explicit mass M, align align = NoAlign,
         string format = defaultmassformat, pen p = currentpen)
{/*<asyxml></code><documentation>Draw a dot with label 'L' as
   label(picture, Label, explicit mass, align, string, pen, filltype) does.
   <look href = "#label(picture, Label, mass, align, string, pen, filltype)"/>.</documentation></function></asyxml>*/
  Label lL = L.copy();
  lL.s = massformat(format, lL.s, M);
  lL.position(locate(M.M));
  lL.align(align, E);
  lL.p(p);
  dot(pic, M.M, p);
  add(pic, lL);
}
// *........................MASSES.........................*
// *=======================================================*

// *=======================================================*
// *.......................TRIANGLES.......................*
/*<asyxml><function type="point" signature="orthocentercenter(point,point,point)"><code></asyxml>*/
point orthocentercenter(point A, point B, point C)
{/*<asyxml></code><documentation>Return the orthocenter of the triangle ABC.</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B, C);
  coordsys R = P[0].coordsys;
  pair pp = extension(A, projection(P[1], P[2]) * P[0], B, projection(P[0], P[2]) * P[1]);
  return point(R, pp/R);
}

/*<asyxml><function type="point" signature="centroid(point,point,point)"><code></asyxml>*/
point centroid(point A, point B, point C)
{/*<asyxml></code><documentation>Return the centroid of the triangle ABC.</documentation></function></asyxml>*/
  return (A + B + C)/3;
}

/*<asyxml><function type="point" signature="incenter(point,point,point)"><code></asyxml>*/
point incenter(point A, point B, point C)
{/*<asyxml></code><documentation>Return the center of the incircle of the triangle ABC.</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B, C);
  coordsys R = P[0].coordsys;
  pair a = A, b = B, c = C;
  pair pp = extension(a, a + dir(a--b, a--c), b, b + dir(b--a, b--c));
  return point(R, pp/R);
}

/*<asyxml><function type="real" signature="inradius(point,point,point)"><code></asyxml>*/
real inradius(point A, point B, point C)
{/*<asyxml></code><documentation>Return the radius of the incircle of the triangle ABC.</documentation></function></asyxml>*/
  point IC = incenter(A, B, C);
  return abs(IC - projection(A, B) * IC);
}

/*<asyxml><function type="circle" signature="incircle(point,point,point)"><code></asyxml>*/
circle incircle(point A, point B, point C)
{/*<asyxml></code><documentation>Return the incircle of the triangle ABC.</documentation></function></asyxml>*/
  point IC = incenter(A, B, C);
  return circle(IC, abs(IC - projection(A, B) * IC));
}

/*<asyxml><function type="point" signature="excenter(point,point,point)"><code></asyxml>*/
point excenter(point A, point B, point C)
{/*<asyxml></code><documentation>Return the center of the excircle of the triangle tangent with (AB).</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B, C);
  coordsys R = P[0].coordsys;
  pair a = A, b = B, c = C;
  pair pp = extension(a, a + rotate(90) * dir(a--b, a--c), b, b + rotate(90) * dir(b--a, b--c));
  return point(R, pp/R);
}

/*<asyxml><function type="real" signature="exradius(point,point,point)"><code></asyxml>*/
real exradius(point A, point B, point C)
{/*<asyxml></code><documentation>Return the radius of the excircle of the triangle ABC with (AB).</documentation></function></asyxml>*/
  point EC = excenter(A, B, C);
  return abs(EC - projection(A, B) * EC);
}

/*<asyxml><function type="circle" signature="excircle(point,point,point)"><code></asyxml>*/
circle excircle(point A, point B, point C)
{/*<asyxml></code><documentation>Return the excircle of the triangle ABC tangent with (AB).</documentation></function></asyxml>*/
  point center = excenter(A, B, C);
  real radius = abs(center - projection(B, C) * center);
  return circle(center, radius);
}

private int[] numarray = {1, 2, 3};
numarray.cyclic = true;

/*<asyxml><struct signature="triangle"><code></asyxml>*/
struct triangle {/*<asyxml></code><documentation></documentation></asyxml>*/

  /*<asyxml><struct signature="vertex"><code></asyxml>*/
  struct vertex {/*<asyxml></code><documentation>Structure used to communicate the vertex of a triangle.</documentation><property type = "int" signature="n"><code></asyxml>*/
    int n;/*<asyxml></code><documentation>1 means VA,2 means VB,3 means VC,4 means VA etc...</documentation></property><property type = "triangle" signature="triangle"><code></asyxml>*/
    triangle t;/*<asyxml></code><documentation>The triangle to which the vertex refers.</documentation></property></asyxml>*/
  }/*<asyxml></struct></asyxml>*/

  /*<asyxml><property type = "point" signature="A,B,C"><code></asyxml>*/
  restricted point A, B, C;/*<asyxml></code><documentation>The vertices of the triangle (as point).</documentation></property><property type = "vertex" signature="VA, VB, VC"><code></asyxml>*/
  restricted vertex VA, VB, VC;/*<asyxml></code><documentation>The vertices of the triangle (as vertex).
                                 Note that the vertex structure contains the triangle to wish it refers.</documentation></property></asyxml>*/
  VA.n = 1;VB.n = 2;VC.n = 3;

  /*<asyxml><method type = "vertex" signature="vertex(int)"><code></asyxml>*/
  vertex vertex(int n)
  {/*<asyxml></code><documentation>Return numbered vertex.
     'n' is 1 means VA, 2 means VB, 3 means VC, 4 means VA etc...</documentation></method></asyxml>*/
    n = numarray[n - 1];
    if(n == 1) return VA;
    else if(n == 2) return VB;
    return VC;
  }

  /*<asyxml><method type = "point" signature="point(int)"><code></asyxml>*/
  point point(int n)
  {/*<asyxml></code><documentation>Return numbered point.
     n is 1 means A, 2 means B, 3 means C, 4 means A etc...</documentation></method></asyxml>*/
    n = numarray[n - 1];
    if(n == 1) return A;
    else if(n == 2) return B;
    return C;
  }

  /*<asyxml><method type = "void" signature="init(point,point,point)"><code></asyxml>*/
  void init(point A, point B, point C)
  {/*<asyxml></code><documentation>Constructor.</documentation></method></asyxml>*/
    point[] P = standardizecoordsys(A, B, C);
    this.A = P[0];
    this.B = P[1];
    this.C = P[2];
    VA.t = this; VB.t = this; VC.t = this;
  }

  /*<asyxml><method type = "void" signature="operator init(point,point,point)"><code></asyxml>*/
  void operator init(point A, point B, point C)
  {/*<asyxml></code><documentation>For backward compatibility.
     Provide the routine 'triangle(point A, point B, point C)'.</documentation></method></asyxml>*/
    this.init(A, B, C);
  }

  /*<asyxml><method type = "void" signature="init(real,real,real,real,point)"><code></asyxml>*/
  void operator init(real b, real alpha, real c, real angle = 0, point A = (0, 0))
  {/*<asyxml></code><documentation>For backward compatibility.
     Provide the routine 'triangle(real b, real alpha, real c, real angle = 0, point A = (0, 0))
     which returns the triangle ABC rotated by 'angle' (in degrees) and where b = AC, degrees(A) = alpha, AB = c.</documentation></method></asyxml>*/
    coordsys R = A.coordsys;
    this.init(A, A + R.polar(c, radians(angle)), A + R.polar(b, radians(angle + alpha)));
  }

  /*<asyxml><method type = "real" signature="a(),b(),c()"><code></asyxml>*/
  real a()
  {/*<asyxml></code><documentation>Return the length BC.
     b() and c() are also defined and return the length AC and AB respectively.</documentation></method></asyxml>*/
    return length(C - B);
  }
  real b() {return length(A - C);}
  real c() {return length(B - A);}

  private real det(pair a, pair b) {return a.x * b.y - a.y * b.x;}

  /*<asyxml><method type = "real" signature="area()"><code></asyxml>*/
  real area()
  {/*<asyxml></code><documentation></documentation></method></asyxml>*/
    pair a = locate(A), b = locate(B), c = locate(C);
    return 0.5 * abs(det(a, b) + det(b, c) + det(c, a));
  }

  /*<asyxml><method type = "real" signature="alpha(),beta(),gamma()"><code></asyxml>*/
  real alpha()
  {/*<asyxml></code><documentation>Return the measure (in degrees) of the angle A.
     beta() and gamma() are also defined and return the measure of the angles B and C respectively.</documentation></method></asyxml>*/
    return degrees(acos((b()^2 + c()^2 - a()^2)/(2b() * c())));
  }
  real beta()  {return degrees(acos((c()^2 + a()^2 - b()^2)/(2c() * a())));}
  real gamma() {return degrees(acos((a()^2 + b()^2 - c()^2)/(2a() * b())));}

  /*<asyxml><method type = "path" signature="Path()"><code></asyxml>*/
  path Path()
  {/*<asyxml></code><documentation>The path of the triangle.</documentation></method></asyxml>*/
    return A--C--B--cycle;
  }

  /*<asyxml><struct signature="side"><code></asyxml>*/
  struct side
  {/*<asyxml></code><documentation>Structure used to communicate the side of a triangle.</documentation><property type = "int" signature="n"><code></asyxml>*/
    int n;/*<asyxml></code><documentation>1 or 0 means [AB],-1 means [BA],2 means [BC],-2 means [CB] etc.</documentation></property><property type = "triangle" signature="triangle"><code></asyxml>*/
    triangle t;/*<asyxml></code><documentation>The triangle to which the side refers.</documentation></property></asyxml>*/
  }/*<asyxml></struct></asyxml>*/

  /*<asyxml><property type = "side" signature="AB"><code></asyxml>*/
  side AB;/*<asyxml></code><documentation>For the routines using the structure 'side', triangle.AB means 'side AB'.
            BA, AC, CA etc are also defined.</documentation></property></asyxml>*/
  AB.n = 1; AB.t = this;
  side BA; BA.n = -1; BA.t = this;
  side BC; BC.n = 2; BC.t = this;
  side CB; CB.n = -2; CB.t = this;
  side CA; CA.n = 3; CA.t = this;
  side AC; AC.n = -3; AC.t = this;

  /*<asyxml><method type = "side" signature="side(int)"><code></asyxml>*/
  side side(int n)
  {/*<asyxml></code><documentation>Return numbered side.
     n is 1 means AB, -1 means BA, 2 means BC, -2 means CB, etc.</documentation></method></asyxml>*/
    if(n == 0) abort('Invalid side number.');
    int an = numarray[abs(n)-1];
    if(an == 1) return n > 0 ? AB : BA;
    else if(an == 2) return n > 0 ? BC : CB;
    return n > 0 ? CA : AC;
  }

  /*<asyxml><method type = "line" signature="line(int)"><code></asyxml>*/
  line line(int n)
  {/*<asyxml></code><documentation>Return the numbered line.</documentation></method></asyxml>*/
    if(n == 0) abort('Invalid line number.');
    int an = numarray[abs(n)-1];
    if(an == 1) return n > 0 ? line(A, B) : line(B, A);
    else if(an == 2) return n > 0 ? line(B, C) : line(C, B);
    return n > 0 ? line(C, A) : line(A, C);
  }

}/*<asyxml></struct></asyxml>*/

from triangle unravel side; // The structure 'side' is now available outside the triangle structure.
from triangle unravel vertex; // The structure 'vertex' is now available outside the triangle structure.

triangle[] operator ^^(triangle[] t1, triangle t2)
{
  triangle[] T;
  for (int i = 0; i < t1.length; ++i) T.push(t1[i]);
  T.push(t2);
  return T;
}

triangle[] operator ^^(... triangle[] t)
{
  triangle[] T;
  for (int i = 0; i < t.length; ++i) {
    T.push(t[i]);
  }
  return T;
}

/*<asyxml><operator type = "line" signature="cast(side)"><code></asyxml>*/
line operator cast(side side)
{/*<asyxml></code><documentation>Cast side to (infinite) line.
   Most routine with line parameters works with side parameters.
   One can use the code 'segment(a_side)' to obtain a line segment.</documentation></operator></asyxml>*/
  triangle t = side.t;
  return t.line(side.n);
}

/*<asyxml><function type="line" signature="line(explicit side)"><code></asyxml>*/
line line(explicit side side)
{/*<asyxml></code><documentation>Return 'side' as line.</documentation></function></asyxml>*/
  return (line)side;
}

/*<asyxml><function type="segment" signature="segment(explicit side)"><code></asyxml>*/
segment segment(explicit side side)
{/*<asyxml></code><documentation>Return 'side' as segment.</documentation></function></asyxml>*/
  return (segment)(line)side;
}

/*<asyxml><operator type = "point" signature="cast(vertex)"><code></asyxml>*/
point operator cast(vertex V)
{/*<asyxml></code><documentation>Cast vertex to point.
   Most routine with point parameters works with vertex parameters.</documentation></operator></asyxml>*/
  return V.t.point(V.n);
}

/*<asyxml><function type="point" signature="point(explicit vertex)"><code></asyxml>*/
point point(explicit vertex V)
{/*<asyxml></code><documentation>Return the point corresponding to the vertex 'V'.</documentation></function></asyxml>*/
  return (point)V;
}

/*<asyxml><function type="side" signature="opposite(vertex)"><code></asyxml>*/
side opposite(vertex V)
{/*<asyxml></code><documentation>Return the opposite side of vertex 'V'.</documentation></function></asyxml>*/
  return V.t.side(numarray[abs(V.n)]);
}

/*<asyxml><function type="vertex" signature="opposite(side)"><code></asyxml>*/
vertex opposite(side side)
{/*<asyxml></code><documentation>Return the opposite vertex of side 'side'.</documentation></function></asyxml>*/
  return side.t.vertex(numarray[abs(side.n) + 1]);
}

/*<asyxml><function type="point" signature="midpoint(side)"><code></asyxml>*/
point midpoint(side side)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return midpoint(segment(side));
}

/*<asyxml><operator type = "triangle" signature="*(transform,triangle)"><code></asyxml>*/
triangle operator *(transform T, triangle t)
{/*<asyxml></code><documentation>Provide transform * triangle.</documentation></operator></asyxml>*/
  return triangle(T * t.A, T * t.B, T * t.C);
}

/*<asyxml><function type="triangle" signature="triangleAbc(real,real,real,real,point)"><code></asyxml>*/
triangle triangleAbc(real alpha, real b, real c, real angle = 0, point A = (0, 0))
{/*<asyxml></code><documentation>Return the triangle ABC rotated by 'angle' with BAC = alpha, AC = b and AB = c.</documentation></function></asyxml>*/
  triangle T;
  coordsys R = A.coordsys;
  T.init(A, A + R.polar(c, radians(angle)), A + R.polar(b, radians(angle + alpha)));
  return T;
}

/*<asyxml><function type="triangle" signature="triangleabc(real,real,real,real,point)"><code></asyxml>*/
triangle triangleabc(real a, real b, real c, real angle = 0, point A = (0, 0))
{/*<asyxml></code><documentation>Return the triangle ABC rotated by 'angle' with BC = a, AC = b and AB = c.</documentation></function></asyxml>*/
  triangle T;
  coordsys R = A.coordsys;
  T.init(A, A + R.polar(c, radians(angle)), A + R.polar(b, radians(angle) + acos((b^2 + c^2 - a^2)/(2 * b * c))));
  return T;
}

/*<asyxml><function type="triangle" signature="triangle(line,line,line)"><code></asyxml>*/
triangle triangle(line l1, line l2, line l3)
{/*<asyxml></code><documentation>Return the triangle defined by three line.</documentation></function></asyxml>*/
  point P1, P2, P3;
  P1 = intersectionpoint(l1, l2);
  P2 = intersectionpoint(l1, l3);
  P3 = intersectionpoint(l2, l3);
  if(!(defined(P1) && defined(P2) && defined(P3))) abort("triangle: two lines are parallel.");
  return triangle(P1, P2, P3);
}

/*<asyxml><function type="point" signature="foot(vertex)"><code></asyxml>*/
point foot(vertex V)
{/*<asyxml></code><documentation>Return the endpoint of the altitude from V.</documentation></function></asyxml>*/
  return projection((line)opposite(V)) * ((point)V);
}

/*<asyxml><function type="point" signature="foot(side)"><code></asyxml>*/
point foot(side side)
{/*<asyxml></code><documentation>Return the endpoint of the altitude on 'side'.</documentation></function></asyxml>*/
  return projection((line)side) * point(opposite(side));
}

/*<asyxml><function type="line" signature="altitude(vertex)"><code></asyxml>*/
line altitude(vertex V)
{/*<asyxml></code><documentation>Return the altitude passing through 'V'.</documentation></function></asyxml>*/
  return line(point(V), foot(V));
}

/*<asyxml><function type="line" signature="altitude(vertex)"><code></asyxml>*/
line altitude(side side)
{/*<asyxml></code><documentation>Return the altitude cutting 'side'.</documentation></function></asyxml>*/
  return altitude(opposite(side));
}

/*<asyxml><function type="point" signature="orthocentercenter(triangle)"><code></asyxml>*/
point orthocentercenter(triangle t)
{/*<asyxml></code><documentation>Return the orthocenter of the triangle t.</documentation></function></asyxml>*/
  return orthocentercenter(t.A, t.B, t.C);
}

/*<asyxml><function type="point" signature="centroid(triangle)"><code></asyxml>*/
point centroid(triangle t)
{/*<asyxml></code><documentation>Return the centroid of the triangle 't'.</documentation></function></asyxml>*/
  return (t.A + t.B + t.C)/3;
}

/*<asyxml><function type="point" signature="circumcenter(triangle)"><code></asyxml>*/
point circumcenter(triangle t)
{/*<asyxml></code><documentation>Return the circumcenter of the triangle 't'.</documentation></function></asyxml>*/
  return circumcenter(t.A, t.B, t.C);
}

/*<asyxml><function type="circle" signature="circle(triangle)"><code></asyxml>*/
circle circle(triangle t)
{/*<asyxml></code><documentation>Return the circumcircle of the triangle 't'.</documentation></function></asyxml>*/
  return circle(t.A, t.B, t.C);
}

/*<asyxml><function type="circle" signature="circumcircle(triangle)"><code></asyxml>*/
circle circumcircle(triangle t)
{/*<asyxml></code><documentation>Return the circumcircle of the triangle 't'.</documentation></function></asyxml>*/
  return circle(t.A, t.B, t.C);
}

/*<asyxml><function type="point" signature="incenter(triangle)"><code></asyxml>*/
point incenter(triangle t)
{/*<asyxml></code><documentation>Return the center of the incircle of the triangle 't'.</documentation></function></asyxml>*/
  return incenter(t.A, t.B, t.C);
}

/*<asyxml><function type="real" signature="inradius(triangle)"><code></asyxml>*/
real inradius(triangle t)
{/*<asyxml></code><documentation>Return the radius of the incircle of the triangle 't'.</documentation></function></asyxml>*/
  return inradius(t.A, t.B, t.C);
}

/*<asyxml><function type="circle" signature="incircle(triangle)"><code></asyxml>*/
circle incircle(triangle t)
{/*<asyxml></code><documentation>Return the the incircle of the triangle 't'.</documentation></function></asyxml>*/
  return incircle(t.A, t.B, t.C);
}

/*<asyxml><function type="point" signature="excenter(side,triangle)"><code></asyxml>*/
point excenter(side side)
{/*<asyxml></code><documentation>Return the center of the excircle tangent with the side 'side' of its triangle.
   side = 0 means AB, 1 means AC, other means BC.
   One must use the predefined sides t.AB, t.AC where 't' is a triangle....</documentation></function></asyxml>*/
  point op;
  triangle t = side.t;
  int n = numarray[abs(side.n) - 1];
  if(n == 1) op = excenter(t.A, t.B, t.C);
  else  if(n == 2) op = excenter(t.B, t.C, t.A);
  else op = excenter(t.C, t.A, t.B);
  return op;
}

/*<asyxml><function type="real" signature="exradius(side,triangle)"><code></asyxml>*/
real exradius(side side)
{/*<asyxml></code><documentation>Return radius of the excircle tangent with the side 'side' of its triangle.
   side = 0 means AB, 1 means BC, other means CA.
   One must use the predefined sides t.AB, t.AC where 't' is a triangle....</documentation></function></asyxml>*/
  real or;
  triangle t = side.t;
  int n = numarray[abs(side.n) - 1];
  if(n == 1) or = exradius(t.A, t.B, t.C);
  else  if(n == 2) or = exradius(t.B, t.C, t.A);
  else or = exradius(t.A, t.C, t.B);
  return or;
}

/*<asyxml><function type="circle" signature="excircle(side,triangle)"><code></asyxml>*/
circle excircle(side side)
{/*<asyxml></code><documentation>Return the excircle tangent with the side 'side' of its triangle.
   side = 0 means AB, 1 means AC, other means BC.
   One must use the predefined sides t.AB, t.AC where 't' is a triangle....</documentation></function></asyxml>*/
  circle oc;
  int n = numarray[abs(side.n) - 1];
  triangle t = side.t;
  if(n == 1) oc = excircle(t.A, t.B, t.C);
  else  if(n == 2) oc = excircle(t.B, t.C, t.A);
  else oc = excircle(t.A, t.C, t.B);
  return oc;
}

/*<asyxml><struct signature="trilinear"><code></asyxml>*/
struct trilinear
{/*<asyxml></code><documentation>Trilinear coordinates 'a:b:c' relative to triangle 't'.
   <url href = "http://mathworld.wolfram.com/TrilinearCoordinates.html"/></documentation><property type = "real" signature="a,b,c"><code></asyxml>*/
  real a,b,c;/*<asyxml></code><documentation>The trilinear coordinates.</documentation></property><property type = "triangle" signature="t"><code></asyxml>*/
  triangle t;/*<asyxml></code><documentation>The reference triangle.</documentation></property></asyxml>*/
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="trilinear" signature="trilinear(triangle,real,real,real)"><code></asyxml>*/
trilinear trilinear(triangle t, real a, real b, real c)
{/*<asyxml></code><documentation>Return the trilinear coordinates relative to 't'.
   <url href = "http://mathworld.wolfram.com/TrilinearCoordinates.html"/></documentation></function></asyxml>*/
  trilinear ot;
  ot.a = a; ot.b = b; ot.c = c;
  ot.t = t;
  return ot;
}

/*<asyxml><function type="trilinear" signature="trilinear(triangle,point)"><code></asyxml>*/
trilinear trilinear(triangle t, point M)
{/*<asyxml></code><documentation>Return the trilinear coordinates of 'M' relative to 't'.
   <url href = "http://mathworld.wolfram.com/TrilinearCoordinates.html"/></documentation></function></asyxml>*/
  trilinear ot;
  pair m = locate(M);
  int sameside(pair A, pair B, pair m, pair p)
  {// Return 1 if 'm' and 'p' are same side of line (AB) else return -1.
    pair mil = (A + B)/2;
    pair mA = rotate(90, mil) * A;
    pair mB = rotate(-90, mil) * A;
    return (abs(m - mA) <= abs(m - mB)) == (abs(p - mA) <= abs(p - mB)) ? 1 : -1;
  }
  real det(pair a, pair b) {return a.x * b.y - a.y * b.x;}
  real area(pair a, pair b, pair c){return 0.5 * abs(det(a, b) + det(b, c) + det(c, a));}
  pair A = t.A, B = t.B, C = t.C;
  real t1 = area(B, C, m), t2 = area(C, A, m), t3 = area(A, B, m);
  ot.a = sameside(B, C, A, m) * t1/t.a();
  ot.b = sameside(A, C, B, m) * t2/t.b();
  ot.c = sameside(A, B, C, m) * t3/t.c();
  ot.t = t;
  return ot;
}

/*<asyxml><function type="void" signature="write(trilinear)"><code></asyxml>*/
void write(trilinear tri)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  write(format("%f : ", tri.a) + format("%f : ", tri.b) + format("%f", tri.c));
}

/*<asyxml><function type="point" signature="trilinear(triangle,real,real,real)"><code></asyxml>*/
point point(trilinear tri)
{/*<asyxml></code><documentation>Return the trilinear coordinates relative to 't'.
   <url href = "http://mathworld.wolfram.com/TrilinearCoordinates.html"/></documentation></function></asyxml>*/
  triangle t = tri.t;
  return masscenter(0.5 * t.a() * mass(t.A, tri.a),
                    0.5 * t.b() * mass(t.B, tri.b),
                    0.5 * t.c() * mass(t.C, tri.c));
}

/*<asyxml><function type="int[]" signature="tricoef(side)"><code></asyxml>*/
int[] tricoef(side side)
{/*<asyxml></code><documentation>Return an array of integer (values are 0 or 1) which represents 'side'.
   For example, side = t.BC will be represented by {0, 1, 1}.</documentation></function></asyxml>*/
  int[] oi;
  int n = numarray[abs(side.n) - 1];
  oi.push((n == 1 || n == 3) ? 1 : 0);
  oi.push((n == 1 || n == 2) ? 1 : 0);
  oi.push((n == 2 || n == 3) ? 1 : 0);
  return oi;
}

/*<asyxml><operator type = "point" signature="cast(trilinear)"><code></asyxml>*/
point operator cast(trilinear tri)
{/*<asyxml></code><documentation>Cast trilinear to point.
   One may use the routine 'point(trilinear)' to force the casting.</documentation></operator></asyxml>*/
  return point(tri);
}

/*<asyxml><typedef type = "centerfunction" return = "real" params = "real, real, real"><code></asyxml>*/
typedef real centerfunction(real, real, real);/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/TriangleCenterFunction.html"/></documentation></typedef></asyxml>*/

/*<asyxml><function type="trilinear" signature="trilinear(triangle,centerfunction,real,real,real)"><code></asyxml>*/
trilinear trilinear(triangle t, centerfunction f, real a = t.a(), real b = t.b(), real c = t.c())
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/TriangleCenterFunction.html"/></documentation></function></asyxml>*/
  return trilinear(t, f(a, b, c), f(b, c, a), f(c, a, b));
}

/*<asyxml><function type="point" signature="symmedian(triangle)"><code></asyxml>*/
point symmedian(triangle t)
{/*<asyxml></code><documentation>Return the symmedian point of 't'.</documentation></function></asyxml>*/
  point A, B, C;
  real a = t.a(), b = t.b(), c = t.c();
  A = trilinear(t, 0, b, c);
  B = trilinear(t, a, 0, c);
  return intersectionpoint(line(t.A, A), line(t.B, B));
}

/*<asyxml><function type="point" signature="symmedian(side)"><code></asyxml>*/
point symmedian(side side)
{/*<asyxml></code><documentation>The symmedian point on the side 'side'.</documentation></function></asyxml>*/
  triangle t = side.t;
  int n = numarray[abs(side.n) - 1];
  if(n == 1) return trilinear(t, t.a(), t.b(), 0);
  if(n == 2) return trilinear(t, 0, t.b(), t.c());
  return trilinear(t, t.a(), 0, t.c());
}

/*<asyxml><function type="line" signature="symmedian(vertex)"><code></asyxml>*/
line symmedian(vertex V)
{/*<asyxml></code><documentation>Return the symmedian passing through 'V'.</documentation></function></asyxml>*/
  return line(point(V), symmedian(V.t));
}

/*<asyxml><function type="triangle" signature="cevian(triangle,point)"><code></asyxml>*/
triangle cevian(triangle t, point P)
{/*<asyxml></code><documentation>Return the Cevian triangle with respect of 'P'
   <url href = "http://mathworld.wolfram.com/CevianTriangle.html"/>.</documentation></function></asyxml>*/
  trilinear tri = trilinear(t, locate(P));
  point A = point(trilinear(t, 0, tri.b, tri.c));
  point B = point(trilinear(t, tri.a, 0, tri.c));
  point C = point(trilinear(t, tri.a, tri.b, 0));
  return triangle(A, B, C);
}

/*<asyxml><function type="point" signature="cevian(side,point)"><code></asyxml>*/
point cevian(side side, point P)
{/*<asyxml></code><documentation>Return the Cevian point on 'side' with respect of 'P'.</documentation></function></asyxml>*/
  triangle t = side.t;
  trilinear tri = trilinear(t, locate(P));
  int[] s = tricoef(side);
  return point(trilinear(t, s[0] * tri.a, s[1] * tri.b, s[2] * tri.c));
}

/*<asyxml><function type="line" signature="cevian(vertex,point)"><code></asyxml>*/
line cevian(vertex V, point P)
{/*<asyxml></code><documentation>Return line passing through 'V' and its Cevian image with respect of 'P'.</documentation></function></asyxml>*/
  return line(point(V), cevian(opposite(V), P));
}

/*<asyxml><function type="point" signature="gergonne(triangle)"><code></asyxml>*/
point gergonne(triangle t)
{/*<asyxml></code><documentation>Return the Gergonne point of 't'.</documentation></function></asyxml>*/
  real f(real a, real b, real c){return 1/(a * (b + c - a));}
  return point(trilinear(t, f));
}

/*<asyxml><function type="point[]" signature="fermat(triangle)"><code></asyxml>*/
point[] fermat(triangle t)
{/*<asyxml></code><documentation>Return the Fermat points of 't'.</documentation></function></asyxml>*/
  point[] P;
  real A = t.alpha(), B = t.beta(), C = t.gamma();
  P.push(point(trilinear(t, 1/Sin(A + 60), 1/Sin(B + 60), 1/Sin(C + 60))));
  P.push(point(trilinear(t, 1/Sin(A - 60), 1/Sin(B - 60), 1/Sin(C - 60))));
  return P;
}

/*<asyxml><function type="point" signature="isotomicconjugate(triangle,point)"><code></asyxml>*/
point isotomicconjugate(triangle t, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsotomicConjugate.html"/></documentation></function></asyxml>*/
  if(!inside(t.Path(), locate(M))) abort("isotomic: the point must be inside the triangle.");
  trilinear tr = trilinear(t, M);
  return point(trilinear(t, 1/(t.a()^2 * tr.a), 1/(t.b()^2 * tr.b), 1/(t.c()^2 * tr.c)));
}

/*<asyxml><function type="line" signature="isotomic(vertex,point)"><code></asyxml>*/
line isotomic(vertex V, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsotomicConjugate.html"/>.</documentation></function></asyxml>*/
  side op = opposite(V);
  return line(V, rotate(180, midpoint(op)) * cevian(op, M));
}

/*<asyxml><function type="point" signature="isotomic(side,point)"><code></asyxml>*/
point isotomic(side side, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsotomicConjugate.html"/></documentation></function></asyxml>*/
  return intersectionpoint(isotomic(opposite(side), M), side);
}

/*<asyxml><function type="triangle" signature="isotomic(triangle,point)"><code></asyxml>*/
triangle isotomic(triangle t, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsotomicConjugate.html"/></documentation></function></asyxml>*/
  return triangle(isotomic(t.BC, M), isotomic(t.CA, M), isotomic(t.AB, M));
}

/*<asyxml><function type="point" signature="isogonalconjugate(triangle,point)"><code></asyxml>*/
point isogonalconjugate(triangle t, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsogonalConjugate.html"/></documentation></function></asyxml>*/
  trilinear tr = trilinear(t, M);
  return point(trilinear(t, 1/tr.a, 1/tr.b, 1/tr.c));
}

/*<asyxml><function type="point" signature="isogonal(side,point)"><code></asyxml>*/
point isogonal(side side, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsogonalConjugate.html"/></documentation></function></asyxml>*/
  return cevian(side, isogonalconjugate(side.t, M));
}

/*<asyxml><function type="line" signature="isogonal(vertex,point)"><code></asyxml>*/
line isogonal(vertex V, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsogonalConjugate.html"/></documentation></function></asyxml>*/
  return line(V, isogonal(opposite(V), M));
}

/*<asyxml><function type="triangle" signature="isogonal(triangle,point)"><code></asyxml>*/
triangle isogonal(triangle t, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/IsogonalConjugate.html"/></documentation></function></asyxml>*/
  return triangle(isogonal(t.BC, M), isogonal(t.CA, M), isogonal(t.AB, M));
}

/*<asyxml><function type="triangle" signature="pedal(triangle,point)"><code></asyxml>*/
triangle pedal(triangle t, point M)
{/*<asyxml></code><documentation>Return the pedal triangle of 'M' in 't'.
   <url href = "http://mathworld.wolfram.com/PedalTriangle.html"/></documentation></function></asyxml>*/
  return triangle(projection(t.BC) * M, projection(t.AC) * M, projection(t.AB) * M);
}

/*<asyxml><function type="triangle" signature="pedal(triangle,point)"><code></asyxml>*/
line pedal(side side, point M)
{/*<asyxml></code><documentation>Return the pedal line of 'M' cutting 'side'.
   <url href = "http://mathworld.wolfram.com/PedalTriangle.html"/></documentation></function></asyxml>*/
  return line(M, projection(side) * M);
}

/*<asyxml><function type="triangle" signature="antipedal(triangle,point)"><code></asyxml>*/
triangle antipedal(triangle t, point M)
{/*<asyxml></code><documentation><url href = "http://mathworld.wolfram.com/AntipedalTriangle.html"/></documentation></function></asyxml>*/
  trilinear Tm = trilinear(t, M);
  real a = Tm.a, b = Tm.b, c = Tm.c;
  real CA = Cos(t.alpha()), CB = Cos(t.beta()), CC = Cos(t.gamma());
  point A = trilinear(t, -(b + a * CC) * (c + a * CB), (c + a * CB) * (a + b * CC), (b + a * CC) * (a + c * CB));
  point B = trilinear(t, (c + b * CA) * (b + a * CC), -(c + b * CA) * (a + b * CC), (a + b * CC) * (b + c * CA));
  point C = trilinear(t, (b + c * CA) * (c + a * CB), (a + c * CB) * (c + b * CA), -(a + c * CB) * (b + c * CA));
  return triangle(A, B, C);
}

/*<asyxml><function type="triangle" signature="extouch(triangle)"><code></asyxml>*/
triangle extouch(triangle t)
{/*<asyxml></code><documentation>Return the extouch triangle of the triangle 't'.
   The extouch triangle of 't' is the triangle formed by the points
   of tangency of a triangle 't' with its excircles.</documentation></function></asyxml>*/
  point A, B, C;
  real a = t.a(), b = t.b(), c = t.c();
  A = trilinear(t, 0, (a - b + c)/b, (a + b - c)/c);
  B = trilinear(t, (-a + b + c)/a, 0, (a + b - c)/c);
  C = trilinear(t, (-a + b + c)/a, (a - b + c)/b, 0);
  return triangle(A, B, C);
}

/*<asyxml><function type="triangle" signature="extouch(triangle)"><code></asyxml>*/
triangle incentral(triangle t)
{/*<asyxml></code><documentation>Return the incentral triangle of the triangle 't'.
   It is the triangle whose vertices are determined by the intersections of the
   reference triangle's angle bisectors with the respective opposite sides.</documentation></function></asyxml>*/
  point A, B, C;
  // real a = t.a(), b = t.b(), c = t.c();
  A = trilinear(t, 0, 1, 1);
  B = trilinear(t, 1, 0, 1);
  C = trilinear(t, 1, 1, 0);
  return triangle(A, B, C);
}

/*<asyxml><function type="triangle" signature="extouch(side)"><code></asyxml>*/
triangle extouch(side side)
{/*<asyxml></code><documentation>Return the triangle formed by the points of tangency of the triangle referenced by 'side' with its excircles.
   One vertex of the returned triangle is on the segment 'side'.</documentation></function></asyxml>*/
  triangle t = side.t;
  transform p1 = projection((line)t.AB);
  transform p2 = projection((line)t.AC);
  transform p3 = projection((line)t.BC);
  point EP = excenter(side);
  return triangle(p3 * EP, p2 * EP, p1 * EP);
}

/*<asyxml><function type="point" signature="bisectorpoint(side)"><code></asyxml>*/
point bisectorpoint(side side)
{/*<asyxml></code><documentation>The intersection point of the angle bisector from the
   opposite point of 'side' with the side 'side'.</documentation></function></asyxml>*/
  triangle t = side.t;
  int n = numarray[abs(side.n) - 1];
  if(n == 1) return trilinear(t, 1, 1, 0);
  if(n == 2) return trilinear(t, 0, 1, 1);
  return trilinear(t, 1, 0, 1);
}

/*<asyxml><function type="line" signature="bisector(vertex,real)"><code></asyxml>*/
line bisector(vertex V, real angle = 0)
{/*<asyxml></code><documentation>Return the interior bisector passing through 'V' rotated by angle (in degrees)
   around 'V'.</documentation></function></asyxml>*/
  return rotate(angle, point(V)) * line(point(V), incenter(V.t));
}

/*<asyxml><function type="line" signature="bisector(side)"><code></asyxml>*/
line bisector(side side)
{/*<asyxml></code><documentation>Return the bisector of the line segment 'side'.</documentation></function></asyxml>*/
  return bisector(segment(side));
}

/*<asyxml><function type="point" signature="intouch(side)"><code></asyxml>*/
point intouch(side side)
{/*<asyxml></code><documentation>The point of tangency on the side 'side' of its incircle.</documentation></function></asyxml>*/
  triangle t = side.t;
  real a = t.a(), b = t.b(), c = t.c();
  int n = numarray[abs(side.n) - 1];
  if(n == 1) return trilinear(t, b * c/(-a + b + c), a * c/(a - b + c), 0);
  if(n == 2) return trilinear(t, 0, a * c/(a - b + c), a * b/(a + b - c));
  return trilinear(t, b * c/(-a + b + c), 0, a * b/(a + b - c));
}

/*<asyxml><function type="triangle" signature="intouch(triangle)"><code></asyxml>*/
triangle intouch(triangle t)
{/*<asyxml></code><documentation>Return the intouch triangle of the triangle 't'.
   The intouch triangle of 't' is the triangle formed by the points
   of tangency of a triangle 't' with its incircles.</documentation></function></asyxml>*/
  point A, B, C;
  real a = t.a(), b = t.b(), c = t.c();
  A = trilinear(t, 0, a * c/(a - b + c), a * b/(a + b - c));
  B = trilinear(t, b * c/(-a + b + c), 0, a * b/(a + b - c));
  C = trilinear(t, b * c/(-a + b + c), a * c/(a - b + c), 0);
  return triangle(A, B, C);
}

/*<asyxml><function type="triangle" signature="tangential(triangle)"><code></asyxml>*/
triangle tangential(triangle t)
{/*<asyxml></code><documentation>Return the tangential triangle of the triangle 't'.
   The tangential triangle of 't' is the triangle formed by the lines
   tangent to the circumcircle of the given triangle 't' at its vertices.</documentation></function></asyxml>*/
  point A, B, C;
  real a = t.a(), b = t.b(), c = t.c();
  A = trilinear(t, -a, b, c);
  B = trilinear(t, a, -b, c);
  C = trilinear(t, a, b, -c);
  return triangle(A, B, C);
}

/*<asyxml><function type="triangle" signature="medial(triangle t)"><code></asyxml>*/
triangle medial(triangle t)
{/*<asyxml></code><documentation>Return the triangle whose vertices are midpoints of the sides of 't'.</documentation></function></asyxml>*/
  return triangle(midpoint(t.BC), midpoint(t.AC), midpoint(t.AB));
}

/*<asyxml><function type="line" signature="median(vertex)"><code></asyxml>*/
line median(vertex V)
{/*<asyxml></code><documentation>Return median from 'V'.</documentation></function></asyxml>*/
  return line(point(V), midpoint(segment(opposite(V))));
}

/*<asyxml><function type="line" signature="median(side)"><code></asyxml>*/
line median(side side)
{/*<asyxml></code><documentation>Return median from the opposite vertex of 'side'.</documentation></function></asyxml>*/
  return median(opposite(side));
}

/*<asyxml><function type="triangle" signature="orthic(triangle)"><code></asyxml>*/
triangle orthic(triangle t)
{/*<asyxml></code><documentation>Return the triangle whose vertices are endpoints of the altitudes from each of the vertices of 't'.</documentation></function></asyxml>*/
  return triangle(foot(t.BC), foot(t.AC), foot(t.AB));
}

/*<asyxml><function type="triangle" signature="symmedial(triangle)"><code></asyxml>*/
triangle symmedial(triangle t)
{/*<asyxml></code><documentation>Return the symmedial triangle of 't'.</documentation></function></asyxml>*/
  point A, B, C;
  real a = t.a(), b = t.b(), c = t.c();
  A = trilinear(t, 0, b, c);
  B = trilinear(t, a, 0, c);
  C = trilinear(t, a, b, 0);
  return triangle(A, B, C);
}

/*<asyxml><function type="triangle" signature="anticomplementary(triangle)"><code></asyxml>*/
triangle anticomplementary(triangle t)
{/*<asyxml></code><documentation>Return the triangle which has the given triangle 't' as its medial triangle.</documentation></function></asyxml>*/
  real a = t.a(), b = t.b(), c = t.c();
  real ab = a * b, bc = b * c, ca = c * a;
  point A = trilinear(t, -bc, ca, ab);
  point B = trilinear(t, bc, -ca, ab);
  point C = trilinear(t, bc, ca, -ab);
  return triangle(A, B, C);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(triangle,line,bool)"><code></asyxml>*/
point[] intersectionpoints(triangle t, line l, bool extended = false)
{/*<asyxml></code><documentation>Return the intersection points.
   If 'extended' is true, the sides are lines else the sides are segments.
   intersectionpoints(line, triangle, bool) is also defined.</documentation></function></asyxml>*/
  point[] OP;
  void addpoint(point P)
  {
    if(defined(P)) {
      bool exist = false;
      for (int i = 0; i < OP.length; ++i) {
        if(P == OP[i]) {exist = true; break;}
      }
      if(!exist) OP.push(P);
    }
  }
  if(extended) {
    for (int i = 1; i <= 3; ++i) {
      addpoint(intersectionpoint(t.line(i), l));
    }
  } else {
    for (int i = 1; i <= 3; ++i) {
      addpoint(intersectionpoint((segment)t.line(i), l));
    }
  }
  return OP;
}

point[] intersectionpoints(line l, triangle t, bool extended = false)
{
  return intersectionpoints(t, l, extended);
}

/*<asyxml><function type="vector" signature="dir(vertex)"><code></asyxml>*/
vector dir(vertex V)
{/*<asyxml></code><documentation>The direction (towards the outside of the triangle) of the interior angle bisector of 'V'.</documentation></function></asyxml>*/
  triangle t = V.t;
  if(V.n == 1) return vector(defaultcoordsys, (-dir(t.A--t.B, t.A--t.C)));
  if(V.n == 2) return vector(defaultcoordsys, (-dir(t.B--t.A, t.B--t.C)));
  return vector(defaultcoordsys, (-dir(t.C--t.A, t.C--t.B)));
}

/*<asyxml><function type="void" signature="lvoid label(picture,Label,vertex,pair,real,pen,filltype)"><code></asyxml>*/
void label(picture pic = currentpicture, Label L, vertex V,
           pair align = dir(V),
           real alignFactor = 1,
           pen p = nullpen, filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw 'L' on picture 'pic' at vertex 'V' aligned by 'alignFactor * align'.</documentation></function></asyxml>*/
  label(pic, L, locate(point(V)), alignFactor * align, p, filltype);
}

/*<asyxml><function type="void" signature="label(picture,Label,Label,Label,triangle,real,real,pen,filltype)"><code></asyxml>*/
void label(picture pic = currentpicture, Label LA = "$A$",
           Label LB = "$B$", Label LC = "$C$",
           triangle t,
           real alignAngle = 0,
           real alignFactor = 1,
           pen p = nullpen, filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw labels LA, LB and LC aligned in the rotated (by 'alignAngle' in degrees) direction
   (towards the outside of the triangle) of the interior angle bisector of vertices.
   One can  individually modify the alignment by setting the Label parameter 'align'.</documentation></function></asyxml>*/
  Label lla = LA.copy();
  lla.align(lla.align, rotate(alignAngle) * locate(dir(t.VA)));
  label(pic, LA, t.VA, align = lla.align.dir, alignFactor = alignFactor, p, filltype);
  Label llb = LB.copy();
  llb.align(llb.align, rotate(alignAngle) * locate(dir(t.VB)));
  label(pic, llb, t.VB, align = llb.align.dir, alignFactor = alignFactor, p, filltype);
  Label llc = LC.copy();
  llc.align(llc.align, rotate(alignAngle) * locate(dir(t.VC)));
  label(pic, llc, t.VC, align = llc.align.dir, alignFactor = alignFactor, p, filltype);
}

/*<asyxml><function type="void" signature="show(picture,Label,Label,Label,Label,Label,Label,triangle,pen,filltype)"><code></asyxml>*/
void show(picture pic = currentpicture,
          Label LA = "$A$", Label LB = "$B$", Label LC = "$C$",
          Label La = "$a$", Label Lb = "$b$", Label Lc = "$c$",
          triangle t, pen p = currentpen, filltype filltype = NoFill)
{/*<asyxml></code><documentation>Draw triangle and labels of sides and vertices.</documentation></function></asyxml>*/
  pair a = locate(t.A), b = locate(t.B), c = locate(t.C);
  draw(pic, a--b--c--cycle, p);
  label(pic, LA, a, -dir(a--b, a--c), p, filltype);
  label(pic, LB, b, -dir(b--a, b--c), p, filltype);
  label(pic, LC, c, -dir(c--a, c--b), p, filltype);
  pair aligna = I * unit(c - b), alignb = I * unit(c - a), alignc = I * unit(b - a);
  pair mAB = locate(midpoint(t.AB)), mAC = locate(midpoint(t.AC)), mBC = locate(midpoint(t.BC));
  label(pic, La, b--c, align = rotate(dot(a - mBC, aligna) > 0 ? 180 :0) * aligna, p);
  label(pic, Lb, a--c, align = rotate(dot(b - mAC, alignb) > 0 ? 180 :0) * alignb, p);
  label(pic, Lc, a--b, align = rotate(dot(c - mAB, alignc) > 0 ? 180 :0) * alignc, p);
}

/*<asyxml><function type="void" signature="draw(picture,triangle,pen,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, triangle t, pen p = currentpen, marker marker = nomarker)
{/*<asyxml></code><documentation>Draw sides of the triangle 't' on picture 'pic' using pen 'p'.</documentation></function></asyxml>*/
  draw(pic, t.Path(), p, marker);
}

/*<asyxml><function type="void" signature="draw(picture,triangle[],pen,marker)"><code></asyxml>*/
void draw(picture pic = currentpicture, triangle[] t, pen p = currentpen, marker marker = nomarker)
{/*<asyxml></code><documentation>Draw sides of the triangles 't' on picture 'pic' using pen 'p'.</documentation></function></asyxml>*/
  for(int i = 0; i < t.length; ++i) draw(pic, t[i], p, marker);
}

/*<asyxml><function type="void" signature="drawline(picture,triangle,pen)"><code></asyxml>*/
void drawline(picture pic = currentpicture, triangle t, pen p = currentpen)
{/*<asyxml></code><documentation>Draw lines of the triangle 't' on picture 'pic' using pen 'p'.</documentation></function></asyxml>*/
  draw(t, p);
  draw(pic, line(t.A, t.B), p);
  draw(pic, line(t.A, t.C), p);
  draw(pic, line(t.B, t.C), p);
}

/*<asyxml><function type="void" signature="dot(picture,triangle,pen)"><code></asyxml>*/
void dot(picture pic = currentpicture, triangle t, pen p = currentpen)
{/*<asyxml></code><documentation>Draw a dot at each vertex of 't'.</documentation></function></asyxml>*/
  dot(pic, t.A^^t.B^^t.C, p);
}
// *.......................TRIANGLES.......................*
// *=======================================================*

// *=======================================================*
// *.......................INVERSIONS......................*
/*<asyxml><function type="point" signature="inverse(real k,point,point)"><code></asyxml>*/
point inverse(real k, point A, point M)
{/*<asyxml></code><documentation>Return the inverse point of 'M' with respect to point A and inversion radius 'k'.</documentation></function></asyxml>*/
  return A + k/conj(M - A);
}

/*<asyxml><function type="point" signature="radicalcenter(circle,circle)"><code></asyxml>*/
point radicalcenter(circle c1, circle c2)
{/*<asyxml></code><documentation><url href = "http://fr.wikipedia.org/wiki/Puissance_d'un_point_par_rapport_%C3%A0_un_cercle"/></documentation></function></asyxml>*/
  point[] P = standardizecoordsys(c1.C, c2.C);
  real k = c1.r^2 - c2.r^2;
  pair C1 = locate(c1.C);
  pair C2 = locate(c2.C);
  pair oop = C2 - C1;
  pair K = (abs(oop) == 0) ?
    (infinity, infinity) :
    midpoint(C1--C2) + 0.5 * k * oop/dot(oop, oop);
  return point(P[0].coordsys, K/P[0].coordsys);
}

/*<asyxml><function type="line" signature="radicalline(circle,circle)"><code></asyxml>*/
line radicalline(circle c1, circle c2)
{/*<asyxml></code><documentation><url href = "http://fr.wikipedia.org/wiki/Puissance_d'un_point_par_rapport_%C3%A0_un_cercle"/></documentation></function></asyxml>*/
  if (c1.C == c2.C) abort("radicalline: the centers must be distinct");
  return perpendicular(radicalcenter(c1, c2), line(c1.C, c2.C));
}

/*<asyxml><function type="point" signature="radicalcenter(circle,circle,circle)"><code></asyxml>*/
point radicalcenter(circle c1, circle c2, circle c3)
{/*<asyxml></code><documentation><url href = "http://fr.wikipedia.org/wiki/Puissance_d'un_point_par_rapport_%C3%A0_un_cercle"/></documentation></function></asyxml>*/
  return intersectionpoint(radicalline(c1, c2), radicalline(c1, c3));
}

/*<asyxml><struct signature="inversion"><code></asyxml>*/
struct inversion
{/*<asyxml></code><documentation>http://mathworld.wolfram.com/Inversion.html</documentation></asyxml>*/
  point C;
  real k;
}/*<asyxml></struct></asyxml>*/

/*<asyxml><function type="inversion" signature="inversion(real,point)"><code></asyxml>*/
inversion inversion(real k, point C)
{/*<asyxml></code><documentation>Return the inversion with respect to 'C' having inversion radius 'k'.</documentation></function></asyxml>*/
  inversion oi;
  oi.k = k;
  oi.C = C;
  return oi;
}
/*<asyxml><function type="inversion" signature="inversion(real,point)"><code></asyxml>*/
inversion inversion(point C, real k)
{/*<asyxml></code><documentation>Return the inversion with respect to 'C' having inversion radius 'k'.</documentation></function></asyxml>*/
  return inversion(k, C);
}

/*<asyxml><function type="inversion" signature="inversion(circle,circle)"><code></asyxml>*/
inversion inversion(circle c1, circle c2, real sgn = 1)
{/*<asyxml></code><documentation>Return the inversion which transforms 'c1' to
   . 'c2' and positive inversion radius if 'sgn > 0';
   . 'c2' and negative inversion radius if 'sgn < 0';
   . 'c1' and 'c2' to 'c2' if 'sgn = 0'.</documentation></function></asyxml>*/
  if(sgn == 0) {
    point O = radicalcenter(c1, c2);
    return inversion(O^c1, O);
  }
  real a = abs(c1.r/c2.r);
  if(sgn > 0) {
    point O = c1.C + a/abs(1 - a) * (c2.C - c1.C);
    return inversion(a * abs(abs(O - c2.C)^2 - c2.r^2), O);
  }
  point O = c1.C + a/abs(1 + a) * (c2.C - c1.C);
  return inversion(-a * abs(abs(O - c2.C)^2 - c2.r^2), O);
}

/*<asyxml><function type="inversion" signature="inversion(circle,circle,circle)"><code></asyxml>*/
inversion inversion(circle c1, circle c2, circle c3)
{/*<asyxml></code><documentation>Return the inversion which transform 'c1' to 'c1', 'c2' to 'c2' and 'c3' to 'c3'.</documentation></function></asyxml>*/
  point Rc = radicalcenter(c1, c2, c3);
  return inversion(Rc, Rc^c1);
}

circle operator cast(inversion i){return circle(i.C, sgn(i.k) * sqrt(abs(i.k)));}
/*<asyxml><function type="circle" signature="circle(inversion)"><code></asyxml>*/
circle circle(inversion i)
{/*<asyxml></code><documentation>Return the inversion circle of 'i'.</documentation></function></asyxml>*/
  return i;
}

inversion operator cast(circle c)
{
  return inversion(sgn(c.r) * c.r^2, c.C);
}
/*<asyxml><function type="inversion" signature="inversion(circle)"><code></asyxml>*/
inversion inversion(circle c)
{/*<asyxml></code><documentation>Return the inversion represented by the circle of 'c'.</documentation></function></asyxml>*/
  return c;
}

/*<asyxml><operator type = "point" signature="*(inversion,point)"><code></asyxml>*/
point operator *(inversion i, point P)
{/*<asyxml></code><documentation>Provide inversion * point.</documentation></operator></asyxml>*/
  return inverse(i.k, i.C, P);
}

void lineinversion()
{
  warning("lineinversion", "the inversion of the line is not a circle.
The returned circle has an infinite radius, circle.l has been set.");
}


/*<asyxml><function type="circle" signature="inverse(real,point,line)"><code></asyxml>*/
circle inverse(real k, point A, line l)
{/*<asyxml></code><documentation>Return the inverse circle of 'l' with
   respect to point 'A' and inversion radius 'k'.</documentation></function></asyxml>*/
  if(A @ l) {
    lineinversion();
    circle C = circle(A, infinity);
    C.l = l;
    return C;
  }
  point Ap = inverse(k, A, l.A), Bp = inverse(k, A, l.B);
  return circle(A, Ap, Bp);
}

/*<asyxml><operator type = "circle" signature="*(inversion,line)"><code></asyxml>*/
circle operator *(inversion i, line l)
{/*<asyxml></code><documentation>Provide inversion * line for lines that don't pass through the inversion center.</documentation></operator></asyxml>*/
  return inverse(i.k, i.C, l);
}

/*<asyxml><function type="circle" signature="inverse(real,point,circle)"><code></asyxml>*/
circle inverse(real k, point A, circle c)
{/*<asyxml></code><documentation>Return the inverse circle of 'c' with
   respect to point A and inversion radius 'k'.</documentation></function></asyxml>*/
  if(degenerate(c)) return inverse(k, A, c.l);
  if(A @ c) {
    lineinversion();
    point M = rotate(180, c.C) * A, Mp = rotate(90, c.C) * A;
    circle oc = circle(A, infinity);
    oc.l = line(inverse(k, A, M), inverse(k, A, Mp));
    return oc;
  }
  point[] P = standardizecoordsys(A, c.C);
  real s = k/((P[1].x - P[0].x)^2 + (P[1].y - P[0].y)^2 - c.r^2);
  return circle(P[0] + s * (P[1]-P[0]), abs(s) * c.r);
}

/*<asyxml><operator type = "circle" signature="*(inversion,circle)"><code></asyxml>*/
circle operator *(inversion i, circle c)
{/*<asyxml></code><documentation>Provide inversion * circle.</documentation></operator></asyxml>*/
  return inverse(i.k, i.C, c);
}
// *.......................INVERSIONS......................*
// *=======================================================*

// *=======================================================*
// *........................FOOTER.........................*
/*<asyxml><function type="point[]" signature="intersectionpoints(line,circle)"><code></asyxml>*/
point[] intersectionpoints(line l, circle c)
{/*<asyxml></code><documentation>Note that the line 'l' may be a segment by casting.
   intersectionpoints(circle, line) is also defined.</documentation></function></asyxml>*/
  if(degenerate(c)) return new point[]{intersectionpoint(l, c.l)};
  point[] op;
  coordsys R = samecoordsys(l.A, c.C) ?
    l.A.coordsys : defaultcoordsys;
  coordsys Rp = defaultcoordsys;
  circle cc = circle(changecoordsys(Rp, c.C), c.r);
  point proj = projection(l) * c.C;
  if(proj @ cc) { // The line is a tangente of the circle.
    if(proj @ l) op.push(proj);// line may be a segement...
  } else {
    coordsys Rc = cartesiansystem(c.C, (1, 0), (0, 1));
    line ll = changecoordsys(Rc, l);
    pair[] P = intersectionpoints(ll.A.coordinates, ll.B.coordinates,
                                1, 0, 1, 0, 0, -c.r^2);
    for (int i = 0; i < P.length; ++i) {
      point inter = changecoordsys(R, point(Rc, P[i]));
      if(inter @ l) op.push(inter);
    }
  }
  return op;
}

point[] intersectionpoints(circle c, line l)
{
  return intersectionpoints(l, c);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(line,ellipse)"><code></asyxml>*/
point[] intersectionpoints(line l, ellipse el)
{/*<asyxml></code><documentation>Note that the line 'l' may be a segment by casting.
   intersectionpoints(ellipse, line) is also defined.</documentation></function></asyxml>*/
  if(el.e == 0) return intersectionpoints(l, (circle)el);
  if(degenerate(el)) return new point[]{intersectionpoint(l, el.l)};
  point[] op;
  coordsys R = samecoordsys(l.A, el.C) ? l.A.coordsys : defaultcoordsys;
  coordsys Rp = defaultcoordsys;
  line ll = changecoordsys(Rp, l);
  ellipse ell = (ellipse) changecoordsys(Rp, el);
  circle C = circle(ell.C, ell.a);
  point[] Ip = intersectionpoints(ll, C);
  if (Ip.length > 0 &&
      (perpendicular(ll, line(ell.F1, Ip[0])) ||
       perpendicular(ll, line(ell.F2, Ip[0])))) {
    // http://www.mathcurve.com/courbes2d/ellipse/ellipse.shtml
    // Definition of the tangent at the antipodal point on the circle.
    // 'l' is a tangent of 'el'
    transform t = scale(el.a/el.b, el.F1, el.F2, el.C, rotate(90, el.C) * el.F1);
    point inter = inverse(t) * intersectionpoints(C, t * ll)[0];
    if(inter @ l) op.push(inter);
  } else {
    coordsys Rc = canonicalcartesiansystem(el);
    line ll = changecoordsys(Rc, l);
    pair[] P = intersectionpoints(ll.A.coordinates, ll.B.coordinates,
                                1/el.a^2, 0, 1/el.b^2, 0, 0, -1);
    for (int i = 0; i < P.length; ++i) {
      point inter = changecoordsys(R, point(Rc, P[i]));
      if(inter @ l) op.push(inter);
    }
  }
  return op;
}

point[] intersectionpoints(ellipse el, line l)
{
  return intersectionpoints(l, el);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(line,parabola)"><code></asyxml>*/
point[] intersectionpoints(line l, parabola p)
{/*<asyxml></code><documentation>Note that the line 'l' may be a segment by casting.
   intersectionpoints(parabola, line) is also defined.</documentation></function></asyxml>*/
  point[] op;
  coordsys R = coordsys(p);
  bool tgt = false;
  line ll = changecoordsys(R, l),
    lv = parallel(p.V, p.D);
  point M = intersectionpoint(lv, ll), tgtp;
  if(finite(M)) {// Test if 'l' is tangent to 'p'
    line l1 = bisector(line(M, p.F));
    line l2 = rotate(90, M) * lv;
    point P = intersectionpoint(l1, l2);
    tgtp = rotate(180, P) * p.F;
    tgt = (tgtp @ l);
  }
  if(tgt) {
    if(tgtp @ l) op.push(tgtp);
  } else {
    real[] eq = changecoordsys(defaultcoordsys, equation(p)).a;
    pair[] tp = intersectionpoints(locate(l.A), locate(l.B), eq);
    point inter;
    for (int i = 0; i < tp.length; ++i) {
      inter = point(R, tp[i]/R);
      if(inter @ l) op.push(inter);
    }
  }
  return op;
}

point[] intersectionpoints(parabola p, line l)
{
  return intersectionpoints(l, p);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(line,hyperbola)"><code></asyxml>*/
point[] intersectionpoints(line l, hyperbola h)
{/*<asyxml></code><documentation>Note that the line 'l' may be a segment by casting.
   intersectionpoints(hyperbola, line) is also defined.</documentation></function></asyxml>*/
  point[] op;
  coordsys R = coordsys(h);
  point A = intersectionpoint(l, h.A1), B = intersectionpoint(l, h.A2);
  point M = midpoint(segment(A, B));
  bool tgt = Finite(M) ? M @ h : false;
  if(tgt) {
    if(M @ l) op.push(M);
  } else {
    real[] eq = changecoordsys(defaultcoordsys, equation(h)).a;
    pair[] tp = intersectionpoints(locate(l.A), locate(l.B), eq);
    point inter;
    for (int i = 0; i < tp.length; ++i) {
      inter = point(R, tp[i]/R);
      if(inter @ l) op.push(inter);
    }
  }
  return op;
}

point[] intersectionpoints(hyperbola h, line l)
{
  return intersectionpoints(l, h);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(line,conic)"><code></asyxml>*/
point[] intersectionpoints(line l, conic co)
{/*<asyxml></code><documentation>Note that the line 'l' may be a segment by casting.
   intersectionpoints(conic, line) is also defined.</documentation></function></asyxml>*/
  point[] op;
  if(co.e < 1) op = intersectionpoints((ellipse)co, l);
  else
    if(co.e == 1) op = intersectionpoints((parabola)co, l);
    else op = intersectionpoints((hyperbola)co, l);
  return op;
}

point[] intersectionpoints(conic co, line l)
{
  return intersectionpoints(l, co);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(bqe,bqe)"><code></asyxml>*/
point[] intersectionpoints(bqe bqe1, bqe bqe2)
{/*<asyxml></code><documentation>Return the intersection of the two conic sections whose equations are 'bqe1' and 'bqe2'.</documentation></function></asyxml>*/
  coordsys R=canonicalcartesiansystem(conic(bqe1));
  real[] a=changecoordsys(R,bqe1).a;
  real[] b=changecoordsys(R,bqe2).a;

  static real e=100 * sqrt(realEpsilon);
  real[] x,y,c;
  point[] P;
  if(abs(a[0]-b[0]) > e || abs(a[1]-b[1]) > e || abs(a[2]-b[2]) > e) {
    c=new real[] {a[0]*a[2]*(-2*b[0]*b[2]+b[1]^2)+a[0]^2*b[2]^2+a[2]^2*b[0]^2,

                  2*a[0]*a[2]*b[1]*b[4]-2*a[2]*a[3]*b[0]*b[2]
                  -2*a[0]*a[2]*b[2]*b[3]+a[2]*a[3]*b[1]^2+2*a[2]^2*b[0]*b[3],

                  a[2]*a[5]*b[1]^2-2*a[2]*a[3]*b[2]*b[3]+2*a[2]^2*b[0]*b[5]
                  +2*a[0]*a[5]*b[2]^2+a[3]^2*b[2]^2-2*a[2]*a[5]*b[0]*b[2]
                  -2*a[0]*a[2]*b[2]*b[5]+a[2]^2*b[3]^2+2*a[2]*a[3]*b[1]*b[4]
                  +a[0]*a[2]*b[4]^2,

                  a[2]*a[3]*b[4]^2+2*a[2]^2*b[3]*b[5]-2*a[2]*a[3]*b[2]*b[5]
                  -2*a[2]*a[5]*b[2]*b[3]+2*a[2]*a[5]*b[1]*b[4],

                  -2*a[2]*a[5]*b[2]*b[5]+a[5]^2*b[2]^2+a[2]*a[5]*b[4]^2
                  +a[2]^2*b[5]^2};
    x=realquarticroots(c[0],c[1],c[2],c[3],c[4]);
  } else {
    if(abs(b[4]) > e) {
      real D=b[4]^2;
      c=new real[] {(a[0]*b[4]^2+a[2]*b[3]^2+
                       (-2*a[2]*a[3])*b[3]+a[2]*a[3]^2)/D,
                    -((-2*a[2]*b[3]+2*a[2]*a[3])*b[5]-a[3]*b[4]^2+
                      (2*a[2]*a[5])*b[3])/D,a[2]*(a[5]-b[5])^2/D+a[5]};
      x=quadraticroots(c[0],c[1],c[2]);
    } else {
      if(abs(a[3]-b[3]) > e) {
        real D=b[3]-a[3];
        c=new real[] {a[2],0,a[0]*(a[5]-b[5])^2/D^2-a[3]*b[5]/D+a[5]};
        y=quadraticroots(c[0],c[1],c[2]);
        for(int i=0; i < y.length; ++i) {
          c=new real[] {a[0],a[3],a[2]*y[i]^2+a[5]};
          x=quadraticroots(c[0],c[1],c[2]);
          for(int j=0; j < x.length; ++j) {
            if(abs(b[0]*x[j]^2+b[1]*x[j]*y[i]+b[2]*y[i]^2+b[3]*x[j]
                   +b[4]*y[i]+b[5]) < 1e-5)
              P.push(changecoordsys(currentcoordsys,point(R,(x[j],y[i]))));
          }
        }
        return P;
      } else {
        if(abs(a[5]-b[5]) < e)
          abort("intersectionpoints: intersection of identical conics.");
      }
    }
  }
  for(int i=0; i < x.length; ++i) {
    c=new real[] {a[2],0,a[0]*x[i]^2+a[3]*x[i]+a[5]};
    y=quadraticroots(c[0],c[1],c[2]);
    for(int j=0; j < y.length; ++j) {
      if(abs(b[0]*x[i]^2+b[1]*x[i]*y[j]+b[2]*y[j]^2+b[3]*x[i]+b[4]*y[j]+b[5])
         < 1e-5)
        P.push(changecoordsys(currentcoordsys,point(R,(x[i],y[j]))));
    }
  }
  return P;
}

/*<asyxml><function type="point[]" signature="intersectionpoints(conic,conic)"><code></asyxml>*/
point[] intersectionpoints(conic co1, conic co2)
{/*<asyxml></code><documentation>Return the intersection points of the two conics.</documentation></function></asyxml>*/
  if(degenerate(co1)) return intersectionpoints(co1.l[0], co2);
  if(degenerate(co2)) return intersectionpoints(co1, co2.l[0]);
  return intersectionpoints(equation(co1), equation(co2));
}

/*<asyxml><function type="point[]" signature="intersectionpoints(triangle,conic,bool)"><code></asyxml>*/
point[] intersectionpoints(triangle t, conic co, bool extended = false)
{/*<asyxml></code><documentation>Return the intersection points.
   If 'extended' is true, the sides are lines else the sides are segments.
   intersectionpoints(conic, triangle, bool) is also defined.</documentation></function></asyxml>*/
  if(degenerate(co)) return intersectionpoints(t, co.l[0], extended);
  point[] OP;
  void addpoint(point P[])
  {
    for (int i = 0; i < P.length; ++i) {
      if(defined(P[i])) {
        bool exist = false;
        for (int j = 0; j < OP.length; ++j) {
          if(P[i] == OP[j]) {exist = true; break;}
        }
        if(!exist) OP.push(P[i]);
      }}}
  if(extended) {
    for (int i = 1; i <= 3; ++i) {
      addpoint(intersectionpoints(t.line(i), co));
    }
  } else {
    for (int i = 1; i <= 3; ++i) {
      addpoint(intersectionpoints((segment)t.line(i), co));
    }
  }
  return OP;
}

point[] intersectionpoints(conic co, triangle t, bool extended = false)
{
  return intersectionpoints(t, co, extended);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(ellipse,ellipse)"><code></asyxml>*/
point[] intersectionpoints(ellipse a, ellipse b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  // if(degenerate(a)) return intersectionpoints(a.l, b);
  // if(degenerate(b)) return intersectionpoints(a, b.l);;
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(ellipse,circle)"><code></asyxml>*/
point[] intersectionpoints(ellipse a, circle b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  // if(degenerate(a)) return intersectionpoints(a.l, b);
  // if(degenerate(b)) return intersectionpoints(a, b.l);;
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(circle,ellipse)"><code></asyxml>*/
point[] intersectionpoints(circle a, ellipse b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints(b, a);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(ellipse,parabola)"><code></asyxml>*/
point[] intersectionpoints(ellipse a, parabola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  // if(degenerate(a)) return intersectionpoints(a.l, b);
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(parabola,ellipse)"><code></asyxml>*/
point[] intersectionpoints(parabola a, ellipse b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints(b, a);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(ellipse,hyperbola)"><code></asyxml>*/
point[] intersectionpoints(ellipse a, hyperbola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  // if(degenerate(a)) return intersectionpoints(a.l, b);
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(hyperbola,ellipse)"><code></asyxml>*/
point[] intersectionpoints(hyperbola a, ellipse b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints(b, a);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(circle,parabola)"><code></asyxml>*/
point[] intersectionpoints(circle a, parabola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(parabola,circle)"><code></asyxml>*/
point[] intersectionpoints(parabola a, circle b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(circle,hyperbola)"><code></asyxml>*/
point[] intersectionpoints(circle a, hyperbola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(hyperbola,circle)"><code></asyxml>*/
point[] intersectionpoints(hyperbola a, circle b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(parabola,parabola)"><code></asyxml>*/
point[] intersectionpoints(parabola a, parabola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(parabola,hyperbola)"><code></asyxml>*/
point[] intersectionpoints(parabola a, hyperbola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(hyperbola,parabola)"><code></asyxml>*/
point[] intersectionpoints(hyperbola a, parabola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}
/*<asyxml><function type="point[]" signature="intersectionpoints(hyperbola,hyperbola)"><code></asyxml>*/
point[] intersectionpoints(hyperbola a, hyperbola b)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  return intersectionpoints((conic)a, (conic)b);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(circle,circle)"><code></asyxml>*/
point[] intersectionpoints(circle c1, circle c2)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  if(degenerate(c1))
    return degenerate(c2) ?
      new point[]{intersectionpoint(c1.l, c2.l)} : intersectionpoints(c1.l, c2);
  if(degenerate(c2)) return intersectionpoints(c1, c2.l);
  return (c1.C == c2.C) ?
    new point[] :
    intersectionpoints(radicalline(c1, c2), c1);
}

/*<asyxml><function type="line" signature="tangent(circle,abscissa)"><code></asyxml>*/
line tangent(circle c, abscissa x)
{/*<asyxml></code><documentation>Return the tangent of 'c' at 'point(c, x)'.</documentation></function></asyxml>*/
  if(c.r == 0) abort("tangent: a circle with a radius equals zero has no tangent.");
  point M = point(c, x);
  return line(rotate(90, M) * c.C, M);
}

/*<asyxml><function type="line[]" signature="tangents(circle,point)"><code></asyxml>*/
line[] tangents(circle c, point M)
{/*<asyxml></code><documentation>Return the tangents of 'c' passing through 'M'.</documentation></function></asyxml>*/
  line[] ol;
  if(inside(c, M)) return ol;
  if(M @ c) {
    ol.push(tangent(c, relabscissa(c, M)));
  } else {
    circle cc = circle(c.C, M);
    point[] inter = intersectionpoints(c, cc);
    for (int i = 0; i < inter.length; ++i)
      ol.push(tangents(c, inter[i])[0]);
  }
  return ol;
}

/*<asyxml><function type="point" signature="point(circle,point)"><code></asyxml>*/
point point(circle c, point M)
{/*<asyxml></code><documentation>Return the intersection point of 'c'
   with the half-line '[c.C M)'.</documentation></function></asyxml>*/
  return intersectionpoints(c, line(c.C, false, M))[0];
}

/*<asyxml><function type="line" signature="tangent(circle,point)"><code></asyxml>*/
line tangent(circle c, point M)
{/*<asyxml></code><documentation>Return the tangent of 'c' at the
   intersection point of the half-line'[c.C M)'.</documentation></function></asyxml>*/
  return tangents(c, point(c, M))[0];
}

/*<asyxml><function type="point" signature="point(circle,explicit vector)"><code></asyxml>*/
point point(circle c, explicit vector v)
{/*<asyxml></code><documentation>Return the intersection point of 'c'
   with the half-line '[c.C v)'.</documentation></function></asyxml>*/
  return point(c, c.C + v);
}

/*<asyxml><function type="line" signature="tangent(circle,explicit vector)"><code></asyxml>*/
line tangent(circle c, explicit vector v)
{/*<asyxml></code><documentation>Return the tangent of 'c' at the
   point M so that vec(c.C M) is collinear to 'v' with the same sense.</documentation></function></asyxml>*/
  line ol = tangent(c, c.C + v);
  return dot(ol.v, v) > 0 ? ol : reverse(ol);
}

/*<asyxml><function type="line" signature="tangent(ellipse,abscissa)"><code></asyxml>*/
line tangent(ellipse el, abscissa x)
{/*<asyxml></code><documentation>Return the tangent of 'el' at 'point(el, x)'.</documentation></function></asyxml>*/
  point M = point(el, x);
  line l1 = line(el.F1, M);
  line l2 = line(el.F2, M);
  line ol = (l1 == l2) ? perpendicular(M, l1) : bisector(l1, l2, 90, false);
  return ol;
}

/*<asyxml><function type="line[]" signature="tangents(ellipse,point)"><code></asyxml>*/
line[] tangents(ellipse el, point M)
{/*<asyxml></code><documentation>Return the tangents of 'el' passing through 'M'.</documentation></function></asyxml>*/
  line[] ol;
  if(inside(el, M)) return ol;
  if(M @ el) {
    ol.push(tangent(el, relabscissa(el, M)));
  } else {
    point Mp = samecoordsys(M, el.F2) ?
      M : changecoordsys(el.F2.coordsys, M);
    circle c = circle(Mp, abs(el.F1 - Mp));
    circle cc = circle(el.F2, 2 * el.a);
    point[] inter = intersectionpoints(c, cc);
    for (int i = 0; i < inter.length; ++i) {
      line tl = line(inter[i], el.F2, false);
      point[] P = intersectionpoints(tl, el);
      ol.push(line(Mp, P[0]));
    }
  }
  return ol;
}

/*<asyxml><function type="line" signature="tangent(parabola,abscissa)"><code></asyxml>*/
line tangent(parabola p, abscissa x)
{/*<asyxml></code><documentation>Return the tangent of 'p' at 'point(p, x)' (use the Wells method).</documentation></function></asyxml>*/
  line lt = rotate(90, p.V) * line(p.V, p.F);
  point P = point(p, x);
  if(P == p.V) return lt;
  point M = midpoint(segment(P, p.F));
  line l = rotate(90, M) * line(P, p.F);
  return line(P, projection(lt) * M);
}

/*<asyxml><function type="line[]" signature="tangents(parabola,point)"><code></asyxml>*/
line[] tangents(parabola p, point M)
{/*<asyxml></code><documentation>Return the tangent of 'p' at 'M' (use the Wells method).</documentation></function></asyxml>*/
  line[] ol;
  if(inside(p, M)) return ol;
  if(M @ p) {
    ol.push(tangent(p, angabscissa(p, M)));
  }
  else {
    point Mt = changecoordsys(coordsys(p), M);
    circle c = circle(Mt, p.F);
    line l = rotate(90, p.V) * line(p.V, p.F);
    point[] R = intersectionpoints(l, c);
    for (int i = 0; i < R.length; ++i) {
      ol.push(line(Mt, R[i]));
    }
    // An other method: http://www.du.edu/~jcalvert/math/parabola.htm
    //   point[] R = intersectionpoints(p.directrix, c);
    //   for (int i = 0; i < R.length; ++i) {
    //     ol.push(bisector(segment(p.F, R[i])));
    //   }
  }
  return ol;
}

/*<asyxml><function type="line" signature="tangent(hyperbola,abscissa)"><code></asyxml>*/
line tangent(hyperbola h, abscissa x)
{/*<asyxml></code><documentation>Return the tangent of 'h' at 'point(p, x)'.</documentation></function></asyxml>*/
  point M = point(h, x);
  line ol = bisector(line(M, h.F1), line(M, h.F2));
  if(sameside(h.F1, h.F2, ol) || ol == line(h.F1, h.F2)) ol = rotate(90, M) * ol;
  return ol;
}

/*<asyxml><function type="line[]" signature="tangents(hyperbola,point)"><code></asyxml>*/
line[] tangents(hyperbola h, point M)
{/*<asyxml></code><documentation>Return the tangent of 'h' at 'M'.</documentation></function></asyxml>*/
  line[] ol;
  if(M @ h) {
    ol.push(tangent(h, angabscissa(h, M, fromCenter)));
  } else {
    coordsys cano = canonicalcartesiansystem(h);
    bqe bqe = changecoordsys(cano, equation(h));
    real a = abs(1/(bqe.a[5] * bqe.a[0])), b = abs(1/(bqe.a[5] * bqe.a[2]));
    point Mp = changecoordsys(cano, M);
    real x0 = Mp.x, y0 = Mp.y;
    if(abs(x0) > epsgeo) {
      real c0 = a * y0^2/(b * x0)^2 - 1/b,
        c1 = 2 * a * y0/(b * x0^2), c2 = a/x0^2 - 1;
      real[] sol = quadraticroots(c0, c1, c2);
      for (real y:sol) {
        point tmp = changecoordsys(coordsys(h), point(cano, (a * (1 + y * y0/b)/x0, y)));
        ol.push(line(M, tmp));
      }
    } else if(abs(y0) > epsgeo) {
      real y = -b/y0, x = sqrt(a * (1 + b/y0^2));
      ol.push(line(M, changecoordsys(coordsys(h), point(cano, (x, y)))));
      ol.push(line(M, changecoordsys(coordsys(h), point(cano, (-x, y)))));
    }}
  return ol;
}

/*<asyxml><function type="point[]" signature="intersectionpoints(conic,arc)"><code></asyxml>*/
point[] intersectionpoints(conic co, arc a)
{/*<asyxml></code><documentation>intersectionpoints(arc, circle) is also defined.</documentation></function></asyxml>*/
  point[] op;
  point[] tp = intersectionpoints(co, (conic)a.el);
  for (int i = 0; i < tp.length; ++i)
    if(tp[i] @ a) op.push(tp[i]);
  return op;
}

point[] intersectionpoints(arc a, conic co)
{
  return intersectionpoints(co, a);
}

/*<asyxml><function type="point[]" signature="intersectionpoints(arc,arc)"><code></asyxml>*/
point[] intersectionpoints(arc a1, arc a2)
{/*<asyxml></code><documentation></documentation></function></asyxml>*/
  point[] op;
  point[] tp = intersectionpoints(a1.el, a2.el);
  for (int i = 0; i < tp.length; ++i)
    if(tp[i] @ a1 && tp[i] @ a2) op.push(tp[i]);
  return op;
}


/*<asyxml><function type="point[]" signature="intersectionpoints(line,arc)"><code></asyxml>*/
point[] intersectionpoints(line l, arc a)
{/*<asyxml></code><documentation>intersectionpoints(arc, line) is also defined.</documentation></function></asyxml>*/
  point[] op;
  point[] tp = intersectionpoints(a.el, l);
  for (int i = 0; i < tp.length; ++i)
    if(tp[i] @ a && tp[i] @ l) op.push(tp[i]);
  return op;
}

point[] intersectionpoints(arc a, line l)
{
  return intersectionpoints(l, a);
}

/*<asyxml><function type="point" signature="arcsubtendedcenter(point,point,real)"><code></asyxml>*/
point arcsubtendedcenter(point A, point B, real angle)
{/*<asyxml></code><documentation>Return the center of the arc retuned
   by the 'arcsubtended' routine.</documentation></function></asyxml>*/
  point OM;
  point[] P = standardizecoordsys(A, B);
  angle = angle%(sgnd(angle) * 180);
  line bis = bisector(P[0], P[1]);
  line AB = line(P[0], P[1]);
  return intersectionpoint(bis, rotate(90 - angle, A) * AB);
}

/*<asyxml><function type="arc" signature="arcsubtended(point,point,real)"><code></asyxml>*/
arc arcsubtended(point A, point B, real angle)
{/*<asyxml></code><documentation>Return the arc circle from which the segment AB is saw with
   the angle 'angle'.
   If the point 'M' is on this arc, the oriented angle (MA, MB) is
   equal to 'angle'.</documentation></function></asyxml>*/
  point[] P = standardizecoordsys(A, B);
  line AB = line(P[0], P[1]);
  angle = angle%(sgnd(angle) * 180);
  point C = arcsubtendedcenter(P[0], P[1], angle);
  real BC = degrees(B - C)%360;
  real AC = degrees(A - C)%360;
  return arc(circle(C, abs(B - C)), BC, AC, angle > 0 ? CCW : CW);
}

/*<asyxml><function type="arc" signature="arccircle(point,point,point)"><code></asyxml>*/
arc arccircle(point A, point M, point B)
{/*<asyxml></code><documentation>Return the CCW arc circle 'AB' passing through 'M'.</documentation></function></asyxml>*/
  circle tc = circle(A, M, B);
  real a = degrees(A - tc.C);
  real b = degrees(B - tc.C);
  real m = degrees(M - tc.C);

  arc oa = arc(tc, a, b);
  // TODO: use cross product to determine CWW or CW
  if (!(M @ oa)) {
    oa.direction = !oa.direction;
  }

  return oa;
}

/*<asyxml><function type="arc" signature="arc(ellipse,abscissa,abscissa,bool)"><code></asyxml>*/
arc arc(ellipse el, explicit abscissa x1, explicit abscissa x2, bool direction = CCW)
{/*<asyxml></code><documentation>Return the arc from 'point(c, x1)' to 'point(c, x2)' in the direction 'direction'.</documentation></function></asyxml>*/
  real a = degrees(point(el, x1) - el.C);
  real b = degrees(point(el, x2) - el.C);
  arc oa = arc(el, a - el.angle, b - el.angle, fromCenter, direction);
  return oa;
}

/*<asyxml><function type="arc" signature="arc(ellipse,point,point,bool)"><code></asyxml>*/
arc arc(ellipse el, point M, point N, bool direction = CCW)
{/*<asyxml></code><documentation>Return the arc from 'M' to 'N' in the direction 'direction'.
   The points 'M' and 'N' must belong to the ellipse 'el'.</documentation></function></asyxml>*/
  return arc(el, relabscissa(el, M), relabscissa(el, N), direction);
}

/*<asyxml><function type="arc" signature="arccircle(point,point,real,bool)"><code></asyxml>*/
arc arccircle(point A, point B, real angle, bool direction = CCW)
{/*<asyxml></code><documentation>Return the arc circle centered on A
   from B to rotate(angle, A) * B in the direction 'direction'.</documentation></function></asyxml>*/
  point M = rotate(angle, A) * B;
  return arc(circle(A, abs(A - B)), B, M, direction);
}

/*<asyxml><function type="arc" signature="arc(explicit arc,abscissa,abscissa)"><code></asyxml>*/
arc arc(explicit arc a, abscissa x1, abscissa x2)
{/*<asyxml></code><documentation>Return the arc from 'point(a, x1)' to 'point(a, x2)' traversed in the direction of the arc direction.</documentation></function></asyxml>*/
  real a1 = angabscissa(a.el, point(a, x1), a.polarconicroutine).x;
  real a2 = angabscissa(a.el, point(a, x2), a.polarconicroutine).x;
  return arc(a.el, a1, a2, a.polarconicroutine, a.direction);
}

/*<asyxml><function type="arc" signature="arc(explicit arc,point,point)"><code></asyxml>*/
arc arc(explicit arc a, point M, point N)
{/*<asyxml></code><documentation>Return the arc from 'M' to 'N'.
   The points 'M' and 'N' must belong to the arc 'a'.</documentation></function></asyxml>*/
  return arc(a, relabscissa(a, M), relabscissa(a, N));
}

/*<asyxml><function type="arc" signature="inverse(real,point,segment)"><code></asyxml>*/
arc inverse(real k, point A, segment s)
{/*<asyxml></code><documentation>Return the inverse arc circle of 's'
   with respect to point A and inversion radius 'k'.</documentation></function></asyxml>*/
  point Ap = inverse(k, A, s.A), Bp = inverse(k, A, s.B),
    M = inverse(k, A, midpoint(s));
  return arccircle(Ap, M, Bp);
}

/*<asyxml><operator type = "arc" signature="*(inversion,segment)"><code></asyxml>*/
arc operator *(inversion i, segment s)
{/*<asyxml></code><documentation>Provide
   inversion * segment.</documentation></operator></asyxml>*/
  return inverse(i.k, i.C, s);
}

/*<asyxml><operator type = "path" signature="*(inversion,triangle)"><code></asyxml>*/
path operator *(inversion i, triangle t)
{/*<asyxml></code><documentation>Provide inversion * triangle.</documentation></operator></asyxml>*/
  return (path)(i * segment(t.AB))--
    (path)(i * segment(t.BC))--
    (path)(i * segment(t.CA))&cycle;
}

/*<asyxml><function type="path" signature="compassmark(pair,pair,real,real)"><code></asyxml>*/
path compassmark(pair O, pair A, real position, real angle = 10)
{/*<asyxml></code><documentation>Return an arc centered on O with the angle 'angle' so that the position
   of 'A' on this arc makes an angle 'position * angle'.</documentation></function></asyxml>*/
  real a = degrees(A - O);
  real pa = (a - position * angle)%360,
    pb = (a - (position - 1) * angle)%360;
  real t1 = intersect(unitcircle, (0, 0)--2 * dir(pa))[0];
  real t2 = intersect(unitcircle, (0, 0)--2 * dir(pb))[0];
  int n = length(unitcircle);
  if(t1 >= t2) t1 -= n;
  return shift(O) * scale(abs(O - A)) * subpath(unitcircle, t1, t2);
}

/*<asyxml><function type="line" signature="tangent(explicit arc,abscissa)"><code></asyxml>*/
line tangent(explicit arc a, abscissa x)
{/*<asyxml></code><documentation>Return the tangent of 'a' at 'point(a, x)'.</documentation></function></asyxml>*/
  abscissa ag = angabscissa(a, point(a, x));
  return tangent(a.el, ag + a.angle1 + (a.el.e == 0 ? a.angle0 : 0));
}

/*<asyxml><function type="line" signature="tangent(explicit arc,point)"><code></asyxml>*/
line tangent(explicit arc a, point M)
{/*<asyxml></code><documentation>Return the tangent of 'a' at 'M'.
   The points 'M' must belong to the arc 'a'.</documentation></function></asyxml>*/
  return tangent(a, angabscissa(a, M));
}

// *=======================================================*
// *.......Routines for compatibility with original geometry module........*

path square(pair z1, pair z2)
{
  pair v = z2 - z1;
  pair z3 = z2 + I * v;
  pair z4 = z3 - v;
  return z1--z2--z3--z4--cycle;
}

// Draw a perpendicular symbol at z aligned in the direction align
// relative to the path z--z + dir.
void perpendicular(picture pic = currentpicture, pair z, pair align,
                   pair dir = E, real size = 0, pen p = currentpen,
                   margin margin = NoMargin, filltype filltype = NoFill)
{
  perpendicularmark(pic, (point) z, align, dir, size, p, margin, filltype);
}


// Draw a perpendicular symbol at z aligned in the direction align
// relative to the path z--z + dir(g, 0)
void perpendicular(picture pic = currentpicture, pair z, pair align, path g,
                   real size = 0, pen p = currentpen, margin margin = NoMargin,
                   filltype filltype = NoFill)
{
  perpendicularmark(pic, (point) z, align, dir(g, 0), size, p, margin, filltype);
}

// Return an interior arc BAC of triangle ABC, given a radius r > 0.
// If r < 0, return the corresponding exterior arc of radius |r|.
path arc(explicit pair B, explicit pair A, explicit pair C, real r)
{
  real BA = degrees(B - A);
  real CA = degrees(C - A);
  return arc(A, abs(r), BA, CA, (r < 0) ^ ((BA-CA) % 360 < 180) ? CW : CCW);
}

// *.......End of compatibility routines........*
// *=======================================================*

// *........................FOOTER.........................*
// *=======================================================*

