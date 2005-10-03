/*****
 * path.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores and returns information on a predefined path.
 *
 * When changing the path algorithms, also update the corresponding 
 * three-dimensional algorithms in three.asy.
 *****/

#include <cfloat>

#include "path.h"
#include "angle.h"
#include "camperror.h"

namespace camp {

struct quad {
  enum { NONE, SINGLE, DOUBLE, ANY } roots;
  double t1,t2;
};

// Accurate computation of sqrt(1+x)-1.
inline double sqrt1pxm1(double x)
{
  return x/(sqrt(1.0+x)+1.0);
}
  
// Solve the quadratic equation ax^2+bx+c=0.
inline quad solveQuadratic(double a, double b, double c)
{
  quad ret;
  
  if(a == 0.0) {
    if(b != 0.0) {
      ret.roots=quad::SINGLE;
      ret.t1=-c/b;
    } else if(c == 0.0) {
      ret.roots=quad::ANY;
      ret.t1=0.0;
      } else
      ret.roots=quad::NONE;
  } else if(b == 0.0) {
    double x=-c/a;
    if(x >= 0.0) {
      ret.roots=quad::DOUBLE;
      ret.t2=sqrt(x);
      ret.t1=-ret.t2;
    } else
      ret.roots=quad::NONE;
  } else {
    double factor=0.5*b/a;
    double x=-2.0*c/(b*factor);
    if(x > -1.0) {
      ret.roots = quad::DOUBLE;
      double sqrtm1=sqrt1pxm1(x);
      double r2=factor*sqrtm1;
      double r1=-r2-2.0*factor;
      if(r1 <= r2) {
	ret.t1=r1;
	ret.t2=r2;
      } else {
	ret.t1=r2;
	ret.t2=r1;
      }
    } else if(x == -1.0) {
      ret.roots=quad::SINGLE;
      ret.t1=ret.t2=-factor;
    } else
      ret.roots=quad::NONE;
  }
  return ret;
}

pair path::point(double t) const
{
  emptyError();
    
  // NOTE: there may be better methods, but let's not split hairs, yet.
  int i = ifloor(t);
  int iplus;
  t = fmod(t,1);
  if (t < 0) t += 1;

  if (cycles) {
    i = imod(i,n);
    iplus = imod(i+1,n);
  }
  else if (i < 0)
    return nodes[0].point;
  else if (i >= n-1)
    return nodes[n-1].point;
  else
    iplus = i+1;

  double one_t = 1.0-t;

  pair a = nodes[i].point,
       b = nodes[i].post,
       c = nodes[iplus].pre,
       d = nodes[iplus].point,
       ab   = one_t*a   + t*b,
       bc   = one_t*b   + t*c,
       cd   = one_t*c   + t*d,
       abc  = one_t*ab  + t*bc,
       bcd  = one_t*bc  + t*cd,
       abcd = one_t*abc + t*bcd;

  return abcd;
}


pair path::precontrol(double t) const
{
  emptyError();
		     
  // NOTE: may be better methods, but let's not split hairs, yet.
  int i = ifloor(t);
  int iplus;
  t = fmod(t,1);
  if (t < 0) t += 1;

  if (cycles) {
    i = imod(i,n);
    iplus = imod(i+1,n);
  }
  else if (i < 0)
    return nodes[0].pre;
  else if (i >= n-1)
    return nodes[n-1].pre;
  else
    iplus = i+1;

  double one_t = 1.0-t;

  pair a = nodes[i].point,
       b = nodes[i].post,
       c = nodes[iplus].pre,
       ab   = one_t*a   + t*b,
       bc   = one_t*b   + t*c,
       abc  = one_t*ab  + t*bc;

  return abc;
}
        
 
pair path::postcontrol(double t) const
{
  emptyError();
  
  // NOTE: may be better methods, but let's not split hairs, yet.
  int i = ifloor(t);
  int iplus;
  t = fmod(t,1);
  if (t < 0) t += 1;

  if (cycles) {
    i = imod(i,n);
    iplus = imod(i+1,n);
  }
  else if (i < 0)
    return nodes[0].post;
  else if (i >= n-1)
    return nodes[n-1].post;
  else
    iplus = i+1;

  double one_t = 1.0-t;

  pair b = nodes[i].post,
       c = nodes[iplus].pre,
       d = nodes[iplus].point,
       bc   = one_t*b   + t*c,
       cd   = one_t*c   + t*d,
       bcd  = one_t*bc  + t*cd;

  return bcd;
}


path path::reverse() const
{
  solvedKnot *nodes = new solvedKnot[n];
  for (int i = 0, j = length(); i < n; i++, j--) {
    nodes[i].pre = postcontrol(j);
    nodes[i].point = point(j);
    nodes[i].post = precontrol(j);
    nodes[i].straight = straight(j-1);
  }
  return path(nodes, n, cycles);
}

path path::subpath(int start, int end) const
{
  if(empty()) return path();

  if (start > end) {
    const path &rp = reverse();
    path result = rp.subpath(length()-start, length()-end);
    return result;
  }

  if (!cycles) {
    if (start < 0)
      start = 0;
    if (end > n-1)
      end = n-1;
  }

  int sn = end-start+1;
  solvedKnot *nodes = new solvedKnot[sn];

  for (int i = 0, j = start; j <= end; i++, j++) {
    nodes[i].pre = precontrol(j);
    nodes[i].point = point(j);
    nodes[i].post = postcontrol(j);
    nodes[i].straight = straight(j);
  }
  nodes[0].pre = nodes[0].point;
  nodes[sn-1].post = nodes[sn-1].point;

  return path(nodes, sn);
}

inline pair split(double t, pair x, pair y) { return x+(y-x)*t; }

inline void splitCubic(solvedKnot sn[], double t, solvedKnot left_,
		       solvedKnot right_)
{
  solvedKnot &left=(sn[0]=left_), &mid=sn[1], &right=(sn[2]=right_);
  pair x=split(t,left.post,right.pre);
  left.post=split(t,left.point,left.post);
  right.pre=split(t,right.pre,right.point);
  mid.pre=split(t,left.post,x);
  mid.post=split(t,x,right.pre);
  mid.point=split(t,mid.pre,mid.post);
}

path path::subpath(double start, double end) const
{
  if(empty()) return path();
  
  if (start > end) {
    const path &rp = reverse();
    path result = rp.subpath(length()-start, length()-end);
    return result;
  }

  solvedKnot startL, startR, endL, endR;
  if (!cycles) {
    if (start < 0)
      start = 0;
    if (end > n-1)
      end = n-1;
    startL = nodes[(int)floor(start)];
    startR = nodes[(int)ceil(start)];
    endL = nodes[(int)floor(end)];
    endR = nodes[(int)ceil(end)];
  } else {
    if(fabs(start) > INT_MAX || fabs(end) > INT_MAX)
      reportError("invalid path index");
    startL = nodes[imod((int) floor(start),n)];
    startR = nodes[imod((int) ceil(start),n)];
    endL = nodes[imod((int) floor(end),n)];
    endR = nodes[imod((int) ceil(end),n)];
  }

  if (start == end) return path(point(start));

  solvedKnot sn[3];
  path p = subpath(iceil(start), ifloor(end));
  if (start > floor(start)) {
    if (end < ceil(start)) {
      splitCubic(sn,start-floor(start),startL,startR);
      splitCubic(sn,(end-start)/(ceil(end)-start),sn[1],sn[2]);
      return path(sn[0],sn[1]);
    }
    splitCubic(sn,start-floor(start),startL,startR);
    p=concat(path(sn[1],sn[2]),p);
  }
  if (ceil(end) > end) {
    splitCubic(sn,end-floor(end),endL,endR);
    p=concat(p,path(sn[0],sn[1]));
  }
  return p;
}

// Calculate coefficients of Bezier derivative.
static inline void derivative(pair& a, pair& b, pair& c,
			      pair z0, pair z0p, pair z1m, pair z1)
{
  a=z1-z0+3.0*(z0p-z1m);
  b=2.0*(z0+z1m)-4.0*z0p;
  c=z0p-z0;
}

bbox path::bounds() const
{
  if (empty()) {
    // No bounds
    return bbox(/* empty */);
  }

  if(!box.empty) return box;
  
  for (int i = 0; i < length(); i++) {
    box += point(i);
    if(straight(i)) continue;
    
    pair a,b,c;
    derivative(a,b,c,point(i),postcontrol(i),precontrol(i+1),point(i+1));
    quad ret;
    
    // Check x coordinate
    ret=solveQuadratic(a.getx(),b.getx(),c.getx());
    if(ret.roots != quad::NONE) box += point(i+ret.t1);
    if(ret.roots == quad::DOUBLE) box += point(i+ret.t2);
    
    // Check y coordinate
    ret=solveQuadratic(a.gety(),b.gety(),c.gety());
    if(ret.roots != quad::NONE) box += point(i+ret.t1);
    if(ret.roots == quad::DOUBLE) box += point(i+ret.t2);
  }
  box += point(length());
  return box;
}

// {{{ Arclength Calculations

static pair a,b,c;

static double ds(double t)
{
  double dx=quadratic(a.getx(),b.getx(),c.getx(),t);
  double dy=quadratic(a.gety(),b.gety(),c.gety(),t);
  return sqrt(dx*dx+dy*dy);
}

// Calculates arclength of a cubic using adaptive simpson integration.
double cubiclength(pair z0, pair z0p, pair z1m, pair z1, double goal=-1)
{
  double L,integral;
  derivative(a,b,c,z0,z0p,z1m,z1);
  
  if(!simpson(integral,ds,0.0,1.0,DBL_EPSILON,1.0))
    reportError("nesting capacity exceeded in computing arclength");
  L=3.0*integral;
  if(goal < 0 || goal > L) return L;
  
  static const double third=1.0/3.0;
  goal *= third;
  double t=0.5;
  if(!unsimpson(goal,ds,0.0,t,100.0*DBL_EPSILON,integral,1.0))
    reportError("nesting capacity exceeded in computing arctime");
  return -t;
}

double path::arclength() const {
  if (cached_length != -1) return cached_length;

  double L=0.0;
  for (int i = 0; i < n-1; i++) {
    L += cubiclength(point(i),postcontrol(i),precontrol(i+1),point(i+1));
  }
  if(cycles) L += cubiclength(point(n-1),postcontrol(n-1),precontrol(n),
			      point(n));
  cached_length = L;
  
  return cached_length;
}

double path::arctime(double goal) const {
  if (cycles) {
    if (goal == 0) return 0;
    if (goal < 0)  {
      const path &rp = this->reverse();
      double result = -rp.arctime(-goal);
      return result;
    }
    if (cached_length > 0 && goal >= cached_length) {
      int loops = (int)(goal / cached_length);
      goal -= loops*cached_length;
      return loops*n+arctime(goal);
    }      
  } else {
    if (goal <= 0)
      return 0;
    if (cached_length > 0 && goal >= cached_length)
      return n-1;
  }
    
  double l,L=0;
  for (int i = 0; i < n-1; i++) {
    l = cubiclength(point(i),postcontrol(i),precontrol(i+1),point(i+1),goal);
    if (l < 0)
      return (-l+i);
    else {
      L += l;
      goal -= l;
      if (goal <= 0)
        return i+1;
    }
  }
  if (cycles) {
    l = cubiclength(point(n-1),postcontrol(n-1),precontrol(n),point(n),goal);
    if (l < 0)
      return -l+n-1;
    if (cached_length > 0 && cached_length != L+l) {
      reportError("arclength != length.\n"
                  "path::arclength(double) must have broken semantics.\n"
                  "Please report this error.");
    }
    cached_length = L += l;
    goal -= l;
    return arctime(goal)+n;
  }
  else {
    cached_length = L;
    return length();
  }
}

// }}}

inline bool goodroot(double a, double b, double c, double t)
{
  return 0.0 <= t && t <= 1.0 && quadratic(a,b,c,t) >= 0.0;
}

// {{{ Direction Time Calulation
// Algorithm Stolen from Knuth's MetaFont
inline double cubicDir(const solvedKnot& left, const solvedKnot& right,
		       const pair& rot)
{
  pair a,b,c;
  derivative(a,b,c,left.point,left.post,right.pre,right.point);
  a *= rot; b *= rot; c *= rot;
  
  quad ret = solveQuadratic(a.gety(),b.gety(),c.gety());
  switch(ret.roots) {
    case quad::ANY:
    case quad::SINGLE:
      {
      if(goodroot(a.getx(),b.getx(),c.getx(),ret.t1)) return ret.t1;
      } break;

    case quad::DOUBLE:
      {
      if(goodroot(a.getx(),b.getx(),c.getx(),ret.t1)) return ret.t1;
      if(goodroot(a.getx(),b.getx(),c.getx(),ret.t2)) return ret.t2;
      } break;

    case quad::NONE:
      break;
  }

  return -1;
}

// TODO: Check that we handle corner cases.
// Velocity(t) == (0,0)
double path::directiontime(pair dir) const {
  if (dir == pair(0,0)) return 0;
  pair rot = pair(1,0)/unit(dir);
    
  double t; double pre,post;
  for (int i = 0; i < n-1+cycles; ) {
    t = cubicDir(this->nodes[i],(cycles && i==n-1) ? nodes[0]:nodes[i+1],rot);
    if (t >= 0) return i+t;
    i++;
    if (cycles || i != n-1) {
      pre = angle((point(i)-precontrol(i))*rot);
      post = angle((postcontrol(i)-point(i))*rot);
      if ((pre <= 0 && post >= 0 && pre >= post - PI) ||
          (pre >= 0 && post <= 0 && pre <= post + PI))
        return i;
    }
  }
  
  return -1;
}
// }}}

// {{{ Path Intersection Calculation

// Algorithm derived from Knuth's MetaFont
pair intersectcubics(solvedKnot left1, solvedKnot right1,
                     solvedKnot left2, solvedKnot right2,
		     double fuzz, int depth=DBL_MANT_DIG)
{
  const pair F(-1,-1);

  bbox box1, box2;
  box1 += left1.point; box1 += left1.post;
  box1 += right1.pre;  box1 += right1.point;
  box2 += left2.point; box2 += left2.post;
  box2 += right2.pre;  box2 += right2.point;
  
  double lambda=box1.diameter()+box2.diameter();
  
  if (box1.Max().getx()+fuzz >= box2.Min().getx() &&
      box1.Max().gety()+fuzz >= box2.Min().gety() &&
      box2.Max().getx()+fuzz >= box1.Min().getx() &&
      box2.Max().gety()+fuzz >= box1.Min().gety()) {
    if(lambda <= fuzz || depth == 0) return pair(0,0);
    solvedKnot sn1[3], sn2[3];
    splitCubic(sn1,0.5,left1,right1);
    splitCubic(sn2,0.5,left2,right2);
    pair t;
    depth--;
    if ((t=intersectcubics(sn1[0],sn1[1],sn2[0],sn2[1],fuzz,depth)) != F)
      return t*0.5;
    if ((t=intersectcubics(sn1[0],sn1[1],sn2[1],sn2[2],fuzz,depth)) != F)
      return t*0.5+pair(0,1);
    if ((t=intersectcubics(sn1[1],sn1[2],sn2[0],sn2[1],fuzz,depth)) != F)
      return t*0.5+pair(1,0);
    if ((t=intersectcubics(sn1[1],sn1[2],sn2[1],sn2[2],fuzz,depth)) != F)
      return t*0.5+pair(1,1);
  }
  return F;
}

// TODO: Handle corner cases. (Done I think)
pair intersectiontime(path p1, path p2, double fuzz=0.0)
{
  fuzz=max(fuzz,10*DBL_EPSILON*(length(p1.max()-p1.min())+
				length(p2.max()-p2.min())));
  const pair F(-1,-1);
  solvedKnot *n1=p1.Nodes();
  solvedKnot *n2=p2.Nodes();
  int L1=p1.length();
  int L2=p2.length();
  int icycle=p1.cyclic() ? p1.size()-1 : -1;
  int jcycle=p2.cyclic() ? p2.size()-1 : -1;
  if(p1.size() == 1) {L1=1; icycle=0;}
  if(p2.size() == 1) {L2=1; jcycle=0;}
  for (int i = 0; i < L1; i++) {
    solvedKnot& left1=n1[i];
    solvedKnot& right1=(i == icycle) ? n1[0] : n1[i+1];
    for (int j = 0; j < L2; j++) {
      pair t=intersectcubics(left1,right1,
			     n2[j],(j == jcycle) ? n2[0] : n2[j+1],fuzz);
      if (t != F) return t*0.5 + pair(i,j);
    }
  }
  return F;  
}
// }}}

ostream& operator<< (ostream& out, const path p)
{
  size_t oldPrec = out.precision(6);
 
  int n = p.n;
  switch(n) {
  case 0:
    out << "<nullpath>";
    return out;
  case 1:
    out << p.point(0);
    return out;
  }

  out << p.point(0) << ".. controls " << p.postcontrol(0) << " and ";

  for (int i = 1; i < n-1; i++)
  {
    out << p.precontrol(i) << newl;

    out << " .." << p.point(i);

    out << ".. controls " << p.postcontrol(i) << " and ";
  }
  out << p.precontrol(n-1) << newl
      << " .." << p.point(n-1);

  if (p.cycles) 
    out << ".. controls " << p.postcontrol(n-1) << " and "
        << p.precontrol(0) << newl
	<< " ..cycle";

  out.precision(oldPrec);

  return out;
}

path concat(path p1, path p2)
{
  int n1 = p1.length(), n2 = p2.length();

  if (n1 == 0) return p2;
  if (n2 == 0) return p1;
  if (p1.point(n1) != p2.point(0))
    reportError("paths in concatenation do not meet");

  solvedKnot *nodes = new solvedKnot[n1+n2+1];

  int i = 0;
  nodes[0].pre = p1.point(0);
  for (int j = 0; j < n1; j++) {
    nodes[i].point = p1.point(j);
    nodes[i].straight = p1.straight(j);
    nodes[i].post = p1.postcontrol(j);
    nodes[i+1].pre = p1.precontrol(j+1);
    i++;
  }
  for (int j = 0; j < n2; j++) {
    nodes[i].point = p2.point(j);
    nodes[i].straight = p2.straight(j);
    nodes[i].post = p2.postcontrol(j);
    nodes[i+1].pre = p2.precontrol(j+1);
    i++;
  }
  nodes[i].point = nodes[i].post = p2.point(n2);

  return path(nodes, i+1);
}

path path::transformed(const transform& t) const
{
  solvedKnot *nodes = new solvedKnot[n];

  for (int i = 0; i < n; ++i) {
    nodes[i].pre = t * this->nodes[i].pre;
    nodes[i].point = t * this->nodes[i].point;
    nodes[i].post = t * this->nodes[i].post;
    nodes[i].straight = this->nodes[i].straight;
  }

  path p(nodes, n, cyclic());
  return p;
}

path transformed(const transform& t, path p)
{
  int n = p.size();
  solvedKnot *nodes = new solvedKnot[n];

  for (int i = 0; i < n; ++i) {
    nodes[i].pre = t * p.precontrol(i);
    nodes[i].point = t * p.point(i);
    nodes[i].post = t * p.postcontrol(i);
    nodes[i].straight = p.straight(i);
  }

  return path(nodes, n, p.cyclic());
}

} //namespace camp
