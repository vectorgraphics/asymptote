/*****
 * path.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores and returns information on a predefined path.
 *
 * When changing the path algorithms, also update the corresponding 
 * three-dimensional algorithms in path3.cc and three.asy.
 *****/

#include "path.h"
#include "util.h"
#include "angle.h"
#include "camperror.h"
#include "mathop.h"

namespace camp {

const double Fuzz=10.0*DBL_EPSILON;
const double Fuzz2=Fuzz*Fuzz;
const double sqrtFuzz=sqrt(Fuzz);

path nullpath;
  
// Accurate computation of sqrt(1+x)-1.
inline double sqrt1pxm1(double x)
{
  return x/(sqrt(1.0+x)+1.0);
}
inline pair sqrt1pxm1(pair x)
{
  return x/(Sqrt(1.0+x)+1.0);
}
  
// Solve for the real roots of the quadratic equation ax^2+bx+c=0.
quadraticroots::quadraticroots(double a, double b, double c)
{
  if(a == 0.0) {
    if(b != 0.0) {
      distinct=quadraticroots::ONE;
      roots=1;
      t1=-c/b;
    } else if(c == 0.0) {
      distinct=quadraticroots::MANY;
      roots=1;
      t1=0.0;
    } else {
      distinct=quadraticroots::NONE;
      roots=0;
    }
  } else if(b == 0.0) {
    double x=-c/a;
    if(x >= 0.0) {
      distinct=quadraticroots::TWO;
      roots=2;
      t2=sqrt(x);
      t1=-t2;
    } else {
      distinct=quadraticroots::NONE;
      roots=0;
    }
  } else {
    double factor=0.5*b/a;
    double x=-2.0*c/(b*factor);
    if(x > -1.0) {
      distinct=quadraticroots::TWO;
      roots=2;
      double sqrtm1=sqrt1pxm1(x);
      double r2=factor*sqrtm1;
      double r1=-r2-2.0*factor;
      if(r1 <= r2) {
	t1=r1;
	t2=r2;
      } else {
	t1=r2;
	t2=r1;
      }
    } else if(x == -1.0) {
      distinct=quadraticroots::ONE;
      roots=2;
      t1=t2=-factor;
    } else {
      distinct=quadraticroots::NONE;
      roots=0;
    }
  }
}

// Solve for the complex roots of the quadratic equation ax^2+bx+c=0.
Quadraticroots::Quadraticroots(pair a, pair b, pair c)
{
  if(a == 0.0) {
    if(b != 0.0) {
      roots=1;
      z1=-c/b;
    } else if(c == 0.0) {
      roots=1;
      z1=0.0;
    } else
      roots=0;
  } else {
    roots=2;
    if(b == 0.0) {
      z1=Sqrt(-c/a);
      z2=-z1;
    } else {
      pair factor=0.5*b/a;
      pair x=-2.0*c/(b*factor);
      pair sqrtm1=sqrt1pxm1(x);
      z1=factor*sqrtm1;
      z2=-z1-2.0*factor;
    }
  }
}

inline bool goodroot(double t)
{
  return 0.0 <= t && t <= 1.0;
}

inline bool goodroot(double a, double b, double c, double t)
{
  return goodroot(t) && quadratic(a,b,c,t) >= 0.0;
}

// Accurate computation of cbrt(sqrt(1+x)+1)-cbrt(sqrt(1+x)-1).
inline double cbrtsqrt1pxm(double x)
{
  double s=sqrt1pxm1(x);
  return 2.0/(cbrt(x+2.0*(sqrt(1.0+x)+1.0))+cbrt(x)+cbrt(s*s));
}
  
// Taylor series of cos((atan(1.0/w)+pi)/3.0).
static inline double costhetapi3(double w)
{
  static const double c1=1.0/3.0;
  static const double c3=-19.0/162.0;
  static const double c5=425.0/5832.0;
  static const double c7=-16829.0/314928.0;
  double w2=w*w;
  double w3=w2*w;
  double w5=w3*w2;
  return c1*w+c3*w3+c5*w5+c7*w5*w2;
}
      
// Solve for the real roots of the cubic equation ax^3+bx^2+cx+d=0.
cubicroots::cubicroots(double a, double b, double c, double d) 
{
  static const double third=1.0/3.0;
  static const double ninth=1.0/9.0;
  static const double fiftyfourth=1.0/54.0;
  
  // Remove roots at numerical infinity.
  if(fabs(a) <= Fuzz*(fabs(b)+fabs(c)*Fuzz+fabs(d)*Fuzz*Fuzz)) {
    quadraticroots q(b,c,d);
    roots=q.roots;
    if(q.roots >= 1) t1=q.t1;
    if(q.roots == 2) t2=q.t2;
    return;
  }
  
  // Detect roots at numerical zero.
  if(fabs(d) <= Fuzz*(fabs(c)+fabs(b)*Fuzz+fabs(a)*Fuzz*Fuzz)) {
    quadraticroots q(a,b,c);
    roots=q.roots+1;
    t1=0;
    if(q.roots >= 1) t2=q.t1;
    if(q.roots == 2) t3=q.t2;
    return;
  }
  
  double ainv=1.0/a;
  b *= ainv; c *= ainv; d *= ainv;
  
  double b2=b*b;
  double Q=3.0*c-b2;
  if(fabs(Q) < Fuzz*(3.0*fabs(c)+fabs(b2)))
    Q=0.0;
  
  double R=(3.0*Q+b2)*b-27.0*d;
  if(fabs(R) < Fuzz*((3.0*fabs(Q)+fabs(b2))*fabs(b)+27.0*fabs(d)))
    R=0.0;
  
  Q *= ninth;
  R *= fiftyfourth;
  
  double Q3=Q*Q*Q;
  double R2=R*R;
  double D=Q3+R2;
  double mthirdb=-b*third;
  
  if(D > 0.0) {
    roots=1;
    t1=mthirdb;
    if(R2 != 0.0) t1 += cbrt(R)*cbrtsqrt1pxm(Q3/R2);
  } else {
    roots=3;
    double v=0.0,theta;
    if(R2 > 0.0) {
      v=sqrt(-D/R2);
      theta=atan(v);
    } else theta=0.5*PI;
    double factor=2.0*sqrt(-Q)*(R >= 0 ? 1 : -1);
      
    t1=mthirdb+factor*cos(third*theta);
    t2=mthirdb-factor*cos(third*(theta-PI));
    t3=mthirdb;
    if(R2 > 0.0)
      t3 -= factor*((v < 100.0) ? cos(third*(theta+PI)) : costhetapi3(1.0/v)); 
  }
}
  
pair path::point(double t) const
{
  checkEmpty(n);
    
  Int i = Floor(t);
  Int iplus;
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
  checkEmpty(n);
		     
  Int i = Floor(t);
  Int iplus;
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

  return (abc == a) ? nodes[i].pre : abc;
}
        
 
pair path::postcontrol(double t) const
{
  checkEmpty(n);
  
  Int i = Floor(t);
  Int iplus;
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

  return (bcd == d) ? nodes[iplus].post : bcd;
}

path path::reverse() const
{
  mem::vector<solvedKnot> nodes(n);
  Int len=length();
  for (Int i = 0, j = len; i < n; i++, j--) {
    nodes[i].pre = postcontrol(j);
    nodes[i].point = point(j);
    nodes[i].post = precontrol(j);
    nodes[i].straight = straight(j-1);
  }
  return path(nodes, n, cycles);
}

path path::subpath(Int a, Int b) const
{
  if(empty()) return path();

  if (a > b) {
    const path &rp = reverse();
    Int len=length();
    path result = rp.subpath(len-a, len-b);
    return result;
  }

  if (!cycles) {
    if (a < 0)
      a = 0;
    if (b > n-1)
      b = n-1;
  }

  Int sn = b-a+1;
  mem::vector<solvedKnot> nodes(sn);

  for (Int i = 0, j = a; j <= b; i++, j++) {
    nodes[i].pre = precontrol(j);
    nodes[i].point = point(j);
    nodes[i].post = postcontrol(j);
    nodes[i].straight = straight(j);
  }
  nodes[0].pre = nodes[0].point;
  nodes[sn-1].post = nodes[sn-1].point;

  return path(nodes, sn);
}

inline pair split(double t, const pair& x, const pair& y) { return x+(y-x)*t; }

inline void splitCubic(solvedKnot sn[], double t, const solvedKnot& left_,
		       const solvedKnot& right_)
{
  solvedKnot &left=(sn[0]=left_), &mid=sn[1], &right=(sn[2]=right_);
  pair x=split(t,left.post,right.pre);
  left.post=split(t,left.point,left.post);
  right.pre=split(t,right.pre,right.point);
  mid.pre=split(t,left.post,x);
  mid.post=split(t,x,right.pre);
  mid.point=split(t,mid.pre,mid.post);
}

path path::subpath(double a, double b) const
{
  if(empty()) return path();
  
  if (a > b) {
    const path &rp = reverse();
    Int len=length();
    return rp.subpath(len-a, len-b);
  }

  solvedKnot aL, aR, bL, bR;
  if (!cycles) {
    if (a < 0) {
      a = 0;
      if (b < 0)
	b = 0;
    }	
    if (b > n-1) {
      b = n-1;
      if (a > n-1)
	a = n-1;
    }
    aL = nodes[(Int)floor(a)];
    aR = nodes[(Int)ceil(a)];
    bL = nodes[(Int)floor(b)];
    bR = nodes[(Int)ceil(b)];
  } else {
    if(run::validInt(a) && run::validInt(b)) {
      aL = nodes[imod((Int) floor(a),n)];
      aR = nodes[imod((Int) ceil(a),n)];
      bL = nodes[imod((Int) floor(b),n)];
      bR = nodes[imod((Int) ceil(b),n)];
    } else reportError("invalid path index");
  }

  if (a == b) return path(point(a));

  solvedKnot sn[3];
  path p = subpath(Ceil(a), Floor(b));
  if (a > floor(a)) {
    if (b < ceil(a)) {
      splitCubic(sn,a-floor(a),aL,aR);
      splitCubic(sn,(b-a)/(ceil(b)-a),sn[1],sn[2]);
      return path(sn[0],sn[1]);
    }
    splitCubic(sn,a-floor(a),aL,aR);
    p=concat(path(sn[1],sn[2]),p);
  }
  if (ceil(b) > b) {
    splitCubic(sn,b-floor(b),bL,bR);
    p=concat(p,path(sn[0],sn[1]));
  }
  return p;
}

// Special case of subpath used by intersect.
void path::halve(path &first, path &second) const
{
  solvedKnot sn[3];
  splitCubic(sn,0.5,nodes[0],nodes[1]);
  first=path(sn[0],sn[1]);
  second=path(sn[1],sn[2]);
}
  
// Calculate coefficients of Bezier derivative.
static inline void derivative(pair& a, pair& b, pair& c,
			      const pair& z0, const pair& z0p,
			      const pair& z1m, const pair& z1)
{
  a=z1-z0+3.0*(z0p-z1m);
  b=2.0*(z0+z1m)-4.0*z0p;
  c=z0p-z0;
}

bbox path::bounds() const
{
  if(!box.empty) return box;
  
  if (empty()) {
    // No bounds
    return bbox();
  }

  Int len=length();
  for (Int i = 0; i < len; i++) {
    box += point(i);
    if(straight(i)) continue;
    
    pair a,b,c;
    derivative(a,b,c,point(i),postcontrol(i),precontrol(i+1),point(i+1));
    
    // Check x coordinate
    quadraticroots x(a.getx(),b.getx(),c.getx());
    if(x.distinct != quadraticroots::NONE && goodroot(x.t1))
      box += point(i+x.t1);
    if(x.distinct == quadraticroots::TWO && goodroot(x.t2))
      box += point(i+x.t2);
    
    // Check y coordinate
    quadraticroots y(a.gety(),b.gety(),c.gety());
    if(y.distinct != quadraticroots::NONE && goodroot(y.t1))
      box += point(i+y.t1);
    if(y.distinct == quadraticroots::TWO && goodroot(y.t2))
      box += point(i+y.t2);
  }
  box += point(len);
  return box;
}

bbox path::bounds(double min, double max) const
{
  bbox box;
  
  static pair I(0,1);
  
  Int len=length();
  for (Int i = 0; i < len; i++) {
    pair v=I*dir(i);
    box += point(i)+min*v;
    box += point(i)+max*v;
    if(straight(i)) continue;
    
    pair a,b,c;
    derivative(a,b,c,point(i),postcontrol(i),precontrol(i+1),point(i+1));
    
    // Check x coordinate
    quadraticroots x(a.getx(),b.getx(),c.getx());
    if(x.distinct != quadraticroots::NONE && goodroot(x.t1)) {
      double t=i+x.t1;
      pair v=I*dir(t);
      box += point(t)+min*v;
      box += point(t)+max*v;
    }
    if(x.distinct == quadraticroots::TWO && goodroot(x.t2)) {
      double t=i+x.t2;
      pair v=I*dir(t);
      box += point(t)+min*v;
      box += point(t)+max*v;
    }
    
    // Check y coordinate
    quadraticroots y(a.gety(),b.gety(),c.gety());
    if(y.distinct != quadraticroots::NONE && goodroot(y.t1)) {
      double t=i+y.t1;
      pair v=I*dir(t);
      box += point(t)+min*v;
      box += point(t)+max*v;
    }
    if(y.distinct == quadraticroots::TWO && goodroot(y.t2)) {
      double t=i+y.t2;
      pair v=I*dir(t);
      box += point(t)+min*v;
      box += point(t)+max*v;
    }
  }
  pair v=I*dir(len);
  box += point(len)+min*v;
  box += point(len)+max*v;
  return box;
}
  
bbox path::internalbounds(const bbox& padding) const
{
  bbox box;
  
  // Check interior nodes.
  Int len=length();
  for (Int i = 1; i < len; i++) {
    
    pair pre=point(i)-precontrol(i);
    pair post=postcontrol(i)-point(i);
    
    // Check node x coordinate
    if((pre.getx() >= 0.0) ^ (post.getx() >= 0)) {
      pair z=point(i);
      box += z+padding.left;
      box += z+padding.right;
    }
			      
    // Check node y coordinate
    if((pre.gety() >= 0.0) ^ (post.gety() >= 0)) {
      pair z=point(i);
      box += z+pair(0,padding.bottom);
      box += z+pair(0,padding.top);
    }
  }
			      
  // Check interior segments.
  for (Int i = 0; i < len; i++) {
    if(straight(i)) continue;
    
    pair a,b,c;
    derivative(a,b,c,point(i),postcontrol(i),precontrol(i+1),point(i+1));
    
    // Check x coordinate
    quadraticroots x(a.getx(),b.getx(),c.getx());
    if(x.distinct != quadraticroots::NONE && goodroot(x.t1)) {
      pair z=point(i+x.t1);
      box += z+padding.left;
      box += z+padding.right;
    }
    if(x.distinct == quadraticroots::TWO && goodroot(x.t2)) {
      pair z=point(i+x.t2);     
      box += z+padding.left;
      box += z+padding.right;
    }
    
    // Check y coordinate
    quadraticroots y(a.gety(),b.gety(),c.gety());
    if(y.distinct != quadraticroots::NONE && goodroot(y.t1)) {
      pair z=point(i+y.t1);     
      box += z+pair(0,padding.bottom);
      box += z+pair(0,padding.top);
    }
    if(y.distinct == quadraticroots::TWO && goodroot(y.t2)) {
      pair z=point(i+y.t2);
      box += z+pair(0,padding.bottom);
      box += z+pair(0,padding.top);
    }
  }
  return box;
}

// {{{ Arclength Calculations

static pair a,b,c;

static double ds(double t)
{
  double dx=quadratic(a.getx(),b.getx(),c.getx(),t);
  double dy=quadratic(a.gety(),b.gety(),c.gety(),t);
//  cout << t << " " << sqrt(dx*dx+dy*dy) << endl;
  return sqrt(dx*dx+dy*dy);
}

// Calculates arclength of a cubic using adaptive simpson integration.
double cubiclength(const pair& z0, const pair& z0p,
		   const pair& z1m, const pair& z1, double goal=-1)
{
  double L,integral;
  derivative(a,b,c,z0,z0p,z1m,z1);
  
  if(!simpson(integral,ds,0.0,1.0,DBL_EPSILON,1.0))
    reportError("nesting capacity exceeded in computing arclength");
  L=3.0*integral;
  if(goal < 0 || goal >= L) return L;
  
  static const double third=1.0/3.0;
  double t=goal/L;
  goal *= third;
  if(!unsimpson(goal,ds,0.0,t,10.0*DBL_EPSILON,integral,1.0,sqrt(DBL_EPSILON)))
    reportError("nesting capacity exceeded in computing arctime");
  return -t;
}

double path::arclength() const {
  if (cached_length != -1) return cached_length;

  double L=0.0;
  for (Int i = 0; i < n-1; i++) {
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
      Int loops = (Int)(goal / cached_length);
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
  for (Int i = 0; i < n-1; i++) {
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

// {{{ Direction Time Calulation
// Algorithm Stolen from Knuth's MetaFont
inline double cubicDir(const solvedKnot& left, const solvedKnot& right,
		       const pair& rot)
{
  pair a,b,c;
  derivative(a,b,c,left.point,left.post,right.pre,right.point);
  a *= rot; b *= rot; c *= rot;
  
  quadraticroots ret(a.gety(),b.gety(),c.gety());
  switch(ret.distinct) {
    case quadraticroots::MANY:
    case quadraticroots::ONE:
      {
      if(goodroot(a.getx(),b.getx(),c.getx(),ret.t1)) return ret.t1;
      } break;

    case quadraticroots::TWO:
      {
      if(goodroot(a.getx(),b.getx(),c.getx(),ret.t1)) return ret.t1;
      if(goodroot(a.getx(),b.getx(),c.getx(),ret.t2)) return ret.t2;
      } break;

    case quadraticroots::NONE:
      break;
  }

  return -1;
}

// TODO: Check that we handle corner cases.
// Velocity(t) == (0,0)
double path::directiontime(const pair& dir) const {
  if (dir == pair(0,0)) return 0;
  pair rot = pair(1,0)/unit(dir);
    
  double t; double pre,post;
  for (Int i = 0; i < n-1+cycles; ) {
    t = cubicDir(this->nodes[i],(cycles && i==n-1) ? nodes[0]:nodes[i+1],rot);
    if (t >= 0) return i+t;
    i++;
    if (cycles || i != n-1) {
      pair Pre = (point(i)-precontrol(i))*rot;
      pair Post = (postcontrol(i)-point(i))*rot;
      static pair zero(0.0,0.0);
      if(Pre != zero && Post != zero) {
	pre = angle(Pre);
	post = angle(Post);
	if ((pre <= 0 && post >= 0 && pre >= post - PI) ||
	    (pre >= 0 && post <= 0 && pre <= post + PI))
	  return i;
      }
    }
  }
  
  return -1;
}
// }}}

// {{{ Path Intersection Calculation

const unsigned maxdepth=DBL_MANT_DIG;
const unsigned mindepth=maxdepth-16;

bool intersect(double& S, double& T, path& p, path& q, double fuzz,
	       unsigned depth)
{
  if(errorstream::interrupt) throw interrupted();
  
  pair maxp=p.max();
  pair minp=p.min();
  pair maxq=q.max();
  pair minq=q.min();
  
  if(maxp.getx()+fuzz >= minq.getx() &&
     maxp.gety()+fuzz >= minq.gety() && 
     maxq.getx()+fuzz >= minp.getx() &&
     maxq.gety()+fuzz >= minp.gety()) {
    // Overlapping bounding boxes

    --depth;
    if((maxp-minp).length()+(maxq-minq).length() <= fuzz || depth == 0) {
      S=0;
      T=0;
      return true;
    }
    
    Int lp=p.length();
    path p1,p2;
    double pscale,poffset;
    
    if(lp == 1) {
      p.halve(p1,p2);
      pscale=poffset=0.5;
    } else {
      Int tp=lp/2;
      p1=p.subpath(0,tp);
      p2=p.subpath(tp,lp);
      poffset=tp;
      pscale=1.0;
    }
      
    Int lq=q.length();
    path q1,q2;
    double qscale,qoffset;
    
    if(lq == 1) {
      q.halve(q1,q2);
      qscale=qoffset=0.5;
    } else {
      Int tq=lq/2;
      q1=q.subpath(0,tq);
      q2=q.subpath(tq,lq);
      qoffset=tq;
      qscale=1.0;
    }
      
    if(intersect(S,T,p1,q1,fuzz,depth)) {
      S=S*pscale;
      T=T*qscale;
      return true;
    }
    if(intersect(S,T,p1,q2,fuzz,depth)) {
      S=S*pscale;
      T=T*qscale+qoffset;
      return true;
    }
    if(intersect(S,T,p2,q1,fuzz,depth)) {
      S=S*pscale+poffset;
      T=T*qscale;
      return true;
    }
    if(intersect(S,T,p2,q2,fuzz,depth)) {
      S=S*pscale+poffset;
      T=T*qscale+qoffset;
      return true;
    }
  }
  return false;
}

void add(std::vector<double>& S, std::vector<double>& T, double s, double t,
	 const path& p, const path& q, double fuzz)
{
  for(unsigned i=0; i < S.size(); ++i)
    if((p.point(S[i])-p.point(s)).length() <= fuzz &&
       (q.point(T[i])-q.point(t)).length() <= fuzz) return;
  S.push_back(s);
  T.push_back(t);
}
  
void add(std::vector<double>& S, std::vector<double>& T,
	 std::vector<double>& S1, std::vector<double>& T1,
	 double pscale, double qscale, double poffset, double qoffset,
	 const path& p, const path& q, double fuzz)
{
  fuzz *= 2.0;
  for(unsigned i=0; i < S1.size(); ++i)
    add(S,T,pscale*S1[i]+poffset,qscale*T1[i]+qoffset,p,q,fuzz);
}

void intersections(std::vector<double>& S, std::vector<double>& T,
		   path& p, path& q, double fuzz, unsigned depth)
{
  if(errorstream::interrupt) throw interrupted();
  
  pair maxp=p.max();
  pair minp=p.min();
  pair maxq=q.max();
  pair minq=q.min();
  
  if(maxp.getx()+fuzz >= minq.getx() &&
     maxp.gety()+fuzz >= minq.gety() && 
     maxq.getx()+fuzz >= minp.getx() &&
     maxq.gety()+fuzz >= minp.gety()) {
    // Overlapping bounding boxes

    --depth;
    if((maxp-minp).length()+(maxq-minq).length() <= fuzz || depth == 0) {
      S.push_back(0.0);
      T.push_back(0.0);
      return;
    }
    
    Int lp=p.length();
    path p1,p2;
    double pscale,poffset;
    
    if(lp <= 1) {
      p.halve(p1,p2);
      pscale=poffset=0.5;
    } else {
      Int tp=lp/2;
      p1=p.subpath(0,tp);
      p2=p.subpath(tp,lp);
      poffset=tp;
      pscale=1.0;
    }
      
    Int lq=q.length();
    path q1,q2;
    double qscale,qoffset;
    
    if(lq <= 1) {
      q.halve(q1,q2);
      qscale=qoffset=0.5;
    } else {
      Int tq=lq/2;
      q1=q.subpath(0,tq);
      q2=q.subpath(tq,lq);
      qoffset=tq;
      qscale=1.0;
    }
      
    std::vector<double> S1,T1;
    intersections(S1,T1,p1,q1,fuzz,depth);
    add(S,T,S1,T1,pscale,qscale,0.0,0.0,p,q,fuzz);

    if(depth <= mindepth && S1.size() > 0)
      return;
    
    S1.clear();
    T1.clear();
    intersections(S1,T1,p1,q2,fuzz,depth);
    add(S,T,S1,T1,pscale,qscale,0.0,qoffset,p,q,fuzz);
    
    if(depth <= mindepth && S1.size() > 0)
      return;
    
    S1.clear();
    T1.clear();
    intersections(S1,T1,p2,q1,fuzz,depth);
    add(S,T,S1,T1,pscale,qscale,poffset,0.0,p,q,fuzz);
    
    if(depth <= mindepth && S1.size() > 0)
      return;
    
    S1.clear();
    T1.clear();
    intersections(S1,T1,p2,q2,fuzz,depth);
    add(S,T,S1,T1,pscale,qscale,poffset,qoffset,p,q,fuzz);
  }
}

// }}}

ostream& operator<< (ostream& out, const path& p)
{
  Int n = p.n;
  switch(n) {
  case 0:
    out << "<nullpath>";
    break;
    
  case 1:
    out << p.point((Int) 0);
    break;

  default:
    out << p.point((Int) 0) << ".. controls " << p.postcontrol((Int) 0) 
	<< " and ";

    for (Int i = 1; i < n-1; i++) {
      out << p.precontrol(i) << newl;

      out << " .." << p.point(i);

      out << ".. controls " << p.postcontrol(i) << " and ";
    }
    
    out << p.precontrol(n-1) << newl
	<< " .." << p.point(n-1);

    if (p.cycles) 
      out << ".. controls " << p.postcontrol(n-1) << " and "
	  << p.precontrol((Int) 0) << newl
	  << " ..cycle";
    break;
  }

  return out;
}

path concat(const path& p1, const path& p2)
{
  Int n1 = p1.length(), n2 = p2.length();

  if (n1 == -1) return p2;
  if (n2 == -1) return p1;
  pair a=p1.point(n1);
  pair b=p2.point((Int) 0);

  mem::vector<solvedKnot> nodes(n1+n2+1);

  Int i = 0;
  nodes[0].pre = p1.point((Int) 0);
  for (Int j = 0; j < n1; j++) {
    nodes[i].point = p1.point(j);
    nodes[i].straight = p1.straight(j);
    nodes[i].post = p1.postcontrol(j);
    nodes[i+1].pre = p1.precontrol(j+1);
    i++;
  }
  for (Int j = 0; j < n2; j++) {
    nodes[i].point = p2.point(j);
    nodes[i].straight = p2.straight(j);
    nodes[i].post = p2.postcontrol(j);
    nodes[i+1].pre = p2.precontrol(j+1);
    i++;
  }
  nodes[i].point = nodes[i].post = p2.point(n2);

  return path(nodes, i+1);
}

// Increment count if the path has a vertical component at t.
bool path::Count(Int& count, double t) const
{
  pair z=point(t);
  pair Pre=z-precontrol(t);
  pair Post=postcontrol(t)-z;
  double pre=unit(Pre).gety();
  double post=unit(Post).gety();
  if(pre == 0.0 && Pre != pair(0.0,0.0)) pre=post;
  if(post == 0.0 && Post != pair(0.0,0.0)) post=pre;
  Int incr=(pre*post > Fuzz) ? sgn1(pre) : 0;
  count += incr;
  return incr != 0.0;
}
  
// Count if t is in (begin,end] and z lies to the left of point(i+t).
void path::countleft(Int& count, double x, Int i, double t, double begin,
		     double end, double& mint, double& maxt) const 
{
  if(t > -Fuzz && t < Fuzz) t=0;
  if(begin < t && t <= end && x < point(i+t).getx() && Count(count,i+t)) {
    if(t > maxt) maxt=t;
    if(t < mint) mint=t;
  }
}

// Return the winding number of the region bounded by the (cyclic) path
// relative to the point z.
Int path::windingnumber(const pair& z) const
{
  if(!cycles)
    reportError("path is not cyclic");
  Int count=0;
  
  double x=z.getx();
  double y=z.gety();
  
  double begin=-Fuzz;
  double end=1.0+Fuzz;
      
  bbox b=bounds();
  
  if(z.getx() < b.left || z.getx() > b.right ||
     z.gety() < b.bottom || z.gety() > b.top) return 0;
  
  for(Int i=0; i < n; ++i) {
    pair a=point(i);
    pair d=point(i+1);
      
    double mint=1.0;
    double maxt=0.0;
    double stop=(i < n-1) ? 1.0+Fuzz : end;
      
    if(straight(i)) {
      double denom=d.gety()-a.gety();
      if(denom != 0.0)
	countleft(count,x,i,(z.gety()-a.gety())/denom,begin,stop,mint,maxt);
    } else {
      pair b=postcontrol(i);
      pair c=precontrol(i+1);
    
      double A=-a.gety()+3.0*(b.gety()-c.gety())+d.gety();
      double B=3.0*(a.gety()-2.0*b.gety()+c.gety());
      double C=3.0*(-a.gety()+b.gety());
      double D=a.gety()-y;
    
      cubicroots r(A,B,C,D);

      if(r.roots >= 1) countleft(count,x,i,r.t1,begin,stop,mint,maxt);
      if(r.roots >= 2) countleft(count,x,i,r.t2,begin,stop,mint,maxt);
      if(r.roots >= 3) countleft(count,x,i,r.t3,begin,stop,mint,maxt);
    }
      
    // Avoid double-counting endpoint roots.      
    if(i == 0)
      end=camp::min(mint-Fuzz,Fuzz)+1.0;
    if(mint <= maxt)
      begin=camp::max(maxt+Fuzz-1.0,-Fuzz); 
    else // no root found
      begin=-Fuzz;
  }
  return count;
}

path path::transformed(const transform& t) const
{
  mem::vector<solvedKnot> nodes(n);

  for (Int i = 0; i < n; ++i) {
    nodes[i].pre = t * this->nodes[i].pre;
    nodes[i].point = t * this->nodes[i].point;
    nodes[i].post = t * this->nodes[i].post;
    nodes[i].straight = this->nodes[i].straight;
  }

  path p(nodes, n, cyclic());
  return p;
}

path transformed(const transform& t, const path& p)
{
  Int n = p.size();
  mem::vector<solvedKnot> nodes(n);

  for (Int i = 0; i < n; ++i) {
    nodes[i].pre = t * p.precontrol(i);
    nodes[i].point = t * p.point(i);
    nodes[i].post = t * p.postcontrol(i);
    nodes[i].straight = p.straight(i);
  }

  return path(nodes, n, p.cyclic());
}

path nurb(pair z0, pair z1, pair z2, pair z3,
	  double w0, double w1, double w2, double w3, Int m)
{
  mem::vector<solvedKnot> nodes(m+1);

  if(m < 1) reportError("invalid sampling interval");

  double step=1.0/m;
  for(Int i=0; i <= m; ++i) { 
    double t=i*step;
    double t2=t*t;
    double onemt=1.0-t;
    double onemt2=onemt*onemt;
    double W0=w0*onemt2*onemt;
    double W1=w1*3.0*t*onemt2;
    double W2=w2*3.0*t2*onemt;
    double W3=w3*t2*t;
    nodes[i].point=(W0*z0+W1*z1+W2*z2+W3*z3)/(W0+W1+W2+W3);
  }
  
  static const double onethird=1.0/3.0;
  static const double twothirds=2.0/3.0;
  pair z=nodes[0].point;
  nodes[0].pre=z;
  nodes[0].post=twothirds*z+onethird*nodes[1].point;
  for(int i=1; i < m; ++i) {
    pair z0=nodes[i].point;
    pair zm=nodes[i-1].point;
    pair zp=nodes[i+1].point;
    pair pre=twothirds*z0+onethird*zm;
    pair pos=twothirds*z0+onethird*zp;
    pair dir=unit(pos-pre);
    nodes[i].pre=z0-length(z0-pre)*dir;
    nodes[i].post=z0+length(pos-z0)*dir;
  }
  z=nodes[m].point;
  nodes[m].pre=twothirds*z+onethird*nodes[m-1].point;
  nodes[m].post=z;
  return path(nodes,m+1);
}

} //namespace camp
