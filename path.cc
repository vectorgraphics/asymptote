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

namespace camp {

const double Fuzz=10.0*DBL_EPSILON;
const double Fuzz2=Fuzz*Fuzz;
const double sqrtFuzz=sqrt(Fuzz);

// Accurate computation of sqrt(1+x)-1.
inline double sqrt1pxm1(double x)
{
  return x/(sqrt(1.0+x)+1.0);
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
      distinct = quadraticroots::TWO;
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
  
  double ainv=1.0/a;
  b *= ainv; c *= ainv; d *= ainv;
  
  double b2=b*b;
  double Q=(3.0*c-b2)*ninth;
  double Q3=Q*Q*Q;
  double R=(9.0*b*c-27.0*d-2.0*b2*b)*fiftyfourth;
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
  emptyError();
    
  // NOTE: there may be better methods, but let's not split hairs, yet.
  int i = Floor(t);
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
  int i = Floor(t);
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

  return (abc == a) ? nodes[i].pre : abc;
}
        
 
pair path::postcontrol(double t) const
{
  emptyError();
  
  // NOTE: may be better methods, but let's not split hairs, yet.
  int i = Floor(t);
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

  return (bcd == d) ? nodes[iplus].post : bcd;
}

path path::reverse() const
{
  solvedKnot *nodes = new solvedKnot[n];
  int len=length();
  for (int i = 0, j = len; i < n; i++, j--) {
    nodes[i].pre = postcontrol(j);
    nodes[i].point = point(j);
    nodes[i].post = precontrol(j);
    nodes[i].straight = straight(j-1);
  }
  return path(nodes, n, cycles);
}

path path::subpath(int a, int b) const
{
  if(empty()) return path();

  if (a > b) {
    const path &rp = reverse();
    int len=length();
    path result = rp.subpath(len-a, len-b);
    return result;
  }

  if (!cycles) {
    if (a < 0)
      a = 0;
    if (b > n-1)
      b = n-1;
  }

  int sn = b-a+1;
  solvedKnot *nodes = new solvedKnot[sn];

  for (int i = 0, j = a; j <= b; i++, j++) {
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

path path::subpath(double a, double b) const
{
  if(empty()) return path();
  
  if (a > b) {
    const path &rp = reverse();
    int len=length();
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
    aL = nodes[(int)floor(a)];
    aR = nodes[(int)ceil(a)];
    bL = nodes[(int)floor(b)];
    bR = nodes[(int)ceil(b)];
  } else {
    if(fabs(a) > INT_MAX || fabs(b) > INT_MAX)
      reportError("invalid path index");
    aL = nodes[imod((int) floor(a),n)];
    aR = nodes[imod((int) ceil(a),n)];
    bL = nodes[imod((int) floor(b),n)];
    bR = nodes[imod((int) ceil(b),n)];
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
  if(!box.empty) return box;
  
  if (empty()) {
    // No bounds
    return bbox();
  }

  int len=length();
  for (int i = 0; i < len; i++) {
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
  
  int len=length();
  for (int i = 0; i < len; i++) {
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
  int len=length();
  for (int i = 1; i < len; i++) {
    
    pair pre=point(i)-precontrol(i);
    pair post=postcontrol(i)-point(i);
    
    // Check node x coordinate
    if(pre.getx() >= 0.0 ^ post.getx() >= 0) {
      pair z=point(i);
      box += z+padding.left;
      box += z+padding.right;
    }
			      
    // Check node y coordinate
    if(pre.gety() >= 0.0 ^ post.gety() >= 0) {
      pair z=point(i);
      box += z+pair(0,padding.bottom);
      box += z+pair(0,padding.top);
    }
  }
			      
  // Check interior segments.
  for (int i = 0; i < len; i++) {
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
  if(goal < 0 || goal >= L) return L;
  
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
double path::directiontime(pair dir) const {
  if (dir == pair(0,0)) return 0;
  pair rot = pair(1,0)/unit(dir);
    
  double t; double pre,post;
  for (int i = 0; i < n-1+cycles; ) {
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

static unsigned count;  
unsigned maxIntersectCount=100000;

// Algorithm derived from Knuth's MetaFont
bool intersectcubics(pair &t, solvedKnot left1, solvedKnot right1,
                     solvedKnot left2, solvedKnot right2,
		     double fuzz, unsigned depth=DBL_MANT_DIG)
{
  bbox box1, box2;
  box1 += left1.point; box1 += left1.post;
  box1 += right1.pre;  box1 += right1.point;
  box2 += left2.point; box2 += left2.post;
  box2 += right2.pre;  box2 += right2.point;
  
  double lambda=box1.diameter()+box2.diameter();
  
  if(box1.Max().getx()+fuzz >= box2.Min().getx() &&
     box1.Max().gety()+fuzz >= box2.Min().gety() &&
     box2.Max().getx()+fuzz >= box1.Min().getx() &&
     box2.Max().gety()+fuzz >= box1.Min().gety()) {
    if(lambda <= fuzz || depth == 0 || count == 0) {
      t=pair(0,0);
      return true;
    }
    --depth;
    --count;
    solvedKnot sn1[3], sn2[3];
    splitCubic(sn1,0.5,left1,right1);
    splitCubic(sn2,0.5,left2,right2);
    pair T;
    if(intersectcubics(T,sn1[0],sn1[1],sn2[0],sn2[1],fuzz,depth)) {
      t=T*0.5;
      return true;
    }
    if(intersectcubics(T,sn1[0],sn1[1],sn2[1],sn2[2],fuzz,depth)) {
      t=T*0.5+pair(0,1);
      return true;
    }
    if(intersectcubics(T,sn1[1],sn1[2],sn2[0],sn2[1],fuzz,depth)) {
      t=T*0.5+pair(1,0);
      return true;
    }
    if(intersectcubics(T,sn1[1],sn1[2],sn2[1],sn2[2],fuzz,depth)) {
      t=T*0.5+pair(1,1);
      return true;
    }
  }
  return false;
}

// TODO: Handle corner cases. (Done I think)
bool intersect(pair &t, path p1, path p2, double fuzz=0.0)
{
  fuzz=max(fuzz,Fuzz*max(max(length(p1.max()),length(p1.min())),
			 max(length(p2.max()),length(p2.min()))));
  solvedKnot *n1=p1.Nodes();
  solvedKnot *n2=p2.Nodes();
  int L1=p1.length();
  int L2=p2.length();
  int icycle=p1.cyclic() ? p1.size()-1 : -1;
  int jcycle=p2.cyclic() ? p2.size()-1 : -1;
  if(p1.size() == 1) {L1=1; icycle=0;}
  if(p2.size() == 1) {L2=1; jcycle=0;}
  for(int i=0; i < L1; ++i) {
    solvedKnot& left1=n1[i];
    solvedKnot& right1=(i == icycle) ? n1[0] : n1[i+1];
    for(int j=0; j < L2; ++j) {
      count=maxIntersectCount;
      pair T;
      if(intersectcubics(T,left1,right1,
			 n2[j],(j == jcycle) ? n2[0] : n2[j+1],fuzz)) {
	t=T*0.5+pair(i,j);
	return true;
      }
    }
  }
  return false;  
}
// }}}

ostream& operator<< (ostream& out, const path p)
{
  size_t oldPrec = out.precision(6);
  
  int n = p.n;
  switch(n) {
  case 0:
    out << "<nullpath>";
    break;
    
  case 1:
    out << p.point(0);
    break;

  default:
    out << p.point(0) << ".. controls " << p.postcontrol(0) << " and ";

    for (int i = 1; i < n-1; i++) {
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
    break;
  }

  out.precision(oldPrec);

  return out;
}

path concat(path p1, path p2)
{
  int n1 = p1.length(), n2 = p2.length();

  if (n1 == -1) return p2;
  if (n2 == -1) return p1;
  pair a=p1.point(n1);
  pair b=p2.point(0);
  if ((a-b).abs2() > Fuzz*max(a.abs2(),b.abs2()))
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

// Increment count if the path has a vertical component at t.
bool path::Count(int& count, double t) const
{
  pair z=point(t);
  pair Pre=z-precontrol(t);
  pair Post=postcontrol(t)-z;
  double pre=unit(Pre).gety();
  double post=unit(Post).gety();
  if(pre == 0.0 && Pre != pair(0.0,0.0)) pre=post;
  if(post == 0.0 && Post != pair(0.0,0.0)) post=pre;
  int incr=(pre*post > Fuzz) ? sgn1(pre) : 0;
  count += incr;
  return incr != 0.0;
}
  
// Count if t is in (begin,end] and z lies to the left of point(i+t).
void path::countleft(int& count, double x, int i, double t, double begin,
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
int path::windingnumber(const pair& z) const
{
  if(!cycles)
    reportError("path is not cyclic");
  int count=0;
  
  double x=z.getx();
  double y=z.gety();
  
  double begin=-Fuzz;
  double end=1.0+Fuzz;
      
  bbox b=bounds();
  
  if(z.getx() < b.left || z.getx() > b.right ||
     z.gety() < b.bottom || z.gety() > b.top) return 0;
  
  for(int i=0; i < n; ++i) {
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
