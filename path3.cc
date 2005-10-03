/*****
 * path.cc
 * John Bowman
 *
 * Compute information for a three-dimensional path.
 *****/

#include <cfloat>

#include "path.h"
#include "triple.h"

namespace camp {

// Calculate coefficients of Bezier derivative.
static inline void derivative(triple& a, triple& b, triple& c,
			      triple z0, triple z0p, triple z1m, triple z1)
{
  a=z1-z0+3.0*(z0p-z1m);
  b=2.0*(z0+z1m)-4.0*z0p;
  c=z0p-z0;
}

static triple a,b,c;

static double ds(double t)
{
  double dx=quadratic(a.getx(),b.getx(),c.getx(),t);
  double dy=quadratic(a.gety(),b.gety(),c.gety(),t);
  double dz=quadratic(a.getz(),b.getz(),c.getz(),t);
  return sqrt(dx*dx+dy*dy+dz*dz);
}

// Calculates arclength of a cubic using adaptive simpson integration.
double cubiclength(triple z0, triple z0p, triple z1m, triple z1, double goal)
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

struct bbox3 {
  bool empty;
  double left,bottom,lower;
  double right,top,upper;
  
public:  
  bbox3()
    : empty(true), left(0.0), bottom(0.0), lower(0.0), right(0.0), top(0.0),
      upper(0.0) {}

  void add(triple v) {
    double x=v.getx(); 
    double y=v.gety();
    double z=v.getz();
    
    if(empty) {
      left=right=x;
      bottom=top=y;
      lower=upper=z;
      empty=false;
    } else {
      if (x < left)
	left = x;  
      if (x > right)
	right = x;  
      if (y < bottom)
	bottom = y;
      if (y > top)
	top = y;
      if (z < lower)
	lower = z;
      if (z > upper)
	upper = z;
    }
  }

  triple Min() {
    return triple(left,bottom,lower);
  }
  triple Max() {
    return triple(right,top,upper);
  }
  
  double diameter() {
    return (Max()-Min()).length();
  }
};
  
inline triple split(double t, triple x, triple y) {return x+t*(y-x);}

inline void splitCubic(node sn[], double t, node left_, node right_)
{
  node &left=(sn[0]=left_), &mid=sn[1], &right=(sn[2]=right_);
  triple x=split(t,left.post,right.pre);
  left.post=split(t,left.point,left.post);
  right.pre=split(t,right.pre,right.point);
  mid.pre=split(t,left.post,x);
  mid.post=split(t,x,right.pre);
  mid.point=split(t,mid.pre,mid.post);
}

pair intersectcubics(node left1, node right1, node left2, node right2,
		     double fuzz, int depth=DBL_MANT_DIG)
{
  const pair F(-1,-1);

  bbox3 box1, box2;
  box1.add(left1.point); box1.add(left1.post);
  box1.add(right1.pre);  box1.add(right1.point);
  box2.add(left2.point); box2.add(left2.post);
  box2.add(right2.pre);  box2.add(right2.point);
  
  double lambda=box1.diameter()+box2.diameter();
  
  if (box1.Max().getx()+fuzz >= box2.Min().getx() &&
      box1.Max().gety()+fuzz >= box2.Min().gety() &&
      box1.Max().getz()+fuzz >= box2.Min().getz() &&
      box2.Max().getx()+fuzz >= box1.Min().getx() &&
      box2.Max().gety()+fuzz >= box1.Min().gety() &&
      box2.Max().getz()+fuzz >= box1.Min().getz()) {
    if(lambda <= fuzz || depth == 0) return pair(0,0);
    node sn1[3], sn2[3];
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
  
pair intersect(int L1, int L2, node n1[], node n2[], double fuzz=0.0)
{
  pair F=pair(-1,-1);
  for (int i=0; i < L1; ++i) {
    node left1=n1[i];
    node right1=n1[i+1];
    for (int j=0; j < L2; ++j) {
      pair t=intersectcubics(left1,right1,n2[j],n2[j+1],fuzz);
      if (t != F) return t*0.5 + pair(i,j);
    }
  }
  return F;
}
  
} //namespace camp
