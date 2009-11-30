// Author : Philippe Ivaldi
// http://www.piprime.fr/
// 2006/11/10

import animation;
import graph;

unitsize(x=2cm,y=1.5cm);

typedef real realfcn(real);

real lambda=4;
real T=2;
real [] k=new real[3];
real [] w=new real[3];
k[0]=2pi/lambda;
w[0]=2pi/T;
real dk=-.5;
k[1]=k[0]-dk;
k[2]=k[0]+dk;
real dw=1;
w[1]=w[0]-dw;
w[2]=w[0]+dw;

real vp=w[1]/k[1];
real vg=dw/dk;

realfcn F(real x) {
  return new real(real t) {
    return cos(k[1]*x-w[1]*t)+cos(k[2]*x-w[2]*t);
  };
};

realfcn G(real x) {
  return new real(real t) {
    return 2*cos(0.5*(k[2]-k[1])*x+0.5*(w[1]-w[2])*t);
  };
};

realfcn operator -(realfcn f) {return new real(real t) {return -f(t);};};

animation A;

real tmax=abs(2pi/dk);
real xmax=abs(2pi/dw);

pen envelope=0.8*blue;
pen fillpen=lightgrey;

int n=50;
real step=tmax/(n-1);
for(int i=0; i < n; ++i) {
  save();
  real t=i*step;
  real a=xmax*t/tmax-xmax/pi;
  real b=xmax*t/tmax;
  path f=graph(F(t),a,b);
  path g=graph(G(t),a,b);
  path h=graph(-G(t),a,b);
  fill(buildcycle(reverse(f),g),fillpen);
  draw(f);
  draw(g,envelope);
  draw(h,envelope);
  A.add();
  restore();
}

for(int i=0; i < n; ++i) {
  save();
  real t=i*step;
  real a=-xmax/pi;
  real b=xmax;
  path f=graph(F(t),a,b);
  path g=graph(G(t),a,b);
  path h=graph(-G(t),a,b);
  path B=box((-xmax/pi,-2),(xmax,2));
  fill(buildcycle(reverse(f),g,B),fillpen);
  fill(buildcycle(f,g,reverse(B)),fillpen);
  draw(f);
  draw(g,envelope);
  draw(h,envelope);
  A.add();
  restore();
}

A.movie(0,10);
