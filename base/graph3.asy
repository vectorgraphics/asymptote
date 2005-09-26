// Three-dimensional graphing routines

import math;
import graph;
import three;

static public int nsub=4;
static public int nmesh=10;

typedef pair direction(real);

pair dir(triple v, triple dir, projection P=currentprojection)
{
  return unit(project(v+dir,P)-project(v,P));
}

direction dir(guide3 G, triple dir, projection P=currentprojection)
{
  return new pair(real t) {
    return dir(point(G,t),dir,P);
  };
}

direction perpendicular(guide3 G, triple normal,
			projection P=currentprojection)
{
  return new pair(real t) {
    return dir(point(G,t),cross(dir(G,t),normal),P);
  };
}

real projecttime(guide3 G, real T, guide g, projection P=currentprojection)
{
  triple v=point(G,T);
  pair z=project(v,P);
  pair dir=dir(v,dir(G,T),P);
  return intersect(g,z).x;
}

real projecttime(guide3 G, real T, projection P=currentprojection)
{
  return projecttime(G,T,project(G,P),P);
}

valuetime linear(picture pic=currentpicture, guide3 G, scalefcn S,
		 real Min, real Max, projection P=currentprojection)
{
  real factor=Max == Min ? 0.0 : 1.0/(Max-Min);
  path g=project(G,P);
  return new real(real v) {
    return projecttime(G,(S(v)-Min)*factor,g,P);
  };
}

// Draw a general three-dimensional axis.
void axis(picture pic=currentpicture, Label L="", guide3 G, pen p=currentpen,
	  ticks ticks, tickspec spec, arrowbar arrow=None,
	  int[] divisor=new int[], bool put=Above, bool opposite=false,
	  projection P=currentprojection) 
{
  divisor=copy(divisor);
  spec=spec.copy();
  Label L=L.copy();
  real t=0.5*length(G);
  if(L.defaultposition) L.position(t);
  
  path g=project(G,P);
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    frame d;
    ticks(d,t,L,0,g,g,p,arrow,spec,opposite,divisor);
    (put ? add : prepend)(f,t*T*inverse(t)*d);
  });
  
  pic.addPath(g,p);
  
  if(L.s != "") {
    frame f;
    Label L0=L.copy();
    L0.position(0);
    add(f,L0);
    pair pos=point(g,L.relative()*length(g));
    pic.addBox(pos,pos,min(f),max(f));
  }
}

bounds autoscale(real min, real max, autoscaleT A)
{
  bounds m;
  if(finite(A.tickMin) && finite(A.tickMax)) {
    m.min=A.tickMin; m.max=A.tickMax;
  } else m=autoscale(min,max,A.scale);
  return m;
}

// Draw an x axis in three dimensions.
void xaxis(picture pic=currentpicture, Label L="", triple min, triple max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  bounds mx=autoscale(min.x,max.x,pic.scale.x);
  guide3 G=min--max;
  valuetime t=linear(pic,G,pic.scale.x.T(),min.x,max.x,P);
  axis(pic,L,G,p,ticks,tickspec(min.x,max.x,mx.min,mx.max,t,dir(G,dir,P)),
       arrow,mx.divisor,put,P);
}


// Draw a y axis in three dimensions.
void yaxis(picture pic=currentpicture, Label L="", triple min, triple max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=X,
	   arrowbar arrow=None, bool put=Above, 
	   projection P=currentprojection) 
{
  bounds my=autoscale(min.y,max.y,pic.scale.y);
  guide3 G=min--max;
  valuetime t=linear(pic,G,pic.scale.y.T(),min.y,max.y,P);
  axis(pic,L,G,p,ticks,tickspec(min.y,max.y,my.min,my.max,t,dir(G,dir,P)),
       arrow,my.divisor,put,P);
}

// Draw a z axis in three dimensions.
void zaxis(picture pic=currentpicture, Label L="", triple min, triple max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  bounds mz=autoscale(min.z,max.z,pic.scale.z);
  guide3 G=min--max;
  valuetime t=linear(pic,G,pic.scale.z.T(),min.z,max.z,P);
  axis(pic,L,G,p,ticks,tickspec(min.z,max.z,mz.min,mz.max,t,dir(G,dir,P)),
       arrow,mz.divisor,put,P);
}

bounds autolimits(real min, real max, autoscaleT A)
{
  bounds m;
  if(A.automin() || A.automax()) {
    m=autoscale(min,max,A.scale);
    if(A.automin()) min=m.min;
    if(A.automax()) max=m.max;
    A.tickMin=m.min;
    A.tickMax=m.max;
  } else {m.min=min; m.max=max;}
  return m;
}

struct limits {
  public triple O,X,Y,Z;
}

limits operator init() {return new limits;}
  
limits autolimits(picture pic=currentpicture, triple min, triple max) 
{
  limits L;
  bounds mx=autolimits(min.x,max.x,pic.scale.x);
  bounds my=autolimits(min.y,max.y,pic.scale.y);
  bounds mz=autolimits(min.z,max.z,pic.scale.z);
  L.O=(mx.min,my.min,mz.min);
  L.X=(mx.max,my.min,mz.min);
  L.Y=(mx.min,my.max,mz.min);
  L.Z=(mx.min,my.min,mz.max);
  return L;
}

typedef guide3 graph(triple F(real), real, real, int);

public graph graph(guide3 join(... guide3[]))
{
  return new guide3(triple F(real), real a, real b, int n) {
    guide3 g;
    real width=n == 0 ? 0 : (b-a)/n;
    for(int i=0; i <= n; ++i) {
      real x=a+width*i;
      g=join(g,F(x));	
    }	
    return g;
  };
}

public guide3 Straight(... guide3[])=operator --;
		       
triple Scale(picture pic, triple v)
{
  return (pic.scale.x.T(v.x),pic.scale.y.T(v.y),pic.scale.z.T(v.z));
}

typedef guide3 interpolate(... guide3[]);

guide3 graph(picture pic=currentpicture, real x(real), real y(real),
	     real z(real), real a, real b, int n=ngraph,
	     interpolate join=operator --)
{
  return graph(join)(new triple(real t) {return Scale(pic,(x(t),y(t),z(t)));},
		     a,b,n);
}

guide3 graph3(picture pic=currentpicture, triple v(real), real a, real b,
	     int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new triple(real t) {return Scale(pic,v(t));},a,b,n);
}

int conditional(triple[] v, bool[] cond)
{
  if(cond.length > 0) {
    if(cond.length != v.length)
      abort("condition array has different length than data");
    return sum(cond)-1;
  } else return v.length-1;
}

guide3 graph(picture pic=currentpicture, triple[] v, bool[] cond={},
	     interpolate join=operator --)
{
  int n=conditional(v,cond);
  int i=-1;
  return graph(join)(new triple(real) {
    i=next(i,cond);
    return Scale(pic,v[i]);},0,0,n);
}

guide3 graph(picture pic=currentpicture, real[] x, real[] y, real[] z,
	    bool[] cond={}, interpolate join=operator --)
{
  if(x.length != y.length || x.length != z.length) abort(differentlengths);
  int n=conditional(x,cond);
  int i=-1;
  return graph(join)(new triple(real) {
    i=next(i,cond);
    return Scale(pic,(x[i],y[i],z[i]));},0,0,n);
}

// The graph of a function along a path.
guide3 graph(triple F(path, real), path p, int n=nsub,
	     interpolate join=operator --)
{
  guide3 g;
  for(int i=0; i < n*length(p); ++i)
    g=join(g,F(p,i/n));
  return cyclic(p) ? join(g,cycle3) : join(g,F(p,length(p)));
}

guide3 graph(triple F(pair), path p, int n=nsub, interpolate join=operator --)
{
  return graph(new triple(path p, real position) 
	       {return F(point(p,position));},p,n,join);
}

guide3 graph(picture pic=currentpicture, real f(pair), path p, int n=nsub,
	     interpolate join=operator --) 
{
  return graph(new triple(pair z) {return Scale(pic,(z.x,z.y,f(z)));},p,n,
	       join);
}

guide3 graph(real f(pair), path p, int n=nsub, real T(pair),
	     interpolate join=operator --)
{
  return graph(new triple(pair z) {pair w=T(z); return (w.x,w.y,f(w));},p,n,
	       join);
}

picture surface(real f(pair z), pair a, pair b, int n=nmesh, int nsub=nsub,
		pen surfacepen=lightgray, pen meshpen=currentpen,
		projection P=currentprojection)
{
  picture pic;
  pair back,front;

  void drawcell(pair a, pair b) {
    guide3 g=graph(f,box(a,b),nsub);
    filldraw(pic,project(g,P),surfacepen,meshpen);
  }

  pair sample(int i, int j) {
    return (interp(back.x,front.x,i/n),
            interp(back.y,front.y,j/n));
  }

  pair camera=(P.camera.x,P.camera.y);
  pair z=b-a;
  int sign=sgn(dot(camera,I*conj(z)));
  
  if(sign >= 0) {
    back=a;
    front=b;
  } else {
    back=b;
    front=a;
  }

  if(sign*sgn(dot(camera,I*z)) >= 0)
    for(int j=0; j < n; ++j)
      for(int i=0; i < n; ++i)
	drawcell(sample(i,j),sample(i+1,j+1));
  else
    for(int i=0; i < n; ++i)
      for(int j=0; j < n; ++j)
	drawcell(sample(i,j),sample(i+1,j+1));

  return pic;
}

triple polar(real r, real theta, real phi)
{
  return r*expi(theta,phi);
}

guide3 polargraph(real r(real,real), real theta(real), real phi(real),
		  int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new triple(real t) {
      return polar(r(theta(t),phi(t)),theta(t),phi(t));
    },0,1,n);
}

// True arc
path3 Arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
	  int ngraph=400)
{
  return shift(c)*polargraph(new real(real theta, real phi){return r;},
			     new real(real t){return interp(theta1,theta2,t);},
			     new real(real t){return interp(phi1,phi2,t);},
			     ngraph,operator ..);
}

// True circle
path3 Circle(triple c, real r, triple normal=Z, int ngraph=400)
{
  path3 p=Arc(O,r,pi/2,0,pi/2,2pi,ngraph)--cycle3;
  if(normal != Z) p=rotate(longitude(normal),Z)*rotate(colatitude(normal),Y)*p;
  return shift(c)*p;

}
