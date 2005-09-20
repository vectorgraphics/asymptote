// Three-dimensional graphing routines

import math;
import graph;
import three;

static public int nsub=4;
static public int nmesh=10;

typedef pair dirfcn(real);
dirfcn perpendicular(guide3 G, triple normal, projection P=currentprojection)
{
  return new pair(real t) {
    triple v=point(G,t);
    return normal == O ? 0 :
      project(v+cross(dir(G,t),normal),P)-project(v,P);};
}

// A general 3d axis.
void axis(picture pic=currentpicture, guide3 G, Label L="", pen p=currentpen,
	  ticks ticks, tickspec spec, arrowbar arrow=None,
	  int[] divisor=new int[], bool put=Above, bool opposite=false,
	  projection P=currentprojection) 
{
  divisor=copy(divisor);
  spec=spec.copy();
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

void xaxis(picture pic=currentpicture, triple min, triple max, triple dir=Y,
	   Label L="", pen p=currentpen, ticks ticks=NoTicks,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  real xmin=min.x;
  real xmax=max.x;
  bounds mx=autoscale(xmin,xmax,pic.scale.x.scale);
  
  guide3 G=min--max;
  path g=project(G,P);
  real factor=1.0/(xmax-xmin);
  axis(pic,G,L,p,ticks,
       tickspec(xmin,xmax,mx.min,mx.max,
		new real(real v) {
		  real T=(pic.scale.x.Tlog(v)-xmin)*factor;
		  pair z=project(point(G,T),P);
		  pair dir=project(dir(G,T),P);
		  return intersect(g,z-I*dir--z+I*dir).x;
		},new pair(real t) {
		  triple v=point(G,t);
		  return project(v+dir,P)-project(v,P);}),
       arrow,mx.divisor,put,P);
}

void yaxis(picture pic=currentpicture, triple min, triple max, triple dir=X,
	   Label L="", pen p=currentpen, ticks ticks=NoTicks,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  real ymin=min.y;
  real ymax=max.y;
  bounds my=autoscale(ymin,ymax,pic.scale.y.scale);
  
  guide3 G=min--max;
  path g=project(G,P);
  real factor=1.0/(ymax-ymin);
  axis(pic,G,L,p,ticks,
       tickspec(ymin,ymax,my.min,my.max,
		new real(real v) {
		  real T=(pic.scale.y.Tlog(v)-ymin)*factor;
		  pair z=project(point(G,T),P);
		  pair dir=project(dir(G,T),P);
		  return intersect(g,z-I*dir--z+I*dir).x;
		},new pair(real t) {
		  triple v=point(G,t);
		  return project(v+dir,P)-project(v,P);}),
       arrow,my.divisor,put,P);
}

void zaxis(picture pic=currentpicture, triple min, triple max, triple dir=X,
	   Label L="", pen p=currentpen, ticks ticks=NoTicks,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  real zmin=min.z;
  real zmax=max.z;
  bounds mz=autoscale(zmin,zmax,pic.scale.z.scale);
  
  guide3 G=min--max;
  path g=project(G,P);
  real factor=1.0/(zmax-zmin);
  axis(pic,G,L,p,ticks,
       tickspec(zmin,zmax,mz.min,mz.max,
		new real(real v) {
		  real T=(pic.scale.z.Tlog(v)-zmin)*factor;
		  pair z=project(point(G,T),P);
		  pair dir=project(dir(G,T),P);
		  return intersect(g,z-I*dir--z+I*dir).x;
		},new pair(real t) {
		  triple v=point(G,t);
		  return project(v+dir,P)-project(v,P);}),
       arrow,mz.divisor,put,P);
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

