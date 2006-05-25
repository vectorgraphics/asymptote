// Three-dimensional graphing routines

private import math;
import graph;
import three;
public import light;

public int nmesh=10;

triple Scale(picture pic, triple v)
{
  return (pic.scale.x.T(v.x),pic.scale.y.T(v.y),pic.scale.z.T(v.z));
}

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
	  ticks ticks, ticklocate locate, arrowbar arrow=None,
	  int[] divisor=new int[], bool put=Above,
	  projection P=currentprojection,  bool opposite=false) 
{
  divisor=copy(divisor);
  locate=locate.copy();
  Label L=L.copy();
  if(L.defaultposition) L.position(0.5*length(G));
  
  path g=project(G,P);
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    frame d;
    ticks(d,t,L,0,g,g,p,arrow,locate,divisor,opposite);
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

// Draw an x axis in three dimensions.
void xaxis(picture pic=currentpicture, Label L="", triple min, triple max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection, bool opposite=false) 
{
  bounds m=autoscale(min.x,max.x,pic.scale.x.scale);
  guide3 G=min--max;
  valuetime t=linear(pic,G,pic.scale.x.T(),min.x,max.x,P);
  axis(pic,opposite ? "" : L,G,p,ticks,
       ticklocate(min.x,max.x,pic.scale.x,m.min,m.max,t,dir(G,dir,P)),
       arrow,m.divisor,put,P,opposite);
}

void xaxis(picture pic=currentpicture, Label L="", triple min, real max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection, bool opposite=false) 
{
  xaxis(pic,L,min,(max,min.y,min.z),p,ticks,dir,arrow,put,P,opposite);
}

// Draw a y axis in three dimensions.
void yaxis(picture pic=currentpicture, Label L="", triple min, triple max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=X,
	   arrowbar arrow=None, bool put=Above, 
	   projection P=currentprojection, bool opposite=false) 
{
  bounds m=autoscale(min.y,max.y,pic.scale.y.scale);
  guide3 G=min--max;
  valuetime t=linear(pic,G,pic.scale.y.T(),min.y,max.y,P);
  axis(pic,L,G,p,ticks,
       ticklocate(min.y,max.y,pic.scale.y,m.min,m.max,t,dir(G,dir,P)),
       arrow,m.divisor,put,P,opposite);
}

void yaxis(picture pic=currentpicture, Label L="", triple min, real max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=X,
	   arrowbar arrow=None, bool put=Above, 
	   projection P=currentprojection, bool opposite=false) 
{
  yaxis(pic,L,min,(min.x,max,min.z),p,ticks,dir,arrow,put,P,opposite);
}

// Draw a z axis in three dimensions.
void zaxis(picture pic=currentpicture, Label L="", triple min, triple max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection, bool opposite=false) 
{
  bounds m=autoscale(min.z,max.z,pic.scale.z.scale);
  guide3 G=min--max;
  valuetime t=linear(pic,G,pic.scale.z.T(),min.z,max.z,P);
  axis(pic,L,G,p,ticks,
       ticklocate(min.z,max.z,pic.scale.z,m.min,m.max,t,dir(G,dir,P)),
       arrow,m.divisor,put,P,opposite);
}

void zaxis(picture pic=currentpicture, Label L="", triple min, real max,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection, bool opposite=false) 
{
  zaxis(pic,L,min,(min.x,min.y,max),p,ticks,dir,arrow,put,P,opposite);
}

// Draw an x axis.
// If all=true, also draw opposing edges of the three-dimensional bounding box.
void xaxis(picture pic=currentpicture, Label L="", bool all=false, bbox3 b,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=Y,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  if(all) {
    bounds m=autoscale(b.min.x,b.max.x,pic.scale.x.scale);
  
    void axis(Label L, triple min, triple max, bool opposite=false,
	      int sign=1) {
      xaxis(pic,L,min,max,p,ticks,sign*dir,arrow,put,P,opposite);
    }
    bool back=dot(b.Y()-b.O(),P.camera)*P.camera.z > 0;
    int sign=back ? -1 : 1;
    axis(L,b.min,b.X(),back,sign);
    axis(L,(b.min.x,b.max.y,b.min.z),(b.max.x,b.max.y,b.min.z),!back,sign);
    axis(L,(b.min.x,b.min.y,b.max.z),(b.max.x,b.min.y,b.max.z),true,-1);
    axis(L,(b.min.x,b.max.y,b.max.z),b.max,true);
  } else xaxis(pic,L,b.O(),b.X(),p,ticks,dir,arrow,put,P);
}

// Draw a y axis.
// If all=true, also draw opposing edges of the three-dimensional bounding box.
void yaxis(picture pic=currentpicture, Label L="", bool all=false, bbox3 b,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=X,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  if(all) {
    bounds m=autoscale(b.min.y,b.max.y,pic.scale.y.scale);
  
    void axis(Label L, triple min, triple max, bool opposite=false,
	      int sign=1) {
      yaxis(pic,L,min,max,p,ticks,sign*dir,arrow,put,P,opposite);
    }
    bool back=dot(b.X()-b.min,P.camera)*P.camera.z > 0;
    int sign=back ? -1 : 1;
    axis(L,b.min,b.Y(),back,sign);
    axis(L,(b.max.x,b.min.y,b.min.z),(b.max.x,b.max.y,b.min.z),!back,sign);
    axis(L,(b.min.x,b.min.y,b.max.z),(b.min.x,b.max.y,b.max.z),true,-1);
    axis(L,(b.max.x,b.min.y,b.max.z),b.max,true);
  } else yaxis(pic,L,b.O(),b.Y(),p,ticks,dir,arrow,put,P);
}

// Draw a z axis.
// If all=true, also draw opposing edges of the three-dimensional bounding box.
void zaxis(picture pic=currentpicture, Label L="", bool all=false, bbox3 b,
	   pen p=currentpen, ticks ticks=NoTicks, triple dir=X,
	   arrowbar arrow=None, bool put=Above,
	   projection P=currentprojection) 
{
  if(all) {
    bounds m=autoscale(b.min.z,b.max.z,pic.scale.z.scale);
  
    void axis(Label L, triple min, triple max, bool opposite=false,
	      int sign=1) {
      zaxis(pic,L,min,max,p,ticks,sign*dir,arrow,put,P,opposite);
    }
    bool back=dot(b.X()-b.min,P.camera)*P.camera.z > 0;
    int sign=back ? -1 : 1;
    axis(L,b.min,b.Z(),back,sign);
    axis(L,(b.max.x,b.min.y,b.min.z),(b.max.x,b.min.y,b.max.z),!back,sign);
    axis(L,(b.min.x,b.max.y,b.min.z),(b.min.x,b.max.y,b.max.z),true,-1);
    axis(L,(b.max.x,b.max.y,b.min.z),b.max,true);
  } else zaxis(pic,L,b.O(),b.Z(),p,ticks,dir,arrow,put,P);
}

bounds autolimits(real min, real max, autoscaleT A)
{
  bounds m;
  min=A.scale.T(min);
  max=A.scale.T(max);
  if(A.automin() || A.automax())
    m=autoscale(min,max,A.scale);
  if(!A.automin()) m.min=min;
  if(!A.automax()) m.max=max;
  return m;
}

bbox3 autolimits(picture pic=currentpicture, triple min, triple max) 
{
  bbox3 b;
  bounds mx=autolimits(min.x,max.x,pic.scale.x);
  bounds my=autolimits(min.y,max.y,pic.scale.y);
  bounds mz=autolimits(min.z,max.z,pic.scale.z);
  b.min=(mx.min,my.min,mz.min);
  b.max=(mx.max,my.max,mz.max);
  return b;
}

bbox3 limits(picture pic=currentpicture, triple min, triple max)
{
  bbox3 b;
  b.min=(pic.scale.x.T(min.x),pic.scale.y.T(min.y),pic.scale.z.T(min.z));
  b.max=(pic.scale.x.T(max.x),pic.scale.y.T(max.y),pic.scale.z.T(max.z));
  return b;
};
  
real crop(real x, real min, real max) {return min(max(x,min),max);}

triple xcrop(triple v, real min, real max) 
{
  return (crop(v.x,min,max),v.y,v.z);
}

triple ycrop(triple v, real min, real max) 
{
  return (v.x,crop(v.y,min,max),v.z);
}

triple zcrop(triple v, real min, real max) 
{
  return (v.x,v.y,crop(v.z,min,max));
}

void xlimits(bbox3 b, real min, real max)
{
  b.min=xcrop(b.min,min,max);
  b.max=xcrop(b.max,min,max);
}

void ylimits(bbox3 b, real min, real max)
{
  b.min=ycrop(b.min,min,max);
  b.max=ycrop(b.max,min,max);
}

void zlimits(bbox3 b, real min, real max)
{
  b.min=zcrop(b.min,min,max);
  b.max=zcrop(b.max,min,max);
}

// Restrict the x, y, and z limits to box(min,max).
void limits(bbox3 b, triple min, triple max)
{
  xlimits(b,min.x,max.x);
  ylimits(b,min.y,max.y);
  zlimits(b,min.z,max.z);
}
  
void axes(Label xlabel="$x$", Label ylabel="$y$", Label zlabel="$z$", 
	  bbox3 b, pen p=currentpen, arrowbar arrow=None,
	  bool put=Below, projection P=currentprojection)
{
  xaxis(xlabel,b,p,arrow,put,P);
  yaxis(ylabel,b,p,arrow,put,P);
  zaxis(zlabel,b,p,arrow,put,P);
}

void axes(Label xlabel="$x$", Label ylabel="$y$", Label zlabel="$z$", 
	  triple min, triple max, pen p=currentpen, arrowbar arrow=None,
	  bool put=Below, projection P=currentprojection)
{
  axes(xlabel,ylabel,zlabel,limits(min,max),p,arrow,put,P);
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
public guide3 Spline(... guide3[])=operator ..;
		       
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
guide3 graph(triple F(path, real), path p, int n=1,
	     interpolate join=operator --)
{
  guide3 g;
  for(int i=0; i < n*length(p); ++i)
    g=join(g,F(p,i/n));
  return cyclic(p) ? join(g,cycle3) : join(g,F(p,length(p)));
}

guide3 graph(triple F(pair), path p, int n=1, interpolate join=operator --)
{
  return graph(new triple(path p, real position) 
	       {return F(point(p,position));},p,n,join);
}

guide3 graph(picture pic=currentpicture, real f(pair), path p, int n=1,
	     interpolate join=operator --) 
{
  return graph(new triple(pair z) {return Scale(pic,(z.x,z.y,f(z)));},p,n,
	       join);
}

guide3 graph(real f(pair), path p, int n=1, real T(pair),
	     interpolate join=operator --)
{
  return graph(new triple(pair z) {pair w=T(z); return (w.x,w.y,f(w));},p,n,
	       join);
}

struct grid {
  pair back,front;
  int n,m;
  bool reverse;
  int sign;
  
  pair sample(int i, int j) {
    return (interp(back.x,front.x,i/n),
            interp(back.y,front.y,j/m));
  }

  static grid set(pair a, pair b, int n, int m=n, 
		  projection P=currentprojection) {
    grid g=new grid;
    g.n=n;
    g.m=m;
    pair camera=(P.camera.x,P.camera.y);
    pair z=b-a;
    g.sign=sgn(dot(camera,I*conj(z)));
    g.reverse=g.sign*sgn(dot(camera,I*z)) >= 0;
  
    if(g.sign >= 0) {
      g.back=a;
      g.front=b;
    } else {
      g.back=b;
      g.front=a;
    }
    return g;
  }
}

// draw the surface described by a matrix f, with lighting
picture surface(real[][] f, pair a, pair b,
		pen surfacepen=lightgray, pen meshpen=nullpen,
		light light=currentlight, projection P=currentprojection)
{
  picture pic;

  if(!rectangular(f)) abort("matrix is not rectangular");
  
  if(light.source == O && meshpen == nullpen) meshpen=currentpen;
  
  int n=f.length-1;
  int m=f[0].length-1;
  
  grid g=grid.set(a,b,n,m,P);

  pair z0=g.sample(0,0);
  real dx=g.sample(1,0).x-z0.x;
  real dy=g.sample(0,1).y-z0.y;
  
// calculate colors at each point
  pen color(int i, int j) {
    real dfx,dfy;
    if(i == 0) dfx=f[1][j]-f[0][j];
    else if(i == n) dfx=f[n][j]-f[n-1][j];
    else dfx=0.5(f[i+1][j]-f[i-1][j]);
    if(j == 0) dfy=f[i][1]-f[i][0];
    else if(j == m) dfy=f[i][m]-f[i][m-1];
    else dfy=0.5(f[i][j+1]-f[i][j-1]);
    return light.intensity(cross((dx,0,dfx),(0,dy,dfy)))*surfacepen;
  }

  int[] edges={0,0,0,2};
  
  void drawcell(int i, int j, int i1, int j1, int i2, int j2) {
    pair a=g.sample(i1,j1);
    pair b=g.sample(i2,j2);
    pair[] v={project((a.x,a.y,f[i][j]),P),
	      project((a.x,b.y,f[i][j+1]),P),
	      project((b.x,b.y,f[i+1][j+1]),P),
	      project((b.x,a.y,f[i+1][j]),P)};
    guide g=v[0]--v[1]--v[2]--v[3]--cycle;
    if(light.source == O)
      filldraw(pic,g,surfacepen,meshpen);
    else {
      pen[] pcell={color(i1,j1),color(i1,j2),color(i2,j2),color(i2,j1)}; 
      gouraudshade(pic,g,pcell,v,edges);
      if(meshpen != nullpen) draw(pic,g,meshpen);
    }
  }

  if(g.sign >= 0)
    if(g.reverse)
      for(int j=0; j < m; ++j)
	for(int i=0; i < n; ++i)
	  drawcell(i,j,i,j,i+1,j+1);
    else
      for(int i=0; i < n; ++i)
	for(int j=0; j < m; ++j)
	  drawcell(i,j,i,j,i+1,j+1);
  else
    if(g.reverse)
      for(int j=0; j < m; ++j)
	for(int i=0; i < n; ++i)
	  drawcell(n-1-i,m-1-j,i+1,j+1,i,j);
    else
      for(int i=0; i < n; ++i)
	for(int j=0; j < m; ++j)
	  drawcell(n-1-i,m-1-j,i+1,j+1,i,j);
  
  return pic;
}

// draw the surface described by a function f, with lighting
picture surface(real f(pair z), pair a, pair b, int n=nmesh, int m=n,
		pen surfacepen=lightgray, pen meshpen=nullpen,
		light light=currentlight, projection P=currentprojection)
{
  real[][] z=new real[n+1][m+1];

  for(int i=0; i <= n; ++i)
    for(int j=0; j <= m; ++j)
      z[i][j]=f((interp(a.x,b.x,i/n),interp(a.y,b.y,j/m)));
  
  return surface(z,a,b,surfacepen,meshpen,light,P);
}

// draw the surface described by f, subsampling nsub times along cell edges
picture surface(real f(pair z), int nsub, pair a, pair b, int n=nmesh, int m=n,
		pen surfacepen=lightgray, pen meshpen=currentpen,
		projection P=currentprojection)
{
  picture pic;

  grid g=grid.set(a,b,n,m,P);
  
  void drawcell(int i, int j) {
    guide3 g=graph(f,box(g.sample(i,j),g.sample(i+1,j+1)),nsub);
    filldraw(pic,project(g,P),surfacepen,meshpen);
  }

  if(g.reverse)
    for(int j=0; j < m; ++j)
      for(int i=0; i < n; ++i)
	drawcell(i,j);
  else
    for(int i=0; i < n; ++i)
      for(int j=0; j < m; ++j)
	drawcell(i,j);

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
	  triple normal=Z, int ngraph=400)
{
  path3 p=polargraph(new real(real theta, real phi) {return r;},
		     new real(real t) {
		       return radians(interp(theta1,theta2,t));},
		      new real(real t) {
			return radians(interp(phi1,phi2,t));},
		      ngraph,operator ..);
  if(normal != Z)
    p=rotate(longitude(normal,warn=false),Z)*rotate(colatitude(normal),Y)*p;
  return shift(c)*p;
}

// True circle
path3 Circle(triple c, real r, triple normal=Z, int ngraph=400)
{
  return Arc(c,r,90,0,90,360,normal,ngraph)..cycle3;
}
