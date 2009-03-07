import graph3;
settings.thick=false;

size3(200,IgnoreAspect);

file in=line(input("filesurface.dat"));
real[] x=in;
real[] y=in;

real[][] f=dimension(in,0,0);

triple f(pair t) {
  int i=round(t.x);
  int j=round(t.y);
  return (x[i],y[j],f[i][j]);
}

draw(surface(f,(0,0),(x.length-1,y.length-1),x.length-1,y.length-1),
     surfacepen=lightgray,meshpen=black);

xaxis3("$x$",Bounds(),red,InTicks);
yaxis3("$y$",Bounds(),red,InTicks(Step=1,step=0.1));
zaxis3("$z$",Bounds(),red,InTicks);
