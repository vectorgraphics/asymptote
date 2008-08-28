import graph; 
import interpolate; 
 
size(15cm,15cm,IgnoreAspect); 
 
real a=1997, b=2002; 
int n=5; 
real[] xpt=a+sequence(n+1)*(b-a)/n; 
real[] ypt={31,36,26,22,21,24}; 
horner h=diffdiv(xpt,ypt);
fhorner L=fhorner(h);
 
scale(false,true);

pen p=linewidth(1);

draw(graph(L,a,b),dashed+black+p,"Lagrange interpolation");
draw(graph(xpt,ypt,Hermite(natural)),red+p,"natural spline");
draw(graph(xpt,ypt,Hermite(monotonic)),blue+p,"monotone spline");
xaxis("$x$",BottomTop,LeftTicks(Step=1,step=0.25));
yaxis("$y$",LeftRight,RightTicks(Step=5));

dot(pairs(xpt,ypt),4bp+gray(0.3));

attach(legend(),point(10S),30S);
