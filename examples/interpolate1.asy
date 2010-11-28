// Lagrange and Hermite interpolation in Asymptote
// Author: Olivier Guibé

import interpolate;
import graph;

// Test 1: The Runge effect in the Lagrange interpolation of 1/(x^2+1).

unitsize(2cm);

real f(real x) {return(1/(x^2+1));}
real df(real x) {return(-2*x/(x^2+1)^2);}

real a=-5, b=5;
int n=15;
real[] x,y,dy;
x=a+(b-a)*sequence(n+1)/n;
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=diffdiv(x,y);
fhorner p=fhorner(h);
draw(graph(p,a,b,n=500),"$x\longmapsto{}L_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\frac{1}{x^2+1}$");

xlimits(-5,5);
ylimits(-1,1,Crop);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

attach(legend(),point(10S),30S);

shipout("runge1");

erase();

// Test 2: The Runge effect in the Hermite interpolation of 1/(x^2+1).

real f(real x) {return(1/(x^2+1));}
real df(real x) {return(-2*x/(x^2+1)^2);}

real a=-5, b=5;
int n=16;
real[] x,y,dy;
x=a+(b-a)*sequence(n+1)/n;
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=hdiffdiv(x,y,dy);
fhorner ph=fhorner(h);
draw(graph(p,a,b,n=500),"$x\longmapsto{}H_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\frac{1}{x^2+1}$");

unitsize(2cm);

xlimits(-5,5);
ylimits(-1,5,Crop);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

attach(legend(),point(10S),30S);

shipout("runge2");

erase();

// Test 3: The Runge effect does not occur for all functions:
// Lagrange interpolation of a function whose successive derivatives
// are bounded by a constant M (here M=1) is shown here to converge.

real f(real x) {return(sin(x));}
real df(real x) {return(cos(x));}

real a=-5, b=5;
int n=16;
real[] x,y,dy;
x=a+(b-a)*sequence(n+1)/n;
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=diffdiv(x,y);
fhorner p=fhorner(h);

draw(graph(p,a,b,n=500),"$x\longmapsto{}L_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\cos(x)$");

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

attach(legend(),point(10S),30S);

shipout("runge3");

erase();

// Test 4: However, one notes here that numerical artifacts may arise
// from limit precision (typically 1e-16).

real f(real x) {return(sin(x));}
real df(real x) {return(cos(x));}

real a=-5, b=5;
int n=72;
real[] x,y,dy;
x=a+(b-a)*sequence(n+1)/n;
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=diffdiv(x,y);
fhorner p=fhorner(h);

draw(graph(p,a,b,n=500),"$x\longmapsto{}L_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\cos(x)$");

ylimits(-1,5,Crop);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

attach(legend(),point(10S),30S);

shipout("runge4");

erase();        

// Test 5: The situation is much better using Tchebychev points. 

unitsize(2cm);

real f(real x) {return(1/(x^2+1));}
real df(real x) {return(-2*x/(x^2+1)^2);}

real a=-5, b=5;
int n=16;
real[] x,y,dy;
fhorner p,ph,ph1;
for(int i=0; i <= n; ++i)
  x[i]=(a+b)/2+(b-a)/2*cos((2*i+1)/(2*n+2)*pi);
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=diffdiv(x,y);
fhorner p=fhorner(h);

draw(graph(p,a,b,n=500),"$x\longmapsto{}T_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\frac{1}{x^2+1}$");

xlimits(-5,5);
ylimits(-1,2,Crop);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);
attach(legend(),point(10S),30S);

shipout("runge5");

erase();

// Test 6: Adding a few more Tchebychev points yields a very good result.

unitsize(2cm);

real f(real x) {return(1/(x^2+1));}
real df(real x) {return(-2*x/(x^2+1)^2);}

real a=-5, b=5;
int n=26;
real[] x,y,dy;
for(int i=0; i <= n; ++i)
  x[i]=(a+b)/2+(b-a)/2*cos((2*i+1)/(2*n+2)*pi);
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=diffdiv(x,y);
fhorner p=fhorner(h);
draw(graph(p,a,b,n=500),"$x\longmapsto{}T_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\frac{1}{x^2+1}$");

xlimits(-5,5);
ylimits(-1,2,Crop);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);
attach(legend(),point(10S),30S);


shipout("runge6");

erase();        

// Test 7: Another Tchebychev example.

unitsize(2cm);

real f(real x) {return(sqrt(abs(x-1)));}

real a=-2, b=2;
int n=30;
real[] x,y,dy;
for(int i=0; i <= n; ++i)
  x[i]=(a+b)/2+(b-a)/2*cos((2*i+1)/(2*n+2)*pi);
y=map(f,x);
dy=map(df,x);
for(int i=0; i <= n; ++i)
  dot((x[i],y[i]),5bp+blue);
horner h=diffdiv(x,y);
fhorner p=fhorner(h);
draw(graph(p,a,b,n=500),"$x\longmapsto{}T_{"+string(n)+"}$");
draw(graph(f,a,b),red,"$x\longmapsto{}\sqrt{|x-1|}$");

xlimits(-2,2);
ylimits(-0.5,2,Crop);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);
attach(legend(),point(10S),30S);

shipout("runge7");
