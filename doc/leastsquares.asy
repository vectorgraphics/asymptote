size(400,200,IgnoreAspect);

import graph;
import stats;

file fin=input("leastsquares.dat").line();

real[][] a=fin;
a=transpose(a);

real[] t=a[0], rho=a[1];

// Read in parameters from the keyboard:
//real first=getreal("first");
//real step=getreal("step");
//real last=getreal("last");

real first=100;
real step=50;
real last=700;

// Remove negative or zero values of rho:
t=rho > 0 ? t : null;
rho=rho > 0 ? rho : null;
    
scale(Log(true),Linear(true));

int n=step > 0 ? ceil((last-first)/step) : 0;

real[] T,xi,dxi;

for(int i=0; i <= n; ++i) {
  real first=first+i*step;
  real[] logrho=(t >= first & t <= last) ? log(rho) : null;
  real[] logt=(t >= first & t <= last) ? -log(t) : null;
  
  if(logt.length < 2) break;
  
  // Fit to the line logt=L.m*logrho+L.b:
  linefit L=leastsquares(logt,logrho);
    
  T.push(first);
  xi.push(L.m);
  dxi.push(L.dm);
} 
    
draw(graph(T,xi),blue);
errorbars(T,xi,dxi,red);

crop();

ylimits(0);

xaxis("$T$",BottomTop,LeftTicks);
yaxis("$\xi$",LeftRight,RightTicks);
