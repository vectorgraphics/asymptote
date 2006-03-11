import graph3;

size(12cm,0);

real sinc(pair z) {
  real r=2pi*abs(z);
  return r != 0 ? sin(r)/r : 1;
}

bbox3 b=limits((-2,-2,-0.2),(2,2,1.2));
currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

aspect(b,1,1,1);

xaxis(rotate(X)*"$x$",b,RightTicks(rotate(X)*Label(fontsize(10pt))));
yaxis(rotate(Y)*"$y$",b.X(),b.XY(),
      LeftTicks(rotate(Y)*Label(fontsize(10pt))));
zaxis("$z$",b,RightTicks(Label(fontsize(10pt))));

layer();

add(surface(sinc,xypart(b.O()),xypart(b.XY()),60));
