import graph3;

size(12cm,0);

currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

real f(pair z) {return abs(z)^2;}

path3 gradient(pair z) {
  static real dx=sqrt(realEpsilon), dy=dx;
  return O--((f(z+dx)-f(z-dx))/dx,(f(z+I*dy)-f(z-I*dy))/dy,0);
}

bbox3 b=limits((-1,-1,0),(1,1,2));
xaxis(rotate(X)*"$x$",b,RightTicks(rotate(X)*Label));
yaxis(rotate(Y)*"$y$",b.X(),b.XY(),LeftTicks(rotate(Y)*Label));
zaxis("$z$",b,RightTicks());

pair A=xypart(b.O());
pair B=xypart(b.XY());

triple F(pair z) {return (z.x,z.y,0);}

add(vectorfield(gradient,F,A,B,red));

add(surface(f,A,B,50,gray+opacity(0.5)));
