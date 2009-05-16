import graph3;

size(12cm,0);

currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

real f(pair z) {return abs(z)^2;}

path3 gradient(pair z) {
  static real dx=sqrtEpsilon, dy=dx;
  return O--((f(z+dx)-f(z-dx))/2dx,(f(z+I*dy)-f(z-I*dy))/2dy,0);
}

pair a=(-1,-1);
pair b=(1,1);

triple F(pair z) {return (z.x,z.y,0);}

add(vectorfield(gradient,F,a,b,red));

draw(surface(f,a,b,Spline),gray+opacity(0.5));

xaxis3(XY()*"$x$",OutTicks(XY()*Label));
yaxis3(XY()*"$y$",InTicks(YX()*Label));
zaxis3("$z$",OutTicks);
