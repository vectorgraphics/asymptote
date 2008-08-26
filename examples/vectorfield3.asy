import graph3;

size(12cm,0);

currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

real f(pair z) {return abs(z)^2;}

path3 gradient(pair z) {
  static real dx=sqrt(realEpsilon), dy=dx;
  return O--((f(z+dx)-f(z-dx))/2dx,(f(z+I*dy)-f(z-I*dy))/2dy,0);
}

xaxis3(XY()*"$x$",RightTicks3(XY()*Label));
yaxis3(XY()*"$y$",LeftTicks3(YX()*Label));
zaxis3("$z$",RightTicks3());

pair A=(-1,-1);
pair B=(1,1);

triple F(pair z) {return (z.x,z.y,0);}

add(vectorfield(gradient,F,A,B,red));

draw(surface(f,A,B,50,gray+opacity(0.5)));
