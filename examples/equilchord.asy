import graph3;

size(0,150);
currentprojection=perspective(5,-4,6);
currentlight=(-1,-1,2);
real t=0.5;

real F(pair z) {
  return (z.x^2+z.y^2 <= 1) ? sqrt(3)*(sqrt(1-z.x^2)-abs(z.y)) : 0; 
}

real a=1.5;
draw((-a,-a,0)--(-a,a,0)--(a,a,0)--(a,-a,0)--cycle,lightgray);

xaxis3(Label("$x$",1),red,Arrow3);
yaxis3(Label("$y$",1),red,Arrow3);
draw(circle((0,0,0),1),dashed);
draw(surface(F,(-1,-1),(t,1),20,monotonic),green,black,render(merge=true));
real y=sqrt(1-t^2);
draw((t,y,0)--(t,-y,0)--(t,0,sqrt(3)*y)--cycle,blue);
label("$1$",(1,0,0),-Y+X);
