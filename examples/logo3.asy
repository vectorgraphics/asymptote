import three;

size(560,320,IgnoreAspect);
size3(140,80,15);
currentprojection=perspective(-3,20,10,up=Y);
currentlight=White;

path[] outline;

real a=-0.4;
real b=0.95;
real y1=-5;
real y2=-3y1/2;
path A=(a,0){dir(10)}::{dir(89.5)}(0,y2);
outline.push(A);
outline.push((0,y1){dir(88.3)}::{dir(20)}(b,0));
real c=0.5*a;
pair z=(0,2.5);
path[] text = shift(0,2)*scale(0.01,0.15)*
  texpath(Label("{\it symptote}",z,0.25*E+0.169S,fontsize(24pt)));
outline.append(text);
pair w=(0,1.7);
outline.push(intersectionpoint(A,w-1--w)--w);
outline.push((0,y1)--(0,y2));
outline.push((a,0)--(b,0));

for(path p : outline)
  draw(extrude(p,-0.1Z),material(lightgray,shininess=1.0));

draw(path3(outline),red+linewidth(0));

draw(surface(text),red,nolight);

