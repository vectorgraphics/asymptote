import three;

size(560,320,IgnoreAspect);
size3(140,80,15);

currentprojection=
  perspective(camera=(-0.7387428806982,-50.4754947040394,10.1433958227456),
              up=(0.0023,0.158056646909439,0.000753078712035578),
              target=(0.275794537878997,1.26294757366264,-0.0498743055531516),
              zoom=1,
              angle=3.26228798100531,
              autoadjust=false);

real a=-0.4;
real b=0.95;
real y1=-5;
real y2=-3y1/2;
path A=(a,0){dir(10)}::{dir(89.5)}(0,y2);
path B=(0,y1){dir(88.3)}::{dir(20)}(b,0);
real c=0.5*a;
pair z=(0,2.5);
transform t=scale(1,15);
transform T=inverse(scale(t.yy,t.xx));
path[] g=shift(0,1.979)*scale(0.01)*t*
  texpath(Label("{\it symptote}",z,0.25*E+0.169S,fontsize(24pt)));
pair w=(0,1.7);
pair u=intersectionpoint(A,w-1--w);

real h=0.25*linewidth();
real hy=(T*(h,h)).x;
g.push(t*((a,hy)--(b,hy)..(b+hy,0)..(b,-hy)--(a,-hy)..(a-hy,0)..cycle));
g.push(T*((h,y1)--(h,y2)..(0,y2+h)..(-h,y2)--(-h,y1)..(0,y1-h)..cycle));
g.push(shift(0,w.y)*t*((u.x,hy)--(w.x,hy)..(w.x+hy,0)..(w.x,-hy)--(u.x,-hy)..(u.x-hy,0)..cycle));
real f=0.75;
g.push(point(A,0)--shift(-f*hy,f*h)*A--point(A,1)--shift(f*hy,-f*h)*reverse(A)--cycle);
g.push(point(B,0)--shift(f*hy,-f*h)*B--point(B,1)--shift(-f*hy,f*h)*reverse(B)--cycle);

triple H=-0.1Z;
material m=material(0.5*red+0.125*purple,shininess=1.0);

for(path p : g)
  draw(extrude(p,H),m,nolight);

surface s=surface(g);
draw(s,red,nolight);
draw(shift(H)*s,m,nolight);
