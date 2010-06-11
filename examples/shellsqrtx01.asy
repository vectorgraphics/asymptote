import graph3;
import solids;
size(0,150);
currentprojection=orthographic(1,0,10,up=Y);
pen color=green;
real alpha=-240;

real f(real x) {return sqrt(x);}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}

path p=graph(F,0,1,n=30,operator ..)--(1,0)--cycle;
path3 p3=path3(p);

revolution a=revolution(p3,X,alpha,0);

render render=render(compression=0,merge=true);
draw(surface(a),color,render);
draw(p3,blue);

surface s=surface(p);
draw(s,color,render);
draw(rotate(alpha,X)*s,color,render);

xaxis3(Label("$x$",1),xmax=1.25,dashed,Arrow3);
yaxis3(Label("$y$",1),Arrow3);

dot("$(1,1)$",(1,1,0));
arrow("$y=\sqrt{x}$",F3(0.8),Y,0.75cm,red);

real r=0.4;
draw(F3(r)--(1,f(r),0),red);
real x=(1+r)/2;

draw("$r$",(x,0,0)--(x,f(r),0),X+0.2Z,red,Arrow3,PenMargin3);
draw(arc(1.1X,0.4,90,90,3,-90),Arrow3);
