import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10,up=X);
currentlight=(3,2,-1);
pen color=green;
real alpha=250;

real f(real x) {return x^2;}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}

pair H(real x) {return (x,x);}

path p=graph(F,0,1,n=10,Spline)--graph(H,1,0,n=10,Spline)--cycle;
surface s=surface(bezulate(p));
path3 p3=path3(p);

revolution a=revolution(p3,X,-alpha,0);
draw(surface(a),color);
//draw(a,blue,false);
draw(s,color);
draw(rotate(-alpha,X)*s,color);
draw(p,blue);

xaxis3(Label("$x$",1,align=-2X+Y),dashed,Arrow3);
yaxis3(Label("$y$",1,align=X),Arrow3);
dot(Label("$(1,1)$"),(1,1,0),Y);
arrow("$y=x$",(0.7,0.7,0),Y,0.75cm,red);
arrow("$y=x^2$",F3(0.7),X,0.75cm,red);
draw(arc(1.25X,0.3,90,90,3,-90),Arrow3);
