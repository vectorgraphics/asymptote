import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10,up=Y);

pen color=green;
real alpha=250;

real f(real x) {return x^2;}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}

path p=graph(F,0,1,n=10,operator ..)--cycle;
path3 p3=path3(p);

revolution a=revolution(p3,X,-alpha,0);
render render=render(compression=0,merge=true);
draw(surface(a),color,render);
surface s=surface(p);
draw(s,color,render);
draw(rotate(-alpha,X)*s,color,render);

draw(p3,blue);

xaxis3(Label("$x$",1),xmax=1.25,dashed,Arrow3);
yaxis3(Label("$y$",1),Arrow3);
dot(Label("$(1,1)$"),(1,1,0),X+Y);
arrow("$y=x$",(0.7,0.7,0),Y,0.75cm,red);
arrow("$y=x^2$",F3(0.7),X,0.75cm,red);
draw(arc(1.1X,0.3,90,90,3,-90),Arrow3);
