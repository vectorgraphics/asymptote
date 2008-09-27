import graph3;
import solids;
size(0,150);
currentprojection=orthographic(0,2,10,up=Y);
currentlight=(0.25,-0.25,5);

pen color=green;

real f(real x) {return x^2;}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}
pair H(real x) {return (x,x);}

path p=graph(F,0,1,n=10)--graph(H,1,0,n=10)--cycle;
path3 p3=path3(p);

revolution a=revolution(-X,p3,Y,0,180);
draw(surface(a),color);
draw(a,1,blue,false);
surface s=surface(bezulate(p));
draw(s,color);
transform3 t=shift(-2X)*rotate(180,Y);
draw(t*s,color);
draw(p3);
draw(t*p3);

draw((-1,0,0)--(-1,1,0),dashed);
xaxis3(Label("$x$",1),Arrow3);
yaxis3(Label("$y$",1),Arrow3);
dot(Label("$(1,1)$"),(1,1,0));
dot(Label("$(-1,1)$"),(-1,1,0),W);
arrow("$y=x^{2}$",F3(0.7),X,1cm); 
arrow("$y=x$",(0.3,0.3,0),X,1.5cm); 
draw(circle((-1,1,0),2,Y),dashed);
draw((-1,1,0)--(1,1,0),dashed);
draw(shift(-X)*arc(0.02Y,0.3,90,0,0,0,CW),Arrow3);
