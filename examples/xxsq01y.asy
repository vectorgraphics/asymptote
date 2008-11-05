import solids;
size(0,150);
currentprojection=perspective(0,0,10);
pen color=green;
real alpha=240;

real f(real x) {return x^2;}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}

path p=graph(F,0,1,n=10,operator ..)--cycle;

revolution a=revolution(path3(p),Y,0,alpha);
draw(surface(a),color);

surface s=surface(bezulate(p));
draw(s,color);
draw(rotate(alpha,(0,1,0))*s,color);

xaxis3(Label("$x$",1),Arrow3);
yaxis3(Label("$y$",1),ymax=1.25,dashed,Arrow3);

dot(Label("$(1,1)$"),(1,1,0),X);
arrow("$y=x^{2}$",F3(0.7),X,0.75cm,red); 
arrow("$y=x$",(0.8,0.8,0),Y,1cm,red); 

real r=0.4;
draw((r,f(r),0)--(r,r,0),red);
draw("$r$",(0,(f(r)+r)*0.5,0)--(r,(f(r)+r)*0.5,0),N,red,Arrows3,PenMargins3);
draw(arc(1.1Y,0.3,90,0,7.5,180),Arrow3);
