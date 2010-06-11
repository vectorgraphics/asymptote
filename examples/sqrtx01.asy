import graph3;
import solids;
size(0,150);
currentprojection=perspective(1.5,0,10,Y);
pen color=green+opacity(0.75);

real f(real x){return sqrt(x);}
pair F(real x){return (x,f(x));}
triple F3(real x){return (x,f(x),0);}

path p=graph(F,0,1,n=20,operator ..);
path3 p3=path3(p);

revolution a=revolution(p3,X,0,360);
draw(surface(a),color,render(compression=Low,merge=true));
draw(p3,blue);

real x=relpoint(p,0.5).x;

xaxis3(Label("$x$",1),xmax=1.5,dashed,Arrow3);
yaxis3(Label("$y$",1),Arrow3);
dot(Label("$(1,1)$"),(1,1,0));
arrow(Label("$y=\sqrt{x}$"),F3(0.7),Y,0.75cm,red);
draw(arc(1.2X,0.4,90,90,175,-40,CW),Arrow3);
draw("$r$",(x,0,0)--F3(x),red,Arrow3,PenMargin3);
