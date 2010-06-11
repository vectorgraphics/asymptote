import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,1,10,up=Y);
currentlight=White;

real f(real x) {return sqrt(x);}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}

path p=graph(F,0,1,n=25,operator ..);
path3 p3=path3(p);

revolution a=revolution(p3,Y,0,360);
draw(surface(a),green,render(compression=Low,merge=true));
draw(p3,blue);

xtick((0,0,0));
xtick((1,0,0));

xaxis3(Label("$x$",1),Arrow3);
yaxis3(Label("$y$",1),ymax=1.5,dashed,Arrow3);
dot(Label("$(1,1)$"),(1,1,0));
arrow("$y=\sqrt{x}$",F3(0.5),X,0.75cm,red);
draw(arc(1.2Y,0.3,90,0,7.5,140),Arrow3);
