import graph3;
import solids;
size(0,150);
currentprojection=perspective(8,10,2);

revolution r=cylinder(-4Z,4,8,Z);
draw(circle(O,4,Z));
draw(surface(r),green+opacity(0.2));

triple F(real x){return (x,sqrt(16-x^2),sqrt((16-x^2)/3));}
path3 p=graph(F,0,4,operator ..);
path3 q=reverse(p)--rotate(180,(0,4,4/sqrt(3)))*p--cycle;
draw(surface(q--cycle),blue);

real t=2;
path3 triangle=(t,0,0)--(t,sqrt(16-t^2),0)--F(t)--cycle;
draw(surface(triangle),red);

xaxis3(Label("$x$",1),Arrow3);
yaxis3(Label("$y$",1),Arrow3);
zaxis3(Label("$z$",1),dashed,Arrow3);
