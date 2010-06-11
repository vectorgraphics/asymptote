import graph3;
import solids;
size(0,150);
currentprojection=perspective(8,10,2);
currentlight=White;

draw(circle(O,4,Z));
draw(shift(-4Z)*scale(4,4,8)*unitcylinder,green+opacity(0.2));

triple F(real x){return (x,sqrt(16-x^2),sqrt((16-x^2)/3));}
path3 p=graph(F,0,4,operator ..);
path3 q=reverse(p)--rotate(180,(0,4,4/sqrt(3)))*p--cycle;

render render=render(merge=true);
draw(surface(q--cycle),red,render);

real t=2;
path3 triangle=(t,0,0)--(t,sqrt(16-t^2),0)--F(t)--cycle;
draw(surface(triangle),blue,render);

xaxis3("$x$",Arrow3,PenMargin3(0,0.25));
yaxis3("$y$",Arrow3,PenMargin3(0,0.25));
zaxis3("$z$",dashed,Arrow3);
