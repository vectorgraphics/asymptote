import graph;
size(0,100);

real f(real x) {return sqrt(1-x*x);}

picture b;
picture a=b;
real x=0.5;
pair zp=(x,f(x));
pair zm=(x,-f(x));
pair zh=(1.5,0.3);
guide g=zm--zp--zh--cycle;

xaxis("$x$");
yaxis(-infinity,1.3,"$y$");

draw("1",(0,0)--dir(135),Arrow,PenMargin);

draw(circle((0,0),1));
filldraw(g,red+white);

picture hidden=new picture;
draw(hidden,circle((0,0),1),dashed);
clip(hidden,g);

add(hidden);
shipout();
