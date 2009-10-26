import graph;
size(300,0);

real f(real x) {return (x != 0.0) ? x * sin(1.0 / x) : 0.0;}
pair F(real x) {return (x,f(x));}

xaxis("$x$",red);
yaxis(red);
draw(graph(f,-1.2/pi,1.2/pi,1000));
label("$x\sin\frac{1}{x}$",F(1.1/pi),NW);

picture pic;
size(pic,50,IgnoreAspect);
xaxis(pic,red);
yaxis(pic,red);
pair z=(0.1/pi,0.1/pi);
draw(pic,graph(pic,f,-z.x,z.x,1000));

add(new void(frame f, transform t) {
    frame G=shift(point(f,N+0.85W))*align(bbox(pic,blue),10SE);
    add(f,G);
    draw(f,t*box(-z,z),blue);
    draw(f,point(G,E)--t*(-z.x,0),blue);
  });

