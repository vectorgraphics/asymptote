import graph;

size(150,0);

real f(real x) {return log(x);}
pair F(real x) {return (x,f(x));}

xaxis(0,"$x$");
yaxis("$y$");

draw(graph(f,0.01,10,operator ..));

labelx(1,SSE);
label("$\log x$",F(7),SE);

