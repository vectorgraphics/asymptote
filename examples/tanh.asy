import graph;
size(100,0);

real f(real x) {return tanh(x);}
pair F(real x) {return (x,f(x));}

xaxis("$x$");
yaxis("$y$");

draw(graph(f,-2.5,2.5,operator ..));

label("$\tanh x$",F(1.5),1.25*N);

