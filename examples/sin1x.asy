import graph;

size(200,0);

real f(real x) {return (x != 0) ? sin(1/x) : 0;}
real T(real x) {return 2/(x*pi);}

real a=-4/pi, b=4/pi;
int n=150,m=5;

guide g=graph(f,a,-T(m),n);
g=graph(g,f,-m,-(m+n),n,T);
g=g--(0,f(0));
g=graph(g,f,m+n,m,n,T);
g=graph(g,f,T(m),b,n);

xaxis("$x$",red);
yaxis(red);

draw(g);

label("$\sin\frac{1}{x}$",(b,f(b)),SW);

