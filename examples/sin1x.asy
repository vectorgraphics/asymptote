import graph;
size(200,0);

real f(real x) {return (x != 0) ? sin(1/x) : 0;}
real T(real x) {return 2/(x*pi);}

real a=-4/pi, b=4/pi;
int n=150,m=5;

xaxis("$x$",red);
yaxis(red);

draw(graph(f,a,-T(m),n)--graph(f,-m,-(m+n),n,T)--(0,f(0))--graph(f,m+n,m,n,T)--
     graph(f,T(m),b,n));

label("$\sin\frac{1}{x}$",(b,f(b)),SW);
