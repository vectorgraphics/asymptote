import graph;
size(0,150);

int a=-1, b=1;

real f(real x) {return x^3-x+2;}
real g(real x) {return x^2;}

draw(graph(f,a,b,operator ..),red);
draw(graph(g,a,b,operator ..),blue);
 
xaxis(); 

int n=5;

real width=(b-a)/(real) n;
for(int i=0; i <= n; ++i) {
  real x=a+width*i;
  draw((x,g(x))--(x,f(x)));
}
 
labelx("$a$",a);
labelx("$b$",b);
draw((a,0)--(a,g(a)),dotted);
draw((b,0)--(b,g(b)),dotted);

real m=a+0.73*(b-a);
arrow("$f(x)$",(m,f(m)),N,red);
arrow("$g(x)$",(m,g(m)),E,0.8cm,blue);

int j=2;
real xi=b-j*width;
real xp=xi+width;
real xm=0.5*(xi+xp);
pair dot=(xm,0.5*(f(xm)+g(xm)));
dot(dot,darkgreen+4.0);
arrow("$\left(x,\frac{f(x)+g(x)}{2}\right)$",dot,NE,2cm,darkgreen);

