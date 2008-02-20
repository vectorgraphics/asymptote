import graph;
size(300,150,IgnoreAspect);

real f(real x) {return 1/x^(1.1);}
pair F(real x) {return (x,f(x));}

dotfactor=7;

void subinterval(real a, real b)
{
  path g=box((a,0),(b,f(b)));
  filldraw(g,lightgray); 
  draw(box((a,f(a)),(b,0)));
}

int a=1, b=9;
  
xaxis("$x$",0,b); 
yaxis("$y$",0); 
 
draw(graph(f,a,b,operator ..),red);
 
int n=2;

for(int i=a; i <= b; ++i) {
  if(i < b) subinterval(i,i+1);
  if(i <= n) labelx(i);
  dot(F(i));
}

int i=n;
labelx("$\ldots$",++i);
labelx("$k$",++i);
labelx("$k+1$",++i);
labelx("$\ldots$",++i);

arrow("$f(x)$",F(i-1.5),NE,1.5cm,red,Margin(0,0.5));

