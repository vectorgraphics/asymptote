import graph;

real f(real x) {return x^3-x+2;}
pair F(real x) {return (x,f(x));}

void rectangle(real a, real b, real c, real h(real,real))
{
  real height=(a < c && c < b) ? f(c) : h(f(a),f(b));
  pair p=(a,0), q=(b,height);
  path g=box(p,q);
  fill(g,lightgray); 
  draw(g); 
}

void partition(real a, real b, real c, real h(real,real))
{
  rectangle(a,a+.4,c,h);
  rectangle(a+.4,a+.6,c,h);
  rectangle(a+.6,a+1.2,c,h);
  rectangle(a+1.2,a+1.6,c,h);
  rectangle(a+1.6,a+1.8,c,h);
  rectangle(a+1.8,b,c,h);

  draw((a,0)--(F(a)));
  draw((b,0)--(F(b)));

  draw(graph(f,a,b,operator ..),red);
  draw((a,0)--(b,0));
  labelx("$a$",a);
  labelx("$b$",b);
}
