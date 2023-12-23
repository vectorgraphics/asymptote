import graph;
size(200,IgnoreAspect);

real f(real x) {return 1/x;}

bool3 branch(real x)
{
  return x != 0;
}

draw(graph(f,-1,1,branch));
axes("$x$","$y$",red);
