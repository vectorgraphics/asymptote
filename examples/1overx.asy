import graph; 
size(200,IgnoreAspect); 

real f(real x) {return 1/x;}; 

bool branch(real x)
{
  static int lastsign=0; 
  if(x == 0) {lastsign=0; return false;}
  int sign=sgn(x); 
  bool b=lastsign == 0 || sign == lastsign; 
  lastsign=sign; 
  return b; 
}

draw(graph(f,-1,1,branch));
axes("$x$","$y$",red);
