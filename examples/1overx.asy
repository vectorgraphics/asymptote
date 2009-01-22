size(200,IgnoreAspect); 
import graph; 
real f(real x) {return 1/x;}; 
draw(graph(f,-1,1,new bool(real x) {
      static int lastsign=0; 
      if(x == 0) {
	lastsign=0;
	return false;
      }
      int sign=sgn(x); 
      bool b=lastsign == 0 || sign == lastsign; 
      lastsign=sign; 
      return b; 
      })); 
axes("$x$","$y$",red);
