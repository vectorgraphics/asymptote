static struct curve {
  public real a=0;
  public real b=8;

  real y2(real x) {
    return x^3+a*x+b;
  }

  real disc() {
    return -16*(4*a*a*a+27*b*b);
  }

  real lowx () {
    return sqrt(-a/3);
  }

  int comps() {
    if (a < 0) {
      real x=sqrt(-a/3);
      return y2(x) < 0 ? 2 : 1;
    }
    return 1;
  }

  void locus(picture pic=currentpicture, real m, real M, int n=100,
             pen p=currentpen) {
    path flip(path p, bool close) {
      path pp=reverse(yscale(-1)*p)..p;
      return close ? pp..cycle : pp;
    }
    path section(real m, real M, int n) {
      guide g;
      real width=(M-m)/n;
      for(int i=0; i <= n; ++i) {
        real x=m+width*i;
        real yy=y2(x);
        if (yy > 0)
          g=g..(x,sqrt(yy));
      }
      return g;
    }

    if (comps() == 1) {
      draw(pic,flip(section(m,M,n),false),p);
    }
    else {
      real x=lowx(); // The minimum on x^3+ax+b
      if (m < x)
        draw(pic,flip(section(m,min(x,M),n),true),p);
      if (x < M)
        draw(pic,flip(section(max(x,m),M,n),false),p);
    }
  }

  pair neg(pair P) {
    return finite(P.y) ? yscale(-1)*P : P;
  }

  pair add(pair P, pair Q) {
    if (P.x == Q.x && P.x != Q.x)
      return (0,infinity);
    else {
      real lambda=P == Q ? (3*P.x^2+a)/(2*P.y) : (Q.y-P.y)/(Q.x-P.x);
      real Rx=lambda^2-P.x-Q.x;
      return (Rx,(P.x-Rx)*lambda-P.y);
    }
  }
}

import graph;
import math;

size(0,200);

curve c=new curve; c.a=-1; c.b=4;

pair oncurve(real x) 
{
  return (x,sqrt(c.y2(x)));
}

picture output=new picture;

axes();
c.locus(-4,3,.3red+.7blue);

pair P=oncurve(-1),Q=oncurve(1.2);
pair PP=c.add(P,P),sum=c.add(P,Q);

save();

drawline(P,Q,dashed);
drawline(c.neg(sum),sum,dashed);
labeldot("$P$", P, NW);
labeldot("$Q$", Q, SE);
dot(c.neg(sum));
labeldot("$P+Q$", sum, SW);

add(output,currentpicture.fit(W));

restore();

save();

drawline(P,c.neg(PP),dashed);
drawline(c.neg(PP),PP,dashed);
labeldot("$P$", P, NW);
dot(c.neg(PP));
labeldot("$2P$", PP, SW);

add(output,currentpicture.fit(E));

shipout(output);
    
restore();
