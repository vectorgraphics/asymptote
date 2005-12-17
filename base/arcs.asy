bool CCW=true;
bool CW=false;						  

guide unitcircle=E..N..W..S..cycle;

public real circleprecision=0.0006;

guide circle(pair c, real r)
{
  return shift(c)*scale(r)*unitcircle;
}

guide ellipse(pair c, real a, real b)
{
  return shift(c)*xscale(a)*yscale(b)*unitcircle;
}

// return an arc centered at c with radius r from angle1 to angle2 in degrees,
// drawing in the given direction.
guide arc(pair c, real r, real angle1, real angle2, bool direction)
{
  real t1=intersect(unitcircle,(0,0)--2*dir(angle1)).x;
  real t2=intersect(unitcircle,(0,0)--2*dir(angle2)).x;
  static int n=length(unitcircle);
  if(t1 >= t2 && direction) t1 -= n;
  if(t2 >= t1 && !direction) t2 -= n;
  return shift(c)*scale(r)*subpath(unitcircle,t1,t2);
}
  
// return an arc centered at c with radius r > 0 from angle1 to angle2 in
// degrees, drawing counterclockwise if angle2 >= angle1 (otherwise clockwise).
// If r < 0, draw the complementary arc of radius |r|.
guide arc(pair c, real r, real angle1, real angle2)
{
  bool pos=angle2 >= angle1;
  if(r > 0) return arc(c,r,angle1,angle2,pos ? CCW : CW);
  else return arc(c,-r,angle1,angle2,pos ? CW : CCW);
}

// return an arc centered at c from pair z1 to z2 (assuming |z2-c|=|z1-c|),
// drawing in the given direction.
guide arc(pair c, explicit pair z1, explicit pair z2, bool direction=CCW)
{
  return arc(c,abs(z1-c),degrees(z1-c),degrees(z2-c),direction);
}

guide ellipse(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
	      pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  pair m=min(src);
  pair M=max(src);
  pair D=M-m;
  static real factor=0.5*sqrt(2);
  int sign=filltype == NoFill ? 1 : -1;
  guide g=ellipse(0.5*(M+m),factor*D.x+0.5*sign*max(p).x+xmargin,
		  factor*D.y+0.5*sign*max(p).y+ymargin);
  frame F;
  if(put == Below) {
    filltype(F,g,p);
    prepend(dest,F);
  } else filltype(dest,g,p);
  return g;
}

guide ellipse(frame f, Label L, real xmargin=0, real ymargin=xmargin,
	      pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  add(f,L);
  return ellipse(f,xmargin,ymargin,p,filltype,put);
}

