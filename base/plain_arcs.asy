bool CCW=true;
bool CW=false;                                            

path circle(pair c, real r)
{
  return shift(c)*scale(r)*unitcircle;
}

path ellipse(pair c, real a, real b)
{
  return shift(c)*scale(a,b)*unitcircle;
}

// return an arc centered at c from pair z1 to z2 (assuming |z2-c|=|z1-c|),
// drawing in the given direction.
path arc(pair c, explicit pair z1, explicit pair z2, bool direction=CCW)
{
  z1 -= c;
  real r=abs(z1);
  z1=unit(z1);
  z2=unit(z2-c);

  real t1=intersect(unitcircle,(0,0)--2*z1)[0];
  real t2=intersect(unitcircle,(0,0)--2*z2)[0];
  static int n=length(unitcircle);
  if(direction) {
    if (t1 >= t2) t1 -= n;
  } else if(t2 >= t1) t2 -= n;
  return shift(c)*scale(r)*subpath(unitcircle,t1,t2);
}

// return an arc centered at c with radius r from angle1 to angle2 in degrees,
// drawing in the given direction.
path arc(pair c, real r, real angle1, real angle2, bool direction)
{
  return arc(c,c+r*dir(angle1),c+r*dir(angle2),direction);
}
  
// return an arc centered at c with radius r > 0 from angle1 to angle2 in
// degrees, drawing counterclockwise if angle2 >= angle1 (otherwise clockwise).
path arc(pair c, real r, real angle1, real angle2)
{
  return arc(c,r,angle1,angle2,angle2 >= angle1 ? CCW : CW);
}
