// Mark routines and markers written by Philippe Ivaldi.

marker operator * (transform T, marker m)
{
  marker M=new marker;
  M.f=T*m.f;
  M.put=m.put;
  M.markroutine=m.markroutine;
  return M;
}

// Like the function markuniform(int n, bool rotated=false) but add
// the frame center between each pair of mark.
markroutine markuniform(int n, frame center, bool rotated=false) {
  return new void(picture pic=currentpicture, frame f, path g) {
    markuniform(n,rotated)(pic,f,g);
    markuniform(centered=true,n-1,rotated)(pic,center,g);
  };
}

// Return a frame containing n copies of the path g shifted by space
// drawn with pen p.
frame duplicate(path g, int n=1, pair space=0, pen p=currentpen)
{
  if(space == 0) space=dotsize(p);
  frame f;
  int pos=0;
  int sign=1;
  int m=(n+1) % 2;
  for(int i=1; i <= n; ++i) {
    draw(f,shift(space*(pos-0.5*m))*g,p);
    pos += i*sign;
    sign *= -1;
  }
  return f;
}

real tildemarksize=6mm;
real tildemarksize(pen p=currentpen) {return tildemarksize+linewidth(p);};
frame tildeframe(int n=1, real size=0, pair space=0,
                 real angle=0, pair offset=0, pen p=currentpen)
{
  size=(size == 0) ? tildemarksize(p) : size;
  space=(space == 0) ? 0.5size : space;
  path g=(-0.5,-1/6)..(-1.5/6,1/6)..(0,0)..(1.5/6,-1/6)..(0.5,1/6);
  return duplicate(shift(offset)*rotate(angle)*scale(size)*g,n,space,p);
}

frame tildeframe=tildeframe();

// A dot in a frame.
frame dotframe(pen p=currentpen) {
  frame f;
  dot(f,(0,0),p);
  return f;
}

frame dotframe=dotframe();

real stickmarkspacefactor=4;
real stickmarksizefactor=10;
real stickmarksize(pen p=currentpen)
{
  return 1mm+stickmarksizefactor*sqrt(linewidth(p));
}
real stickmarkspace(pen p=currentpen)
{
  return stickmarkspacefactor*sqrt(linewidth(p));
}
frame stickframe(int n=1, real size=0, pair space=0, real angle=0,
                 pair offset=0, pen p=currentpen)
{
  if(size == 0) size=stickmarksize(p);
  if(space == 0) space=stickmarkspace(p);
  return duplicate(shift(offset)*rotate(angle)*scale(0.5*size)*(N--S),n,
                   space,p);
}

frame stickframe=stickframe();

real circlemarkradius=dotsize(currentpen);
real barmarksize=4*dotsize(currentpen);
frame circlebarframe(int n=1, real barsize=0,
                     real radius=0,real angle=0,
                     pair offset=0, pen p=currentpen,
                     filltype filltype=NoFill, bool above=false)
{
  if(barsize == 0) barsize=barmarksize+2*linewidth(p);
  if(radius == 0) radius=circlemarkradius+linewidth(p);
  frame opic;
  guide g=circle(offset,radius);
  frame f=stickframe(n,barsize,space=2*radius/(n+1),angle,offset,p);
  if(above) {
    add(opic,f);
    filltype(opic,g,p);
  } else {
    filltype(opic,g,p);
    add(opic,f);
  }
  return opic;
}

real crossmarksizefactor=5;
real crossmarksize(pen p=currentpen)
{
  return crossmarksizefactor*sqrt(linewidth(p));
}
frame crossframe(int n=3, real size=0, pair space=0,
                 real angle=0, pair offset=0, pen p=currentpen)
{
  if(size == 0) size=crossmarksize(p)+2*linewidth(p);
  frame opic;
  draw(opic,shift(offset)*rotate(angle)*scale(size)*cross(n),p);
  return opic;
}

real markanglespacefactor=4;
real markangleradiusfactor=8;
real markangleradius(pen p=currentpen)
{
  return 8mm+markangleradiusfactor*sqrt(linewidth(p));
}
real markangleradius=markangleradius();
real markanglespace(pen p=currentpen)
{
  return markanglespacefactor*sqrt(linewidth(p));
}
real markanglespace=markanglespace();
// Mark the angle AOB with optional Label, arrows, and markers.
void markangle(picture pic=currentpicture, Label L="",
               int n=1, real radius=0, real space=0,
               pair A, pair O, pair B,
               arrowbar arrow=None,
               pen p=currentpen,
               margin margin=NoMargin,
               marker marker=nomarker)
{
  if(space == 0) space=markanglespace(p);
  if(radius == 0) radius=markangleradius(p);
  picture lpic,phantom;
  path lpth;
  p=squarecap+p;
  real xob=degrees(B-O,false);
  real xoa=degrees(A-O,false);
  if(abs(xob-xoa)>180) radius=-radius;
  bool drawarrow = !arrow(phantom,arc((0,0),radius,xoa,xob),p,margin);
  if(drawarrow && margin == NoMargin) margin=TrueMargin(0,0.5linewidth(p));
  for(int i=0; i < n; ++i) {
    lpth=margin(arc((0,0),radius+sgn(radius)*i*space,xoa,xob),p).g;
    draw(lpic,lpth,p=p,arrow=arrow,margin=NoMargin,marker=marker);
  }
  Label lL=L.copy();
  real position=lL.position.position.x;
  if(lL.defaultposition) {lL.position.relative=true; position=0.5;}
  if(lL.position.relative) position=reltime(lpth,position);
  if(lL.align.default) {
    lL.align.relative=true;
    lL.align.dir=unit(point(lpth,position));
  }
  label(lpic,lL,point(lpth,position),align=NoAlign, p=p);
  add(pic,lpic,O);
}
