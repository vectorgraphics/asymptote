// Mark routines and markers written by Philippe Ivaldi.

marker operator *(transform T, marker m)
{
  marker M=new marker;
  M.f=T*m.f;
  M.put=m.put;
  M.markroutine=m.markroutine;
  return M;
}

// On picture pic, add to path g the frame f, evenly spaced in arclength without these ends.
// If rotated=true, the frame will be rotated by the angle of the tangent
// to the path at the local points.
markroutine markuniforminterval(int n, bool rotated=false) {
  return new void(picture pic=currentpicture, path g, frame f) {
    if(n <= 0) return;
    void add(real x) {
      real t=reltime(g,x);
      add(pic,rotated ? rotate(degrees(dir(g,t)))*f : f, point(g,t));
    }
    real width=1/n;
    for(int i=0; i < n; ++i) add((i+0.5)*width);
  };
}

// Like the function markuniform(int n, bool rotated=false) but add
// the frame 'finterval' between each mark.
markroutine markuniform(int n, bool rotated=false,
                        frame finterval) {
  return new void(picture pic=currentpicture, path g, frame f) {
    markuniform(n,rotated)(pic,g,f);
    markuniforminterval(n-1,rotated)(pic,g,finterval);
  };
}

// Duplicate 'n' times the path 'g' by spacing it of 'space', rotating by
// 'angle' around 0. if the voffset is 0, the position of origin
// (0.0) relative with 'g' remains the same one.
picture duplicate(path g, int n=1,
                  real space=dotsize(currentpen),
                  real angle=0, real voffset=0, pen p=currentpen)
{
  picture opic;
  int pos=0;
  for(int i=1; i <= n; ++i)
    {
      draw (opic,shift((space*(pos-0.5*((n+1)%2)),voffset))*rotate(angle)*g,p);
      pos+=i*(-1)^(i+1);
    }
  return opic;
}

real tildemarksize=6mm;
real tildemarksize(pen p=currentpen){return tildemarksize+linewidth(p);};
frame tildeframe(int n=1, real size=0, real space=0,
                real angle=0, real voffset=0, pen p=currentpen)
{
  size=(size == 0) ? tildemarksize(p) : size;
  space=(space == 0) ? size/2 : space; 
  return duplicate(size*(-1/2,-1/6)..size*(-1.5/6,1/6)..(0,0)..size*(1.5/6,-1/6)..size*(1/2,1/6),n,space,angle,voffset,p).fit();
}

frame tildeframe=tildeframe();

// A dot in a frame.
frame dotframe(pen p=currentpen){
  frame f;
  dot(f,(0,0),p);
  return f;
}

frame dotframe=dotframe();

real stickmarkspacefactor=4;
real stickmarksizefactor=10;
real stickmarksize(pen p=currentpen){return 1mm+stickmarksizefactor*sqrt(linewidth(p));}
real stickmarkspace(pen p=currentpen){return stickmarkspacefactor*sqrt(linewidth(p));}
frame stickframe(int n=1, real size=0, real space=0,
                 real angle=0, real voffset=0, pen p=currentpen)
{
  if(size == 0) size=stickmarksize(p);
  if(space == 0) space=stickmarkspace(p);
  return duplicate(size*N/2--size*S/2,n,space,angle,voffset,p).fit();
}

frame stickframe=stickframe();

real circlemarkradius=dotsize(currentpen);
real barmarksize=4*dotsize(currentpen);
frame circlebarframe(int n=1, real barsize=0,
                     real radius=0,real angle=0,
                     real voffset=0, pen p=currentpen,
                     filltype filltype=NoFill, bool above=false)
{
  if(barsize == 0) barsize=barmarksize+2*linewidth(p);
  if(radius == 0) radius=circlemarkradius+linewidth(p);
  frame opic;
  if (above)
    {
      add(opic,stickframe(n,barsize,space=2*radius/(n+1),angle,voffset,p));
      filltype(opic,shift(0,voffset)*scale(radius)*unitcircle,p);
    }
  else
    {
      filltype(opic,shift(0,voffset)*scale(radius)*unitcircle,p);
      add(opic,stickframe(n,barsize,space=2*radius/(n+1),angle,voffset,p));
    }
  return opic;
}

real crossmarksizefactor=5;
real crossmarksize(pen p=currentpen){return crossmarksizefactor*sqrt(linewidth(p));}
frame crossframe(int n, real size=0, real space=0,
                 real angle=0, real voffset=0, pen p=currentpen)
{
  if(size == 0) size=crossmarksize(p)+2*linewidth(p);
  frame opic;
  draw (opic,shift((0,voffset))*rotate(angle)*scale(size)*cross(n),p);
  return opic;
}

// Mark uniformly with 'markuniform' and between each marks with 'markinterval'
marker markersuniform(int n=2,frame markuniform, frame markinterval,
                      bool rotated=true,bool put=Above)
{
  return marker(markuniform, markuniform(n, rotated, markinterval), put);
}

real markanglespacefactor=4;
real markangleradiusfactor=8;
real markangleradius(pen p=currentpen){return 8mm+markangleradiusfactor*sqrt(linewidth(p));}
real markangleradius=markangleradius();
real markanglespace(pen p=currentpen){return markanglespacefactor*sqrt(linewidth(p));}
real markanglespace=markanglespace();
// Mark the angle (OA,OB) with optionals markers (as frame), arrows and label.
void markangle(picture pic=currentpicture,
	       Label L="",
	       int n=1, real radius=0, real space=0,
	       pair O=0, pair A, pair B,
	       arrowbar arrow=None,
	       pen p=currentpen,
	       margin margin=NoMargin,
	       frame markerframe=newframe)
{
  if(space == 0) space=markanglespace(p);
  if(radius == 0) radius=markangleradius(p);
  marker marker=(markerframe == newframe) ? nomarker:
    marker(markerframe,markuniform(1,true,markerframe));
  picture lpic,phantom;
  path lpth;
  p=squarecap+p;
  real xob=degrees(B-O,false);
  real xoa=degrees(A-O,false);
  if (abs(xob-xoa)>180) radius=-radius;
  bool drawarrow = !arrow(phantom,arc((0,0), radius, xoa, xob),p,margin);
  if (drawarrow && margin == NoMargin) margin=TrueMargin(0,linewidth(p)/2);
  for (int i=0; i < n; ++i)
    {
      lpth=margin(arc((0,0), radius+sgn(radius)*i*space, xoa, xob),p).g;
      draw(lpic,lpth, p=p, arrow=arrow, margin=NoMargin, marker=marker);
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
