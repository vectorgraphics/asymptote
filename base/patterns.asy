// Create a tiling named name from picture pic
// with optional left-bottom margin lb and right-top margin rt.
frame tiling(string name, picture pic, pair lb=0, pair rt=0)
{
  frame tiling;
  frame f=pic.fit(identity());
  pair pmin=min(f)-lb;
  pair pmax=max(f)+rt;
  postscript(tiling,"<< /PaintType 1 /PatternType 1 /TilingType 1 
/BBox ["+format(pmin.x)+" "+format(pmin.y)+" "+format(pmax.x)+" "
	     +format(pmax.y)+"]
/XStep "+format(pmax.x-pmin.x)+"
/YStep "+format(pmax.y-pmin.y)+"
/PaintProc {pop");
  add(tiling,f);
  postscript(tiling,"} >>
 matrix makepattern
/"+name+" exch def");
  return tiling;
}

public real hatchepsilon=1e-4;
picture hatch(real H=5mm, pair dir=NE) 
{
  picture tiling=new picture;
  real theta=angle(dir);
  real s=sin(theta);
  real c=cos(theta);
  if(abs(s) <= hatchepsilon) {
    path p=(0,0)--(H,0);
    draw(tiling,p);
    draw(tiling,shift(0,H)*p);
    clip(tiling,scale(H)*unitsquare);
  } else if(abs(c) <= hatchepsilon) {
    path p=(0,0)--(0,H);
    draw(tiling,p);
    draw(tiling,shift(H,0)*p);
    clip(tiling,scale(H)*unitsquare);
  } else {
    real h=H/s;
    real y=H/c;
    path p=(0,0)--(h,y);
    draw(tiling,p);
    draw(tiling,shift(-h/2,y/2)*p);
    draw(tiling,shift(h/2,-y/2)*p);
    clip(tiling,box((0,0),(h,y)));
  }
  return tiling;
}

picture crosshatch(real H=5mm)
{
  picture tiling=new picture;
  add(tiling,hatch(H));
  add(tiling,shift(H*sqrt(2))*rotate(90)*hatch(H));
  return tiling;
}

picture tile(real Hx=5mm, real Hy=0)
{
  picture tiling=new picture;
  if(Hy == 0) Hy=Hx;
  guide tile=box((0,0),(Hx,Hy));
  draw(tiling,tile);
  clip(tiling,tile);
  return tiling;
}

picture brick(real Hx=5mm, real Hy=0)
{
  picture tiling=new picture;
  if(Hy == 0) Hy=Hx/2;
  guide tile=box((0,0),(Hx,Hy));
  draw(tiling,tile);
  draw(tiling,(Hx/2,Hy)--(Hx/2,2Hy));
  draw(tiling,(0,2Hy)--(Hx,2Hy));
  clip(tiling,box((0,0),(Hx,2Hy)));
  return tiling;
}

// Add to frame preamble a tiling name constructed from picture pic
// with optional left-bottom margin lb and right-top margin rt.
void add(frame preamble=patterns, string name, picture pic, pair lb=0,
	 pair rt=0)
{
  add(preamble,tiling(name,pic,lb,rt));
}
