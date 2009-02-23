// Create a tiling named name from picture pic
// with optional left-bottom margin lb and right-top margin rt.
frame tiling(string name, picture pic, pair lb=0, pair rt=0)
{
  frame tiling;
  frame f=pic.fit(identity());
  pair pmin=min(f)-lb;
  pair pmax=max(f)+rt;
  string s="%.6f";
  postscript(tiling,"<< /PaintType 1 /PatternType 1 /TilingType 1 
/BBox ["+format(s,pmin.x,"C")+" "+format(s,pmin.y,"C")+" "+
             format(s,pmax.x,"C")+" "+format(s,pmax.y,"C")+"]
/XStep "+format(s,pmax.x-pmin.x,"C")+"
/YStep "+format(s,pmax.y-pmin.y,"C")+"
/PaintProc {pop");
  add(tiling,f);
  postscript(tiling,"} >>
 matrix makepattern
/"+name+" exch def");
  return tiling;
}

// Add to frame preamble a tiling name constructed from picture pic
// with optional left-bottom margin lb and right-top margin rt.
void add(string name, picture pic, pair lb=0, pair rt=0)
{
  add(currentpatterns,tiling(name,pic,lb,rt));
}

picture tile(real Hx=5mm, real Hy=0, pen p=currentpen,
             filltype filltype=NoFill)
{
  picture tiling;
  if(Hy == 0) Hy=Hx;
  path tile=box((0,0),(Hx,Hy));
  tiling.add(new void (frame f, transform t) {
      filltype.fill(f,t*tile,p);
    });
  clip(tiling,tile);
  return tiling;
}

picture checker(real Hx=5mm, real Hy=0, pen p=currentpen)
{
  picture tiling;
  if(Hy == 0) Hy=Hx;
  path tile=box((0,0),(Hx,Hy));
  fill(tiling,tile,p);
  fill(tiling,shift(Hx,Hy)*tile,p);
  clip(tiling,box((0,0),(2Hx,2Hy)));
  return tiling;
}

picture brick(real Hx=5mm, real Hy=0, pen p=currentpen)
{
  picture tiling;
  if(Hy == 0) Hy=Hx/2;
  path tile=box((0,0),(Hx,Hy));
  draw(tiling,tile,p);
  draw(tiling,(Hx/2,Hy)--(Hx/2,2Hy),p);
  draw(tiling,(0,2Hy)--(Hx,2Hy),p);
  clip(tiling,box((0,0),(Hx,2Hy)));
  return tiling;
}

real hatchepsilon=1e-4;
picture hatch(real H=5mm, pair dir=NE, pen p=currentpen) 
{
  picture tiling;
  real theta=angle(dir);
  real s=sin(theta);
  real c=cos(theta);
  if(abs(s) <= hatchepsilon) {
    path g=(0,0)--(H,0);
    draw(tiling,g,p);
    draw(tiling,shift(0,H)*g,p);
    clip(tiling,scale(H)*unitsquare);
  } else if(abs(c) <= hatchepsilon) {
    path g=(0,0)--(0,H);
    draw(tiling,g,p);
    draw(tiling,shift(H,0)*g,p);
    clip(tiling,scale(H)*unitsquare);
  } else {
    real h=H/s;
    real y=H/c;
    path g=(0,0)--(h,y);
    draw(tiling,g,p);
    draw(tiling,shift(-h/2,y/2)*g,p);
    draw(tiling,shift(h/2,-y/2)*g,p);
    clip(tiling,box((0,0),(h,y)));
  }
  return tiling;
}

picture crosshatch(real H=5mm, pen p=currentpen)
{
  picture tiling;
  add(tiling,hatch(H,p));
  add(tiling,shift(H*sqrt(2))*rotate(90)*hatch(H,p));
  return tiling;
}

