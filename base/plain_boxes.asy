// Draw and/or fill a box on frame dest using the dimensions of frame src.
guide box(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
          pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  pair z=(xmargin,ymargin);
  int sign=filltype == NoFill ? 1 : -1;
  guide g=box(min(src)+0.5*sign*min(p)-z,max(src)+0.5*sign*max(p)+z);
  frame F;
  if(put == Below) {
    filltype(F,g,p);
    prepend(dest,F);
  } else filltype(dest,g,p);
  return g;
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

guide box(frame f, Label L, real xmargin=0, real ymargin=xmargin,
          pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  add(f,L);
  return box(f,xmargin,ymargin,p,filltype,put);
}

guide ellipse(frame f, Label L, real xmargin=0, real ymargin=xmargin,
              pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  add(f,L);
  return ellipse(f,xmargin,ymargin,p,filltype,put);
}

typedef guide envelope(frame dest, frame src=dest, real xmargin=0,
		       real ymargin=xmargin, pen p=currentpen,
		       filltype filltype=NoFill, bool put=Above);

frame enclose(picture pic=currentpicture, envelope e, Label L,
	      real xmargin=0, real ymargin=xmargin, pen p=currentpen,
	      filltype filltype=NoFill, bool put=Above) 
{
  Label L=L.copy();
  pic.add(new void (frame f, transform t) {
      frame d;
      add(d,t,L);
      e(f,d,xmargin,ymargin,p,filltype,put);
      add(f,d);
    });
  Label L0=L.copy();
  L0.position(0);
  L0.p(p);
  frame f;
  add(f,L0);
  e(f,xmargin,ymargin,p,filltype);
  pic.addBox(L.position,L.position,min(f),max(f));
  return f;
}

object draw(picture pic=currentpicture, Label L, envelope S,
	    real xmargin=0, real ymargin=xmargin, pen p=currentpen,
	    filltype filltype=NoFill, bool put=Above)
{
  object F;
  F.L=L.copy();
  F.f=enclose(pic,S,L,xmargin,ymargin,p,filltype,put);
  return F;
}

object draw(picture pic=currentpicture, Label L, envelope S, pair position,
	    real xmargin=0, real ymargin=xmargin, pen p=currentpen,
	    filltype filltype=NoFill, bool put=Above)
{
  return draw(pic,Label(L,position),S,xmargin,ymargin,p,filltype,put);
}

pair point(object F, pair dir, transform t=identity()) 
{
  return t*F.L.position+point(F.f,dir);
}

void box(picture pic=currentpicture, Label L,
         real xmargin=0, real ymargin=xmargin, pen p=currentpen,
         filltype filltype=NoFill, bool put=Above)
{
  enclose(pic,box,L,xmargin,ymargin,p,filltype,put);
}

frame bbox(picture pic=currentpicture, real xmargin=0, real ymargin=xmargin,
           pen p=currentpen, filltype filltype=NoFill)
{
  frame f=pic.fit(max(pic.xsize-2*xmargin,0),max(pic.ysize-2*ymargin,0));
  box(f,xmargin,ymargin,p,filltype,Below);
  return f;
}

