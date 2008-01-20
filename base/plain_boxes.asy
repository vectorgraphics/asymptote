// Draw and/or fill a box on frame dest using the dimensions of frame src.
path box(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
         pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  pair z=(xmargin,ymargin);
  int sign=filltype == NoFill ? 1 : -1;
  path g=box(min(src)+0.5*sign*min(p)-z,max(src)+0.5*sign*max(p)+z);
  frame F;
  if(put == Below) {
    filltype(F,g,p);
    prepend(dest,F);
  } else filltype(dest,g,p);
  return g;
}


path ellipse(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
             pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  pair m=min(src);
  pair M=max(src);
  pair D=M-m;
  static real factor=0.5*sqrt(2);
  int sign=filltype == NoFill ? 1 : -1;
  path g=ellipse(0.5*(M+m),factor*D.x+0.5*sign*max(p).x+xmargin,
                 factor*D.y+0.5*sign*max(p).y+ymargin);
  frame F;
  if(put == Below) {
    filltype(F,g,p);
    prepend(dest,F);
  } else filltype(dest,g,p);
  return g;
}

path box(frame f, Label L, real xmargin=0, real ymargin=xmargin,
         pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  add(f,L);
  return box(f,xmargin,ymargin,p,filltype,put);
}

path ellipse(frame f, Label L, real xmargin=0, real ymargin=xmargin,
             pen p=currentpen, filltype filltype=NoFill, bool put=Above)
{
  add(f,L);
  return ellipse(f,xmargin,ymargin,p,filltype,put);
}

typedef path envelope(frame dest, frame src=dest, real xmargin=0,
                      real ymargin=xmargin, pen p=currentpen,
                      filltype filltype=NoFill, bool put=Above);

object draw(picture pic=currentpicture, Label L, envelope e, 
	    real xmargin=0, real ymargin=xmargin, pen p=currentpen,
	    filltype filltype=NoFill, bool put=Above) 
{
  object F;
  Label L=L.copy();
  F.L=L;
  pic.add(new void (frame f, transform t) {
      frame d;
      add(d,t,L);
      e(f,d,xmargin,ymargin,p,filltype,put);
      add(f,d);
    },true);
  Label L0=L.copy();
  L0.position(0);
  L0.p(p);
  frame f;
  add(f,L0);
  F.g=e(f,xmargin,ymargin,p,filltype);
  pic.addBox(L.position,L.position,min(f),max(f));
  return F;
}

object draw(picture pic=currentpicture, Label L, envelope e, pair position,
            real xmargin=0, real ymargin=xmargin, pen p=currentpen,
            filltype filltype=NoFill, bool put=Above)
{
  return draw(pic,Label(L,position),e,xmargin,ymargin,p,filltype,put);
}

pair point(object F, pair dir, transform t=identity()) 
{
  pair m=min(F.g);
  pair M=max(F.g);
  pair c=0.5*(m+M);
  pair z=t*F.L.position;
  real[] T=intersect(F.g,c--2*(m+realmult(rectify(dir),M-m))-c);
  if(T.length == 0) return z;
  return z+point(F.g,T[0]);
}

frame bbox(picture pic=currentpicture, real xmargin=0, real ymargin=xmargin,
           pen p=currentpen, filltype filltype=NoFill)
{
  frame f=pic.fit(max(pic.xsize-2*xmargin,0),max(pic.ysize-2*ymargin,0));
  box(f,xmargin,ymargin,p,filltype,Below);
  return f;
}
