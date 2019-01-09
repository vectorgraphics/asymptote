// Draw and/or fill a box on frame dest using the dimensions of frame src.
path box(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
         pen p=currentpen, filltype filltype=NoFill, bool above=true)
{
  pair z=(xmargin,ymargin);
  int sign=filltype == NoFill ? 1 : -1;
  pair h=0.5*sign*(max(p)-min(p));
  path g=box(min(src)-h-z,max(src)+h+z);
  frame F;
  if(above == false) {
    filltype.fill(F,g,p);
    prepend(dest,F);
  } else filltype.fill(dest,g,p);
  return g;
}

path roundbox(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
	      pen p=currentpen, filltype filltype=NoFill, bool above=true)
{
  pair m=min(src);
  pair M=max(src);
  pair bound=M-m;
  int sign=filltype == NoFill ? 1 : -1;
  real a=bound.x+2*xmargin;
  real b=bound.y+2*ymargin;
  real ds=0;
  real dw=min(a,b)*0.3;
  path g=shift(m-(xmargin,ymargin))*((0,dw)--(0,b-dw){up}..{right}
  (dw,b)--(a-dw,b){right}..{down}
  (a,b-dw)--(a,dw){down}..{left}
  (a-dw,0)--(dw,0){left}..{up}cycle);
  
  frame F;
  if(above == false) {
    filltype.fill(F,g,p);
    prepend(dest,F);
  } else filltype.fill(dest,g,p);
  return g;
}

path ellipse(frame dest, frame src=dest, real xmargin=0, real ymargin=xmargin,
             pen p=currentpen, filltype filltype=NoFill, bool above=true)
{
  pair m=min(src);
  pair M=max(src);
  pair D=M-m;
  static real factor=0.5*sqrt(2);
  int sign=filltype == NoFill ? 1 : -1;
  pair h=0.5*sign*(max(p)-min(p));
  path g=ellipse(0.5*(M+m),factor*D.x+h.x+xmargin,factor*D.y+h.y+ymargin);
  frame F;
  if(above == false) {
    filltype.fill(F,g,p);
    prepend(dest,F);
  } else filltype.fill(dest,g,p);
  return g;
}

path box(frame f, Label L, real xmargin=0, real ymargin=xmargin,
         pen p=currentpen, filltype filltype=NoFill, bool above=true)
{
  add(f,L);
  return box(f,xmargin,ymargin,p,filltype,above);
}

path roundbox(frame f, Label L, real xmargin=0, real ymargin=xmargin,
	      pen p=currentpen, filltype filltype=NoFill, bool above=true)
{
  add(f,L);
  return roundbox(f,xmargin,ymargin,p,filltype,above);
}

path ellipse(frame f, Label L, real xmargin=0, real ymargin=xmargin,
             pen p=currentpen, filltype filltype=NoFill, bool above=true)
{
  add(f,L);
  return ellipse(f,xmargin,ymargin,p,filltype,above);
}

typedef path envelope(frame dest, frame src=dest, real xmargin=0,
                      real ymargin=xmargin, pen p=currentpen,
                      filltype filltype=NoFill, bool above=true);

object object(Label L, envelope e, real xmargin=0, real ymargin=xmargin,
	      pen p=currentpen, filltype filltype=NoFill, bool above=true) 
{
  object F;
  F.L=L.copy();
  Label L0=L.copy();
  L0.position(0);
  L0.p(p);
  add(F.f,L0);
  F.g=e(F.f,xmargin,ymargin,p,filltype);
  return F;
}

object draw(picture pic=currentpicture, Label L, envelope e, 
	    real xmargin=0, real ymargin=xmargin, pen p=currentpen,
	    filltype filltype=NoFill, bool above=true) 
{
  object F=object(L,e,xmargin,ymargin,p,filltype,above);
  pic.add(new void (frame f, transform t) {
      frame d;
      add(d,t,F.L);
      e(f,d,xmargin,ymargin,p,filltype,above);
      add(f,d);
    },true);
  pic.addBox(L.position,L.position,min(F.f),max(F.f));
  return F;
}

object draw(picture pic=currentpicture, Label L, envelope e, pair position,
            real xmargin=0, real ymargin=xmargin, pen p=currentpen,
            filltype filltype=NoFill, bool above=true)
{
  return draw(pic,Label(L,position),e,xmargin,ymargin,p,filltype,above);
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

frame bbox(picture pic=currentpicture,
           real xmargin=0, real ymargin=xmargin,
           pen p=currentpen, filltype filltype=NoFill)
{
  real penwidth=linewidth(p);
  frame f=pic.fit(max(pic.xsize-2*(xmargin+penwidth),0),
                  max(pic.ysize-2*(ymargin+penwidth),0));
  box(f,xmargin,ymargin,p,filltype,above=false);
  return f;
}
