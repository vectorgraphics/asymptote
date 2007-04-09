real arrowlength=0.75cm;
real arrowfactor=15;
real arrowangle=15;
real arcarrowfactor=0.5*arrowfactor;
real arcarrowangle=2*arrowangle;
real arrowsizelimit=0.5;
real arrow2sizelimit=1/3;

real barfactor=arrowfactor;

real arrowsize(pen p=currentpen) 
{
  return arrowfactor*linewidth(p);
}

real arcarrowsize(pen p=currentpen)
{
  return arcarrowfactor*linewidth(p);
}

real barsize(pen p=currentpen)
{
  return barfactor*linewidth(p);
}

path arrowhead(path g, position position=EndPoint, pen p=currentpen,
               real size=0, real angle=arrowangle)
{
  if(size == 0) size=arrowsize(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);
  path r=subpath(g,position,0.0);
  pair x=point(r,0);
  real t=arctime(r,size);
  pair y=point(r,t);
  path base=y+2*size*I*dir(r,t)--y-2*size*I*dir(r,t);
  path left=rotate(-angle,x)*r, right=rotate(angle,x)*r;
  real[] tl=intersect(left,base), tr=intersect(right,base);
  pair denom=point(right,tr[0])-y;
  real factor=denom != 0 ? length((point(left,tl[0])-y)/denom) : 1;
  left=rotate(-angle,x)*r; right=rotate(angle*factor,x)*r;
  tl=intersect(left,base); tr=intersect(right,base);
  return subpath(left,0,tl.length > 0 ? tl[0] : 0)--
    subpath(right,tr.length > 0 ? tr[0] : 0,0)
    ..cycle;
}

void arrowheadbbox(picture pic=currentpicture, path g,
                   position position=EndPoint,
                   pen p=currentpen, real size=0,
                   real angle=arrowangle)
{
  // Estimate the bounding box contribution using the local slope at endpoint
  // and ignoring margin.
  if(size == 0) size=arrowsize(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);
  path r=subpath(g,position,0.0);
  pair x=point(r,0);
  pair y=point(r,arctime(r,size))-x;
  pair dz1=rotate(-angle)*y;
  pair dz2=rotate(angle)*y;
  pic.addPoint(x,p);
  pic.addPoint(x,dz1,p);
  pic.addPoint(x,dz2,p);
}

void arrow(frame f, path g, pen p=currentpen, real size=0,
           real angle=arrowangle, filltype filltype=FillDraw,
           position position=EndPoint, bool forwards=true,
           margin margin=NoMargin, bool center=false)
{
  if(size == 0) size=arrowsize(p);
  size=min(arrowsizelimit*arclength(g),size);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) {
    position *= arclength(g);
    if(center) position += 0.5*size;
    position=arctime(g,position);
  } else if(center) 
    position=arctime(g,arclength(subpath(g,0,position))+0.5*size);
  g=margin(g,p).g;
  int L=length(g);
  if(!forwards) {
    g=reverse(g);
    position=L-position;
  }
  path r=subpath(g,position,0.0);
  path s=subpath(g,position,L);

  size=min(arrowsizelimit*arclength(r),size);
  if(filltype == NoFill || position == L) {
    draw(f,subpath(r,arctime(r,size),length(r)),p);
    if(position < L) draw(f,s,p);
  } else draw(f,g,p);
  path head=arrowhead(g,position,p,size,angle);
  filltype(f,head,p+solid);
}

void arrow2(frame f, path g, pen p=currentpen, real size=0,
            real angle=arrowangle, filltype filltype=FillDraw,
            margin margin=NoMargin)
{
  if(size == 0) size=arrowsize(p);
  g=margin(g,p).g;
  size=min(arrow2sizelimit*arclength(g),size);
  path r=reverse(g);
  draw(f,subpath(r,arctime(r,size),length(r)-arctime(g,size)),p);
  path head=arrowhead(g,length(g),p,size,angle);
  path tail=arrowhead(r,length(r),p,size,angle);
  filltype(f,head,p+solid);
  filltype(f,tail,p+solid);
}

picture arrow(path g, pen p=currentpen, real size=0,
              real angle=arrowangle, filltype filltype=FillDraw,
              position position=EndPoint, bool forwards=true,
              margin margin=NoMargin, bool center=false)
{
  picture pic;
  pic.add(new void (frame f, transform t) {
      arrow(f,t*g,p,size,angle,filltype,position,forwards,margin,center);
    });
  
  pic.addPath(g,p);
  arrowheadbbox(pic,forwards ? g : reverse(g),position,p,size,angle);
  return pic;
}

picture arrow2(path g, pen p=currentpen, real size=0,
               real angle=arrowangle, filltype filltype=FillDraw,
               margin margin=NoMargin)
{
  picture pic;
  pic.add(new void (frame f, transform t) {
      arrow2(f,t*g,p,size,angle,filltype,margin);
    });
  
  pic.addPath(g,p);
  arrowheadbbox(pic,g,p,size,angle);
  arrowheadbbox(pic,reverse(g),p,size,angle);
  return pic;
}

void bar(picture pic, pair a, pair d, pen p=currentpen)
{
  picture opic;
  Draw(opic,-0.5d--0.5d,p+solid);
  add(pic,opic,a);
}
                                                      
picture bar(pair a, pair d, pen p=currentpen)
{
  picture pic;
  bar(pic,a,d,p);
  return pic;
}

typedef bool arrowbar(picture, path, pen, margin);

arrowbar Blank()
{
  return new bool(picture pic, path g, pen p, margin margin) {
    return false;
  };    
}

arrowbar None()
{
  return new bool(picture pic, path g, pen p, margin margin) {
    return true;
  };    
}

arrowbar BeginArrow(real size=0, real angle=arrowangle,
                    filltype filltype=FillDraw, position position=BeginPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(g,p,size,angle,filltype,position,false,margin));
    return false;
  };
}

arrowbar Arrow(real size=0, real angle=arrowangle,
               filltype filltype=FillDraw, position position=EndPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(g,p,size,angle,filltype,position,margin));
    return false;
  };
}

arrowbar EndArrow(real size=0, real angle=arrowangle,
                  filltype filltype=FillDraw, position position=EndPoint)=Arrow;

arrowbar MidArrow(real size=0, real angle=arrowangle, filltype filltype=FillDraw)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(g,p,size,angle,filltype,MidPoint,margin,true));
    return false;
  };
}
  
arrowbar Arrows(real size=0, real angle=arrowangle, filltype filltype=FillDraw)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow2(g,p,size,angle,filltype,margin));
    return false;
  };
}

arrowbar BeginArcArrow(real size=0, real angle=arcarrowangle,
                       filltype filltype=FillDraw, position position=BeginPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,position,false,margin));
    return false;
  };
}

arrowbar ArcArrow(real size=0, real angle=arcarrowangle,
                  filltype filltype=FillDraw, position position=EndPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,position,margin));
    return false;
  };
}

arrowbar EndArcArrow(real size=0, real angle=arcarrowangle,
                     filltype filltype=FillDraw,
                     position position=EndPoint)=ArcArrow;
  
arrowbar MidArcArrow(real size=0, real angle=arcarrowangle,
                     filltype filltype=FillDraw)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,MidPoint,margin,true));
    return false;
  };
}
  
arrowbar ArcArrows(real size=0, real angle=arcarrowangle,
                   filltype filltype=FillDraw)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow2(g,p,size,angle,filltype,margin));
    return false;
  };
}
  
arrowbar BeginBar(real size=0) 
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? barsize(p) : size;
    bar(pic,point(g,0),size*dir(g,0)*I,p);
    return true;
  };
}

arrowbar Bar(real size=0) 
{
  return new bool(picture pic, path g, pen p, margin margin) {
    int L=length(g);
    real size=size == 0 ? barsize(p) : size;
    bar(pic,point(g,L),size*dir(g,L)*I,p);
    return true;
  };
}

arrowbar EndBar(real size=0)=Bar; 

arrowbar Bars(real size=0) 
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? barsize(p) : size;
    BeginBar(size)(pic,g,p,margin);
    EndBar(size)(pic,g,p,margin);
    return true;
  };
}

arrowbar Blank=Blank(),
  None=None(),
  BeginArrow=BeginArrow(),
  MidArrow=MidArrow(),
  Arrow=Arrow(),
  EndArrow=Arrow(),
  Arrows=Arrows(),
  BeginArcArrow=BeginArcArrow(),
  MidArcArrow=MidArcArrow(),
  ArcArrow=ArcArrow(),
  EndArcArrow=ArcArrow(),
  ArcArrows=ArcArrows(),
  BeginBar=BeginBar(),
  Bar=Bar(),
  EndBar=Bar(),
  Bars=Bars();

void draw(frame f, path g, pen p=currentpen, arrowbar arrow)
{
  picture pic;
  if(arrow(pic,g,p,NoMargin)) draw(f,g,p);
  add(f,pic.fit());
}

void draw(picture pic=currentpicture, Label L="", path g, align align=NoAlign,
          pen p=currentpen, arrowbar arrow=None, arrowbar bar=None,
          margin margin=NoMargin, Label legend="", marker marker=nomarker)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  legend.p(p);
  if(marker != nomarker && !marker.put) marker.mark(pic,g);
  bool drawpath=arrow(pic,g,p,margin);
  if(bar(pic,g,p,margin) && drawpath) _draw(pic,g,p,margin);
  if(L.s != "") L.out(pic,g);
  if(legend.s != "") {
    Legend l; l.init(legend.s,legend.p,p,marker.f,marker.put);
    pic.legend.push(l);
  }
  if(marker != nomarker && marker.put) marker.mark(pic,g);
}

// Draw a fixed-size line about the user-coordinate 'origin'.
void draw(pair origin, picture pic=currentpicture, Label L="", path g,
          align align=NoAlign, pen p=currentpen, arrowbar arrow=None,
          arrowbar bar=None, margin margin=NoMargin, Label legend="",
          marker marker=nomarker)
{
  picture opic;
  draw(opic,L,g,align,p,arrow,bar,margin,legend,marker);
  add(pic,opic,origin);
}

void draw(picture pic=currentpicture, explicit path[] g, pen p=currentpen,
          Label legend="", marker marker=nomarker)
{ 
  for(int i=0; i < g.length-1; ++i) 
    draw(pic,g[i],p,marker);
  if(g.length > 0) draw(pic,g[g.length-1],p,legend,marker);
} 

void draw(pair origin, picture pic=currentpicture, explicit path[] g,
          pen p=currentpen, Label legend="", marker marker=nomarker)
{
  picture opic;
  draw(opic,g,p,legend,marker);
  add(pic,opic,origin);
}

// Align an arrow pointing to b from the direction dir. The arrow is
// 'length' PostScript units long.
void arrow(picture pic=currentpicture, Label L="", pair b, pair dir,
           real length=arrowlength, align align=NoAlign,
           pen p=currentpen, arrowbar arrow=Arrow, margin margin=EndMargin)
{
  Label L=L.copy();
  if(L.defaultposition) L.position(0);
  L.align(L.align,dir);
  L.p(p);
  marginT margin=margin(b--b,p); // Extract margin.begin and margin.end
  pair a=(margin.begin+length+margin.end)*unit(dir);
  draw(b,pic,L,a--(0,0),align,p,arrow,margin);
}
