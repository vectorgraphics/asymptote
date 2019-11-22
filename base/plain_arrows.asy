real arrowlength=0.75cm;
real arrowfactor=15;
real arrowangle=15;
real arcarrowfactor=0.5*arrowfactor;
real arcarrowangle=2*arrowangle;
real arrowsizelimit=0.5;
real arrow2sizelimit=1/3;
real arrowdir=5;
real arrowbarb=3;
real arrowhookfactor=1.5;
real arrowtexfactor=1;

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

struct arrowhead
{
  path head(path g, position position=EndPoint, pen p=currentpen,
            real size=0, real angle=arrowangle);
  real size(pen p)=arrowsize;
  real arcsize(pen p)=arcarrowsize;
  filltype defaultfilltype(pen) {return FillDraw;}
}

real[] arrowbasepoints(path base, path left, path right, real default=0)
{
  real[][] Tl=transpose(intersections(left,base));
  real[][] Tr=transpose(intersections(right,base));
  return new real[] {Tl.length > 0 ? Tl[0][0] : default,
      Tr.length > 0 ? Tr[0][0] : default};
}

path arrowbase(path r, pair y, real t, real size)
{
  pair perp=2*size*I*dir(r,t);
  return size == 0 ? y : y+perp--y-perp;
}

arrowhead DefaultHead;
DefaultHead.head=new path(path g, position position=EndPoint, pen p=currentpen,
                          real size=0, real angle=arrowangle) {
  if(size == 0) size=DefaultHead.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);
  path r=subpath(g,position,0);
  pair x=point(r,0);
  real t=arctime(r,size);
  pair y=point(r,t);
  path base=arrowbase(r,y,t,size);
  path left=rotate(-angle,x)*r;
  path right=rotate(angle,x)*r;
  real[] T=arrowbasepoints(base,left,right);
  pair denom=point(right,T[1])-y;
  real factor=denom != 0 ? length((point(left,T[0])-y)/denom) : 1;
  path left=rotate(-angle*factor,x)*r;
  path right=rotate(angle*factor,x)*r;
  real[] T=arrowbasepoints(base,left,right);
  return subpath(left,0,T[0])--subpath(right,T[1],0)&cycle;
};

arrowhead SimpleHead;
SimpleHead.head=new path(path g, position position=EndPoint, pen p=currentpen,
                         real size=0, real angle=arrowangle) {
  if(size == 0) size=SimpleHead.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);
  path r=subpath(g,position,0);
  pair x=point(r,0);
  real t=arctime(r,size);
  path left=rotate(-angle,x)*r;
  path right=rotate(angle,x)*r;
  return subpath(left,t,0)--subpath(right,0,t);
};

arrowhead HookHead(real dir=arrowdir, real barb=arrowbarb)
{
  arrowhead a;
  a.head=new path(path g, position position=EndPoint, pen p=currentpen,
                  real size=0, real angle=arrowangle)
    {
      if(size == 0) size=a.size(p);
      angle=min(angle*arrowhookfactor,45);
      bool relative=position.relative;
      real position=position.position.x;
      if(relative) position=reltime(g,position);
      path r=subpath(g,position,0);
      pair x=point(r,0);
      real t=arctime(r,size);
      pair y=point(r,t);
      path base=arrowbase(r,y,t,size);
      path left=rotate(-angle,x)*r;
      path right=rotate(angle,x)*r;
      real[] T=arrowbasepoints(base,left,right,1);
      pair denom=point(right,T[1])-y;
      real factor=denom != 0 ? length((point(left,T[0])-y)/denom) : 1;
      path left=rotate(-angle*factor,x)*r;
      path right=rotate(angle*factor,x)*r;
      real[] T=arrowbasepoints(base,left,right,1);
      left=subpath(left,0,T[0]);
      right=subpath(right,T[1],0);
      pair pl0=point(left,0), pl1=relpoint(left,1);
      pair pr0=relpoint(right,0), pr1=relpoint(right,1);
      pair M=(pl1+pr0)/2;
      pair v=barb*unit(M-pl0);
      pl1=pl1+v; pr0=pr0+v;
      left=pl0{dir(-dir+degrees(M-pl0,false))}..pl1--M;
      right=M--pr0..pr1{dir(dir+degrees(pr1-M,false))};
      return left--right&cycle;
    };
  return a;
}
arrowhead HookHead=HookHead();

arrowhead TeXHead;
TeXHead.size=new real(pen p)
{
  static real hcoef=2.1; // 84/40=abs(base-hint)/base_height
  return hcoef*arrowtexfactor*linewidth(p);
};
TeXHead.arcsize=TeXHead.size;

TeXHead.head=new path(path g, position position=EndPoint, pen p=currentpen,
                      real size=0, real angle=arrowangle) {
  static real wcoef=1/84; // 1/abs(base-hint)
  static path texhead=scale(wcoef)*
  ((0,20)     .. controls (-75,75)    and (-108,158) ..
   (-108,166) .. controls (-108,175)  and (-100,178) ..
   (-93,178)  .. controls (-82,178)   and (-80,173)  ..
   (-77,168)  .. controls (-62,134)   and (-30,61)   ..
   (70,14)    .. controls (82,8)      and (84,7)     ..
   (84,0)     .. controls (84,-7)     and (82,-8)    ..
   (70,-14)   .. controls (-30,-61)   and (-62,-134) ..
   (-77,-168) .. controls (-80,-173)  and (-82,-178) ..
   (-93,-178) .. controls (-100,-178) and (-108,-175)..
   (-108,-166).. controls (-108,-158) and (-75,-75)  ..
   (0,-20)--cycle);
  if(size == 0) size=TeXHead.size(p);
  path gp=scale(size)*texhead;
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);
  path r=subpath(g,position,0);
  pair y=point(r,arctime(r,size));
  return shift(y)*rotate(degrees(-dir(r,arctime(r,0.5*size))))*gp;
};
TeXHead.defaultfilltype=new filltype(pen p) {return Fill(p);};

private real position(position position, real size, path g, bool center)
{
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) {
    position *= arclength(g);
    if(center) position += 0.5*size;
    position=arctime(g,position);
  } else if(center) 
    position=arctime(g,arclength(subpath(g,0,position))+0.5*size);
  return position;
}

void drawarrow(frame f, arrowhead arrowhead=DefaultHead,
               path g, pen p=currentpen, real size=0,
               real angle=arrowangle,
               filltype filltype=null,
               position position=EndPoint, bool forwards=true,
               margin margin=NoMargin, bool center=false)
{
  if(size == 0) size=arrowhead.size(p);
  if(filltype == null) filltype=arrowhead.defaultfilltype(p);
  size=min(arrowsizelimit*arclength(g),size);
  real position=position(position,size,g,center);

  g=margin(g,p).g;
  int L=length(g);
  if(!forwards) {
    g=reverse(g);
    position=L-position;
  }
  path r=subpath(g,position,0);
  size=min(arrowsizelimit*arclength(r),size);
  path head=arrowhead.head(g,position,p,size,angle);
  bool endpoint=position > L-sqrtEpsilon;
  if(cyclic(head) && (filltype == NoFill || endpoint)) {
    if(position > 0)
      draw(f,subpath(r,arctime(r,size),length(r)),p);
    if(!endpoint)
      draw(f,subpath(g,position,L),p);
  } else draw(f,g,p);
  filltype.fill(f,head,p+solid);
}

void drawarrow2(frame f, arrowhead arrowhead=DefaultHead,
                path g, pen p=currentpen, real size=0,
                real angle=arrowangle, filltype filltype=null,
                margin margin=NoMargin)
{
  if(size == 0) size=arrowhead.size(p);
  if(filltype == null) filltype=arrowhead.defaultfilltype(p);
  g=margin(g,p).g;
  size=min(arrow2sizelimit*arclength(g),size);

  path r=reverse(g);
  int L=length(g);
  path head=arrowhead.head(g,L,p,size,angle);
  path tail=arrowhead.head(r,L,p,size,angle);
  if(cyclic(head))
    draw(f,subpath(r,arctime(r,size),L-arctime(g,size)),p);
  else draw(f,g,p);
  filltype.fill(f,head,p+solid);
  filltype.fill(f,tail,p+solid);
}

// Add to picture an estimate of the bounding box contribution of arrowhead
// using the local slope at endpoint and ignoring margin.
void addArrow(picture pic, arrowhead arrowhead, path g, pen p, real size,
              real angle, filltype filltype, real position)
{
  if(filltype == null) filltype=arrowhead.defaultfilltype(p);
  pair z=point(g,position);
  path g=z-(size+linewidth(p))*dir(g,position)--z;
  frame f;
  filltype.fill(f,arrowhead.head(g,position,p,size,angle),p);
  pic.addBox(z,z,min(f)-z,max(f)-z);
}

picture arrow(arrowhead arrowhead=DefaultHead,
              path g, pen p=currentpen, real size=0,
              real angle=arrowangle, filltype filltype=null,
              position position=EndPoint, bool forwards=true,
              margin margin=NoMargin, bool center=false)
{
  if(size == 0) size=arrowhead.size(p);
  picture pic;
  pic.add(new void(frame f, transform t) {
      drawarrow(f,arrowhead,t*g,p,size,angle,filltype,position,forwards,margin,
                center);
    });
  
  pic.addPath(g,p);

  real position=position(position,size,g,center);
  path G;
  if(!forwards) {
    G=reverse(g);
    position=length(g)-position;
  } else G=g;
  addArrow(pic,arrowhead,G,p,size,angle,filltype,position);

  return pic;
}

picture arrow2(arrowhead arrowhead=DefaultHead,
               path g, pen p=currentpen, real size=0,
               real angle=arrowangle, filltype filltype=null,
               margin margin=NoMargin)
{
  if(size == 0) size=arrowhead.size(p);
  picture pic;
  pic.add(new void(frame f, transform t) {
      drawarrow2(f,arrowhead,t*g,p,size,angle,filltype,margin);
    });
  
  pic.addPath(g,p);

  int L=length(g);
  addArrow(pic,arrowhead,g,p,size,angle,filltype,L);
  addArrow(pic,arrowhead,reverse(g),p,size,angle,filltype,L);

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

bool Blank(picture, path, pen, margin)
{
  return false;
}

bool None(picture, path, pen, margin)
{
  return true;
}

arrowbar BeginArrow(arrowhead arrowhead=DefaultHead,
                    real size=0, real angle=arrowangle,
                    filltype filltype=null, position position=BeginPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(arrowhead,g,p,size,angle,filltype,position,forwards=false,
                  margin));
    return false;
  };
}

arrowbar Arrow(arrowhead arrowhead=DefaultHead,
               real size=0, real angle=arrowangle,
               filltype filltype=null, position position=EndPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(arrowhead,g,p,size,angle,filltype,position,margin));
    return false;
  };
}

arrowbar EndArrow(arrowhead arrowhead=DefaultHead,
                  real size=0, real angle=arrowangle,
                  filltype filltype=null, position position=EndPoint)=Arrow;

arrowbar MidArrow(arrowhead arrowhead=DefaultHead,
                  real size=0, real angle=arrowangle, filltype filltype=null)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(arrowhead,g,p,size,angle,filltype,MidPoint,margin,
                  center=true));
    return false;
  };
}
  
arrowbar Arrows(arrowhead arrowhead=DefaultHead,
                real size=0, real angle=arrowangle,
                filltype filltype=null)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow2(arrowhead,g,p,size,angle,filltype,margin));
    return false;
  };
}

arrowbar BeginArcArrow(arrowhead arrowhead=DefaultHead,
                       real size=0, real angle=arcarrowangle,
                       filltype filltype=null, position position=BeginPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arrowhead.arcsize(p) : size;
    add(pic,arrow(arrowhead,g,p,size,angle,filltype,position,
                  forwards=false,margin));
    return false;
  };
}

arrowbar ArcArrow(arrowhead arrowhead=DefaultHead,
                  real size=0, real angle=arcarrowangle,
                  filltype filltype=null, position position=EndPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arrowhead.arcsize(p) : size;
    add(pic,arrow(arrowhead,g,p,size,angle,filltype,position,margin));
    return false;
  };
}

arrowbar EndArcArrow(arrowhead arrowhead=DefaultHead,
                     real size=0, real angle=arcarrowangle,
                     filltype filltype=null,
                     position position=EndPoint)=ArcArrow;
  
arrowbar MidArcArrow(arrowhead arrowhead=DefaultHead,
                     real size=0, real angle=arcarrowangle,
                     filltype filltype=null)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arrowhead.arcsize(p) : size;
    add(pic,arrow(arrowhead,g,p,size,angle,filltype,MidPoint,margin,
                  center=true));
    return false;
  };
}
  
arrowbar ArcArrows(arrowhead arrowhead=DefaultHead,
                   real size=0, real angle=arcarrowangle,
                   filltype filltype=null)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arrowhead.arcsize(p) : size;
    add(pic,arrow2(arrowhead,g,p,size,angle,filltype,margin));
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

arrowbar BeginArrow=BeginArrow(),
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
  if(arrow(pic,g,p,NoMargin))
    draw(f,g,p);
  add(f,pic.fit());
}

void draw(picture pic=currentpicture, Label L=null, path g,
          align align=NoAlign, pen p=currentpen, arrowbar arrow=None,
          arrowbar bar=None, margin margin=NoMargin, Label legend=null,
          marker marker=nomarker)
{
  // These if statements are ordered in such a way that the most common case
  // (with just a path and a pen) executes the least bytecode.
  if (marker == nomarker)
  {
    if (arrow == None && bar == None)
    {
      if (margin == NoMargin && size(nib(p)) == 0)
      {
        pic.addExactAbove(
            new void(frame f, transform t, transform T, pair, pair) {
              _draw(f,t*T*g,p);
            });
        pic.addPath(g,p);

        // Jumping over else clauses takes time, so test if we can return
        // here.
        if (L == null && legend == null)
          return;
      }
      else // With margin or polygonal pen.
      {
        _draw(pic, g, p, margin);
      }
    }
    else /* arrow or bar */
    {
      // Note we are using & instead of && as both arrow and bar need to be
      // called.
      if (arrow(pic, g, p, margin) & bar(pic, g, p, margin))
        _draw(pic, g, p, margin);
    }

    if(L != null && L.s != "") {
      L=L.copy();
      L.align(align);
      L.p(p);
      L.out(pic,g);
    }

    if(legend != null && legend.s != "") {
      legend.p(p);
      pic.legend.push(Legend(legend.s,legend.p,p,marker.f,marker.above));
    }
  }
  else /* marker != nomarker */
  {
    if(marker != nomarker && !marker.above) marker.mark(pic,g);

    // Note we are using & instead of && as both arrow and bar need to be
    // called.
    if ((arrow == None || arrow(pic, g, p, margin)) &
        (bar == None || bar(pic, g, p, margin)))
      {
        _draw(pic, g, p, margin);
      }

    if(L != null && L.s != "") {
      L=L.copy();
      L.align(align);
      L.p(p);
      L.out(pic,g);
    }

    if(legend != null && legend.s != "") {
      legend.p(p);
      pic.legend.push(Legend(legend.s,legend.p,p,marker.f,marker.above));
    }

    if(marker != nomarker && marker.above) marker.mark(pic,g);
  }
}

// Draw a fixed-size line about the user-coordinate 'origin'.
void draw(pair origin, picture pic=currentpicture, Label L=null, path g,
          align align=NoAlign, pen p=currentpen, arrowbar arrow=None,
          arrowbar bar=None, margin margin=NoMargin, Label legend=null,
          marker marker=nomarker)
{
  picture opic;
  draw(opic,L,g,align,p,arrow,bar,margin,legend,marker);
  add(pic,opic,origin);
}

void draw(picture pic=currentpicture, explicit path[] g, pen p=currentpen,
          Label legend=null, marker marker=nomarker)
{ 
  // This could be optimized to size and draw the entire array as a batch.
  for(int i=0; i < g.length-1; ++i) 
    draw(pic,g[i],p,marker);
  if(g.length > 0) draw(pic,g[g.length-1],p,legend,marker);
} 

void draw(picture pic=currentpicture, guide[] g, pen p=currentpen,
          Label legend=null, marker marker=nomarker)
{
  draw(pic,(path[]) g,p,legend,marker);
}

void draw(pair origin, picture pic=currentpicture, explicit path[] g,
          pen p=currentpen, Label legend=null, marker marker=nomarker)
{
  picture opic;
  draw(opic,g,p,legend,marker);
  add(pic,opic,origin);
}

void draw(pair origin, picture pic=currentpicture, guide[] g, pen p=currentpen,
          Label legend=null, marker marker=nomarker)
{
  draw(origin,pic,(path[]) g,p,legend,marker);
}

// Align an arrow pointing to b from the direction dir. The arrow is
// 'length' PostScript units long.
void arrow(picture pic=currentpicture, Label L=null, pair b, pair dir,
           real length=arrowlength, align align=NoAlign,
           pen p=currentpen, arrowbar arrow=Arrow, margin margin=EndMargin)
{
  if(L != null && L.s != "") {
    L=L.copy();
    if(L.defaultposition) L.position(0);
    L.align(L.align,dir);
    L.p(p);
  }
  marginT margin=margin(b--b,p); // Extract margin.begin and margin.end
  pair a=(margin.begin+length+margin.end)*unit(dir);
  draw(b,pic,L,a--(0,0),align,p,arrow,margin);
}

// Fit an array of pictures simultaneously using the sizing of picture all.
frame[] fit2(picture[] pictures, picture all)
{
  frame[] out;
  if(!all.empty2()) {
    transform t=all.calculateTransform();
    pair m=all.min(t);
    pair M=all.max(t);
    for(picture pic : pictures) {
      frame f=pic.fit(t);
      draw(f,m,nullpen);
      draw(f,M,nullpen);
      out.push(f);
    }
  }
  return out;
}

// Fit an array of pictures simultaneously using the size of the first picture.
// TODO: Remove unused arguments.
frame[] fit(string prefix="", picture[] pictures, string format="",
            bool view=true, string options="", string script="",
            projection P=currentprojection)
{
  if(pictures.length == 0)
    return new frame[];
 
  picture all;
  size(all,pictures[0]);
  for(picture pic : pictures)
    add(all,pic);

  return fit2(pictures,all);
}

// Pad a picture to a specified size
frame pad(picture pic=currentpicture, real xsize=pic.xsize,
          real ysize=pic.ysize, filltype filltype=NoFill)
{
  picture P;
  size(P,xsize,ysize,IgnoreAspect);
  draw(P,(0,0),invisible+thin());
  draw(P,(xsize,ysize),invisible+thin());
  add(P,pic.fit(xsize,ysize),(xsize,ysize)/2);
  frame f=P.fit();
  if(filltype != NoFill) {
    frame F;
    filltype.fill(F,box(min(f),max(f)),invisible);
    prepend(f,F);
  }
  return f;
}
