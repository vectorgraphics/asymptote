// A transformation that bends points along a path
transform3 bend(path3 g, real t)
{
  triple dir=dir(g,t);
  triple a=point(g,0), b=postcontrol(g,0);
  triple c=precontrol(g,1), d=point(g,1);
  triple dir1=b-a;
  triple dir2=c-b;
  triple dir3=d-c;

  triple u = unit(cross(dir1,dir3));
  real eps=1000*realEpsilon;
  if(abs(u) < eps) {
    u = unit(cross(dir1,dir2));
    if(abs(u) < eps) {
      u = unit(cross(dir2,dir3));
      if(abs(u) < eps)
        // linear segment: use any direction perpendicular to initial direction
        u = perp(dir1);
    }
  }
  u = unit(perp(u,dir));

  triple w=cross(dir,u);
  triple q=point(g,t);
  return new real[][] {
    {u.x,w.x,dir.x,q.x},
      {u.y,w.y,dir.y,q.y},
        {u.z,w.z,dir.z,q.z},
          {0,0,0,1}
  };
}

// bend a point along a path; assumes that p.z is in [0,scale]
triple bend(triple p, path3 g, real scale)
{
  return bend(g,arctime(g,arclength(g)+p.z-scale))*(p.x,p.y,0);
}

void bend(surface s, path3 g, real L) 
{
  for(patch p : s.s) {
    for(int i=0; i < 4; ++i) {
      for(int j=0; j < 4; ++j) {
        p.P[i][j]=bend(p.P[i][j],g,L);
      }
    }
  }
}

// Refine a noncyclic path3 g so that it approaches its endpoint in
// geometrically spaced steps.
path3 approach(path3 p, int n, real radix=3)
{
  guide3 G;
  real L=length(p);
  real tlast=0;
  real r=1/radix;
  for(int i=1; i < n; ++i) {
    real t=L*(1-r^i);
    G=G&subpath(p,tlast,t);
    tlast=t;
  }
  return G&subpath(p,tlast,L);
}

struct arrowhead3
{
  arrowhead arrowhead2=DefaultHead;
  real size(pen p)=arrowsize;
  real arcsize(pen p)=arcarrowsize;
  real gap=1;
  real size;
  bool splitpath=false;

  surface head(path3 g, position position=EndPoint,
               pen p=currentpen, real size=0, real angle=arrowangle,
               filltype filltype=null, bool forwards=true,
               projection P=currentprojection);

  static surface surface(path3 g, position position, real size,
                         path[] h, pen p, filltype filltype,
                         triple normal, projection P) {
    bool relative=position.relative;
    real position=position.position.x;
    if(relative) position=reltime(g,position);
    path3 r=subpath(g,position,0);
    path3 s=subpath(r,arctime(r,size),0);
    if(filltype == null) filltype=FillDraw(p);
    bool draw=filltype.type != filltype.Fill;
    triple v=point(s,length(s));
    triple N=normal == O ? P.normal : normal;
    triple w=unit(v-point(s,0));
    transform3 t=transform3(w,unit(cross(w,N)));
    path3[] H=t*path3(h);
    surface s;
    real width=linewidth(p);
    if(filltype != NoFill && filltype.type != filltype.UnFill &&
       filltype.type != filltype.Draw) {
      triple n=0.5*width*unit(t*Z);
      s=surface(shift(n)*H,planar=true);
      s.append(surface(shift(-n)*H,planar=true));
      if(!draw)
        for(path g : h)
          s.append(shift(-n)*t*extrude(g,width*Z));
    }
    if(draw)
      for(path3 g : H) {
        tube T=tube(g,width);
        for(surface S : T.s)
          s.append(S);
      }
    return shift(v)*s;
  }

  static path project(path3 g, bool forwards, projection P) {
    path h=project(forwards ? g : reverse(g),P);
    return shift(-point(h,length(h)))*h;
  }

  static path[] align(path H, path h) {
    static real fuzz=1000*realEpsilon;
    real[][] t=intersections(H,h,fuzz*max(abs(max(h)),abs(min(h))));
    return t.length >= 2 ?
      rotate(-degrees(point(H,t[0][0])-point(H,t[1][0]),warn=false))*H : H;
  }
}

arrowhead3 DefaultHead3;
DefaultHead3.head=new surface(path3 g, position position=EndPoint,
                              pen p=currentpen, real size=0,
                              real angle=arrowangle, filltype filltype=null,
                              bool forwards=true,
                              projection P=currentprojection)
{
  if(size == 0) size=DefaultHead3.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  path3 s=subpath(r,arctime(r,size),0);
  int n=length(s);      
  bool straight1=n == 1 && straight(g,0);
  real aspect=Tan(angle);
  real width=size*aspect;
  surface head;
  if(straight1) {
    triple v=point(s,0);
    triple u=point(s,1)-v;
    return shift(v)*align(unit(u))*scale(width,width,size)*unitsolidcone;
  } else {
    real remainL=size;
    bool first=true;
    for(int i=0; i < n; ++i) {
      path3 q=subpath(s,i,i+1);
      if(remainL > 0) {
        real l=arclength(q);
        real w=remainL*aspect;
        surface segment=scale(w,w,l)*unitcylinder;
        if(first) { // add base
          first=false;
          segment.append(scale(w,w,1)*unitdisk);
        }
        for(patch p : segment.s) {
          for(int i=0; i < 4; ++i) {
            for(int j=0; j < 4; ++j) {
              real k=1-p.P[i][j].z/remainL;
              p.P[i][j]=bend((k*p.P[i][j].x,k*p.P[i][j].y,p.P[i][j].z),q,l);
            }
          }
        }
        head.append(segment);
        remainL -= l;
      }
    }
  }
  return head;
};

arrowhead3 HookHead3(real dir=arrowdir, real barb=arrowbarb)
{
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
                     pen p=currentpen, real size=0, real angle=arrowangle,
                     filltype filltype=null, bool forwards=true,
                     projection P=currentprojection) {
    if(size == 0) size=a.size(p);
    
    bool relative=position.relative;
    real position=position.position.x;
    if(relative) position=reltime(g,position);

    path3 r=subpath(g,position,0);
    path3 s=subpath(r,arctime(r,size),0);
    bool straight1=length(s) == 1 && straight(g,0);
    path3 H=path3(HookHead(dir,barb).head((0,0)--(0,size),p,size,angle),
                  YZplane);
    surface head=surface(O,reverse(approach(subpath(H,1,0),7,1.5))&
                         approach(subpath(H,1,2),4,2),Z);
  
    if(straight1) {
      triple v=point(s,0);
      triple u=point(s,1)-v;
      return shift(v)*align(unit(u))*head;
    } else {
      bend(head,s,size);
      return head;
    }
  };
  a.arrowhead2=HookHead;
  a.gap=0.7;
  return a;
}
arrowhead3 HookHead3=HookHead3();

arrowhead3 TeXHead3;
TeXHead3.size=TeXHead.size;
TeXHead3.arcsize=TeXHead.size;
TeXHead3.arrowhead2=TeXHead;
TeXHead3.head=new surface(path3 g, position position=EndPoint,
                          pen p=currentpen, real size=0, real angle=arrowangle,
                          filltype filltype=null, bool forwards=true,
                          projection P=currentprojection)
{
  real texsize=TeXHead3.size(p);
  if(size == 0) size=texsize;
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  path3 s=subpath(r,arctime(r,size),0);
  bool straight1=length(s) == 1 && straight(g,0);

  surface head=surface(O,approach(subpath(path3(TeXHead.head((0,0)--(0,1),p,
                                                             size),
                                                YZplane),5,0),8,1.5),Z);
  if(straight1) {
    triple v=point(s,0);
    triple u=point(s,1)-v;
    return shift(v)*align(unit(u))*head;
  } else {
    path3 s=subpath(r,arctime(r,size/texsize*arrowsize(p)),0);
    bend(head,s,size);
    return head;
  }
};

path3 arrowbase(path3 r, triple y, real t, real size)
{
  triple perp=2*size*perp(dir(r,t));
  return size == 0 ? y : y+perp--y-perp;
}

arrowhead3 DefaultHead2(triple normal=O) {
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
                     pen p=currentpen, real size=0,
                     real angle=arrowangle,
                     filltype filltype=null, bool forwards=true,
                     projection P=currentprojection) {
    if(size == 0) size=a.size(p);
    path h=a.project(g,forwards,P);
    a.size=min(size,arclength(h));
    path[] H=a.align(DefaultHead.head(h,p,size,angle),h);
    H=forwards ? yscale(-1)*H : H;
    return a.surface(g,position,size,H,p,filltype,normal,P);
  };
  a.gap=1.005;
  return a;
}
arrowhead3 DefaultHead2=DefaultHead2();

arrowhead3 HookHead2(real dir=arrowdir, real barb=arrowbarb, triple normal=O)
{
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
                     pen p=currentpen, real size=0, real angle=arrowangle,
                     filltype filltype=null, bool forwards=true,
                     projection P=currentprojection) {
    if(size == 0) size=a.size(p);
    path h=a.project(g,forwards,P);
    a.size=min(size,arclength(h));
    path[] H=a.align(HookHead.head(h,p,size,angle),h);
    H=forwards ? yscale(-1)*H : H;
    return a.surface(g,position,size,H,p,filltype,normal,P);
  };
  a.arrowhead2=HookHead;
  a.gap=1.005;
  a.splitpath=true;
  return a;
}
arrowhead3 HookHead2=HookHead2();

arrowhead3 TeXHead2(triple normal=O) {
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
                     pen p=currentpen, real size=0,
                     real angle=arrowangle, filltype filltype=null,
                     bool forwards=true, projection P=currentprojection) {
    if(size == 0) size=a.size(p);
    path h=a.project(g,forwards,P);
    a.size=min(size,arclength(h));
    h=rotate(-degrees(dir(h,length(h)),warn=false))*h;
    path[] H=TeXHead.head(h,p,size,angle);
    H=forwards ? yscale(-1)*H : H;
    return a.surface(g,position,size,H,p,
                     filltype == null ? TeXHead.defaultfilltype(p) : filltype,
                     normal,P);
  };
  a.arrowhead2=TeXHead;
  a.size=TeXHead.size;
  a.gap=1.005;
  return a;
}
arrowhead3 TeXHead2=TeXHead2();

private real position(position position, real size, path3 g, bool center)
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

void drawarrow(picture pic, arrowhead3 arrowhead=DefaultHead3,
               path3 g, material p=currentpen, material arrowheadpen=nullpen,
               real size=0, real angle=arrowangle, position position=EndPoint,
               filltype filltype=null, bool forwards=true,
               margin3 margin=NoMargin3, bool center=false, light light=nolight,
               light arrowheadlight=currentlight,
               projection P=currentprojection)
{
  pen q=(pen) p;
  if(filltype != null) {
    if(arrowheadpen == nullpen && filltype != null)
      arrowheadpen=filltype.fillpen;
    if(arrowheadpen == nullpen && filltype != null)
      arrowheadpen=filltype.drawpen;
  }
  if(arrowheadpen == nullpen) arrowheadpen=p;
  if(size == 0) size=arrowhead.size(q);
  size=min(arrowsizelimit*arclength(g),size);
  real position=position(position,size,g,center);

  g=margin(g,q).g;
  int L=length(g);
  if(!forwards) {
    g=reverse(g);
    position=L-position;
  }
  path3 r=subpath(g,position,0);
  size=min(arrowsizelimit*arclength(r),size);
  surface head=arrowhead.head(g,position,q,size,angle,filltype,forwards,P);
  if(arrowhead.size > 0) size=arrowhead.size;
  bool endpoint=position > L-sqrtEpsilon;
  if(arrowhead.splitpath || endpoint) {
    if(position > 0) {
      real Size=size*arrowhead.gap;
      draw(pic,subpath(r,arctime(r,Size),length(r)),p,light);
    }
    if(!endpoint)
      draw(pic,subpath(g,position,L),p,light);
  } else draw(pic,g,p,light);
  draw(pic,head,arrowheadpen,arrowheadlight);
}

void drawarrow2(picture pic, arrowhead3 arrowhead=DefaultHead3,
                path3 g, material p=currentpen, material arrowheadpen=nullpen,
                real size=0, real angle=arrowangle, filltype filltype=null,
                margin3 margin=NoMargin3, light light=nolight,
                light arrowheadlight=currentlight,
                projection P=currentprojection)
{
  pen q=(pen) p;
  if(filltype != null) {
    if(arrowheadpen == nullpen && filltype != null)
      arrowheadpen=filltype.fillpen;
    if(arrowheadpen == nullpen && filltype != null)
      arrowheadpen=filltype.drawpen;
  }
  if(arrowheadpen == nullpen) arrowheadpen=p;
  if(size == 0) size=arrowhead.size(q);
  g=margin(g,q).g;
  size=min(arrow2sizelimit*arclength(g),size);

  path3 r=reverse(g);
  int L=length(g);
  real Size=size*arrowhead.gap;
  draw(pic,subpath(r,arctime(r,Size),L-arctime(g,Size)),p,light);
  draw(pic,arrowhead.head(g,L,q,size,angle,filltype,forwards=true,P),
       arrowheadpen,arrowheadlight);
  draw(pic,arrowhead.head(r,L,q,size,angle,filltype,forwards=false,P),
       arrowheadpen,arrowheadlight);
}

// Add to picture an estimate of the bounding box contribution of arrowhead
// using the local slope at endpoint.
void addArrow(picture pic, arrowhead3 arrowhead, path3 g, pen p, real size,
              real angle, filltype filltype, real position)
{
  triple v=point(g,position);
  path3 g=v-(size+linewidth(p))*dir(g,position)--v;
  surface s=arrowhead.head(g,position,p,size,angle);
  if(s.s.length > 0) {
    pic.addPoint(v,min(s)-v);
    pic.addPoint(v,max(s)-v);
  } else pic.addPoint(v);
}

picture arrow(arrowhead3 arrowhead=DefaultHead3,
              path3 g, material p=currentpen, material arrowheadpen=p,
              real size=0, real angle=arrowangle,
              filltype filltype=null, position position=EndPoint,
              bool forwards=true, margin3 margin=NoMargin3,
              bool center=false, light light=nolight,
              light arrowheadlight=currentlight)
{
  pen q=(pen) p;
  if(size == 0) size=arrowhead.size(q);
  picture pic;
  if(is3D())
    pic.add(new void(frame f, transform3 t, picture pic2, projection P) {
        picture opic;
        drawarrow(opic,arrowhead,t*g,p,arrowheadpen,size,angle,position,
                  filltype,forwards,margin,center,light,arrowheadlight,P);
        add(f,opic.fit3(identity4,pic2,P));
      });

  addPath(pic,g,q);

  real position=position(position,size,g,center);
  path3 G;
  if(!forwards) {
    G=reverse(g);
    position=length(g)-position;
  } else G=g;
  addArrow(pic,arrowhead,G,q,size,angle,filltype,position);

  return pic;
}

picture arrow2(arrowhead3 arrowhead=DefaultHead3,
               path3 g, material p=currentpen, material arrowheadpen=p,
               real size=0, real angle=arrowangle, filltype filltype=null,
               margin3 margin=NoMargin3, light light=nolight,
               light arrowheadlight=currentlight) 
{
  pen q=(pen) p;
  if(size == 0) size=arrowhead.size(q);
  picture pic;

  if(is3D())
    pic.add(new void(frame f, transform3 t, picture pic2, projection P) {
        picture opic;
        drawarrow2(opic,arrowhead,t*g,p,arrowheadpen,size,angle,filltype,
                   margin,light,arrowheadlight,P);
        add(f,opic.fit3(identity4,pic2,P));
      });

  addPath(pic,g,q);

  int L=length(g);
  addArrow(pic,arrowhead,g,q,size,angle,filltype,L);
  addArrow(pic,arrowhead,reverse(g),q,size,angle,filltype,L);

  return pic;
}

void add(picture pic, arrowhead3 arrowhead, real size, real angle,
         filltype filltype, position position, material arrowheadpen,
         path3 g, material p, bool forwards=true, margin3 margin,
         bool center=false, light light, light arrowheadlight)
{
  add(pic,arrow(arrowhead,g,p,arrowheadpen,size,angle,filltype,position,
                forwards,margin,center,light,arrowheadlight));
  if(!is3D()) {
    pic.add(new void(frame f, transform3 t, picture pic, projection P) {
        if(pic != null) {
          pen q=(pen) p;
          path3 G=t*g;
          marginT3 m=margin(G,q);
          add(pic,arrow(arrowhead.arrowhead2,project(G,P),q,size,angle,
                        filltype == null ?
                        arrowhead.arrowhead2.defaultfilltype
                        ((pen) arrowheadpen) : filltype,position,
                        forwards,TrueMargin(m.begin,m.end),center));
        }
      },true);
  }
}

void add2(picture pic, arrowhead3 arrowhead, real size, real angle,
          filltype filltype, material arrowheadpen, path3 g, material p,
          margin3 margin, light light, light arrowheadlight)
{
  add(pic,arrow2(arrowhead,g,p,arrowheadpen,size,angle,filltype,margin,light,
                 arrowheadlight));
  if(!is3D()) {
    pic.add(new void(frame f, transform3 t, picture pic, projection P) {
        if(pic != null) {
          pen q=(pen) p;
          path3 G=t*g;
          marginT3 m=margin(G,q);
          add(pic,arrow2(arrowhead.arrowhead2,project(G,P),q,size,angle,
                         filltype == null ?
                         arrowhead.arrowhead2.defaultfilltype
                         ((pen) arrowheadpen) : filltype,
                         TrueMargin(m.begin,m.end)));
        }
      },true);
  }
}

void bar(picture pic, triple a, triple d, triple perp=O,
         material p=currentpen, light light=nolight)
{
  d *= 0.5;
  perp *= 0.5;
  pic.add(new void(frame f, transform3 t, picture pic2, projection P) {
      picture opic;
      triple A=t*a;
      triple v=d == O ? abs(perp)*unit(cross(P.normal,perp)) : d;
      draw(opic,A-v--A+v,p,light);
      add(f,opic.fit3(identity4,pic2,P));
    });
  triple v=d == O ? cross(currentprojection.normal,perp) : d;
  pen q=(pen) p;
  triple m=min3(q);
  triple M=max3(q);
  pic.addPoint(a,-v-m);
  pic.addPoint(a,-v+m);
  pic.addPoint(a,v-M);
  pic.addPoint(a,v+M);
}
                                                      
picture bar(triple a, triple dir, triple perp=O, material p=currentpen)
{
  picture pic;
  bar(pic,a,dir,perp,p);
  return pic;
}

typedef bool arrowbar3(picture, path3, material, margin3, light, light);

bool Blank(picture, path3, material, margin3, light, light)
{
  return false;
}

bool None(picture, path3, material, margin3, light, light)
{
  return true;
}

arrowbar3 BeginArrow3(arrowhead3 arrowhead=DefaultHead3,
                      real size=0, real angle=arrowangle,
                      filltype filltype=null, position position=BeginPoint,
                      material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    add(pic,arrowhead,size,angle,filltype,position,arrowheadpen,g,p,
        forwards=false,margin,light,arrowheadlight);
    return false;
  };
}

arrowbar3 Arrow3(arrowhead3 arrowhead=DefaultHead3,
                 real size=0, real angle=arrowangle,
                 filltype filltype=null, position position=EndPoint,
                 material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    add(pic,arrowhead,size,angle,filltype,position,arrowheadpen,g,p,margin,
        light,arrowheadlight);
    return false;
  };
}

arrowbar3 EndArrow3(arrowhead3 arrowhead=DefaultHead3,
                    real size=0, real angle=arrowangle,
                    filltype filltype=null, position position=EndPoint,
                    material arrowheadpen=nullpen)=Arrow3;

arrowbar3 MidArrow3(arrowhead3 arrowhead=DefaultHead3,
                    real size=0, real angle=arrowangle,
                    filltype filltype=null, material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    add(pic,arrowhead,size,angle,filltype,MidPoint,
        arrowheadpen,g,p,margin,center=true,light,arrowheadlight);
    return false;
  };
}

arrowbar3 Arrows3(arrowhead3 arrowhead=DefaultHead3,
                  real size=0, real angle=arrowangle,
                  filltype filltype=null, material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    add2(pic,arrowhead,size,angle,filltype,arrowheadpen,g,p,margin,light,
         arrowheadlight);
    return false;
  };
}

arrowbar3 BeginArcArrow3(arrowhead3 arrowhead=DefaultHead3,
                         real size=0, real angle=arcarrowangle,
                         filltype filltype=null, position position=BeginPoint,
                         material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) { 
    real size=size == 0 ? arrowhead.arcsize((pen) p) : size;
    add(pic,arrowhead,size,angle,filltype,position,arrowheadpen,g,p,
        forwards=false,margin,light,arrowheadlight);
    return false; 
  };
}

arrowbar3 ArcArrow3(arrowhead3 arrowhead=DefaultHead3,
                    real size=0, real angle=arcarrowangle,
                    filltype filltype=null, position position=EndPoint,
                    material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    real size=size == 0 ? arrowhead.arcsize((pen) p) : size;
    add(pic,arrowhead,size,angle,filltype,position,arrowheadpen,g,p,margin,
        light,arrowheadlight);
    return false;
  };
}

arrowbar3 EndArcArrow3(arrowhead3 arrowhead=DefaultHead3,
                       real size=0, real angle=arcarrowangle,
                       filltype filltype=null,
                       position position=EndPoint,
                       material arrowheadpen=nullpen)=ArcArrow3;


arrowbar3 MidArcArrow3(arrowhead3 arrowhead=DefaultHead3,
                       real size=0, real angle=arcarrowangle,
                       filltype filltype=null, material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    real size=size == 0 ? arrowhead.arcsize((pen) p) : size;
    add(pic,arrowhead,size,angle,filltype,MidPoint,arrowheadpen,g,p,margin,
        center=true,light,arrowheadlight);
    return false;
  };
}

arrowbar3 ArcArrows3(arrowhead3 arrowhead=DefaultHead3,
                     real size=0, real angle=arcarrowangle,
                     filltype filltype=null, material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
                  light light, light arrowheadlight) {
    real size=size == 0 ? arrowhead.arcsize((pen) p) : size;
    add2(pic,arrowhead,size,angle,filltype,arrowheadpen,g,p,margin,light,
         arrowheadlight);
    return false;
  };
}

arrowbar3 BeginBar3(real size=0, triple dir=O)
{
  return new bool(picture pic, path3 g, material p, margin3 margin, light light,
                  light) {
    real size=size == 0 ? barsize((pen) p) : size;
    bar(pic,point(g,0),size*unit(dir),size*dir(g,0),p,light);
    return true;
  };
}

arrowbar3 Bar3(real size=0, triple dir=O) 
{
  return new bool(picture pic, path3 g, material p, margin3 margin, light light,
                  light) {
    int L=length(g);
    real size=size == 0 ? barsize((pen) p) : size;
    bar(pic,point(g,L),size*unit(dir),size*dir(g,L),p,light);
    return true;
  };
}

arrowbar3 EndBar3(real size=0, triple dir=O)=Bar3; 

arrowbar3 Bars3(real size=0, triple dir=O) 
{
  return new bool(picture pic, path3 g, material p, margin3 margin, light light,
                  light) {
    real size=size == 0 ? barsize((pen) p) : size;
    BeginBar3(size,dir)(pic,g,p,margin,light,nolight);
    EndBar3(size,dir)(pic,g,p,margin,light,nolight);
    return true;
  };
}

arrowbar3 BeginArrow3=BeginArrow3(),
MidArrow3=MidArrow3(),
Arrow3=Arrow3(),
EndArrow3=Arrow3(),
Arrows3=Arrows3(),
BeginArcArrow3=BeginArcArrow3(),
MidArcArrow3=MidArcArrow3(),
ArcArrow3=ArcArrow3(),
EndArcArrow3=ArcArrow3(),
ArcArrows3=ArcArrows3(),
BeginBar3=BeginBar3(),
Bar3=Bar3(),
EndBar3=Bar3(),
Bars3=Bars3();
