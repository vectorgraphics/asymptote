// transformation that bends points along a path
// assumes that p.z is in [0,scale]
triple bend0(triple p, path3 g, real time)
{
  triple dir=dir(g,time);
  triple dir2=unit(cross(dir(g,0),dir(g,1)));
  if(abs(dir2) < 1000*realEpsilon) {
    if(abs(dir-X) < 1000*realEpsilon || abs(dir+X) < 1000*realEpsilon)
      dir2=Y;
    else dir2=X;
  }
  dir2=unit(dir2-dot(dir2,dir)*dir);

  triple w=cross(dir,dir2);
  triple q=point(g,time);
  transform3 t=new real[][] {
    {dir2.x,w.x,dir.x,0},
    {dir2.y,w.y,dir.y,0},
    {dir2.z,w.z,dir.z,0},
    {0,0,0,1}
  };
  return shift(q-t*(0,0,p.z))*t*p;
}

triple bend(triple p, path3 g, real scale, real endtime)
{
  return bend0(p,g,arctime(g,arclength(subpath(g,0,endtime))+p.z-scale));
}

triple bend(triple p, path3 g, real scale)
{
  return bend0(p,g,arctime(g,arclength(g)+p.z-scale));
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

real takeStep(path3 s, real T, real width)
{
  int L=length(s);
  path3 si=subpath(s,T,L);
  // Find minimum radius over [0,t]
  real minradius(real t) {
    real R=infinity;
    int nsamples=8;
    real step=t/nsamples;
    for(int i=0; i < nsamples; ++i) {
      real r=radius(si,i*step);
      if(r >= 0) R=min(R,r);
    }
    return R;
  }
  int niterations=8;
  static real K=0.08;
  real R;
  real lastt;
  real t=0;
  for(int i=0; i < niterations; ++ i) {
    R=minradius(t);
    t=0.5*(t+arctime(si,K*(width+R)));
  }
  // Compute new interval
  real t=arctime(s,arclength(subpath(s,0,T))+K*(width+minradius(t)));
  if(t < L) t=min(t,0.5*(T+L));
  return t;
}

surface tube(path3 g, real width)
{
  surface tube;
  real r=0.5*width;

  transform3 t=scale3(r);

  static real epsilon=sqrt(realEpsilon);

  int n=length(g);
  for(int i=0; i < n; ++i) {
    real S=straightness(g,i);
    if(S < epsilon*r) {
      triple v=point(g,i);
      triple u=point(g,i+1)-v;
      tube.append(shift(v)*align(unit(u))*scale(r,r,abs(u))*unitcylinder);
    } else {
      path3 s=subpath(g,i,i+1);
      real endtime=0;
      while(endtime < 1) {
        real newend=takeStep(s,endtime,r);
        path3 si=subpath(s,endtime,newend);
        real L=arclength(si);
        surface segment=scale(r,r,L)*unitcylinder;
	bend(segment,si,L);
        tube.s.append(segment.s);
        endtime=newend;
      }
    }

    if((cyclic(g) || i > 0) && abs(dir(g,i,1)-dir(g,i,-1)) > epsilon)
      tube.append(shift(point(g,i))*t*align(dir(g,i,-1))*unithemisphere);
  }
  return tube;
}

struct arrowhead3
{
  surface head(path3 g, position position=EndPoint, pen p=currentpen,
               real size=0, real angle=arrowangle,
	       projection P=currentprojection);

  static surface surface(triple v, path3 s, path[] h, real size, pen p,
			 filltype filltype, projection P) {
    path3[] H=path3(rotate(degrees(dir(v,dir(s,arctime(s,0.5*size)),P),
				  warn=false))*h);
    return shift(v)*transform3(P)*(filltype != NoFill ?
				   surface(H) : tube(H[0],linewidth(p)));
  }

  arrowhead arrowhead2=DefaultHead;
  filltype filltype=Fill;
  real size(pen p)=arrowsize;
  real gap=1;
  bool lighting=true;
}

arrowhead3 DefaultHead3;

DefaultHead3.head=new surface(path3 g, position position=EndPoint,
			      pen p=currentpen, real size=0,
			      real angle=arrowangle,
			      projection P=currentprojection)
{
  if(size == 0) size=DefaultHead3.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  real t=arctime(r,size);
  path3 s=subpath(r,t,0);
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
    for(int i=0; i < n; ++i) {
      path3 s=subpath(s,i,i+1);
      real endtime=0;
      while(endtime < 1) {
	real newend=takeStep(s,endtime,width);
	path3 si=subpath(s,endtime,newend);
	real l=arclength(si);
	real w=remainL*aspect;
	surface segment=scale(w,w,l)*unitcylinder;
	if(endtime == 0) // add base
	  segment.append(scale(w,w,1)*unitdisk);
	for(patch p : segment.s) {
	  for(int i=0; i < 4; ++i) {
	    for(int j=0; j < 4; ++j) {
	      real k=1-p.P[i][j].z/remainL;
	      p.P[i][j]=(k*p.P[i][j].x,k*p.P[i][j].y,p.P[i][j].z);
	      p.P[i][j]=bend(p.P[i][j],si,l);
	    }
	  }
	}
	head.append(segment);
	endtime=newend;
	remainL -= l;
      }
    }
  }
  return head;
};

real[] arrowbasepoints(path3 base, path3 left, path3 right)
{
  real[][] Tl=transpose(intersections(left,base));
  real[][] Tr=transpose(intersections(right,base));
  return new real[] {Tl.length > 0 ? Tl[0][0] : 1,
      Tr.length > 0 ? Tr[0][0] : 1};
}

path3 arrowbase(path3 r, triple y, real t, real size)
{
  triple perp=2*size*perp(dir(r,t));
  return size == 0 ? y : y+perp--y-perp;
}

// Refine a noncyclic path3 g so that it approaches its endpoint in
// geometrically spaced steps.
path3 approach(path3 g, int n, real radix=3)
{
  guide3 G;
  real L=length(g);
  real tlast=0;
  real r=1/radix;
  for(int i=1; i < n; ++i) {
    real t=L*(1-r^i);
    G=G&subpath(g,tlast,t);
    tlast=t;
 }
  return G&subpath(g,tlast,L);
}

arrowhead3 DefaultHead2(filltype filltype=Fill)
{
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
		     pen p=currentpen, real size=0, real angle=arrowangle,
		     projection P=currentprojection) {
    if(size == 0) size=DefaultHead.size(p);
    bool relative=position.relative;
    real position=position.position.x;
    if(relative) position=reltime(g,position);

    path3 r=subpath(g,position,0);
    real t=arctime(r,size);
    path3 s=subpath(r,t,0);
    triple v=point(s,0);
    size=abs(project(v+size*dir(s,0),P)-project(v,P));
    return a.surface(v,s,DefaultHead.head((0,0)--(size,0),p,size,angle),size,p,
		     filltype,P);
  };
  a.filltype=filltype;
  a.gap=0.966;
  a.lighting=false;
  return a;
}
arrowhead3 DefaultHead2=DefaultHead2();

arrowhead3 HookHead2(real dir=arrowdir, real barb=arrowbarb,
		     filltype filltype=Fill)
{
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
		     pen p=currentpen, real size=0, real angle=arrowangle,
		     projection P=currentprojection) {
  if(size == 0) size=a.size(p);

  angle=min(angle*arrowhookfactor,45);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  real t=arctime(r,size);
  path3 s=subpath(r,t,0);
  triple v=point(s,0);
  size=abs(project(v+size*dir(s,0),P)-project(v,P));
  return a.surface(v,s,HookHead.head((0,0)--(size,0),p,size,angle),size,p,
		   filltype,P);
  };
  a.filltype=filltype;
  a.arrowhead2=HookHead;
  a.gap=0.85;
  a.lighting=false;
  return a;
}
arrowhead3 HookHead2=HookHead2();

arrowhead3 TeXHead2;
TeXHead2.size=TeXHead.size;
TeXHead2.gap=0.5;
TeXHead2.lighting=false;
TeXHead2.arrowhead2=TeXHead;
TeXHead2.head=new surface(path3 g, position position=EndPoint,
			  pen p=currentpen, real size=0,
			  real angle=arrowangle, projection P=currentprojection)
{
  if(size == 0) size=TeXHead2.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  real t=arctime(r,size);
  path3 s=subpath(r,t,0);
  triple v=point(s,0);
  size=abs(project(v+size*dir(s,0.5),P)-project(v,P));
  return TeXHead2.surface(v,s,bezulate(TeXHead.head((0,0)--(1,0),p,size)),
			  size,p,Fill,P);
};

arrowhead3 HookHead3(real dir=arrowdir, real barb=arrowbarb)
{
  arrowhead3 a;
  a.head=new surface(path3 g, position position=EndPoint,
		     pen p=currentpen, real size=0, real angle=arrowangle,
		     projection P=currentprojection) {
  if(size == 0) size=a.size(p);

  angle=min(angle*arrowhookfactor,45);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  real t=arctime(r,size);
  path3 s=subpath(r,t,0);
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
  a.gap=0.85;
  return a;
}

arrowhead3 HookHead3=HookHead3();

arrowhead3 TeXHead3;
TeXHead3.size=TeXHead.size;
TeXHead3.arrowhead2=TeXHead;
TeXHead3.head=new surface(path3 g, position position=EndPoint,
			  pen p=currentpen, real size=0,
			  real angle=arrowangle, projection P=currentprojection)
{
  if(size == 0) size=TeXHead3.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,position,0);
  real t=arctime(r,size);
  path3 s=subpath(r,t,0);
  bool straight1=length(s) == 1 && straight(g,0);

  surface head=surface(O,approach(subpath(path3(TeXHead.head((0,0)--(0,1),p,
							     size),
						YZplane),5,0),8,1.5),Z);
  if(straight1) {
    triple v=point(s,0);
    triple u=point(s,1)-v;
    return shift(v)*align(unit(u))*head;
  } else {
    bend(head,s,size);
    return head;
  }
};

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
	       path3 g, material p=currentpen, material arrowheadpen=p,
	       real size=0, real angle=arrowangle, position position=EndPoint,
	       bool forwards=true, margin3 margin=NoMargin3,
	       bool center=false, light light=nolight,
	       light arrowheadlight=currentlight,
	       projection P=currentprojection)
{
  pen q=(pen) p;
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
  static real fuzz=sqrt(realEpsilon);
  if(!cyclic(g) && position > L-fuzz)
    draw(pic,subpath(r,arctime(r,size*arrowhead.gap),length(r)),p,light);
  else draw(pic,g,p,light);
  draw(pic,arrowhead.head(g,position,q,size,angle,P),arrowheadpen,
       arrowhead.lighting ? arrowheadlight : nolight);
 }

void drawarrow2(picture pic, arrowhead3 arrowhead=DefaultHead3,
		path3 g, material p=currentpen, material arrowheadpen=p,
		real size=0, real angle=arrowangle, margin3 margin=NoMargin3,
		light light=nolight, light arrowheadlight=currentlight,
		projection P=currentprojection)
{
  pen q=(pen) p;
  if(arrowheadpen == nullpen) arrowheadpen=p;
  if(size == 0) size=arrowhead.size(q);
  g=margin(g,q).g;
  size=min(arrow2sizelimit*arclength(g),size);

  path3 r=reverse(g);
  int L=length(g);
  real Size=size*arrowhead.gap;
  draw(pic,subpath(r,arctime(r,Size),L-arctime(g,Size)),p,light);
  draw(pic,arrowhead.head(g,L,q,size,angle,P),arrowheadpen,
       arrowhead.lighting ? arrowheadlight : nolight);
  draw(pic,arrowhead.head(r,L,q,size,angle,P),arrowheadpen,
       arrowhead.lighting ? arrowheadlight : nolight);
}

// Add to picture an estimate of the bounding box contribution of arrowhead
// using the local slope at endpoint.
void addArrow(picture pic, arrowhead3 arrowhead, path3 g, pen p, real size,
	      real angle, real position)
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
	      real size=0, real angle=arrowangle, position position=EndPoint,
              bool forwards=true, margin3 margin=NoMargin3,
	      bool center=false, light light=nolight,
	      light arrowheadlight=currentlight)
{
  pen q=(pen) p;
  if(size == 0) size=arrowhead.size(q);
  picture pic;
  if(is3D())
    pic.add(new void(frame f, transform3 t, picture pic2, projection P) {
	picture opic=new picture;
	drawarrow(opic,arrowhead,t*g,p,arrowheadpen,size,angle,position,
		  forwards,margin,center,light,arrowheadlight,P);
	add(f,opic.fit3(identity4,pic2,P));
      });

  addPath(pic,g,q);

  real position=position(position,size,g,center);
  path3 G;
  if(!forwards) {
    G=reverse(g);
    position=length(g)-position;
  } else G=g;
  addArrow(pic,arrowhead,G,q,size,angle,position);

  return pic;
}

picture arrow2(arrowhead3 arrowhead=DefaultHead3,
               path3 g, material p=currentpen, material arrowheadpen=p,
	       real size=0, real angle=arrowangle, margin3 margin=NoMargin3,
	       light light=nolight, light arrowheadlight=currentlight)
{
  pen q=(pen) p;
  if(size == 0) size=arrowhead.size(q);
  picture pic;

  if(is3D())
    pic.add(new void(frame f, transform3 t, picture pic2, projection P) {
	picture opic=new picture;
	drawarrow2(opic,arrowhead,t*g,p,arrowheadpen,size,angle,margin,light,
		   arrowheadlight,P);
	add(f,opic.fit3(identity4,pic2,P));
      });

  addPath(pic,g,q);

  int L=length(g);
  addArrow(pic,arrowhead,g,q,size,angle,L);
  addArrow(pic,arrowhead,reverse(g),q,size,angle,L);

  return pic;
}

void bar(picture pic, triple a, triple d, material p=currentpen,
	 light light=nolight)
{
  d *= 0.5;
  pic.add(new void(frame f, transform3 t, picture pic2, projection P) {
      picture opic=new picture;
      triple A=t*a;
      triple v=abs(d)*unit(cross(P.vector(),d));
      draw(opic,A-v--A+v,p,light);
      add(f,opic.fit3(identity4,pic2,P));
    });
  triple v=cross(currentprojection.vector(),d);
  pen q=(pen) p;
  triple m=min3(q);
  triple M=max3(q);
  pic.addPoint(a,-v-m);
  pic.addPoint(a,-v+m);
  pic.addPoint(a,v-M);
  pic.addPoint(a,v+M);
}
                                                      
picture bar(triple a, triple d, material p=currentpen)
{
  picture pic;
  bar(pic,a,d,p);
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

void add(picture pic, arrowhead3 arrowhead, real size, real angle,
	 position position, material arrowheadpen, path3 g, material p,
	 bool forwards=true, margin3 margin, bool center=false, light light,
	 light arrowheadlight)
{
  add(pic,arrow(arrowhead,g,p,arrowheadpen,size,angle,position,forwards,
		margin,center,light,arrowheadlight));
  if(!is3D()) {
    pic.add(new void(frame f, transform3 t, picture pic, projection P) {
	if(pic != null) {
	  pen q=(pen) p;
	  marginT3 m=margin(g,q);
	  add(pic,arrow(arrowhead.arrowhead2,project(t*g,P),q,size,angle,
			arrowhead.filltype,position,forwards,
			TrueMargin(m.begin,m.end),center));
	}
      },true);
  }
}

void add2(picture pic, arrowhead3 arrowhead, real size, real angle,
	  material arrowheadpen, path3 g, material p, margin3 margin,
	  light light, light arrowheadlight)
{
  add(pic,arrow2(arrowhead,g,p,arrowheadpen,size,angle,margin,light,
		 arrowheadlight));
  if(!is3D()) {
    pic.add(new void(frame f, transform3 t, picture pic, projection P) {
	if(pic != null) {
	  pen q=(pen) p;
	  marginT3 m=margin(g,q);
	  add(pic,arrow2(arrowhead.arrowhead2,project(t*g,P),q,size,angle,
			 arrowhead.filltype,TrueMargin(m.begin,m.end)));
	}
      },true);
  }
}

arrowbar3 BeginArrow3(arrowhead3 arrowhead=DefaultHead3,
		      real size=0, real angle=arrowangle,
		      position position=BeginPoint,
		      material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    add(pic,arrowhead,size,angle,position,arrowheadpen,g,p,forwards=false,
	margin,light,arrowheadlight);
    return false;
  };
}

arrowbar3 Arrow3(arrowhead3 arrowhead=DefaultHead3,
		 real size=0, real angle=arrowangle,
		 position position=EndPoint,
		 material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    add(pic,arrowhead,size,angle,position,arrowheadpen,g,p,margin,light,
	arrowheadlight);
    return false;
  };
}

arrowbar3 EndArrow3(arrowhead3 arrowhead=DefaultHead3,
		    real size=0, real angle=arrowangle,
		    position position=EndPoint,
		    material arrowheadpen=nullpen)=Arrow3;

arrowbar3 MidArrow3(arrowhead3 arrowhead=DefaultHead3,
		    real size=0, real angle=arrowangle,
		    material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    add(pic,arrowhead,size,angle,MidPoint,arrowheadpen,g,p,margin,center=true,
	light,arrowheadlight);
    return false;
  };
}

arrowbar3 Arrows3(arrowhead3 arrowhead=DefaultHead3,
		  real size=0, real angle=arrowangle,
		  material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    add2(pic,arrowhead,size,angle,arrowheadpen,g,p,margin,light,arrowheadlight);
    return false;
  };
}

arrowbar3 BeginArcArrow3(arrowhead3 arrowhead=DefaultHead3,
			 real size=0, real angle=arcarrowangle,
			 position position=BeginPoint,
			 material arrowheadpen=nullpen) {
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) { 
    real size=(size == 0) ? arcarrowsize((pen) p) : size; 
    add(pic,arrowhead,size,angle,position,arrowheadpen,g,p,forwards=false,
	margin,light,arrowheadlight);
    return false; 
  };
}

arrowbar3 ArcArrow3(arrowhead3 arrowhead=DefaultHead3,
		    real size=0, real angle=arcarrowangle,
		    position position=EndPoint,
		    material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    real size=(size == 0) ? arcarrowsize((pen) p) : size; 
    add(pic,arrowhead,size,angle,position,arrowheadpen,g,p,margin,light,
	arrowheadlight);
    return false;
  };
}

arrowbar3 EndArcArrow3(arrowhead3 arrowhead=DefaultHead3,
		       real size=0, real angle=arcarrowangle,
		       position position=EndPoint,
		       material arrowheadpen=nullpen)=ArcArrow3;


arrowbar3 MidArcArrow3(arrowhead3 arrowhead=DefaultHead3,
		    real size=0, real angle=arcarrowangle,
		    material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    real size=(size == 0) ? arcarrowsize((pen) p) : size; 
    add(pic,arrowhead,size,angle,MidPoint,arrowheadpen,g,p,margin,center=true,
	light,arrowheadlight);
    return false;
  };
}

arrowbar3 ArcArrows3(arrowhead3 arrowhead=DefaultHead3,
		  real size=0, real angle=arcarrowangle,
		  material arrowheadpen=nullpen)
{
  return new bool(picture pic, path3 g, material p, margin3 margin,
		  light light, light arrowheadlight) {
    real size=(size == 0) ? arcarrowsize((pen) p) : size; 
    add2(pic,arrowhead,size,angle,arrowheadpen,g,p,margin,light,arrowheadlight);
    return false;
  };
}

arrowbar3 BeginBar3(real size=0) 
{
  return new bool(picture pic, path3 g, material p, margin3 margin, light light,
		  light) {
    real size=size == 0 ? barsize((pen) p) : size;
    bar(pic,point(g,0),size*dir(g,0),p,light);
    return true;
  };
}

arrowbar3 Bar3(real size=0) 
{
  return new bool(picture pic, path3 g, material p, margin3 margin, light light,
		  light) {
    int L=length(g);
    real size=size == 0 ? barsize((pen) p) : size;
    bar(pic,point(g,L),size*dir(g,L),p,light);
    return true;
  };
}

arrowbar3 EndBar3(real size=0)=Bar3; 

arrowbar3 Bars3(real size=0) 
{
  return new bool(picture pic, path3 g, material p, margin3 margin, light light,
		  light) {
    real size=size == 0 ? barsize((pen) p) : size;
    BeginBar3(size)(pic,g,p,margin,light,nolight);
    EndBar3(size)(pic,g,p,margin,light,nolight);
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
