// Three-dimensional graphing routines

private import math;
import graph;
import three;

typedef triple direction3(real);
direction3 Dir(triple dir) {return new triple(real) {return dir;};}

ticklocate ticklocate(real a, real b, autoscaleT S=defaultS,
                      real tickmin=-infinity, real tickmax=infinity,
                      real time(real)=null, direction3 dir) 
{
  if((valuetime) time == null) time=linear(S.T(),a,b);
  ticklocate locate;
  locate.a=a;
  locate.b=b;
  locate.S=S.copy();
  if(finite(tickmin)) locate.S.tickMin=tickmin;
  if(finite(tickmax)) locate.S.tickMax=tickmax;
  locate.time=time;
  locate.dir=zero;
  locate.dir3=dir;
  return locate;
}
                             
private struct locateT {
  real t;         // tick location time
  triple V;       // tick location in frame coordinates
  triple pathdir; // path direction in frame coordinates
  triple dir;     // tick direction in frame coordinates
  
  void dir(transform3 T, path3 g, ticklocate locate, real t) {
    pathdir=unit(shiftless(T)*dir(g,t));
    triple Dir=locate.dir3(t);
    dir=unit(Dir);
  }
  // Locate the desired position of a tick along a path.
  void calc(transform3 T, path3 g, ticklocate locate, real val) {
    t=locate.time(val);
    V=T*point(g,t);
    dir(T,g,locate,t);
  }
}

void drawtick(picture pic, transform3 T, path3 g, path3 g2,
              ticklocate locate, real val, real Size, int sign, pen p,
              bool extend)
{
  locateT locate1,locate2;
  locate1.calc(T,g,locate,val);
  path3 G;
  if(extend && size(g2) > 0) {
    locate2.calc(T,g2,locate,val);
    G=locate1.V--locate2.V;
  } else
    G=(sign == 0) ?
      locate1.V-Size*locate1.dir--locate1.V+Size*locate1.dir :
      locate1.V--locate1.V+Size*sign*locate1.dir;
  draw(pic,G,p,name="tick");
}

triple ticklabelshift(triple align, pen p=currentpen) 
{
  return 0.25*unit(align)*labelmargin(p);
}

// Signature of routines that draw labelled paths with ticks and tick labels.
typedef void ticks3(picture, transform3, Label, path3, path3, pen,
                    arrowbar3, margin3, ticklocate, int[], bool opposite=false,
                    bool primary=true);

// Label a tick on a frame.
void labeltick(picture pic, transform3 T, path3 g,
               ticklocate locate, real val, int sign, real Size,
               ticklabel ticklabel, Label F, real norm=0)
{
  locateT locate1;
  locate1.calc(T,g,locate,val);
  triple align=F.align.dir3;
  if(align == O) align=sign*locate1.dir;

  triple shift=align*labelmargin(F.p);
  if(dot(align,sign*locate1.dir) >= 0)
    shift=sign*(Size)*locate1.dir;

  real label;
  if(locate.S.scale.logarithmic)
    label=locate.S.scale.Tinv(val);
  else {
    label=val;
    if(abs(label) < zerotickfuzz*norm) label=0;
    // Fix epsilon errors at +/-1e-4
    // default format changes to scientific notation here
    if(abs(abs(label)-1e-4) < epsilon) label=sgn(label)*1e-4;
  }

  string s=ticklabel(label);
  triple v=locate1.V+shift;
  if(s != "")
    label(pic,F.defaulttransform3 ? baseline(s,baselinetemplate) : F.T3*s,v,
          align,F.p);
}  

// Add axis label L to frame f.
void labelaxis(picture pic, transform3 T, Label L, path3 g, 
               ticklocate locate=null, int sign=1, bool ticklabels=false)
{
  triple m=pic.min(identity4);
  triple M=pic.max(identity4);
  triple align=L.align.dir3;
  Label L=L.copy();

  pic.add(new void(frame f, transform3 T, picture pic2, projection P) {
      path3 g=T*g;
      real t=relative(L,g);
      triple v=point(g,t);
      picture F;
      if(L.align.dir3 == O)
        align=unit(invert(L.align.dir,v,P))*abs(L.align.dir);
      
      if(ticklabels && locate != null && piecewisestraight(g)) {
        locateT locate1;
        locate1.dir(T,g,locate,t);
        triple pathdir=locate1.pathdir;

        triple perp=cross(pathdir,P.normal);
        if(align == O)
          align=unit(sgn(dot(sign*locate1.dir,perp))*perp);
        path[] g=project(box(T*m,T*M),P);
        pair z=project(v,P);
        pair Ppathdir=project(v+pathdir,P)-z;
        pair Perp=unit(I*Ppathdir);
        real angle=degrees(Ppathdir,warn=false);
        transform S=rotate(-angle,z);
        path[] G=S*g;
        pair Palign=project(v+align,P)-z;
        pair Align=rotate(-angle)*dot(Palign,Perp)*Perp;
        pair offset=unit(Palign)*
          abs((Align.y >= 0 ? max(G).y : (Align.y < 0 ? min(G).y : 0))-z.y);
        triple normal=cross(pathdir,align);
        if(normal != O) v=invert(z+offset,normal,v,P);
      }

      label(F,L,v);
      add(f,F.fit3(identity4,pic2,P));
    },exact=false);

  path3[] G=path3(texpath(L,bbox=true));
  if(G.length > 0) {
    G=L.align.is3D ? align(G,O,align,L.p) : L.T3*G;
    triple v=point(g,relative(L,g));
    pic.addBox(v,v,min(G),max(G));
  }
}

// Tick construction routine for a user-specified array of tick values.
ticks3 Ticks3(int sign, Label F="", ticklabel ticklabel=null,
              bool beginlabel=true, bool endlabel=true,
              real[] Ticks=new real[], real[] ticks=new real[], int N=1,
              bool begin=true, bool end=true,
              real Size=0, real size=0, bool extend=false,
              pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(picture pic, transform3 t, Label L, path3 g, path3 g2, pen p,
                  arrowbar3 arrow, margin3 margin, ticklocate locate,
                  int[] divisor, bool opposite, bool primary) {
    // Use local copy of context variables:
    int Sign=opposite ? -1 : 1;
    int sign=Sign*sign;
    pen pTick=pTick;
    pen ptick=ptick;
    ticklabel ticklabel=ticklabel;
    
    real Size=Size;
    real size=size;
    if(Size == 0) Size=Ticksize;
    if(size == 0) size=ticksize;
    
    Label L=L.copy();
    Label F=F.copy();
    L.p(p);
    F.p(p);
    if(pTick == nullpen) pTick=p;
    if(ptick == nullpen) ptick=pTick;
    
    bool ticklabels=false;
    path3 G=t*g;
    path3 G2=t*g2;
    
    scalefcn T;
    
    real a,b;
    if(locate.S.scale.logarithmic) {
      a=locate.S.postscale.Tinv(locate.a);
      b=locate.S.postscale.Tinv(locate.b);
      T=locate.S.scale.T;
    } else {
      a=locate.S.Tinv(locate.a);
      b=locate.S.Tinv(locate.b);
      T=identity;
    }
    
    if(a > b) {real temp=a; a=b; b=temp;}

    real norm=max(abs(a),abs(b));
    
    string format=autoformat(F.s,norm...Ticks);
    if(F.s == "%") F.s="";
    if(ticklabel == null) {
      if(locate.S.scale.logarithmic) {
        int base=round(locate.S.scale.Tinv(1));
        ticklabel=format == "%" ? Format("") : DefaultLogFormat(base);
      } else ticklabel=Format(format);
    }

    bool labelaxis=L.s != "" && primary;

    begingroup3(pic,"axis");

    if(primary) draw(pic,margin(G,p).g,p,arrow);
    else draw(pic,G,p);

    for(int i=(begin ? 0 : 1); i < (end ? Ticks.length : Ticks.length-1); ++i) {
      real val=T(Ticks[i]);
      if(val >= a && val <= b)
        drawtick(pic,t,g,g2,locate,val,Size,sign,pTick,extend);
    }
    for(int i=0; i < ticks.length; ++i) {
      real val=T(ticks[i]);
      if(val >= a && val <= b)
        drawtick(pic,t,g,g2,locate,val,size,sign,ptick,extend);
    }

    if(N == 0) N=1;
    if(Size > 0 && primary) {
      for(int i=(beginlabel ? 0 : 1);
          i < (endlabel ? Ticks.length : Ticks.length-1); i += N) {
        real val=T(Ticks[i]);
        if(val >= a && val <= b) {
          ticklabels=true;
          labeltick(pic,t,g,locate,val,Sign,Size,ticklabel,F,norm);
        }
      }
    }
    if(labelaxis)
      labelaxis(pic,t,L,G,locate,Sign,ticklabels);

    endgroup3(pic);
  };
}

// Automatic tick construction routine.
ticks3 Ticks3(int sign, Label F="", ticklabel ticklabel=null,
              bool beginlabel=true, bool endlabel=true,
              int N, int n=0, real Step=0, real step=0,
              bool begin=true, bool end=true, tickmodifier modify=None,
              real Size=0, real size=0, bool extend=false,
              pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(picture pic, transform3 T, Label L,
                  path3 g, path3 g2, pen p,
                  arrowbar3 arrow, margin3 margin=NoMargin3, ticklocate locate,
                  int[] divisor, bool opposite, bool primary) {
    path3 G=T*g;
    real limit=Step == 0 ? axiscoverage*arclength(G) : 0;
    tickvalues values=modify(generateticks(sign,F,ticklabel,N,n,Step,step,
                                           Size,size,identity(),1,
                                           project(G,currentprojection),
                                           limit,p,locate,divisor,
                                           opposite));
    Ticks3(sign,F,ticklabel,beginlabel,endlabel,values.major,values.minor,
           values.N,begin,end,Size,size,extend,pTick,ptick)
      (pic,T,L,g,g2,p,arrow,margin,locate,divisor,opposite,primary);
  };
}

ticks3 NoTicks3()
{
  return new void(picture pic, transform3 T, Label L, path3 g,
                  path3, pen p, arrowbar3 arrow, margin3 margin,
                  ticklocate, int[], bool opposite, bool primary) {
    path3 G=T*g;
    if(primary) draw(pic,margin(G,p).g,p,arrow,margin);
    else draw(pic,G,p);
    if(L.s != "" && primary) {
      Label L=L.copy();
      L.p(p);
      labelaxis(pic,T,L,G,opposite ? -1 : 1);
    }
  };
}

ticks3 InTicks(Label format="", ticklabel ticklabel=null,
               bool beginlabel=true, bool endlabel=true,
               int N=0, int n=0, real Step=0, real step=0,
               bool begin=true, bool end=true, tickmodifier modify=None,
               real Size=0, real size=0, bool extend=false,
               pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks3(-1,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
                begin,end,modify,Size,size,extend,pTick,ptick);
}

ticks3 OutTicks(Label format="", ticklabel ticklabel=null,
                bool beginlabel=true, bool endlabel=true,
                int N=0, int n=0, real Step=0, real step=0,
                bool begin=true, bool end=true, tickmodifier modify=None,
                real Size=0, real size=0, bool extend=false,
                pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks3(1,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
                begin,end,modify,Size,size,extend,pTick,ptick);
}

ticks3 InOutTicks(Label format="", ticklabel ticklabel=null,
                  bool beginlabel=true, bool endlabel=true,
                  int N=0, int n=0, real Step=0, real step=0,
                  bool begin=true, bool end=true, tickmodifier modify=None,
                  real Size=0, real size=0, bool extend=false,
                  pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks3(0,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
                begin,end,modify,Size,size,extend,pTick,ptick);
}

ticks3 InTicks(Label format="", ticklabel ticklabel=null, 
               bool beginlabel=true, bool endlabel=true, 
               real[] Ticks, real[] ticks=new real[],
               real Size=0, real size=0, bool extend=false,
               pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks3(-1,format,ticklabel,beginlabel,endlabel,
                Ticks,ticks,Size,size,extend,pTick,ptick);
}

ticks3 OutTicks(Label format="", ticklabel ticklabel=null, 
                bool beginlabel=true, bool endlabel=true, 
                real[] Ticks, real[] ticks=new real[],
                real Size=0, real size=0, bool extend=false,
                pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks3(1,format,ticklabel,beginlabel,endlabel,
                Ticks,ticks,Size,size,extend,pTick,ptick);
}

ticks3 InOutTicks(Label format="", ticklabel ticklabel=null, 
                  bool beginlabel=true, bool endlabel=true, 
                  real[] Ticks, real[] ticks=new real[],
                  real Size=0, real size=0, bool extend=false,
                  pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks3(0,format,ticklabel,beginlabel,endlabel,
                Ticks,ticks,Size,size,extend,pTick,ptick);
}

ticks3 NoTicks3=NoTicks3(),
InTicks=InTicks(),
OutTicks=OutTicks(),
InOutTicks=InOutTicks();

triple tickMin3(picture pic)
{
  return minbound(pic.userMin(),(pic.scale.x.tickMin,pic.scale.y.tickMin,
                                  pic.scale.z.tickMin));
}
  
triple tickMax3(picture pic)
{
  return maxbound(pic.userMax(),(pic.scale.x.tickMax,pic.scale.y.tickMax,
                                  pic.scale.z.tickMax));
}
                                               
axis Bounds(int type=Both, int type2=Both, triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=type;
    axis.type2=type2;
    axis.position=0.5;
    axis.align=align;
    axis.extend=extend;
  };
}

axis YZEquals(real y, real z, triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.type2=Value;
    axis.value=pic.scale.y.T(y);
    axis.value2=pic.scale.z.T(z);
    axis.position=1;
    axis.align=align;
    axis.extend=extend;
  };
}

axis XZEquals(real x, real z, triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.type2=Value;
    axis.value=pic.scale.x.T(x);
    axis.value2=pic.scale.z.T(z);
    axis.position=1;
    axis.align=align;
    axis.extend=extend;
  };
}

axis XYEquals(real x, real y, triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.type2=Value;
    axis.value=pic.scale.x.T(x);
    axis.value2=pic.scale.y.T(y);
    axis.position=1;
    axis.align=align;
    axis.extend=extend;
  };
}

axis YZZero(triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.type2=Value;
    axis.value=pic.scale.y.T(pic.scale.y.scale.logarithmic ? 1 : 0);
    axis.value2=pic.scale.z.T(pic.scale.z.scale.logarithmic ? 1 : 0);
    axis.position=1;
    axis.align=align;
    axis.extend=extend;
  };
}

axis XZZero(triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.type2=Value;
    axis.value=pic.scale.x.T(pic.scale.x.scale.logarithmic ? 1 : 0);
    axis.value2=pic.scale.z.T(pic.scale.z.scale.logarithmic ? 1 : 0);
    axis.position=1;
    axis.align=align;
    axis.extend=extend;
  };
}

axis XYZero(triple align=O, bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.type2=Value;
    axis.value=pic.scale.x.T(pic.scale.x.scale.logarithmic ? 1 : 0);
    axis.value2=pic.scale.y.T(pic.scale.y.scale.logarithmic ? 1 : 0);
    axis.position=1;
    axis.align=align;
    axis.extend=extend;
  };
}

axis
Bounds=Bounds(),
YZZero=YZZero(),
XZZero=XZZero(),
XYZero=XYZero();

// Draw a general three-dimensional axis.
void axis(picture pic=currentpicture, Label L="", path3 g, path3 g2=nullpath3,
          pen p=currentpen, ticks3 ticks, ticklocate locate,
          arrowbar3 arrow=None, margin3 margin=NoMargin3,
          int[] divisor=new int[], bool above=false, bool opposite=false) 
{
  Label L=L.copy();
  real t=reltime(g,0.5);
  if(L.defaultposition) L.position(t);
  divisor=copy(divisor);
  locate=locate.copy();
  
  pic.add(new void (picture f, transform3 t, transform3 T, triple, triple) {
      picture d;
      ticks(d,t,L,g,g2,p,arrow,margin,locate,divisor,opposite,true);
      add(f,t*T*inverse(t)*d);
    },above=above);
  
  addPath(pic,g,p);
  
  if(L.s != "") {
    frame f;
    Label L0=L.copy();
    L0.position(0);
    add(f,L0);
    triple pos=point(g,L.relative()*length(g));
    pic.addBox(pos,pos,min3(f),max3(f));
  }
}

real xtrans(transform3 t, real x)
{
  return (t*(x,0,0)).x;
}

real ytrans(transform3 t, real y)
{
  return (t*(0,y,0)).y;
}

real ztrans(transform3 t, real z)
{
  return (t*(0,0,z)).z;
}

private triple defaultdir(triple X, triple Y, triple Z, bool opposite=false,
                          projection P) {
  triple u=cross(P.normal,Z);
  return abs(dot(u,X)) > abs(dot(u,Y)) ? -X : (opposite ? Y : -Y);
}

// An internal routine to draw an x axis at a particular y value.
void xaxis3At(picture pic=currentpicture, Label L="", axis axis,
              real xmin=-infinity, real xmax=infinity, pen p=currentpen,
              ticks3 ticks=NoTicks3,
              arrowbar3 arrow=None, margin3 margin=NoMargin3, bool above=true,
              bool opposite=false, bool opposite2=false, bool primary=true)
{
  int type=axis.type;
  int type2=axis.type2;
  triple dir=axis.align.dir3 == O ?
    defaultdir(Y,Z,X,opposite^opposite2,currentprojection) : axis.align.dir3;
  Label L=L.copy();
  if(L.align.dir3 == O && L.align.dir == 0) L.align(opposite ? -dir : dir);

  real y=axis.value;
  real z=axis.value2;
  real y2,z2;
  int[] divisor=copy(axis.xdivisor);

  pic.add(new void(picture f, transform3 t, transform3 T, triple lb,
                   triple rt) {
            transform3 tinv=inverse(t);
            triple a=xmin == -infinity ? tinv*(lb.x-min3(p).x,ytrans(t,y),
                                               ztrans(t,z)) : (xmin,y,z);
            triple b=xmax == infinity ? tinv*(rt.x-max3(p).x,ytrans(t,y),
                                              ztrans(t,z)) : (xmax,y,z);
            real y0;
            real z0;
            if(abs(dir.y) < abs(dir.z)) {
              y0=y;
              z0=z2;
            } else {
              y0=y2;
              z0=z;
            }
            
            triple a2=xmin == -infinity ? tinv*(lb.x-min3(p).x,ytrans(t,y0),
                                                ztrans(t,z0)) : (xmin,y0,z0);
            triple b2=xmax == infinity ? tinv*(rt.x-max3(p).x,ytrans(t,y0),
                                               ztrans(t,z0)) : (xmax,y0,z0);

            if(xmin == -infinity || xmax == infinity) {
              bounds mx=autoscale(a.x,b.x,pic.scale.x.scale);
              pic.scale.x.tickMin=mx.min;
              pic.scale.x.tickMax=mx.max;
              divisor=mx.divisor;
            }
      
            triple fuzz=X*epsilon*max(abs(a.x),abs(b.x));
            a -= fuzz;
            b += fuzz;

            picture d;
            ticks(d,t,L,a--b,finite(y0) && finite(z0) ? a2--b2 : nullpath3,
                  p,arrow,margin,
                  ticklocate(a.x,b.x,pic.scale.x,Dir(dir)),divisor,
                  opposite,primary);
            add(f,t*T*tinv*d);
          },above=above);

  void bounds() {
    if(type == Min)
      y=pic.scale.y.automin() ? tickMin3(pic).y : pic.userMin().y;
    else if(type == Max)
      y=pic.scale.y.automax() ? tickMax3(pic).y : pic.userMax().y;
    else if(type == Both) {
      y2=pic.scale.y.automax() ? tickMax3(pic).y : pic.userMax().y;
      y=opposite ? y2 : 
        (pic.scale.y.automin() ? tickMin3(pic).y : pic.userMin().y);
    }

    if(type2 == Min)
      z=pic.scale.z.automin() ? tickMin3(pic).z : pic.userMin().z;
    else if(type2 == Max)
      z=pic.scale.z.automax() ? tickMax3(pic).z : pic.userMax().z;
    else if(type2 == Both) {
      z2=pic.scale.z.automax() ? tickMax3(pic).z : pic.userMax().z;
      z=opposite2 ? z2 : 
        (pic.scale.z.automin() ? tickMin3(pic).z : pic.userMin().z);
    }

    real Xmin=finite(xmin) ? xmin : pic.userMin().x;
    real Xmax=finite(xmax) ? xmax : pic.userMax().x;

    triple a=(Xmin,y,z);
    triple b=(Xmax,y,z);
    triple a2=(Xmin,y2,z2);
    triple b2=(Xmax,y2,z2);

    if(finite(a)) {
      pic.addPoint(a,min3(p));
      pic.addPoint(a,max3(p));
    }
  
    if(finite(b)) {
      pic.addPoint(b,min3(p));
      pic.addPoint(b,max3(p));
    }

    if(finite(a) && finite(b)) {
      picture d;
      ticks(d,pic.scaling3(warn=false),L,
            (a.x,0,0)--(b.x,0,0),(a2.x,0,0)--(b2.x,0,0),p,arrow,margin,
            ticklocate(a.x,b.x,pic.scale.x,Dir(dir)),divisor,
            opposite,primary);
      frame f;
      if(L.s != "") {
        Label L0=L.copy();
        L0.position(0);
        add(f,L0);
      }
      triple pos=a+L.relative()*(b-a);
      triple m=min3(d);
      triple M=max3(d);
      pic.addBox(pos,pos,(min3(f).x,m.y,m.z),(max3(f).x,m.y,m.z));
    }
  }

  // Process any queued y and z axes bound calculation requests.
  for(int i=0; i < pic.scale.y.bound.length; ++i)
    pic.scale.y.bound[i]();
  for(int i=0; i < pic.scale.z.bound.length; ++i)
    pic.scale.z.bound[i]();

  pic.scale.y.bound.delete();
  pic.scale.z.bound.delete();

  bounds();

  // Request another x bounds calculation before final picture scaling.
  pic.scale.x.bound.push(bounds);
}

// An internal routine to draw an x axis at a particular y value.
void yaxis3At(picture pic=currentpicture, Label L="", axis axis,
              real ymin=-infinity, real ymax=infinity, pen p=currentpen,
              ticks3 ticks=NoTicks3,
              arrowbar3 arrow=None, margin3 margin=NoMargin3, bool above=true,
              bool opposite=false, bool opposite2=false, bool primary=true)
{
  int type=axis.type;
  int type2=axis.type2;
  triple dir=axis.align.dir3 == O ?
    defaultdir(X,Z,Y,opposite^opposite2,currentprojection) : axis.align.dir3;
  Label L=L.copy();
  if(L.align.dir3 == O && L.align.dir == 0) L.align(opposite ? -dir : dir);

  real x=axis.value;
  real z=axis.value2;
  real x2,z2;
  int[] divisor=copy(axis.ydivisor);

  pic.add(new void(picture f, transform3 t, transform3 T, triple lb,
                   triple rt) {
            transform3 tinv=inverse(t);
            triple a=ymin == -infinity ? tinv*(xtrans(t,x),lb.y-min3(p).y,
                                               ztrans(t,z)) : (x,ymin,z);
            triple b=ymax == infinity ? tinv*(xtrans(t,x),rt.y-max3(p).y,
                                              ztrans(t,z)) : (x,ymax,z);
            real x0;
            real z0;
            if(abs(dir.x) < abs(dir.z)) {
              x0=x;
              z0=z2;
            } else {
              x0=x2;
              z0=z;
            }
            
            triple a2=ymin == -infinity ? tinv*(xtrans(t,x0),lb.y-min3(p).y,
                                                ztrans(t,z0)) : (x0,ymin,z0);
            triple b2=ymax == infinity ? tinv*(xtrans(t,x0),rt.y-max3(p).y,
                                               ztrans(t,z0)) : (x0,ymax,z0);
 
            if(ymin == -infinity || ymax == infinity) {
              bounds my=autoscale(a.y,b.y,pic.scale.y.scale);
              pic.scale.y.tickMin=my.min;
              pic.scale.y.tickMax=my.max;
              divisor=my.divisor;
            }
      
            triple fuzz=Y*epsilon*max(abs(a.y),abs(b.y));
            a -= fuzz;
            b += fuzz;

            picture d;
            ticks(d,t,L,a--b,finite(x0) && finite(z0) ? a2--b2 : nullpath3,
                  p,arrow,margin,
                  ticklocate(a.y,b.y,pic.scale.y,Dir(dir)),divisor,
                  opposite,primary);
            add(f,t*T*tinv*d);
          },above=above);

  void bounds() {
    if(type == Min)
      x=pic.scale.x.automin() ? tickMin3(pic).x : pic.userMin().x;
    else if(type == Max)
      x=pic.scale.x.automax() ? tickMax3(pic).x : pic.userMax().x;
    else if(type == Both) {
      x2=pic.scale.x.automax() ? tickMax3(pic).x : pic.userMax().x;
      x=opposite ? x2 : 
        (pic.scale.x.automin() ? tickMin3(pic).x : pic.userMin().x);
    }

    if(type2 == Min)
      z=pic.scale.z.automin() ? tickMin3(pic).z : pic.userMin().z;
    else if(type2 == Max)
      z=pic.scale.z.automax() ? tickMax3(pic).z : pic.userMax().z;
    else if(type2 == Both) {
      z2=pic.scale.z.automax() ? tickMax3(pic).z : pic.userMax().z;
      z=opposite2 ? z2 : 
        (pic.scale.z.automin() ? tickMin3(pic).z : pic.userMin().z);
    }

    real Ymin=finite(ymin) ? ymin : pic.userMin().y;
    real Ymax=finite(ymax) ? ymax : pic.userMax().y;

    triple a=(x,Ymin,z);
    triple b=(x,Ymax,z);
    triple a2=(x2,Ymin,z2);
    triple b2=(x2,Ymax,z2);

    if(finite(a)) {
      pic.addPoint(a,min3(p));
      pic.addPoint(a,max3(p));
    }
  
    if(finite(b)) {
      pic.addPoint(b,min3(p));
      pic.addPoint(b,max3(p));
    }

    if(finite(a) && finite(b)) {
      picture d;
      ticks(d,pic.scaling3(warn=false),L,
            (0,a.y,0)--(0,b.y,0),(0,a2.y,0)--(0,a2.y,0),p,arrow,margin,
            ticklocate(a.y,b.y,pic.scale.y,Dir(dir)),divisor,
            opposite,primary);
      frame f;
      if(L.s != "") {
        Label L0=L.copy();
        L0.position(0);
        add(f,L0);
      }
      triple pos=a+L.relative()*(b-a);
      triple m=min3(d);
      triple M=max3(d);
      pic.addBox(pos,pos,(m.x,min3(f).y,m.z),(m.x,max3(f).y,m.z));
    }
  }

  // Process any queued x and z axis bound calculation requests.
  for(int i=0; i < pic.scale.x.bound.length; ++i)
    pic.scale.x.bound[i]();
  for(int i=0; i < pic.scale.z.bound.length; ++i)
    pic.scale.z.bound[i]();

  pic.scale.x.bound.delete();
  pic.scale.z.bound.delete();

  bounds();

  // Request another y bounds calculation before final picture scaling.
  pic.scale.y.bound.push(bounds);
}

// An internal routine to draw an x axis at a particular y value.
void zaxis3At(picture pic=currentpicture, Label L="", axis axis,
              real zmin=-infinity, real zmax=infinity, pen p=currentpen,
              ticks3 ticks=NoTicks3,
              arrowbar3 arrow=None, margin3 margin=NoMargin3, bool above=true,
              bool opposite=false, bool opposite2=false, bool primary=true)
{
  int type=axis.type;
  int type2=axis.type2;
  triple dir=axis.align.dir3 == O ?
    defaultdir(X,Y,Z,opposite^opposite2,currentprojection) : axis.align.dir3;
  Label L=L.copy();
  if(L.align.dir3 == O && L.align.dir == 0) L.align(opposite ? -dir : dir);

  real x=axis.value;
  real y=axis.value2;
  real x2,y2;
  int[] divisor=copy(axis.zdivisor);

  pic.add(new void(picture f, transform3 t, transform3 T, triple lb,
                   triple rt) {
            transform3 tinv=inverse(t);
            triple a=zmin == -infinity ? tinv*(xtrans(t,x),ytrans(t,y),
                                               lb.z-min3(p).z) : (x,y,zmin);
            triple b=zmax == infinity ? tinv*(xtrans(t,x),ytrans(t,y),
                                              rt.z-max3(p).z) : (x,y,zmax);
            real x0;
            real y0;
            if(abs(dir.x) < abs(dir.y)) {
              x0=x;
              y0=y2;
            } else {
              x0=x2;
              y0=y;
            }
            
            triple a2=zmin == -infinity ? tinv*(xtrans(t,x0),ytrans(t,y0),
                                                lb.z-min3(p).z) : (x0,y0,zmin);
            triple b2=zmax == infinity ? tinv*(xtrans(t,x0),ytrans(t,y0),
                                               rt.z-max3(p).z) : (x0,y0,zmax);

            if(zmin == -infinity || zmax == infinity) {
              bounds mz=autoscale(a.z,b.z,pic.scale.z.scale);
              pic.scale.z.tickMin=mz.min;
              pic.scale.z.tickMax=mz.max;
              divisor=mz.divisor;
            }
      
            triple fuzz=Z*epsilon*max(abs(a.z),abs(b.z));
            a -= fuzz;
            b += fuzz;

            picture d;
            ticks(d,t,L,a--b,finite(x0) && finite(y0) ? a2--b2 : nullpath3,
                  p,arrow,margin,
                  ticklocate(a.z,b.z,pic.scale.z,Dir(dir)),divisor,
                  opposite,primary);
            add(f,t*T*tinv*d);
          },above=above);

  void bounds() {
    if(type == Min)
      x=pic.scale.x.automin() ? tickMin3(pic).x : pic.userMin().x;
    else if(type == Max)
      x=pic.scale.x.automax() ? tickMax3(pic).x : pic.userMax().x;
    else if(type == Both) {
      x2=pic.scale.x.automax() ? tickMax3(pic).x : pic.userMax().x;
      x=opposite ? x2 : 
        (pic.scale.x.automin() ? tickMin3(pic).x : pic.userMin().x);
    }

    if(type2 == Min)
      y=pic.scale.y.automin() ? tickMin3(pic).y : pic.userMin().y;
    else if(type2 == Max)
      y=pic.scale.y.automax() ? tickMax3(pic).y : pic.userMax().y;
    else if(type2 == Both) {
      y2=pic.scale.y.automax() ? tickMax3(pic).y : pic.userMax().y;
      y=opposite2 ? y2 : 
        (pic.scale.y.automin() ? tickMin3(pic).y : pic.userMin().y);
    }

    real Zmin=finite(zmin) ? zmin : pic.userMin().z;
    real Zmax=finite(zmax) ? zmax : pic.userMax().z;

    triple a=(x,y,Zmin);
    triple b=(x,y,Zmax);
    triple a2=(x2,y2,Zmin);
    triple b2=(x2,y2,Zmax);

    if(finite(a)) {
      pic.addPoint(a,min3(p));
      pic.addPoint(a,max3(p));
    }
  
    if(finite(b)) {
      pic.addPoint(b,min3(p));
      pic.addPoint(b,max3(p));
    }

    if(finite(a) && finite(b)) {
      picture d;
      ticks(d,pic.scaling3(warn=false),L,
            (0,0,a.z)--(0,0,b.z),(0,0,a2.z)--(0,0,a2.z),p,arrow,margin,
            ticklocate(a.z,b.z,pic.scale.z,Dir(dir)),divisor,
            opposite,primary);
      frame f;
      if(L.s != "") {
        Label L0=L.copy();
        L0.position(0);
        add(f,L0);
      }
      triple pos=a+L.relative()*(b-a);
      triple m=min3(d);
      triple M=max3(d);
      pic.addBox(pos,pos,(m.x,m.y,min3(f).z),(m.x,m.y,max3(f).z));
    }
  }

  // Process any queued x and y axes bound calculation requests.
  for(int i=0; i < pic.scale.x.bound.length; ++i)
    pic.scale.x.bound[i]();
  for(int i=0; i < pic.scale.y.bound.length; ++i)
    pic.scale.y.bound[i]();

  pic.scale.x.bound.delete();
  pic.scale.y.bound.delete();

  bounds();

  // Request another z bounds calculation before final picture scaling.
  pic.scale.z.bound.push(bounds);
}

// Internal routine to autoscale the user limits of a picture.
void autoscale3(picture pic=currentpicture, axis axis)
{
  bool set=pic.scale.set;
  autoscale(pic,axis);

  if(!set) {
    bounds mz;
    if(pic.userSetz()) {
      mz=autoscale(pic.userMin().z,pic.userMax().z,pic.scale.z.scale);
      if(pic.scale.z.scale.logarithmic &&
         floor(pic.userMin().z) == floor(pic.userMax().z)) {
        if(pic.scale.z.automin())
          pic.userMinz3(floor(pic.userMin().z));
        if(pic.scale.z.automax())
          pic.userMaxz3(ceil(pic.userMax().z));
      }
    } else {mz.min=mz.max=0; pic.scale.set=false;}
    
    pic.scale.z.tickMin=mz.min;
    pic.scale.z.tickMax=mz.max;
    axis.zdivisor=mz.divisor;
  }
}

// Draw an x axis in three dimensions.
void xaxis3(picture pic=currentpicture, Label L="", axis axis=YZZero,
            real xmin=-infinity, real xmax=infinity, pen p=currentpen,
            ticks3 ticks=NoTicks3,
            arrowbar3 arrow=None, margin3 margin=NoMargin3, bool above=false)
{
  if(xmin > xmax) return;
  
  if(pic.scale.x.automin && xmin > -infinity) pic.scale.x.automin=false;
  if(pic.scale.x.automax && xmax < infinity) pic.scale.x.automax=false;

  if(!pic.scale.set) {
    axis(pic,axis);
    autoscale3(pic,axis);
  }
  
  bool newticks=false;
  
  if(xmin != -infinity) {
    xmin=pic.scale.x.T(xmin);
    newticks=true;
  }
  
  if(xmax != infinity) {
    xmax=pic.scale.x.T(xmax);
    newticks=true;
  }
  
  if(newticks && pic.userSetx() && ticks != NoTicks3) {
    if(xmin == -infinity) xmin=pic.userMin().x;
    if(xmax == infinity) xmax=pic.userMax().x;
    bounds mx=autoscale(xmin,xmax,pic.scale.x.scale);
    pic.scale.x.tickMin=mx.min;
    pic.scale.x.tickMax=mx.max;
    axis.xdivisor=mx.divisor;
  }
  
  axis(pic,axis);
  
  if(xmin == -infinity && !axis.extend) {
    if(pic.scale.set)
      xmin=pic.scale.x.automin() ? pic.scale.x.tickMin :
        max(pic.scale.x.tickMin,pic.userMin().x);
    else xmin=pic.userMin().x;
  }
  
  if(xmax == infinity && !axis.extend) {
    if(pic.scale.set)
      xmax=pic.scale.x.automax() ? pic.scale.x.tickMax :
        min(pic.scale.x.tickMax,pic.userMax().x);
    else xmax=pic.userMax().x;
  }

  if(L.defaultposition) {
    L=L.copy();
    L.position(axis.position);
  }
  
  bool back=false;
  if(axis.type == Both) {
    triple v=currentprojection.normal;
    back=dot((0,pic.userMax().y-pic.userMin().y,0),v)*sgn(v.z) > 0;
  }

  xaxis3At(pic,L,axis,xmin,xmax,p,ticks,arrow,margin,above,false,false,!back);
  if(axis.type == Both)
    xaxis3At(pic,L,axis,xmin,xmax,p,ticks,arrow,margin,above,true,false,back);
  if(axis.type2 == Both) {
    xaxis3At(pic,L,axis,xmin,xmax,p,ticks,arrow,margin,above,false,true,false);
    if(axis.type == Both)
      xaxis3At(pic,L,axis,xmin,xmax,p,ticks,arrow,margin,above,true,true,false);
  }
}

// Draw a y axis in three dimensions.
void yaxis3(picture pic=currentpicture, Label L="", axis axis=XZZero,
            real ymin=-infinity, real ymax=infinity, pen p=currentpen,
            ticks3 ticks=NoTicks3,
            arrowbar3 arrow=None, margin3 margin=NoMargin3, bool above=false)
{
  if(ymin > ymax) return;

  if(pic.scale.y.automin && ymin > -infinity) pic.scale.y.automin=false;
  if(pic.scale.y.automax && ymax < infinity) pic.scale.y.automax=false;
  
  if(!pic.scale.set) {
    axis(pic,axis);
    autoscale3(pic,axis);
  }
  
  bool newticks=false;
  
  if(ymin != -infinity) {
    ymin=pic.scale.y.T(ymin);
    newticks=true;
  }
  
  if(ymax != infinity) {
    ymax=pic.scale.y.T(ymax);
    newticks=true;
  }
  
  if(newticks && pic.userSety() && ticks != NoTicks3) {
    if(ymin == -infinity) ymin=pic.userMin().y;
    if(ymax == infinity) ymax=pic.userMax().y;
    bounds my=autoscale(ymin,ymax,pic.scale.y.scale);
    pic.scale.y.tickMin=my.min;
    pic.scale.y.tickMax=my.max;
    axis.ydivisor=my.divisor;
  }
  
  axis(pic,axis);
  
  if(ymin == -infinity && !axis.extend) {
    if(pic.scale.set)
      ymin=pic.scale.y.automin() ? pic.scale.y.tickMin :
        max(pic.scale.y.tickMin,pic.userMin().y);
    else ymin=pic.userMin().y;
  }
  
  
  if(ymax == infinity && !axis.extend) {
    if(pic.scale.set)
      ymax=pic.scale.y.automax() ? pic.scale.y.tickMax :
        min(pic.scale.y.tickMax,pic.userMax().y);
    else ymax=pic.userMax().y;
  }

  if(L.defaultposition) {
    L=L.copy();
    L.position(axis.position);
  }
  
  bool back=false;
  if(axis.type == Both) {
    triple v=currentprojection.normal;
    back=dot((pic.userMax().x-pic.userMin().x,0,0),v)*sgn(v.z) > 0;
  }

  yaxis3At(pic,L,axis,ymin,ymax,p,ticks,arrow,margin,above,false,false,!back);

  if(axis.type == Both)
    yaxis3At(pic,L,axis,ymin,ymax,p,ticks,arrow,margin,above,true,false,back);
  if(axis.type2 == Both) {
    yaxis3At(pic,L,axis,ymin,ymax,p,ticks,arrow,margin,above,false,true,false);
    if(axis.type == Both)
      yaxis3At(pic,L,axis,ymin,ymax,p,ticks,arrow,margin,above,true,true,false);
  }
}
// Draw a z axis in three dimensions.
void zaxis3(picture pic=currentpicture, Label L="", axis axis=XYZero,
            real zmin=-infinity, real zmax=infinity, pen p=currentpen,
            ticks3 ticks=NoTicks3,
            arrowbar3 arrow=None, margin3 margin=NoMargin3, bool above=false)
{
  if(zmin > zmax) return;

  if(pic.scale.z.automin && zmin > -infinity) pic.scale.z.automin=false;
  if(pic.scale.z.automax && zmax < infinity) pic.scale.z.automax=false;
  
  if(!pic.scale.set) {
    axis(pic,axis);
    autoscale3(pic,axis);
  }
  
  bool newticks=false;
  
  if(zmin != -infinity) {
    zmin=pic.scale.z.T(zmin);
    newticks=true;
  }
  
  if(zmax != infinity) {
    zmax=pic.scale.z.T(zmax);
    newticks=true;
  }
  
  if(newticks && pic.userSetz() && ticks != NoTicks3) {
    if(zmin == -infinity) zmin=pic.userMin().z;
    if(zmax == infinity) zmax=pic.userMax().z;
    bounds mz=autoscale(zmin,zmax,pic.scale.z.scale);
    pic.scale.z.tickMin=mz.min;
    pic.scale.z.tickMax=mz.max;
    axis.zdivisor=mz.divisor;
  }
  
  axis(pic,axis);
  
  if(zmin == -infinity && !axis.extend) {
    if(pic.scale.set)
      zmin=pic.scale.z.automin() ? pic.scale.z.tickMin :
        max(pic.scale.z.tickMin,pic.userMin().z);
    else zmin=pic.userMin().z;
  }
  
  if(zmax == infinity && !axis.extend) {
    if(pic.scale.set)
      zmax=pic.scale.z.automax() ? pic.scale.z.tickMax :
        min(pic.scale.z.tickMax,pic.userMax().z);
    else zmax=pic.userMax().z;
  }

  if(L.defaultposition) {
    L=L.copy();
    L.position(axis.position);
  }
  
  bool back=false;
  if(axis.type == Both) {
    triple v=currentprojection.vector();
    back=dot((pic.userMax().x-pic.userMin().x,0,0),v)*sgn(v.y) > 0;
  }

  zaxis3At(pic,L,axis,zmin,zmax,p,ticks,arrow,margin,above,false,false,!back);
  if(axis.type == Both)
    zaxis3At(pic,L,axis,zmin,zmax,p,ticks,arrow,margin,above,true,false,back);
  if(axis.type2 == Both) {
    zaxis3At(pic,L,axis,zmin,zmax,p,ticks,arrow,margin,above,false,true,false);
    if(axis.type == Both)
      zaxis3At(pic,L,axis,zmin,zmax,p,ticks,arrow,margin,above,true,true,false);
  }
}

// Set the z limits of a picture.
void zlimits(picture pic=currentpicture, real min=-infinity, real max=infinity,
             bool crop=NoCrop)
{
  if(min > max) return;
  
  pic.scale.z.automin=min <= -infinity;
  pic.scale.z.automax=max >= infinity;
  
  bounds mz;
  if(pic.scale.z.automin() || pic.scale.z.automax())
    mz=autoscale(pic.userMin().z,pic.userMax().z,pic.scale.z.scale);
  
  if(pic.scale.z.automin) {
    if(pic.scale.z.automin()) pic.userMinz(mz.min);
  } else pic.userMinz(min(pic.scale.z.T(min),pic.scale.z.T(max)));
  
  if(pic.scale.z.automax) {
    if(pic.scale.z.automax()) pic.userMaxz(mz.max);
  } else pic.userMaxz(max(pic.scale.z.T(min),pic.scale.z.T(max)));
}

// Restrict the x, y, and z limits to box(min,max).
void limits(picture pic=currentpicture, triple min, triple max)
{
  xlimits(pic,min.x,max.x);
  ylimits(pic,min.y,max.y);
  zlimits(pic,min.z,max.z);
}
  
// Draw x, y and z axes.
void axes3(picture pic=currentpicture,
           Label xlabel="", Label ylabel="", Label zlabel="", 
           bool extend=false,
           triple min=(-infinity,-infinity,-infinity),
           triple max=(infinity,infinity,infinity),
           pen p=currentpen, arrowbar3 arrow=None, margin3 margin=NoMargin3)
{
  xaxis3(pic,xlabel,YZZero(extend),min.x,max.x,p,arrow,margin);
  yaxis3(pic,ylabel,XZZero(extend),min.y,max.y,p,arrow,margin);
  zaxis3(pic,zlabel,XYZero(extend),min.z,max.z,p,arrow,margin);
}

triple Scale(picture pic=currentpicture, triple v)
{
  return (pic.scale.x.T(v.x),pic.scale.y.T(v.y),pic.scale.z.T(v.z));
}

real ScaleZ(picture pic=currentpicture, real z)
{
  return pic.scale.z.T(z);
}

// Draw a tick of length size at triple v in direction dir using pen p.
void tick(picture pic=currentpicture, triple v, triple dir, real size=Ticksize,
          pen p=currentpen)
{
  triple v=Scale(pic,v);
  pic.add(new void (picture f, transform3 t) {
      triple tv=t*v;
      draw(f,tv--tv+unit(dir)*size,p);
    });
  pic.addPoint(v,p);
  pic.addPoint(v,unit(dir)*size,p);
}

void xtick(picture pic=currentpicture, triple v, triple dir=Y,
           real size=Ticksize, pen p=currentpen)
{
  tick(pic,v,dir,size,p);
}

void xtick3(picture pic=currentpicture, real x, triple dir=Y,
            real size=Ticksize, pen p=currentpen)
{
  tick(pic,(x,pic.scale.y.scale.logarithmic ? 1 : 0,
             pic.scale.z.scale.logarithmic ? 1 : 0),dir,size,p);
}

void ytick(picture pic=currentpicture, triple v, triple dir=X,
           real size=Ticksize, pen p=currentpen) 
{
  tick(pic,v,dir,size,p);
}

void ytick3(picture pic=currentpicture, real y, triple dir=X,
            real size=Ticksize, pen p=currentpen)
{
  tick(pic,(pic.scale.x.scale.logarithmic ? 1 : 0,y,
            pic.scale.z.scale.logarithmic ? 1 : 0),dir,size,p);
}

void ztick(picture pic=currentpicture, triple v, triple dir=X,
           real size=Ticksize, pen p=currentpen) 
{
  xtick(pic,v,dir,size,p);
}

void ztick3(picture pic=currentpicture, real z, triple dir=X,
            real size=Ticksize, pen p=currentpen)
{
  xtick(pic,(pic.scale.x.scale.logarithmic ? 1 : 0,
             pic.scale.y.scale.logarithmic ? 1 : 0,z),dir,size,p);
}

void tick(picture pic=currentpicture, Label L, real value, triple v,
          triple dir, string format="", real size=Ticksize, pen p=currentpen)
{
  Label L=L.copy();
  L.align(L.align,-dir);
  if(shift(L.T3)*O == O)
    L.T3=shift(dot(dir,L.align.dir3) > 0 ? dir*size :
               ticklabelshift(L.align.dir3,p))*L.T3;
  L.p(p);
  if(L.s == "") L.s=format(format == "" ? defaultformat : format,value);
  L.s=baseline(L.s,baselinetemplate);
  label(pic,L,Scale(pic,v));
  tick(pic,v,dir,size,p);
}

void xtick(picture pic=currentpicture, Label L, triple v, triple dir=Y,
           string format="", real size=Ticksize, pen p=currentpen)
{
  tick(pic,L,v.x,v,dir,format,size,p);
}

void xtick3(picture pic=currentpicture, Label L, real x, triple dir=Y,
            string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,(x,pic.scale.y.scale.logarithmic ? 1 : 0,
              pic.scale.z.scale.logarithmic ? 1 : 0),dir,size,p);
}

void ytick(picture pic=currentpicture, Label L, triple v, triple dir=X,
           string format="", real size=Ticksize, pen p=currentpen)
{
  tick(pic,L,v.y,v,dir,format,size,p);
}

void ytick3(picture pic=currentpicture, Label L, real y, triple dir=X,
            string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,(pic.scale.x.scale.logarithmic ? 1 : 0,y,
              pic.scale.z.scale.logarithmic ? 1 : 0),dir,format,size,p);
}

void ztick(picture pic=currentpicture, Label L, triple v, triple dir=X,
           string format="", real size=Ticksize, pen p=currentpen)
{
  tick(pic,L,v.z,v,dir,format,size,p);
}

void ztick3(picture pic=currentpicture, Label L, real z, triple dir=X,
            string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,(pic.scale.x.scale.logarithmic ? 1 : 0,
              pic.scale.z.scale.logarithmic ? 1 : 0,z),dir,format,size,p);
}

private void label(picture pic, Label L, triple v, real x, align align,
                   string format, pen p)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  if(shift(L.T3)*O == O)
    L.T3=shift(ticklabelshift(L.align.dir3,L.p))*L.T3;
  if(L.s == "") L.s=format(format == "" ? defaultformat : format,x);
  L.s=baseline(L.s,baselinetemplate);
  label(pic,L,v);
}

void labelx(picture pic=currentpicture, Label L="", triple v,
            align align=-Y, string format="", pen p=currentpen)
{
  label(pic,L,Scale(pic,v),v.x,align,format,p);
}

void labelx3(picture pic=currentpicture, Label L="", real x,
             align align=-Y, string format="", pen p=currentpen)
{
  labelx(pic,L,(x,pic.scale.y.scale.logarithmic ? 1 : 0,
                pic.scale.z.scale.logarithmic ? 1 : 0),align,format,p);
}

void labely(picture pic=currentpicture, Label L="", triple v,
            align align=-X, string format="", pen p=currentpen)
{
  label(pic,L,Scale(pic,v),v.y,align,format,p);
}

void labely3(picture pic=currentpicture, Label L="", real y,
             align align=-X, string format="", pen p=currentpen)
{
  labely(pic,L,(pic.scale.x.scale.logarithmic ? 1 : 0,y,
                pic.scale.z.scale.logarithmic ? 1 : 0),align,format,p);
}

void labelz(picture pic=currentpicture, Label L="", triple v,
            align align=-X, string format="", pen p=currentpen)
{
  label(pic,L,Scale(pic,v),v.z,align,format,p);
}

void labelz3(picture pic=currentpicture, Label L="", real z,
             align align=-X, string format="", pen p=currentpen)
{
  labelz(pic,L,(pic.scale.x.scale.logarithmic ? 1 : 0,
                pic.scale.y.scale.logarithmic ? 1 : 0,z),align,format,p);
}

typedef guide3 graph(triple F(real), real, real, int);
typedef guide3[] multigraph(triple F(real), real, real, int);

graph graph(interpolate3 join)
{
  return new guide3(triple f(real), real a, real b, int n) {
    real width=b-a;
    return n == 0 ? join(f(a)) :
      join(...sequence(new guide3(int i) {return f(a+(i/n)*width);},n+1));
  };
}

multigraph graph(interpolate3 join, bool3 cond(real))
{
  return new guide3[](triple f(real), real a, real b, int n) {
    real width=b-a;
    if(n == 0) return new guide3[] {join(cond(a) ? f(a) : nullpath3)};
    guide3[] G;
    guide3[] g;
    for(int i=0; i < n+1; ++i) {
      real t=a+(i/n)*width;
      bool3 b=cond(t);
      if(b)
        g.push(f(t));
      else {
        if(g.length > 0) {
          G.push(join(...g));
          g=new guide3[] {};
        }
        if(b == default)
          g.push(f(t));
      }
    }
    if(g.length > 0)
      G.push(join(...g));
    return G;
  };
}

guide3 Straight(... guide3[])=operator --;
guide3 Spline(... guide3[])=operator ..;
                       
guide3 graph(picture pic=currentpicture, real x(real), real y(real),
             real z(real), real a, real b, int n=ngraph,
             interpolate3 join=operator --)
{
  return graph(join)(new triple(real t) {return Scale(pic,(x(t),y(t),z(t)));},
                     a,b,n);
}

guide3[] graph(picture pic=currentpicture, real x(real), real y(real),
               real z(real), real a, real b, int n=ngraph,
               bool3 cond(real), interpolate3 join=operator --)
{
  return graph(join,cond)(new triple(real t) {
      return Scale(pic,(x(t),y(t),z(t)));
    },a,b,n);
}

guide3 graph(picture pic=currentpicture, triple v(real), real a, real b,
             int n=ngraph, interpolate3 join=operator --)
{
  return graph(join)(new triple(real t) {return Scale(pic,v(t));},a,b,n);
}

guide3[] graph(picture pic=currentpicture, triple v(real), real a, real b,
               int n=ngraph, bool3 cond(real), interpolate3 join=operator --)
{
  return graph(join,cond)(new triple(real t) {
      return Scale(pic,v(t));
    },a,b,n);
}

guide3 graph(picture pic=currentpicture, triple[] v,
             interpolate3 join=operator --)
{
  int i=0;
  return graph(join)(new triple(real) {
      triple w=Scale(pic,v[i]);
      ++i;
      return w;
    },0,0,v.length-1);
}

guide3[] graph(picture pic=currentpicture, triple[] v, bool3[] cond,
               interpolate3 join=operator --)
{
  int n=v.length;
  int i=0;
  triple w;
  checkconditionlength(cond.length,n);
  bool3 condition(real) {
    bool b=cond[i];
    if(b) w=Scale(pic,v[i]);
    ++i;
    return b;
  }
  return graph(join,condition)(new triple(real) {return w;},0,0,n-1);
}

guide3 graph(picture pic=currentpicture, real[] x, real[] y, real[] z,
             interpolate3 join=operator --)
{
  int n=x.length;
  checklengths(n,y.length);
  checklengths(n,z.length);
  int i=0;
  return graph(join)(new triple(real) {
      triple w=Scale(pic,(x[i],y[i],z[i]));
      ++i;
      return w;
    },0,0,n-1);
}

guide3[] graph(picture pic=currentpicture, real[] x, real[] y, real[] z,
               bool3[] cond, interpolate3 join=operator --)
{
  int n=x.length;
  checklengths(n,y.length);
  checklengths(n,z.length);
  int i=0;
  triple w;
  checkconditionlength(cond.length,n);
  bool3 condition(real) {
    bool3 b=cond[i];
    if(b != false) w=Scale(pic,(x[i],y[i],z[i]));
    ++i;
    return b;
  }
  return graph(join,condition)(new triple(real) {return w;},0,0,n-1);
}

// The graph of a function along a path.
guide3 graph(triple F(path, real), path p, int n=1,
             interpolate3 join=operator --)
{
  guide3 g=join(...sequence(new guide3(int i) {
        return F(p,i/n);
      },n*length(p)));
  return cyclic(p) ? join(g,cycle) : join(g,F(p,length(p)));
}

guide3 graph(triple F(pair), path p, int n=1, interpolate3 join=operator --)
{
  return graph(new triple(path p, real position) 
               {return F(point(p,position));},p,n,join);
}

guide3 graph(picture pic=currentpicture, real f(pair), path p, int n=1,
             interpolate3 join=operator --) 
{
  return graph(new triple(pair z) {return Scale(pic,(z.x,z.y,f(z)));},p,n,
               join);
}

guide3 graph(real f(pair), path p, int n=1, real T(pair),
             interpolate3 join=operator --)
{
  return graph(new triple(pair z) {pair w=T(z); return (w.x,w.y,f(w));},p,n,
               join);
}

// Connect points in v into segments corresponding to consecutive true elements
// of b using interpolation operator join. 
path3[] segment(triple[] v, bool[] cond, interpolate3 join=operator --)
{
  checkconditionlength(cond.length,v.length);
  int[][] segment=segment(cond);
  return sequence(new path3(int i) {return join(...v[segment[i]]);},
                  segment.length);
}

bool uperiodic(real[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length;
  real[] a0=a[0];
  real[] a1=a[n-1];
  for(int j=0; j < m; ++j) {
    real norm=0;
    for(int i=0; i < n; ++i)
      norm=max(norm,abs(a[i][j]));
    real epsilon=sqrtEpsilon*norm;
    if(abs(a0[j]-a1[j]) > epsilon) return false;
  }
  return true;
}
bool vperiodic(real[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length-1;
  for(int i=0; i < n; ++i) {
    real[] ai=a[i];
    real epsilon=sqrtEpsilon*norm(ai);
    if(abs(ai[0]-ai[m]) > epsilon) return false;
  }
  return true;
}

bool uperiodic(triple[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length;
  triple[] a0=a[0];
  triple[] a1=a[n-1];
  real epsilon=sqrtEpsilon*norm(a);
  for(int j=0; j < m; ++j)
    if(abs(a0[j]-a1[j]) > epsilon) return false;
  return true;
}
bool vperiodic(triple[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length-1;
  real epsilon=sqrtEpsilon*norm(a);
  for(int i=0; i < n; ++i)
    if(abs(a[i][0]-a[i][m]) > epsilon) return false;
  return true;
}

// return the surface described by a matrix f
surface surface(triple[][] f, bool[][] cond={})
{
  if(!rectangular(f)) abort("matrix is not rectangular");
  
  int nx=f.length-1;
  int ny=nx > 0 ? f[0].length-1 : 0;
  
  bool all=cond.length == 0;

  int count;
  if(all)
    count=nx*ny;
  else {
    count=0;
    for(int i=0; i < nx; ++i) {
      bool[] condi=cond[i];
      bool[] condp=cond[i+1];
      for(int j=0; j < ny; ++j)
        if(condi[j] && condi[j+1] && condp[j] && condp[j+1]) ++count;
    }
  }

  surface s=surface(count);
  s.index=new int[nx][ny];
  int k=-1;
  for(int i=0; i < nx; ++i) {
    bool[] condi,condp;
    if(!all) {
      condi=cond[i];
      condp=cond[i+1];
    }
    triple[] fi=f[i];
    triple[] fp=f[i+1];
    int[] indexi=s.index[i];
    for(int j=0; j < ny; ++j) {
      if(all || (condi[j] && condi[j+1] && condp[j] && condp[j+1]))
        s.s[++k]=patch(new triple[] {fi[j],fp[j],fp[j+1],fi[j+1]});
      indexi[j]=k;
    }
  }

  if(count == nx*ny) {
    if(uperiodic(f)) s.ucyclic(true);
    if(vperiodic(f)) s.vcyclic(true);
  }

  return s;
}

surface bispline(real[][] z, real[][] p, real[][] q, real[][] r,
                 real[] x, real[] y, bool[][] cond={})
{ // z[i][j] is the value at (x[i],y[j])
  // p and q are the first derivatives with respect to x and y, respectively
  // r is the second derivative ddu/dxdy
  int n=x.length-1;
  int m=y.length-1;

  bool all=cond.length == 0;

  int count;
  if(all)
    count=n*m;
  else {
    count=0;
    for(int i=0; i < n; ++i) {
      bool[] condi=cond[i];
      for(int j=0; j < m; ++j)
        if(condi[j]) ++count;
    }
  }

  surface s=surface(count);
  s.index=new int[n][m];
  int k=0;
  for(int i=0; i < n; ++i) {
    int ip=i+1;
    real xi=x[i];
    real xp=x[ip];
    real x1=interp(xi,xp,1/3);
    real x2=interp(xi,xp,2/3);
    real hx=x1-xi;
    real[] zi=z[i];
    real[] zp=z[ip];
    real[] ri=r[i];
    real[] rp=r[ip];
    real[] pi=p[i];
    real[] pp=p[ip];
    real[] qi=q[i];
    real[] qp=q[ip];
    int[] indexi=s.index[i];
    bool[] condi=all ? null : cond[i];
    for(int j=0; j < m; ++j) {
      if(all || condi[j]) {
        real yj=y[j];
        int jp=j+1;
        real yp=y[jp];
        real y1=interp(yj,yp,1/3);
        real y2=interp(yj,yp,2/3);
        real hy=y1-yj;
        real hxy=hx*hy;
        real zij=zi[j];
        real zip=zi[jp];
        real zpj=zp[j];
        real zpp=zp[jp];
        real pij=hx*pi[j];
        real ppj=hx*pp[j];
        real qip=hy*qi[jp];
        real qpp=hy*qp[jp];
        real zippip=zip+hx*pi[jp];
        real zppmppp=zpp-hx*pp[jp];
        real zijqij=zij+hy*qi[j];
        real zpjqpj=zpj+hy*qp[j];
        
        s.s[k]=patch(new triple[][] {
          {(xi,yj,zij),(xi,y1,zijqij),(xi,y2,zip-qip),(xi,yp,zip)},
          {(x1,yj,zij+pij),(x1,y1,zijqij+pij+hxy*ri[j]),
           (x1,y2,zippip-qip-hxy*ri[jp]),(x1,yp,zippip)},
          {(x2,yj,zpj-ppj),(x2,y1,zpjqpj-ppj-hxy*rp[j]),
           (x2,y2,zppmppp-qpp+hxy*rp[jp]),(x2,yp,zppmppp)},
          {(xp,yj,zpj),(xp,y1,zpjqpj),(xp,y2,zpp-qpp),(xp,yp,zpp)}},copy=false);
        indexi[j]=k;
        ++k;
      }
    }
  }

  return s;
}

private real[][][] bispline0(real[][] z, real[][] p, real[][] q, real[][] r,
                             real[] x, real[] y, bool[][] cond={})
{ // z[i][j] is the value at (x[i],y[j])
  // p and q are the first derivatives with respect to x and y, respectively
  // r is the second derivative ddu/dxdy
  int n=x.length-1;
  int m=y.length-1;

  bool all=cond.length == 0;

  int count;
  if(all)
    count=n*m;
  else {
    count=0;
    for(int i=0; i < n; ++i) {
      bool[] condi=cond[i];
      bool[] condp=cond[i+1];
      for(int j=0; j < m; ++j)
        if(all || (condi[j] && condi[j+1] && condp[j] && condp[j+1]))
          ++count;
    }
  }

  real[][][] s=new real[count][][];
  int k=0;
  for(int i=0; i < n; ++i) {
    int ip=i+1;
    real xi=x[i];
    real xp=x[ip];
    real hx=(xp-xi)/3;
    real[] zi=z[i];
    real[] zp=z[ip];
    real[] ri=r[i];
    real[] rp=r[ip];
    real[] pi=p[i];
    real[] pp=p[ip];
    real[] qi=q[i];
    real[] qp=q[ip];
    bool[] condi=all ? null : cond[i];
    bool[] condp=all ? null : cond[i+1];
    for(int j=0; j < m; ++j) {
      if(all || (condi[j] && condi[j+1] && condp[j] && condp[j+1])) {
        real yj=y[j];
        int jp=j+1;
        real yp=y[jp];
        real hy=(yp-yj)/3;
        real hxy=hx*hy;
        real zij=zi[j];
        real zip=zi[jp];
        real zpj=zp[j];
        real zpp=zp[jp];
        real pij=hx*pi[j];
        real ppj=hx*pp[j];
        real qip=hy*qi[jp];
        real qpp=hy*qp[jp];
        real zippip=zip+hx*pi[jp];
        real zppmppp=zpp-hx*pp[jp];
        real zijqij=zij+hy*qi[j];
        real zpjqpj=zpj+hy*qp[j];

        s[k]=new real[][] {{zij,zijqij,zip-qip,zip},
                           {zij+pij,zijqij+pij+hxy*ri[j],
                            zippip-qip-hxy*ri[jp],zippip},
                           {zpj-ppj,zpjqpj-ppj-hxy*rp[j],
                            zppmppp-qpp+hxy*rp[jp],zppmppp},
                           {zpj,zpjqpj,zpp-qpp,zpp}};
        ++k;
      }
    }
  }

  return s;
}

// return the surface values described by a real matrix f, interpolated with
// xsplinetype and ysplinetype.
real[][][] bispline(real[][] f, real[] x, real[] y,
                    splinetype xsplinetype=null,
                    splinetype ysplinetype=xsplinetype, bool[][] cond={})
{
  real epsilon=sqrtEpsilon*norm(y);
  if(xsplinetype == null)
    xsplinetype=(abs(x[0]-x[x.length-1]) <= epsilon) ? periodic : notaknot;
  if(ysplinetype == null)
    ysplinetype=(abs(y[0]-y[y.length-1]) <= epsilon) ? periodic : notaknot;
  int n=x.length; int m=y.length;
  real[][] ft=transpose(f);
  real[][] tp=new real[m][];
  for(int j=0; j < m; ++j)
    tp[j]=xsplinetype(x,ft[j]);
  real[][] q=new real[n][];
  for(int i=0; i < n; ++i)
    q[i]=ysplinetype(y,f[i]);
  real[][] qt=transpose(q);
  real[] d1=xsplinetype(x,qt[0]);
  real[] d2=xsplinetype(x,qt[m-1]);
  real[][] r=new real[n][];
  real[][] p=transpose(tp);
  for(int i=0; i < n; ++i)
    r[i]=clamped(d1[i],d2[i])(y,p[i]);
  return bispline0(f,p,q,r,x,y,cond);
}

// return the surface described by a real matrix f, interpolated with
// xsplinetype and ysplinetype.
surface surface(real[][] f, real[] x, real[] y,
                splinetype xsplinetype=null, splinetype ysplinetype=xsplinetype,
                bool[][] cond={})
{
  real epsilon=sqrtEpsilon*norm(y);
  if(xsplinetype == null)
    xsplinetype=(abs(x[0]-x[x.length-1]) <= epsilon) ? periodic : notaknot;
  if(ysplinetype == null)
    ysplinetype=(abs(y[0]-y[y.length-1]) <= epsilon) ? periodic : notaknot;
  int n=x.length; int m=y.length;
  real[][] ft=transpose(f);
  real[][] tp=new real[m][];
  for(int j=0; j < m; ++j)
    tp[j]=xsplinetype(x,ft[j]);
  real[][] q=new real[n][];
  for(int i=0; i < n; ++i)
    q[i]=ysplinetype(y,f[i]);
  real[][] qt=transpose(q);
  real[] d1=xsplinetype(x,qt[0]);
  real[] d2=xsplinetype(x,qt[m-1]);
  real[][] r=new real[n][];
  real[][] p=transpose(tp);
  for(int i=0; i < n; ++i)
    r[i]=clamped(d1[i],d2[i])(y,p[i]);
  surface s=bispline(f,p,q,r,x,y,cond);
  if(xsplinetype == periodic) s.ucyclic(true);
  if(ysplinetype == periodic) s.vcyclic(true);
  return s;
}

// return the surface described by a real matrix f, interpolated with
// xsplinetype and ysplinetype.
surface surface(real[][] f, pair a, pair b, splinetype xsplinetype,
                splinetype ysplinetype=xsplinetype, bool[][] cond={})
{
  if(!rectangular(f)) abort("matrix is not rectangular");

  int nx=f.length-1;
  int ny=nx > 0 ? f[0].length-1 : 0;

  if(nx == 0 || ny == 0) return nullsurface;

  real[] x=uniform(a.x,b.x,nx);
  real[] y=uniform(a.y,b.y,ny);
  return surface(f,x,y,xsplinetype,ysplinetype,cond);
}

// return the surface described by a real matrix f, interpolated linearly.
surface surface(real[][] f, pair a, pair b, bool[][] cond={})
{
  if(!rectangular(f)) abort("matrix is not rectangular");

  int nx=f.length-1;
  int ny=nx > 0 ? f[0].length-1 : 0;

  if(nx == 0 || ny == 0) return nullsurface;

  bool all=cond.length == 0;

  triple[][] v=new triple[nx+1][ny+1];
  for(int i=0; i <= nx; ++i) {
    real x=interp(a.x,b.x,i/nx);
    bool[] condi=all ? null : cond[i];
    triple[] vi=v[i];
    real[] fi=f[i];
    for(int j=0; j <= ny; ++j)
      if(all || condi[j])
        vi[j]=(x,interp(a.y,b.y,j/ny),fi[j]);
  }
  return surface(v,cond);
}

// return the surface described by a parametric function f over box(a,b),
// interpolated linearly.
surface surface(triple f(pair z), pair a, pair b, int nu=nmesh, int nv=nu,
                bool cond(pair z)=null)
{
  if(nu <= 0 || nv <= 0) return nullsurface;

  bool[][] active;
  bool all=cond == null;
  if(!all) active=new bool[nu+1][nv+1];

  real du=1/nu;
  real dv=1/nv;
  pair Idv=(0,dv);
  pair dz=(du,dv);

  triple[][] v=new triple[nu+1][nv+1];

  for(int i=0; i <= nu; ++i) {
    real x=interp(a.x,b.x,i*du);
    bool[] activei=all ? null : active[i];
    triple[] vi=v[i];
    for(int j=0; j <= nv; ++j) {
      pair z=(x,interp(a.y,b.y,j*dv));
      if(all || (activei[j]=cond(z))) vi[j]=f(z);
    }
  }
  return surface(v,active);
}
  
// return the surface described by a parametric function f evaluated at u and v
// and interpolated with usplinetype and vsplinetype.
surface surface(triple f(pair z), real[] u, real[] v,
                splinetype[] usplinetype, splinetype[] vsplinetype=Spline,
                bool cond(pair z)=null)
{
  int nu=u.length-1;
  int nv=v.length-1;
  real[] ipt=sequence(u.length);
  real[] jpt=sequence(v.length);
  real[][] fx=new real[u.length][v.length];
  real[][] fy=new real[u.length][v.length];
  real[][] fz=new real[u.length][v.length];

  bool[][] active;
  bool all=cond == null;
  if(!all) active=new bool[u.length][v.length];

  for(int i=0; i <= nu; ++i) {
    real ui=u[i];
    real[] fxi=fx[i];
    real[] fyi=fy[i];
    real[] fzi=fz[i];
    bool[] activei=all ? null : active[i];
    for(int j=0; j <= nv; ++j) {
      pair z=(ui,v[j]);
      if(!all) activei[j]=cond(z);
      triple f=f(z);
      fxi[j]=f.x;
      fyi[j]=f.y;
      fzi[j]=f.z;
    }
  }

  if(usplinetype.length == 0) {
    usplinetype=new splinetype[] {uperiodic(fx) ? periodic : notaknot,
                                  uperiodic(fy) ? periodic : notaknot,
                                  uperiodic(fz) ? periodic : notaknot};
  } else if(usplinetype.length != 3) abort("usplinetype must have length 3");

  if(vsplinetype.length == 0) {
    vsplinetype=new splinetype[] {vperiodic(fx) ? periodic : notaknot,
                                  vperiodic(fy) ? periodic : notaknot,
                                  vperiodic(fz) ? periodic : notaknot};
  } else if(vsplinetype.length != 3) abort("vsplinetype must have length 3");

  real[][][] sx=bispline(fx,ipt,jpt,usplinetype[0],vsplinetype[0],active);
  real[][][] sy=bispline(fy,ipt,jpt,usplinetype[1],vsplinetype[1],active);
  real[][][] sz=bispline(fz,ipt,jpt,usplinetype[2],vsplinetype[2],active);

  surface s=surface(sx.length);
  s.index=new int[nu][nv];
  int k=-1;
  for(int i=0; i < nu; ++i) {
    int[] indexi=s.index[i];
    for(int j=0; j < nv; ++j)
      indexi[j]=++k;
  }

  for(int k=0; k < sx.length; ++k) {
    triple[][] Q=new triple[4][];
    real[][] Px=sx[k];
    real[][] Py=sy[k];
    real[][] Pz=sz[k];
    for(int i=0; i < 4 ; ++i) {
      real[] Pxi=Px[i];
      real[] Pyi=Py[i];
      real[] Pzi=Pz[i];
      Q[i]=new triple[] {(Pxi[0],Pyi[0],Pzi[0]),
                         (Pxi[1],Pyi[1],Pzi[1]),
                         (Pxi[2],Pyi[2],Pzi[2]),
                         (Pxi[3],Pyi[3],Pzi[3])};
    }
    s.s[k]=patch(Q);
  }

  if(usplinetype[0] == periodic && usplinetype[1] == periodic &&
     usplinetype[1] == periodic) s.ucyclic(true);

  if(vsplinetype[0] == periodic && vsplinetype[1] == periodic &&
     vsplinetype[1] == periodic) s.vcyclic(true);

  return s;
}

// return the surface described by a parametric function f over box(a,b),
// interpolated with usplinetype and vsplinetype.
surface surface(triple f(pair z), pair a, pair b, int nu=nmesh, int nv=nu,
                splinetype[] usplinetype, splinetype[] vsplinetype=Spline,
                bool cond(pair z)=null)
{
  return surface(f,uniform(a.x,b.x,nu),uniform(a.y,b.y,nv),
                 usplinetype,vsplinetype,cond);
}

// return the surface described by a real function f over box(a,b),
// interpolated linearly.
surface surface(real f(pair z), pair a, pair b, int nx=nmesh, int ny=nx,
                bool cond(pair z)=null)
{
  return surface(new triple(pair z) {return (z.x,z.y,f(z));},a,b,nx,ny,cond);
}

// return the surface described by a real function f over box(a,b),
// interpolated with xsplinetype and ysplinetype.
surface surface(real f(pair z), pair a, pair b, int nx=nmesh, int ny=nx,
                splinetype xsplinetype, splinetype ysplinetype=xsplinetype,
                bool cond(pair z)=null)
{
  bool[][] active;
  bool all=cond == null;
  if(!all) active=new bool[nx+1][ny+1];

  real dx=1/nx;
  real dy=1/ny;
  pair Idy=(0,dy);
  pair dz=(dx,dy);

  real[][] F=new real[nx+1][ny+1];
  real[] x=uniform(a.x,b.x,nx);
  real[] y=uniform(a.y,b.y,ny);
  for(int i=0; i <= nx; ++i) {
    bool[] activei=all ? null : active[i];
    real[] Fi=F[i];
    real x=x[i];
    for(int j=0; j <= ny; ++j) {
      pair z=(x,y[j]);
      Fi[j]=f(z);
      if(!all) activei[j]=cond(z);
    }
  }
  return surface(F,x,y,xsplinetype,ysplinetype,active);
}

guide3[][] lift(real f(real x, real y), guide[][] g,
                interpolate3 join=operator --)
{
  guide3[][] G=new guide3[g.length][];
  for(int cnt=0; cnt < g.length; ++cnt) {
    guide[] gcnt=g[cnt];
    guide3[] Gcnt=new guide3[gcnt.length];
    for(int i=0; i < gcnt.length; ++i) {
      guide gcnti=gcnt[i];
      guide3 Gcnti=join(...sequence(new guide3(int j) {
            pair z=point(gcnti,j);
            return (z.x,z.y,f(z.x,z.y));
          },size(gcnti)));
      if(cyclic(gcnti)) Gcnti=Gcnti..cycle;
      Gcnt[i]=Gcnti;
    }
    G[cnt]=Gcnt;
  }
  return G;
}

guide3[][] lift(real f(pair z), guide[][] g, interpolate3 join=operator --)
{
  return lift(new real(real x, real y) {return f((x,y));},g,join);
}

void draw(picture pic=currentpicture, Label[] L=new Label[],
          guide3[][] g, pen[] p, light light=currentlight, string name="",
          render render=defaultrender,
          interaction interaction=LabelInteraction())
{
  pen thin=is3D() ? thin() : defaultpen;
  bool group=g.length > 1 && (name != "" || render.defaultnames);
  if(group)
    begingroup3(pic,name == "" ? "contours" : name,render);
  for(int cnt=0; cnt < g.length; ++cnt) {
    guide3[] gcnt=g[cnt];
    pen pcnt=thin+p[cnt];
    for(int i=0; i < gcnt.length; ++i)
      draw(pic,gcnt[i],pcnt,light,name);
    if(L.length > 0) {
      Label Lcnt=L[cnt];
      for(int i=0; i < gcnt.length; ++i) {
        if(Lcnt.s != "" && size(gcnt[i]) > 1)
          label(pic,Lcnt,gcnt[i],pcnt,name,interaction);
      }
    }
  }
  if(group)
    endgroup3(pic);
}

void draw(picture pic=currentpicture, Label[] L=new Label[],
          guide3[][] g, pen p=currentpen, light light=currentlight,
          string name="", render render=defaultrender,
          interaction interaction=LabelInteraction())
{
  draw(pic,L,g,sequence(new pen(int) {return p;},g.length),light,name,
       render,interaction);
}

real maxlength(triple f(pair z), pair a, pair b, int nu, int nv) 
{
  return min(abs(f((b.x,a.y))-f(a))/nu,abs(f((a.x,b.y))-f(a))/nv);
}

// return a vector field on a parametric surface f over box(a,b).
picture vectorfield(path3 vector(pair v), triple f(pair z), pair a, pair b,
                    int nu=nmesh, int nv=nu, bool truesize=false,
                    real maxlength=truesize ? 0 : maxlength(f,a,b,nu,nv),
                    bool cond(pair z)=null, pen p=currentpen,
                    arrowbar3 arrow=Arrow3, margin3 margin=PenMargin3,
                    string name="", render render=defaultrender)
{
  picture pic;
  real du=1/nu;
  real dv=1/nv;
  bool all=cond == null;
  real scale;

  if(maxlength > 0) {
    real size(pair z) {
      path3 g=vector(z);
      return abs(point(g,size(g)-1)-point(g,0));
    }
    real max=size((0,0));
    for(int i=0; i <= nu; ++i) {
      real x=interp(a.x,b.x,i*du);
      for(int j=0; j <= nv; ++j)
        max=max(max,size((x,interp(a.y,b.y,j*dv))));
    }
    scale=max > 0 ? maxlength/max : 1;
  } else scale=1;

  bool group=name != "" || render.defaultnames;
  if(group)
    begingroup3(pic,name == "" ? "vectorfield" : name,render);
  for(int i=0; i <= nu; ++i) {
    real x=interp(a.x,b.x,i*du);
    for(int j=0; j <= nv; ++j) {
      pair z=(x,interp(a.y,b.y,j*dv));
      if(all || cond(z)) {
        path3 g=scale3(scale)*vector(z);
        string name="vector";
        if(truesize) {
          picture opic;
          draw(opic,g,p,arrow,margin,name,render);
          add(pic,opic,f(z));
        } else
          draw(pic,shift(f(z))*g,p,arrow,margin,name,render);
      }
    }
  }
  if(group)
    endgroup3(pic);
  return pic;
}

triple polar(real r, real theta, real phi)
{
  return r*expi(theta,phi);
}

guide3 polargraph(real r(real,real), real theta(real), real phi(real),
                  int n=ngraph, interpolate3 join=operator --)
{
  return graph(join)(new triple(real t) {
      return polar(r(theta(t),phi(t)),theta(t),phi(t));
    },0,1,n);
}

// True arc
path3 Arc(triple c, triple v1, triple v2, triple normal=O, bool direction=CCW,
          int n=nCircle)
{
  v1 -= c;
  real r=abs(v1);
  v1=unit(v1);
  v2=unit(v2-c);

  if(normal == O) {
    normal=cross(v1,v2);
    if(normal == O) abort("explicit normal required for these endpoints");
  }

  transform3 T=align(unit(normal));
  transform3 Tinv=transpose(T);
  v1=Tinv*v1;
  v2=Tinv*v2;

  real fuzz=sqrtEpsilon*max(abs(v1),abs(v2));
  if(abs(v1.z) > fuzz || abs(v2.z) > fuzz)
    abort("invalid normal vector");

  real phi1=radians(longitude(v1,warn=false));
  real phi2=radians(longitude(v2,warn=false));
  if(direction) {
    if(phi1 >= phi2) phi1 -= 2pi;
  } else if(phi2 >= phi1) phi2 -= 2pi;

  static real piby2=pi/2;
  return shift(c)*T*polargraph(new real(real theta, real phi) {return r;},
                               new real(real t) {return piby2;},
                               new real(real t) {return interp(phi1,phi2,t);},
                               n,operator ..);
}

path3 Arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
          triple normal=O, bool direction, int n=nCircle)
{
  return Arc(c,c+r*dir(theta1,phi1),c+r*dir(theta2,phi2),normal,direction,n);
}

path3 Arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
          triple normal=O, int n=nCircle)
{
  return Arc(c,r,theta1,phi1,theta2,phi2,normal,
             theta2 > theta1 || (theta2 == theta1 && phi2 >= phi1) ? CCW : CW,
             n);
}

// True circle
path3 Circle(triple c, real r, triple normal=Z, int n=nCircle)
{
  static real piby2=pi/2;
  return shift(c)*align(unit(normal))*
    polargraph(new real(real theta, real phi) {return r;},
               new real(real t) {return piby2;},
               new real(real t) {return interp(0,2pi,t);},n,operator ..);

}
