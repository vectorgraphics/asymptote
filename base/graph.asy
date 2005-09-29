import math;

static public real ticksize=1mm;
static public real Ticksize=2*ticksize;
static public real ylabelwidth=2.0;
static public real axislabelmargin=1;
static public real axislabelfactor=1.5;
static public real axiscoverage=0.9;
static public int ngraph=100;

static public real epsilon=10*realEpsilon();

static bool Crop=true;
static bool NoCrop=false;

static scaleT Linear;

static scaleT Log;
static scaleT Logarithmic;
Log.init(log10,pow10,logarithmic=true);
Logarithmic=Log;

// A linear scale, with optional autoscaling of minimum and maximum values,
// scaling factor s and intercept.
public scaleT Linear(bool automin=true, bool automax=true, real s=1,
		     real intercept=0)
{
  real sinv=1/s;
  scaleT scale;
  real T(real x) {return (x-intercept)*s;}
  real Tinv(real x) {return x*sinv+intercept;}
  scale.init(T,Tinv,logarithmic=false,automin,automax);
  return scale;
}

// A logarithmic scale, with optional autoscaling of minimum and maximum
// values.
public scaleT Log(bool automin=true, bool automax=true)
{
  scaleT scale;
  scale.init(Log.T,Log.Tinv,logarithmic=true,automin,automax);
  return scale;
}

void scale(picture pic=currentpicture, scaleT x, scaleT y, scaleT z=Linear)
{
  pic.scale.x.scale=x;
  pic.scale.y.scale=y;
  pic.scale.z.scale=z;
}

void scale(picture pic=currentpicture, bool autoscale=true)
{
  scale(pic,Linear(autoscale,autoscale),Linear(autoscale,autoscale),
	Linear(autoscale,autoscale));
}

struct scientific 
{
  public int sign;
  public real mantissa;
  public int exponent;
  int ceil() {return sign*ceil(mantissa);}
  real scale(real x, real exp) {return exp > 0 ? x/10^exp : x*10^-exp;}
  real ceil(real x, real exp) {return ceil(sign*scale(abs(x),exp));}
  real floor(real x, real exp) {return floor(sign*scale(abs(x),exp));}
}

scientific operator init() {return new scientific;}
  
// Convert x to scientific notation
scientific scientific(real x) 
{
    scientific s;
    s.sign=sgn(x);
    x=abs(x);
    if(x == 0) {s.mantissa=0; s.exponent=-intMax(); return s;}
    real logx=log10(x);
    s.exponent=floor(logx);
    s.mantissa=s.scale(x,s.exponent);
    return s;
}

// Autoscale limits and tick divisor.
struct bounds {
  public real min;
  public real max;
  // Possible tick intervals;
  public int[] divisor;
}

bounds operator init() {return new bounds;}
  
// Compute tick divisors.
int[] divisors(int a, int b)
{
  int[] dlist;
  int n=b-a;
  dlist[0]=1;
  if(n == 1) {dlist[1]=10; dlist[2]=100; return dlist;}
  if(n == 2) {dlist[1]=2; return dlist;}
  int sqrtn=floor(sqrt(n));
  int i=0;
  for(int d=2; d <= sqrtn; ++d)
    if(n % d == 0 && (a*b >= 0 || b % (n/d) == 0)) dlist[++i]=d;
  for(int d=sqrtn; d >= 1; --d)
    if(n % d == 0 && (a*b >= 0 || b % d == 0)) dlist[++i]=quotient(n,d);
  return dlist;
}

real upscale(real b, real a)
{
  if(b <= 5) b=5; 
  else if (b > 10 && a >= 0 && b <= 12) b=12;
  else if (b > 10 && (a >= 0 || 15 % -a == 0) && b <= 15) b=15;
  else b=ceil(b/10)*10;
  return b;
}

// Compute autoscale limits and tick divisor.
bounds autoscale(real Min, real Max, scaleT scale=Linear)
{
  bounds m;
  if(scale.logarithmic) {
    m.min=floor(Min);
    m.max=ceil(Max);
    return m;
  }
  if(!(finite(Min) && finite(Max)))
    abort("autoscale requires finite limits");
  Min=scale.Tinv(Min);
  Max=scale.Tinv(Max);
  m.min=Min;
  m.max=Max;
  if(Min > Max) {real temp=Min; Min=Max; Max=temp;}
  if(Min == Max) {
    if(Min == 0) {m.max=1; return m;}
    if(Min > 0) {Min=0; Max *= 2;}
    else {Min *= 2; Max=0;}
  }
  
  int sign;
  if(Min < 0 && Max <= 0) {real temp=-Min; Min=-Max; Max=temp; sign=-1;}
  else sign=1;
  scientific sa=scientific(Min);
  scientific sb=scientific(Max);
  int exp=max(sa.exponent,sb.exponent);
  real a=sa.floor(Min,exp);
  real b=sb.ceil(Max,exp);
  if(sb.mantissa <= 1.5) {
    --exp;
    a=sa.floor(Min,exp);
    b=sb.ceil(Max,exp);
  }
  
  real bsave=b;
  if(b-a > (a >= 0 ? 8 : 6)) {
    b=upscale(b,a);
    if(a >= 0) {
      if(a <= 5) a=0; else a=floor(a/10)*10;
    } else a=-upscale(-a,-1);
  }
  
  // Redo b in case the value of a has changed
  if(bsave-a > (a >= 0 ? 8 : 6))
    b=upscale(bsave,a);
  
  if(sign == -1) {real temp=-a; a=-b; b=temp;}
  real Scale=10.0^exp;
  m.min=scale.T(a*Scale);
  m.max=scale.T(b*Scale);
  if(m.min > m.max) {real temp=m.min; m.min=m.max; m.max=temp;}
  m.divisor=divisors(round(a),round(b));
  return m;
}

typedef string ticklabel(real);

ticklabel ticklabel(string s) {
  return new string(real x) {return format(s,x);};
}

ticklabel LogFormat=new string(real x) {
  return format("$10^{%d}$",round(x));
};

// The default direction specifier.
pair zero(real) {return 0;}

struct ticklocate {
  public real a,b;	    // Tick values at point(g,0), point(g,length(g)).
  public autoscaleT S;	    // Autoscaling transformation.
  public real time(real v); // Returns the time corresponding to the value v. 
  public pair dir(real t);  // Returns the absolute tick direction as a
                            // function of t (zero means perpendicular).
  ticklocate copy() {
    ticklocate T=new ticklocate;
    T.a=a;
    T.b=b;
    T.S=S.copy();
    T.time=time;
    T.dir=dir;
    return T;
  }
}

ticklocate operator init() {return new ticklocate;}

autoscaleT defaultS;
  
typedef real valuetime(real);
real valuetime(real x) {return 0;}

valuetime linear(picture pic=currentpicture, scalefcn S=identity,
		 real Min, real Max)
{
  real factor=Max == Min ? 0.0 : 1.0/(Max-Min);
  return new real(real v) {return (S(v)-Min)*factor;};
}

ticklocate ticklocate(real a, real b, autoscaleT S=defaultS,
		      real tickmin=-infinity, real tickmax=infinity,
		      real time(real)=null, pair dir(real)=zero) 
{
  if(time == null) time=linear(S.T(),a,b);
  ticklocate locate;
  locate.a=a;
  locate.b=b;
  locate.S=S.copy();
  if(finite(tickmin)) locate.S.tickMin=tickmin;
  if(finite(tickmax)) locate.S.tickMax=tickmax;
  locate.time=time;
  locate.dir=dir;
  return locate;
}
			     
private struct locateT {
  real t;       // tick location time
  pair Z;       // tick location in frame coordinates
  pair pathdir; // path direction in frame coordinates
  pair dir;     // tick direction in frame coordinates
  
  void dir(transform T, guide g, ticklocate locate, real t) {
    pathdir=unit(T*dir(g,t));
    pair Dir=locate.dir(t);
    dir=Dir == 0 ? -I*pathdir : unit(Dir);
  }
// Locate the desired position of a tick along a path.
  void calc(transform T, guide g, ticklocate locate, real val) {
    t=locate.time(val);
    Z=T*point(g,t);
    dir(T,g,locate,t);
  }
}

locateT operator init() {return new locateT;}
  
pair ticklabelshift(pair align, pen p=currentpen) 
{
  return 0.25*unit(align)*labelmargin(p);
}

void drawtick(frame f, transform T, path g, path g2, ticklocate locate,
	      real val, real Size, int sign, pen p, bool extend)
{
  locateT locate1,locate2;
  locate1.calc(T,g,locate,val);
  if(extend && size(g2) > 0) {
    locate2.calc(T,g2,locate,val);
    draw(f,locate1.Z--locate2.Z,p);
  } else
    draw(f,locate1.Z--locate1.Z+Size*sign*locate1.dir,p);
}

// Label a tick on a frame.
pair labeltick(frame d, transform T, guide g, ticklocate locate, real val,
	       pair side, int sign, real Size,  ticklabel ticklabel, Label F,
	       real norm=0)
{
  locateT locate1;
  locate1.calc(T,g,locate,val);
  pair rot=dir(-F.angle);
  pair align=rot*side*locate1.dir;
  pair shift=dot(align,-sign*locate1.dir) < 0 ? align*Size :
    ticklabelshift(align,F.p);
  pair Z=locate1.Z+shift;
  if(abs(val) < epsilon*norm) val=0;
  pair perp=I*locate1.pathdir;
  // Adjust tick label alignment
  pair adjust=unit(align+0.5*unit(perp*sgn(dot(align,perp))));
  // Project align onto adjusted direction.
  align=adjust*dot(align,adjust);
  string s=ticklabel(val);
  s=baseline(s,align,"$10^4$");
  label(d,rotate(F.angle)*s,Z,align,F.p);
  return locate1.pathdir;
}  

// Add axis label L to frame f.
void labelaxis(frame f, transform T, Label L, guide g, 
	       ticklocate locate=null, int sign=1, bool ticklabels=false)
{
  Label L0=L.copy();
  real t=L0.relative(g);
  pair z=point(g,t);
  pair dir=dir(g,t);
  if(locate != null) {
    locateT locate1;
    locate1.dir(T,g,locate,t);
    pair perp=I*dir;
    L0.align(L0.align,unit(-sgn(dot(sign*locate1.dir,perp))*perp));
  }		     
  pair align=L0.align.dir;
  if(L0.align.relative) align *= -I*dir;
  pair offset;
  if(ticklabels) {
    if(straight(g)) {
      real angle=degrees(dir(g,t));
      transform t=rotate(-angle,point(g,t));
      frame F=t*f;
      pair Z=t*z;
      pair Align=rotate(-angle)*align;
      offset=abs((Align.y >= 0 ? max(F).y : (Align.y < 0 ? min(F).y : 0))-Z.y)*
	unit(align);
    }
    align=axislabelmargin*align;
  }
  z += offset;

  L0.align(align);
  L0.position(z);
  frame d;
  add(d,L0);
  pair width=0.5*(max(d)-min(d));
  int n=length(g);
  real t=L.relative();
  pair s=realmult(width,dir(g,t));
  if(t == 0) {
    if(L.align.default) s *= -axislabelfactor;
    d=shift(s)*d;
  } else if(t == n) {
    if(L.align.default) s *= -axislabelfactor;
    d=shift(-s)*d;
  } else if(offset == 0 && L.align.default) {
    pair s=realmult(width,I*dir(g,t));
    s=axislabelfactor*s;
    d=shift(s)*d;
  }
  add(f,d);
}

// Compute the fractional coverage of a linear axis.
real axiscoverage(int N, transform T, path g, ticklocate locate, real Step,
		  pair side, int sign, real Size, ticklabel ticklabel, Label F,
		  real norm, real limit)
{
  real coverage=0;
  bool loop=cyclic(g);
  real a=locate.S.Tinv(locate.a);
  real b=locate.S.Tinv(locate.b);
  real tickmin=finite(locate.S.tickMin) ? locate.S.Tinv(locate.S.tickMin) : a;
  if(Size > 0) for(int i=0; i <= N; ++i) {
    real val=tickmin+i*Step;
    if(loop || (val >= a && val <= b)) {
      frame d;
      pair dir=labeltick(d,T,g,locate,val,side,sign,Size,ticklabel,F,norm);
      coverage += abs(dot(max(d)-min(d),dir));
      if(coverage > limit) return coverage;
    }
  }
  return coverage;
}

// Compute the fractional coverage of a logarithmic axis.
real logaxiscoverage(int N, transform T, path g, ticklocate locate, pair side,
		     int sign, real Size, Label F, real limit,
		     int first, int last)
{
  real coverage=0;
  real a=locate.a;
  real b=locate.b;
  for(int i=first-1; i <= last+1; i += N) {
    if(i >= a && i <= b) {
      frame d;
      pair dir=labeltick(d,T,g,locate,i,side,sign,Size,LogFormat,F);
      coverage += abs(dot(max(d)-min(d),dir));
      if(coverage > limit) return coverage;
    }
  }
  return coverage;
}

// Signature of routines that draw labelled paths with ticks and tick labels.
typedef void ticks(frame, transform, Label, pair, path, path, pen, arrowbar,
		   ticklocate, int[], bool opposite=false);
					  
private void ticks(frame, transform, Label, pair, path, path, pen, arrowbar,
		   ticklocate, int[], bool opposite=false) {};

// Automatic tick construction routine.
ticks Ticks(bool begin=true, bool end=true, int sign, int N, int n=0,
	    real Step=0, real step=0, real Size=0, real size=0,
	    bool beginlabel=true, bool endlabel=true, Label F="",
	    bool extend=false, pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(frame f, transform T, Label L, pair side, path g, path g2,
		  pen p, arrowbar arrow, ticklocate locate, int[] divisor,
		  bool opposite) {
    // Use local copy of context variables:
    int sign=opposite ? -sign : sign;
    int N=N;
    int n=n;
    real Step=Step;
    real step=step;
    pen pTick=pTick;
    pen ptick=ptick;
    
    real Size=Size;
    real size=size;
    if(Size == 0) Size=Ticksize;
    if(size == 0) size=ticksize;
    
    Label L=L.copy();
    Label F=F.copy();
    L.p(p);
    F.p(p);
    if(pTick == nullpen) pTick=p;
    if(ptick == nullpen) ptick=p;
    
    ticklabel ticklabel=ticklabel(F.s == "" ? defaultformat : F.s);
    if(F.align.dir != 0) side=F.align.dir;
    else if(side == 0) side=rotate(F.angle)*((sign == 1) ? left : right);
    
    bool ticklabels=false;
    guide G=T*g;
    guide G2=T*g2;
    
    if(!locate.S.scale.logarithmic) {
      real a=locate.S.Tinv(locate.a);
      real b=locate.S.Tinv(locate.b);
      if(a > b) {real temp=a; a=b; b=temp;}
      
      real tickmin=finite(locate.S.tickMin) ? 
	locate.S.Tinv(locate.S.tickMin) : a;
      real tickmax=finite(locate.S.tickMax) ?
	locate.S.Tinv(locate.S.tickMax) : b;
      if(tickmin > tickmax) {real temp=tickmin; tickmin=tickmax; tickmax=temp;}
      
      bool singletick=false;
      real len=tickmax-tickmin;
      real norm=max(abs(a),abs(b));
      if(Step == 0 && N == 0) {
	if(divisor.length > 0) {
	  real limit=axiscoverage*arclength(G);
	  for(int d=divisor.length-1; d >= 0; --d) {
	    N=divisor[d];
	    Step=len/N;
	    if(axiscoverage(N,T,g,locate,Step,side,sign,Size,ticklabel,F,norm,
			    limit) <= limit) {
	      if(N == 1 && !(locate.S.automin && locate.S.automax) 
		 && d < divisor.length-1) {
		// Try using 2 ticks (otherwise 1);
		int div=divisor[d+1];
		singletick=true; Step=quotient(div,2)*len/div;
		if(axiscoverage(2,T,g,locate,Step,side,sign,Size,ticklabel,F,
				norm,limit) <= limit) N=2;
	      }
	      // Found a good divisor; now compute subtick divisor
	      if(n == 0) {
		n=quotient(divisor[divisor.length-1],N);
		if(N == 1) n=(a*b >= 0) ? 2 : 1;
		if(n == 1) n=2;
	      }
	      break;
	    }
	  }
	} else N=1;
      }
      
      if(!singletick) {
	if(N == 0) N=(int) (len/Step);
	else Step=len/N;
      }

      if(n == 0) {
	if(step != 0) n=ceil(Step/step);
      } else step=Step/n;
      
      b += epsilon*norm;
      
      int count=Step > 0 ? floor(b/Step)-ceil(a/Step)+1 : 0;
      
      begingroup(f);
      if(opposite) draw(f,G,p);
      else draw(f,G,p,arrow);
      
      if(Size > 0) {
	int c=0;
	for(int i=0; i <= N; ++i) {
	  real val=tickmin+i*Step;
	  if(val >= a && val <= b) {
	    ++c;
	    if((begin || c > 1) && (end || c < count))
	      drawtick(f,T,g,g2,locate,val,Size,sign,pTick,extend);
	  }
	  if(size > 0 && step > 0) {
	    real iStep=i*Step;
	    real jstop=(len-iStep)/step;
	    for(int j=1; j < n && j <= jstop; ++j) {
	      real val=tickmin+iStep+j*step;
	      if(val >= a && val <= b)
		drawtick(f,T,g,g2,locate,val,size,sign,ptick,extend);
	    }
	  }
	}
      }
      endgroup(f);
    
      if(Size > 0 && !opposite) {
	int c=0;
	for(int i=0; i <= N; ++i) {
	  real val=tickmin+i*Step;
	  if(val >= a && val <= b) {
	    ++c;
	    if((beginlabel || c > 1) && (endlabel || c < count)) {
	      ticklabels=true;
	      labeltick(f,T,g,locate,val,side,sign,Size,ticklabel,F,norm);
	    }
	  }
	}
      }

    } else { // Logarithmic
      real a=locate.S.postscale.Tinv(locate.a);
      real b=locate.S.postscale.Tinv(locate.b);
      if(a > b) {real temp=a; a=b; b=temp;}
      
      int first=ceil(a-epsilon);
      int last=floor(b+epsilon);
      
      if(N == 0) {
	real limit=axiscoverage*arclength(G);
	N=1;
	while(N <= last-first) {
	  if(logaxiscoverage(N,T,g,locate,side,sign,Size,F,limit,first,last)
	     <= limit) break;
	  ++N;
	}
      }
      
      if(N <= 2 && n == 0) n=10;
      
      int count=floor(b)-ceil(a)+1;
      
      begingroup(f);
      if(opposite) draw(f,G,p);
      else draw(f,G,p,arrow);

      if(N > 0) {
	int c=0;
	for(int i=first-1; i <= last+1; ++i) {
	  if(i >= a && i <= b) {
	    ++c;
	    if((begin || c > 1) && (end || c < count)) {
	      real Size0=((i-first) % N == 0 || n != 0) ? Size : size;
	      drawtick(f,T,g,g2,locate,i,Size0,sign,pTick,extend);
	    }
	  }
	  if(n > 0) {
	    for(int j=2; j < n; ++j) {
	      real val=(i+1+log10(j/n));
	      if(val >= a && val <= b)
		drawtick(f,T,g,g2,locate,val,size,sign,ptick,extend);
	    }
	  }
	}
      }
      endgroup(f);
      
      if(!opposite && N > 0) {
	int c=0;
	for(int i=first-N; i <= last+N; i += N) {
	  if(i >= a && i <= b) {
	    ++c;
	    if((beginlabel || c > 1) && (endlabel || c < count)) {
	      ticklabels=true;
	      labeltick(f,T,g,locate,i,side,sign,Size,LogFormat,F);
	    }
	  }
	}
      }
    }
    
    if(L.s != "" && !opposite) 
      labelaxis(f,T,L,G,locate,sign,ticklabels);
  };
}

// Tick construction routine for a user-specified array of tick values.
ticks Ticks(int sign, real[] Ticks, real[] ticks=new real[],
	    real Size=0, real size=0,
	    bool beginlabel=true, bool endlabel=true, Label F="",
	    bool extend=false, pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(frame f, transform T, Label L, pair side, path g, path g2, 
		  pen p, arrowbar arrow, ticklocate locate, int[] divisor,
		  bool opposite) {
    // Use local copy of context variables:
    int sign=opposite ? -sign : sign;
    pen pTick=pTick;
    pen ptick=ptick;
    
    real Size=Size;
    real size=size;
    if(Size == 0) Size=Ticksize;
    if(size == 0) size=ticksize;
    
    Label L=L.copy();
    Label F=F.copy();
    L.p(p);
    F.p(p);
    if(pTick == nullpen) pTick=p;
    if(ptick == nullpen) ptick=p;
    
    ticklabel ticklabel=ticklabel(F.s == "" ? defaultformat : F.s);
    if(F.align.dir != 0) side=F.align.dir;
    else if(side == 0) side=rotate(F.angle)*((sign == 1) ? left : right);
    
    bool ticklabels=false;
    guide G=T*g;
    guide G2=T*g2;
    
    real a=locate.a;
    real b=locate.b;

    real norm=max(abs(a),abs(b));
    if(locate.S.scale.logarithmic) ticklabel=new string(real x) {
      return format("$10^{%g}$",x);
    };

    begingroup(f);
    if(opposite) draw(f,G,p);
    else draw(f,G,p,arrow);
    for(int i=0; i < Ticks.length; ++i) {
      real val=Ticks[i];
      if(val >= a && val <= b)
	drawtick(f,T,g,g2,locate,val,Size,sign,pTick,extend);
    }
    for(int i=0; i < ticks.length; ++i) {
      real val=ticks[i];
      if(val >= a && val <= b)
      drawtick(f,T,g,g2,locate,val,size,sign,ptick,extend);
    }
    endgroup(f);
    
    if(Size > 0 && !opposite) {
      for(int i=(beginlabel ? 0 : 1);
	  i < (endlabel ? Ticks.length : Ticks.length-1); ++i) {
	real val=Ticks[i];
	if(val >= a && val <= b) {
	  ticklabels=true;
	  labeltick(f,T,g,locate,val,side,sign,Size,ticklabel,F,norm);
	}
      }
    }
    if(L.s != "" && !opposite) 
      labelaxis(f,T,L,G,locate,sign,ticklabels);
  };
}

ticks NoTicks()
{
  return new void(frame f, transform T, Label L, pair, path g, path, pen p,
		  arrowbar arrow, ticklocate, int[], bool opposite) {
    path G=T*g;
    if(opposite) draw(f,G,p);
    else {
      draw(f,G,p,arrow);
      if(L.s != "") {
	L.p(p);
	labelaxis(f,T,L,G);
      }
    }
  };
}

ticks LeftTicks(bool begin=true, bool end=true, int N=0, int n=0,
		real Step=0, real step=0,
		real Size=0, real size=0,
		bool beginlabel=true, bool endlabel=true,
		Label format="", bool extend=false,
		pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(begin,end,-1,N,n,Step,step,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

ticks RightTicks(bool begin=true, bool end=true, int N=0, int n=0,
		 real Step=0, real step=0,
		 real Size=0, real size=0,
		 bool beginlabel=true, bool endlabel=true,
		 Label format="", bool extend=false,
		 pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(begin,end,1,N,n,Step,step,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

ticks LeftTicks(real[] Ticks, real[] ticks=new real[],
		real Size=0, real size=0,
		bool beginlabel=true, bool endlabel=true, 
		Label format="", bool extend=false,
		pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(-1,Ticks,ticks,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

ticks RightTicks(real[] Ticks, real[] ticks=new real[],
		 real Size=0, real size=0,
		 bool beginlabel=true, bool endlabel=true,
		 Label format="", bool extend=false,
		 pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(1,Ticks,ticks,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

public ticks
  NoTicks=NoTicks(),
  LeftTicks=LeftTicks(),
  RightTicks=RightTicks();

					       
// Structure used to communicate axis and autoscale settings to tick routines. 
private struct axisT {
  public pair value;
  public real position;
  public pair side;
  public pair align;
  public pair value2;
  
  public pair userMin;
  public pair userMax;
  public int[] xdivisor;
  public int[] ydivisor;
  public bool extend;
};

axisT operator init() {return new axisT;}
					       
public axisT axis;
typedef void axis(picture, axisT);
void axis(picture, axisT) {};

axis Bottom(bool extend=false) {
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.y.automin() ?
      (pic.scale.x.tickMin,pic.scale.y.tickMin) : axis.userMin;
    axis.position=0.5;
    axis.side=right;
    axis.align=S;
    axis.value2=Infinity;
    axis.extend=extend;
  };
}

axis Top(bool extend=false) {
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.y.automax() ?
    (pic.scale.x.tickMax,pic.scale.y.tickMax) : axis.userMax;
    axis.position=0.5;
    axis.side=left;
    axis.align=N;
    axis.value2=Infinity;
    axis.extend=extend;
  };
}

axis BottomTop(bool extend=false) {
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.y.automin() ?
    (pic.scale.x.tickMin,pic.scale.y.tickMin) : axis.userMin;
    axis.position=0.5;
    axis.side=right;
    axis.align=S;
    axis.value2=pic.scale.y.automax() ?
    (pic.scale.x.tickMax,pic.scale.y.tickMax) : axis.userMax;
    axis.extend=extend;
  };
}

axis Left(bool extend=false) {
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.x.automin() ? 
    (pic.scale.x.tickMin,pic.scale.y.tickMin) : axis.userMin;
    axis.position=0.5;
    axis.side=left;
    axis.align=W;
    axis.value2=Infinity;
    axis.extend=extend;
  };
}

axis Right(bool extend=false) {
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.x.automax() ?
    (pic.scale.x.tickMax,pic.scale.y.tickMax) : axis.userMax;
    axis.position=0.5;
    axis.side=right;
    axis.align=E;
    axis.value2=Infinity;
    axis.extend=extend;
  };
}

axis LeftRight(bool extend=false) {
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.x.automin() ?
      (pic.scale.x.tickMin,pic.scale.y.tickMin) : axis.userMin;
    axis.position=0.5;
    axis.side=left;
    axis.align=W;
    axis.value2=pic.scale.x.automax() ?
      (pic.scale.x.tickMax,pic.scale.y.tickMax) : axis.userMax;
    axis.extend=extend;
  };
}

axis XEquals(real x, bool extend=true)
{
  return new void(picture pic, axisT axis) {
    axis.value=pic.scale.x.T(x);
    axis.position=1;
    axis.side=left;
    axis.align=W;
    axis.value2=Infinity;
    axis.extend=extend;
  };
}

axis YEquals(real y, bool extend=true)
{
    return new void(picture pic, axisT axis) {
    axis.value=I*pic.scale.y.T(y);
    axis.position=1;
    axis.side=right;
    axis.align=S;
    axis.value2=Infinity;
    axis.extend=extend;
  };
}

axis XZero(bool extend=true)
{
  return XEquals(0,extend);
}

axis YZero(bool extend=true)
{
  return YEquals(0,extend);
}

public axis
  Bottom=Bottom(),
  Top=Top(),
  BottomTop=BottomTop(),
  Left=Left(),
  Right=Right(),
  LeftRight=LeftRight(),
  XZero=XZero(),
  YZero=YZero();

// Draw a general axis.
void axis(picture pic=currentpicture, Label L="", guide g, pen p=currentpen,
	  ticks ticks, ticklocate locate, arrowbar arrow=None,
	  int[] divisor=new int[], bool put=Above, bool opposite=false) 
{
  Label L=L.copy();
  real t=reltime(g,0.5);
  if(L.defaultposition) L.position(t);
  divisor=copy(divisor);
  locate=locate.copy();
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    frame d;
    ticks(d,t,L,0,g,g,p,arrow,locate,divisor,opposite);
    (put ? add : prepend)(f,t*T*inverse(t)*d);
  });
  
  pic.addPath(g,p);
  
  if(L.s != "") {
    frame f;
    Label L0=L.copy();
    L0.position(0);
    add(f,L0);
    pair pos=point(g,L.relative()*length(g));
    pic.addBox(pos,pos,min(f),max(f));
  }
}

// An internal routine to draw an x axis at a particular y value.
void xaxisAt(picture pic=currentpicture, Label L="", axis axis,
	     real xmin=-infinity, real xmax=infinity, pen p=currentpen,
	     ticks ticks=NoTicks, arrowbar arrow=None, bool put=Above,
	     bool opposite=false)
{
  real y=opposite ? axis.value2.y : axis.value.y;
  real y2=axis.value2.y;
  Label L=L.copy();
  int[] divisor=copy(axis.xdivisor);
  pair side=axis.side;
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    transform tinv=inverse(t);
    pair a=xmin == -infinity ? tinv*(lb.x-min(p).x,ytrans(t,y)) : (xmin,y);
    pair b=xmax == infinity ? tinv*(rt.x-max(p).x,ytrans(t,y)) : (xmax,y);
    pair a2=xmin == -infinity ? tinv*(lb.x-min(p).x,ytrans(t,y2)) : (xmin,y2);
    pair b2=xmax == infinity ? tinv*(rt.x-max(p).x,ytrans(t,y2)) : (xmax,y2);
    frame d;
    ticks(d,t,L,side,a--b,finite(y2) ? a2--b2 : nullpath,p,arrow,
	  ticklocate(a.x,b.x,pic.scale.x),divisor,opposite);
    (put ? add : prepend)(f,t*T*tinv*d);
  });

  pair a=(finite(xmin) ? xmin : pic.userMin.x,y);
  pair b=(finite(xmax) ? xmax : pic.userMax.x,y);
  pair a2=(finite(xmin) ? xmin : pic.userMin.x,y2);
  pair b2=(finite(xmax) ? xmax : pic.userMax.x,y2);
  
  if(finite(a)) {
    pic.addPoint(a,min(p));
    pic.addPoint(a,max(p));
  }
  
  if(finite(b)) {
    pic.addPoint(b,min(p));
    pic.addPoint(b,max(p));
  }

  if(finite(a) && finite(b)) {
    frame d;
    ticks(d,identity(),L,side,(a.x,0)--(b.x,0),(a2.x,0)--(b2.x,0),p,arrow,
	  ticklocate(a.x,b.x,pic.scale.x),divisor,opposite);
    frame f;
    if(L.s != "") {
      Label L0=L.copy();
      L0.position(0);
      add(f,L0);
    }
    pair pos=a+L.relative()*(b-a);
    pic.addBox(pos,pos,(min(f).x,min(d).y),(max(f).x,max(d).y));
  }
}

// An internal routine to draw a y axis at a particular x value.
void yaxisAt(picture pic=currentpicture, Label L="", axis axis,
	     real ymin=-infinity, real ymax=infinity, pen p=currentpen,
	     ticks ticks=NoTicks, arrowbar arrow=None, bool put=Above,
	     bool opposite=false)
{
  real x=opposite ? axis.value2.x : axis.value.x;
  real x2=axis.value2.x;
  Label L=L.copy();
  int[] divisor=copy(axis.ydivisor);
  pair side=axis.side;
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    transform tinv=inverse(t);
    pair a=ymin == -infinity ? tinv*(xtrans(t,x),lb.y-min(p).y) : (x,ymin);
    pair b=ymax == infinity ? tinv*(xtrans(t,x),rt.y-max(p).y) : (x,ymax);
    pair a2=ymin == -infinity ? tinv*(xtrans(t,x2),lb.y-min(p).y) : (x2,ymin);
    pair b2=ymax == infinity ? tinv*(xtrans(t,x2),rt.y-max(p).y) : (x2,ymax);
    frame d;
    ticks(d,t,L,side,a--b,finite(x2) ? a2--b2 : nullpath,p,arrow,
	  ticklocate(a.y,b.y,pic.scale.y),divisor,opposite);
    (put ? add : prepend)(f,t*T*tinv*d);
  });
  
  pair a=(x,finite(ymin) ? ymin : pic.userMin.y);
  pair b=(x,finite(ymax) ? ymax : pic.userMax.y);
  pair a2=(x2,finite(ymin) ? ymin : pic.userMin.y);
  pair b2=(x2,finite(ymax) ? ymax : pic.userMax.y);
  
  if(finite(a)) {
    pic.addPoint(a,min(p));
    pic.addPoint(a,max(p));
  }
  
  if(finite(b)) {
    pic.addPoint(b,min(p));
    pic.addPoint(b,max(p));
  }
  
  if(finite(a) && finite(b)) {
    frame d;
    ticks(d,identity(),L,side,(0,a.y)--(0,b.y),(0,a2.y)--(0,b2.y),p,arrow,
	  ticklocate(a.y,b.y,pic.scale.y),divisor,opposite);
    frame f;
    if(L.s != "") {
      Label L0=L.copy();
      L0.position(0);
      add(f,L0);
    }
    pair pos=a+L.relative()*(b-a);
    pic.addBox(pos,pos,(min(d).x,min(f).y),(max(d).x,max(f).y));
  }
}

// Restrict the x limits of a picture.
void xlimits(picture pic=currentpicture, real Min=-infinity, real Max=infinity,
	     bool crop=Crop)
{
  if(!(finite(pic.userMin.x) && finite(pic.userMax.x))) return;
  
  pic.scale.x.automin=Min <= -infinity;
  pic.scale.x.automax=Max >= infinity;
  
  bounds mx;
  if(pic.scale.x.automin() || pic.scale.x.automax())
    mx=autoscale(pic.userMin.x,pic.userMax.x,pic.scale.x.scale);
  
  if(pic.scale.x.automin) {
    if(pic.scale.x.automin()) pic.userMin=(mx.min,pic.userMin.y);
  } else pic.userMin=(pic.scale.x.T(Min),pic.userMin.y);
  
  if(pic.scale.x.automax) {
    if(pic.scale.x.automax()) pic.userMax=(mx.max,pic.userMax.y);
  } else pic.userMax=(pic.scale.x.T(Max),pic.userMax.y);
  
  if(crop) {
    pair userMin=pic.userMin;
    pair userMax=pic.userMax;
    pic.bounds.xclip(userMin.x,userMax.x);
    pic.clip(new void (frame f, transform t) {
      clip(f,box(((t*userMin).x,min(f).y),((t*userMax).x,max(f).y)));
  });
  }
}

// Restrict the y limits of a picture.
void ylimits(picture pic=currentpicture, real Min=-infinity, real Max=infinity,
	     bool crop=Crop)
{
  if(!(finite(pic.userMin.y) && finite(pic.userMax.y))) return;
  
  pic.scale.y.automin=Min <= -infinity;
  pic.scale.y.automax=Max >= infinity;
  
  bounds my;
  if(pic.scale.y.automin() || pic.scale.y.automax())
    my=autoscale(pic.userMin.y,pic.userMax.y,pic.scale.y.scale);
  
  if(pic.scale.y.automin) {
    if(pic.scale.y.automin()) pic.userMin=(pic.userMin.x,my.min);
  } else pic.userMin=(pic.userMin.x,pic.scale.y.T(Min));
  
  if(pic.scale.y.automax) {
    if(pic.scale.y.automax()) pic.userMax=(pic.userMax.x,my.max);
  } else pic.userMax=(pic.userMax.x,pic.scale.y.T(Max));
  
  if(crop) {
    pair userMin=pic.userMin;
    pair userMax=pic.userMax;
    pic.bounds.yclip(userMin.y,userMax.y);
    pic.clip(new void (frame f, transform t) {
      clip(f,box((min(f).x,(t*userMin).y),(max(f).x,(t*userMax).y)));
  });
  }
}

// Crop a picture to the current user-space picture limits.
void crop(picture pic=currentpicture) 
{
  xlimits(pic,false);
  ylimits(pic,false);
  if(finite(pic.userMin) && finite(pic.userMax))
    clip(pic,box(pic.userMin,pic.userMax));
}

// Restrict the x and y limits to box(min,max).
void limits(picture pic=currentpicture, pair min, pair max)
{
  xlimits(pic,min.x,max.x);
  ylimits(pic,min.y,max.y);
}
  
// Internal routine to autoscale an axis.
void autoscale(picture pic=currentpicture, axis axis)
{
  if(!pic.scale.set) {
    bounds mx,my;
    pic.scale.set=true;
    
    if(finite(pic.userMin.x) && finite(pic.userMax.x)) {
      mx=autoscale(pic.userMin.x,pic.userMax.x,pic.scale.x.scale);
      if(pic.scale.x.scale.logarithmic && 
	 floor(pic.userMin.x) == floor(pic.userMax.x)) {
	pic.userMin=(floor(pic.userMin.x),pic.userMin.y);
	pic.userMax=(ceil(pic.userMax.x),pic.userMax.y);
      }
    } else {mx.min=mx.max=0; pic.scale.set=false;}
    
    if(finite(pic.userMin.y) && finite(pic.userMax.y)) {
      my=autoscale(pic.userMin.y,pic.userMax.y,pic.scale.y.scale);
      if(pic.scale.y.scale.logarithmic && 
	 floor(pic.userMin.y) == floor(pic.userMax.y)) {
	pic.userMin=(pic.userMin.x,floor(pic.userMin.y));
	pic.userMax=(pic.userMax.x,ceil(pic.userMax.y));
      }
    } else {my.min=my.max=0; pic.scale.set=false;}
    
    pic.scale.x.tickMin=mx.min;
    pic.scale.x.tickMax=mx.max;
    pic.scale.y.tickMin=my.min;
    pic.scale.y.tickMax=my.max;
    axis.xdivisor=mx.divisor;
    axis.ydivisor=my.divisor;
    axis.userMin=(pic.scale.x.automin() ? mx.min : pic.userMin.x,
		  pic.scale.y.automin() ? my.min : pic.userMin.y);
    axis.userMax=(pic.scale.x.automax() ? mx.max : pic.userMax.x,
		  pic.scale.y.automax() ? my.max : pic.userMax.y);
  }
}

void checkaxis(picture pic, axis axis, bool ticks) 
{
  axis(pic,axis);
  if((!axis.extend || ticks) && pic.empty())
    abort("axis with ticks or unextended axis called on empty picture");
}

// Draw an x axis.
void xaxis(picture pic=currentpicture, Label L="", axis axis=YZero,
	   real xmin=-infinity, real xmax=infinity, pen p=currentpen,
	   ticks ticks=NoTicks, arrowbar arrow=None, bool put=Below)
{
  Label L=L.copy();
  bool newticks=false;
  
  if(xmin != -infinity) {
    pic.addPoint((xmin,pic.userMin.y));
    newticks=true;
  }
  
  if(xmax != infinity) {
    pic.addPoint((xmax,pic.userMax.y));
    newticks=true;
  }
  
  if(pic.scale.set && newticks) {
    if(xmin == -infinity) xmin=pic.userMin.x;
    if(xmax == infinity) xmax=pic.userMax.x;
    bounds mx=autoscale(xmin,xmax,pic.scale.x.scale);
    pic.scale.x.tickMin=mx.min;
    pic.scale.x.tickMax=mx.max;
    axis.xdivisor=mx.divisor;
  } else {
    if(xmin == -infinity || xmax == infinity)
      checkaxis(pic,axis,ticks != NoTicks);
    autoscale(pic,axis);
  }
  
  axis(pic,axis);
  if(axis.extend) put=Above;
  
  if(xmin == -infinity && !axis.extend) {
    if(pic.scale.set && pic.scale.x.automin())
      xmin=pic.scale.x.tickMin;
    else xmin=pic.userMin.x;
  }
  
  if(xmax == infinity && !axis.extend) {
    if(pic.scale.set && pic.scale.x.automax())
      xmax=pic.scale.x.tickMax;
    else xmax=pic.userMax.x;
  }

  if(L.defaultposition) L.position(axis.position);
  L.align(L.align,axis.align);
  
  xaxisAt(pic,L,axis,xmin,xmax,p,ticks,arrow,put);
  if(axis.value2 != Infinity)
    xaxisAt(pic,L,axis,xmin,xmax,p,ticks,arrow,put,true);
}

// Draw a y axis.
void yaxis(picture pic=currentpicture, Label L="", axis axis=XZero,
	   real ymin=-infinity, real ymax=infinity, pen p=currentpen,
	   ticks ticks=NoTicks, arrowbar arrow=None, bool put=Below)
{
  Label L=L.copy();
  bool newticks=false;
  
  if(ymin != -infinity) {
    pic.addPoint((pic.userMin.x,ymin));
    newticks=true;
  }
  
  if(ymax != infinity) {
    pic.addPoint((pic.userMax.x,ymax));
    newticks=true;
  }
  
  if(pic.scale.set && newticks) {
    if(ymin == -infinity) ymin=pic.userMin.y;
    if(ymax == infinity) ymax=pic.userMax.y;
    bounds my=autoscale(ymin,ymax,pic.scale.y.scale);
    pic.scale.y.tickMin=my.min;
    pic.scale.y.tickMax=my.max;
    axis.ydivisor=my.divisor;
  } else {
    if(ymin == -infinity || ymax == infinity) 
      checkaxis(pic,axis,ticks != NoTicks);
    autoscale(pic,axis);
  }
  
  axis(pic,axis);
  if(axis.extend) put=Above;
  
  if(ymin == -infinity && !axis.extend) {
    if(pic.scale.set && pic.scale.y.automin())
      ymin=pic.scale.y.tickMin;
    else ymin=pic.userMin.y;
  }
  
  if(ymax == infinity && !axis.extend) {
    if(pic.scale.set && pic.scale.y.automax())
      ymax=pic.scale.y.tickMax;
    else ymax=pic.userMax.y;
  }

  if(L.defaultposition) L.position(axis.position);
  L.align(L.align,axis.align);
  
  if(L.defaultangle) {
    frame f;
    add(f,Label(L.s,(0,0),L.p));
    L.angle(length(max(f)-min(f)) > ylabelwidth*fontsize(L.p) ? 90 : 0);
  }
  
  yaxisAt(pic,L,axis,ymin,ymax,p,ticks,arrow,put);
  if(axis.value2 != Infinity)
    yaxisAt(pic,L,axis,ymin,ymax,p,ticks,arrow,put,true);
}

// Draw x and y axes.
void axes(picture pic=currentpicture, pen p=currentpen, arrowbar arrow=None,
	  bool put=Below)
{
  xaxis(pic,p,arrow,put);
  yaxis(pic,p,arrow,put);
}

// Draw a yaxis at x.
void xequals(picture pic=currentpicture, Label L="", real x,
	     bool extend=false, real ymin=-infinity, real ymax=infinity,
	     pen p=currentpen, ticks ticks=NoTicks, bool put=Above,
	     arrowbar arrow=None)
{
  yaxis(pic,L,XEquals(x,extend),ymin,ymax,p,ticks,arrow,put);
}

// Draw an xaxis at y.
void yequals(picture pic=currentpicture, Label L="", real y,
	     bool extend=false, real xmin=-infinity, real xmax=infinity,
	     pen p=currentpen, ticks ticks=NoTicks, bool put=Above,
	     arrowbar arrow=None)
{
  xaxis(pic,L,YEquals(y,extend),xmin,xmax,p,ticks,arrow,put);
}

// Draw a tick of length size at pair z in direction dir using pen p.
void tick(picture pic=currentpicture, pair z, pair dir, real size=Ticksize,
	  pen p=currentpen)
{
  pic.add(new void (frame f, transform t) {
    pair tz=t*z;
    draw(f,tz--tz+unit(dir)*size,p);
  });
  pic.addPoint(z,p);
  pic.addPoint(z,unit(dir)*size,p);
}

void xtick(picture pic=currentpicture, pair z, pair dir=N,
	   real size=Ticksize, pen p=currentpen)=tick;

void xtick(picture pic=currentpicture, Label L, pair z, pair dir=N,
	   string format="", real size=Ticksize, pen p=currentpen)
{
  Label L=L.copy();
  if(L.defaultposition) L.position(z);
  if(L.align.default)
    L.align(-dir);
  if(L.shift == 0) 
    L.shift(dot(dir,L.align.dir) > 0 ? dir*size :
	    ticklabelshift(L.align.dir,p));
  L.p(p);
  if(L.s == "") L.s=format(format,z.x);
  L.s=baseline(L.s,L.align,"$10^4$");
  add(pic,L);
  tick(pic,z,dir,size,p);
}

void ytick(picture pic=currentpicture, Label L, explicit pair z, pair dir=E,
	   string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,z,dir,format,size,p);
}

void ytick(picture pic=currentpicture, Label L, real y, pair dir=E,
	   string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,(0,y),dir,format,size,p);
}

void ytick(picture pic=currentpicture, explicit pair z, pair dir=E,
	   real size=Ticksize, pen p=currentpen)=tick;

void ytick(picture pic=currentpicture, real y, pair dir=E,
	   real size=Ticksize, pen p=currentpen)
{
  tick(pic,(0,y),dir,size,p);
}

private void label(picture pic, Label L, pair z, real x, align align,
		   string format, pen p)
{
  Label L=L.copy();
  L.position(z);
  L.align(align);
  L.p(p);
  if(L.shift == 0) L.shift(ticklabelshift(L.align.dir,L.p));
  if(L.s == "") L.s=format(format,x);
  L.s=baseline(L.s,L.align,"$10^4$");
  add(pic,L);
}

// Label and draw an x tick.
void labelx(picture pic=currentpicture, Label L="", pair z, align align=S,
	    string format="", pen p=nullpen)
{
  label(pic,L,z,z.x,align,format,p);
}

void labelx(picture pic=currentpicture, Label L,
	    string format="", explicit pen p=currentpen)
{
  labelx(pic,L,L.position,format,p);
}

// Label and draw a y tick.
void labely(picture pic=currentpicture, Label L="", explicit pair z,
	    align align=W, string format="", pen p=nullpen)
{
  label(pic,L,z,z.y,align,format,p);
}

void labely(picture pic=currentpicture, Label L="", real y,
	    align align=W, string format="", pen p=nullpen)
{
  labely(pic,L,(0,y),align,format,p);
}

void labely(picture pic=currentpicture, Label L,
	    string format="", explicit pen p=nullpen)
{
  labely(pic,L,L.position.position,format,p);
}

private string noprimary="Primary axis must be drawn before secondary axis";

// Construct a secondary X axis
picture secondaryX(picture primary=currentpicture, void f(picture))
{
  if(!primary.scale.set) abort(noprimary);
  picture pic;
  f(pic);
  if(pic.empty()) abort("empty secondaryX picture");
  bounds a=autoscale(pic.userMin.x,pic.userMax.x,pic.scale.x.scale);
  real bmin=pic.scale.x.automin() ? a.min : pic.userMin.x;
  real bmax=pic.scale.x.automax() ? a.max : pic.userMax.x;
  
  real denom=bmax-bmin;
  if(denom != 0.0) {
    pic.erase();
    real m=(primary.userMax.x-primary.userMin.x)/denom;
    pic.scale.x.postscale=Linear(m,bmin-primary.userMin.x/m);
    pic.scale.set=true;
    pic.scale.x.tickMin=pic.scale.x.postscale.T(a.min);
    pic.scale.x.tickMax=pic.scale.x.postscale.T(a.max);
    pic.scale.y.tickMin=primary.scale.y.tickMin;
    pic.scale.y.tickMax=primary.scale.y.tickMax;
    axis.userMin=(bmin,axis.userMin.y);
    axis.userMax=(bmax,axis.userMax.y);
    axis.xdivisor=a.divisor;
    f(pic);
  }
  pic.userMin=primary.userMin;
  pic.userMax=primary.userMax;
  return pic;
}

// Construct a secondary Y axis
picture secondaryY(picture primary=currentpicture, void f(picture))
{
  if(!primary.scale.set) abort(noprimary);
  picture pic;
  f(pic);
  if(pic.empty()) abort("empty secondaryY picture");
  bounds a=autoscale(pic.userMin.y,pic.userMax.y,pic.scale.y.scale);
  real bmin=pic.scale.y.automin() ? a.min : pic.userMin.y;
  real bmax=pic.scale.y.automax() ? a.max : pic.userMax.y;

  real denom=bmax-bmin;
  if(denom != 0.0) {
    pic.erase();
    real m=(primary.userMax.y-primary.userMin.y)/denom;
    pic.scale.y.postscale=Linear(m,bmin-primary.userMin.y/m);
    pic.scale.set=true;
    pic.scale.x.tickMin=primary.scale.x.tickMin;
    pic.scale.x.tickMax=primary.scale.x.tickMax;
    pic.scale.y.tickMin=pic.scale.y.postscale.T(a.min);
    pic.scale.y.tickMax=pic.scale.y.postscale.T(a.max);
    axis.userMin=(axis.userMin.x,bmin);
    axis.userMax=(axis.userMax.x,bmax);
    axis.ydivisor=a.divisor;
    f(pic);
  }
  pic.userMin=primary.userMin;
  pic.userMax=primary.userMax;
  return pic;
}

typedef guide graph(pair F(real), real, real, int);
		       
public graph graph(guide join(... guide[]))
{
  return new guide(pair F(real), real a, real b, int n) {
    guide g;
    real width=n == 0 ? 0 : (b-a)/n;
    for(int i=0; i <= n; ++i) {
      real x=a+width*i;
      g=join(g,F(x));	
    }	
    return g;
  };
}

guide Straight(... guide[])=operator --;
guide Spline(... guide[])=operator ..;

pair Scale(picture pic, pair z)
{
  return (pic.scale.x.T(z.x),pic.scale.y.T(z.y));
}

typedef guide interpolate(... guide[]);

guide graph(picture pic=currentpicture, real f(real), real a, real b,
	    int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new pair(real x) {
    return (x,pic.scale.y.T(f(pic.scale.x.Tinv(x))));},
		     pic.scale.x.T(a),pic.scale.x.T(b),n);
}

guide graph(picture pic=currentpicture, real x(real), real y(real), real a,
	    real b, int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new pair(real t) {return Scale(pic,(x(t),y(t)));},a,b,n);
}

guide graph(picture pic=currentpicture, pair z(real), real a, real b,
	    int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new pair(real t) {return Scale(pic,z(t));},a,b,n);
}

int next(int i, bool[] cond)
{
  ++i;
  if(cond.length > 0) while(!cond[i]) ++i;
  return i;
}

int conditional(pair[] z, bool[] cond)
{
  if(cond.length > 0) {
    if(cond.length != z.length)
      abort("condition array has different length than data");
    return sum(cond)-1;
  } else return z.length-1;
}

guide graph(picture pic=currentpicture, pair[] z, bool[] cond={},
	    interpolate join=operator --)
{
  int n=conditional(z,cond);
  int i=-1;
  return graph(join)(new pair(real) {
    i=next(i,cond);
    return Scale(pic,z[i]);},0,0,n);
}

string differentlengths="attempt to graph arrays of different lengths";

guide graph(picture pic=currentpicture, real[] x, real[] y, bool[] cond={},
	    interpolate join=operator --)
{
  if(x.length != y.length) abort(differentlengths);
  int n=conditional(x,cond);
  int i=-1;
  return graph(join)(new pair(real) {
    i=next(i,cond);
    return Scale(pic,(x[i],y[i]));},0,0,n);
}

guide graph(real f(real), real a, real b, int n=ngraph,
	    real T(real), interpolate join=operator --)
{
  return graph(join)(new pair(real x) {return (T(x),f(T(x)));},a,b,n);
}

pair polar(real r, real theta)
{
  return r*expi(theta);
}

guide polargraph(real r(real), real a, real b, int n=ngraph,
		 interpolate join=operator --)
{
  return graph(join)(new pair(real theta) {return polar(r(theta),theta);},
		     a,b,n);
}

void errorbar(picture pic, pair z, pair dp, pair dm, pen p=currentpen,
	      real size=0)
{
  real dmx=-abs(dm.x);
  real dmy=-abs(dm.y);
  real dpx=abs(dp.x);
  real dpy=abs(dp.y);
  if(dmx != dpx) draw(pic,Scale(pic,z+(dmx,0))--Scale(pic,z+(dpx,0)),p,
		      Bars(size));
  if(dmy != dpy) draw(pic,Scale(pic,z+(0,dmy))--Scale(pic,z+(0,dpy)),p,
		      Bars(size));
}
  
void errorbars(picture pic=currentpicture, pair[] z, pair[] dp, pair[] dm={},
	       bool[] cond={}, pen p=currentpen, real size=0)
{
  if(dm.length == 0) dm=dp;
  if(z.length != dm.length || z.length != dp.length) abort(differentlengths);
  int n=conditional(z,cond);
  int i=-1;
  for(int I=0; I <= n; ++I) {
    i=next(i,cond);
    errorbar(pic,z[i],dp[i],dm[i],p,size);
  }
}

void errorbars(picture pic=currentpicture, real[] x, real[] y,
	       real[] dpx, real[] dpy, real[] dmx={}, real[] dmy={},
	       bool[] cond={}, pen p=currentpen, real size=0)
{
  if(dmx.length == 0) dmx=dpx;
  if(dmy.length == 0) dmy=dpy;
  if(x.length != y.length || 
     x.length != dpx.length || x.length != dmx.length ||
     x.length != dpy.length || x.length != dmy.length) abort(differentlengths);
  int n=conditional(x,cond);
  int i=-1;
  for(int I=0; I <= n; ++I) {
    i=next(i,cond);
    errorbar(pic,(x[i],y[i]),(dpx[i],dpy[i]),(dmx[i],dmy[i]),p,size);
  }
}

void errorbars(picture pic=currentpicture, real[] x, real[] y,
	       real[] dpy, bool[] cond={}, pen p=currentpen, real size=0)
{
  errorbars(pic,x,y,0*x,dpy,cond,p,size);
}

// A path as a function of a relative position parameter.
typedef path vector(real);

void vectorfield(picture pic=currentpicture, path g, int n, 
		 vector vector, real arrowsize=0, real arrowlength=0,
		 pen p=currentpen) 
{
  if(arrowsize == 0) arrowsize=arrowsize(p);
  if(n == 0) return;
  for(int i=0; i <= n; ++i) {
    real x=i/n;
    pair z=relpoint(g,x);
    draw(z,pic,vector(x),p,Arrow(arrowsize));
  }
}

// True arc
guide Arc(pair c, real r, real angle1, real angle2, int ngraph=400)
{
  return shift(c)*polargraph(new real(real t){return r;},angle1,angle2,ngraph,
			     operator ..);
}

// True circle
guide Circle(pair c, real r, int ngraph=400)
{
  return Arc(c,r,0,2pi,ngraph)..cycle;
}

