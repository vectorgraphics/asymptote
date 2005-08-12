static public real ticksize=1mm;
static public real Ticksize=2*ticksize;
static public real ylabelwidth=2.0;
static public real axislabelmargin=1;
static public real axiscoverage=0.9;
static public int ngraph=100;

static public real epsilon=1000*realEpsilon();

static bool Crop=true;
static bool NoCrop=false;

static scaleT Linear;
Linear.init(identity,identity);

static scaleT Log;
Log.init(log10,pow10);

public scaleT Linear(bool automin=true, bool automax=true, real s=1,
		     real intercept=0)
{
  real sinv=1/s;
  scaleT scale;
  real Tinv(real x) {return x*sinv+intercept;}
  scale.init(new real(real x) {return (x-intercept)*s;},
	    Tinv,Tinv,automin,automax);
  return scale;
}

public scaleT Log(bool automin=true, bool automax=true)
{
  scaleT scale;
  scale.init(Log.T,Log.Tinv,Log.Label,automin,automax);
  return scale;
}

real scaleT(real x) {return 0;}

bool linear(scaleT S) 
{
  return S.T == identity;
}

bool logarithmic(scaleT S) 
{
  return S.T == log10 && S.Label == identity;
}

void scale(picture pic=currentpicture, scaleT x, scaleT y)
{
  pic.scale.x.scale=x;
  pic.scale.y.scale=y;
}

void scale(picture pic=currentpicture, bool autoscale=true)
{
  scale(Linear(autoscale,autoscale),Linear(autoscale,autoscale));
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

struct bounds {
  public real min;
  public real max;
  // Possible tick intervals;
  public int[] divisor;
}

bounds operator init() {return new bounds;}
  
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

bounds autoscale(real Min, real Max, scaleT scale=Linear)
{
  bounds m;
  if(logarithmic(scale)) {
    m.min=floor(Min);
    m.max=ceil(Max);
    return m;
  }
  if(Min == infinity && Max == -infinity) {m.min=Min; m.max=Max; return m;}
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

typedef real part(pair);

typedef string ticklabel(real);

ticklabel ticklabel(string s) {
  return new string(real x) {return format(s,x);};
}

ticklabel LogFormat=new string(real x) {
  return format("$10^{%d}$",round(x));
};

void labelaxis(frame f, Label L, guide g, bool ticklabels=false)
{
  Label L0=L.copy();
  pair z=point(g,L0.position.x);
  pair dir=dir(g,L0.position.x);
  pair align=L0.align.dir;
  if(!L0.align.side.alias(NoSide)) align=L0.align.side.align(-I*dir);
  if(ticklabels) {
    pair minf=min(f);
    pair maxf=max(f);
    pair offset=(align.x >= 0 ? maxf.x : 
		 (align.x < 0 ? minf.x : 0),
		 align.y >= 0 ? maxf.y : 
		 (align.y < 0 ? minf.y : 0))-z;
    pair Idir=I*dir;
    z += dot(offset,Idir)*Idir;
    align=axislabelmargin*align;
  }
  L0.align(align);
  L0.position(z);
  frame d;
  add(d,L0);
  pair width=0.5*dot(max(d)-min(d),dir)*dir;
  if(L.position.x == 0) d=shift(width)*d;
  if(L.position.x == length(g)) d=shift(-width)*d;
  add(f,d);
}

private struct locateT {
  pair z; // User location 
  pair Z; // Frame location
  pair dir; // Frame direction
  bool calc(transform T, guide g, real pos) {
    real t=arctime(g,pos);
    z=point(g,t);
    Z=T*z;
    dir=unit(T*dir(g,t));
    return true;
  }
}

locateT operator init() {return new locateT;}
  
pair ticklabelshift(pair align, pen p=currentpen) 
{
  return 0.25*unit(align)*labelmargin(p);
}

pair labeltick(frame d, transform T, guide g, real pos, pair side,
	       int sign, real Size, ticklabel ticklabel, pen pTicklabel,
	       part part, real norm=0)
{
  locateT locate;
  locate.calc(T,g,pos);
  pair align=-side*I*locate.dir;
  pair shift=dot(align,I*sign*locate.dir) < 0 ? align*Size :
    ticklabelshift(align,pTicklabel);
  pair Z=locate.Z+shift;
  real v=part(locate.z);
  if(abs(v) < epsilon*norm) v=0;
  string s=ticklabel(v);
  s=baseline(s,align,"$10^4$");
  label(d,s,Z,align,pTicklabel);
  return locate.dir;
}  

real axiscoverage(int N, transform T, path g, real Step, pair side, int sign,
		  real Size, ticklabel ticklabel, pen pTicklabel, part part,
		  real norm, real limit, real offset, real firstpos,
		  real lastpos)
{
  real coverage=0;
  bool loop=cyclic(g);
  if(Size > 0) for(int i=0; i <= N; ++i) {
    real pos=i*Step;
    if(loop || (pos >= firstpos && pos <= lastpos)) {
      frame d;
      pair dir=labeltick(d,T,g,pos+offset,side,sign,Size,ticklabel,pTicklabel,
			 part,norm);
      coverage += abs(dot(max(d)-min(d),dir));
      if(coverage > limit) return coverage;
    }
  }
  return coverage;
}

real logaxiscoverage(int N, transform T, path g, real Step, real offset, 
		     pair side, int sign, real Size, pen pTicklabel,
		     part part, real limit, int first, int last,
		     real firstpos, real lastpos)
{
  real coverage=0;
  for(int i=first-1; i <= last+1; i += N) {
    frame d;
    real pos=i*Step;
    if(pos >= firstpos && pos <= lastpos) {
      pair dir=labeltick(d,T,g,pos+offset,side,sign,Size,LogFormat,pTicklabel,
			 part);
      coverage += abs(dot(max(d)-min(d),dir));
      if(coverage > limit) return coverage;
    }
  }
  return coverage;
}

void drawtick(frame f, transform T, path g, path g2, real pos, real Size,
	      int sign, pen p, bool extend) 
{
  locateT locate,locate2;
  locate.calc(T,g,pos);
  if(extend) {
    locate2.calc(T,g2,pos);
    draw(f,locate.Z--locate2.Z,p);
  } else
    draw(f,locate.Z--locate.Z-Size*I*sign*locate.dir,p);
}

typedef void ticks(frame, transform, Label, pair, path, path, pen, arrowbar,
		   autoscaleT, part, bool, int[], real, real);

private void ticks(frame, transform, Label, pair, path, path, pen, arrowbar,
		   autoscaleT, part, bool, int[], real, real) {};

ticks Ticks(bool begin=true, bool end=true, int sign, int N, int n=0,
	    real Step=0, real step=0, real Size=0, real size=0,
	    bool beginlabel=true, bool endlabel=true,
	    Label F=defaultformat, bool extend=false, pen pTick=nullpen,
	    pen ptick=nullpen)
{
  return new void(frame f, transform T, Label L, pair side, path G, path G2,
		  pen p, arrowbar arrow, autoscaleT S, part part,
		  bool opposite, int[] divisor, real tickmin, real tickmax) {
    // Use local copy of context variables:
    int sign=opposite ? -sign : sign;
    int N=N;
    int n=n;
    real Step=Step;
    real step=step;
    pen pTicklabel=F.p;
    pen pTick=pTick;
    pen ptick=ptick;
    
    Label L=L.copy();
    L.p(p);
    if(pTicklabel == nullpen) pTicklabel=p;
    if(pTick == nullpen) pTick=p;
    if(ptick == nullpen) ptick=p;
    
    ticklabel ticklabel=ticklabel(F.s);
    if(!F.align.side.alias(NoSide)) side=F.align.side.align((1,0));
    if(side == 0 && !F.align.side.alias(Center))
      side=(sign == 1) ? left : right;
    
    bool ticklabels=false;
    guide g=inverse(T)*G;
    guide g2=inverse(T)*G2;
    
    if(!logarithmic(S.scale)) {
      real a=part(point(g,0));
      real b=part(point(g,length(g)));
      real offset;
      bool singletick=false;
      if(finite(tickmin)) offset=tickmin-S.T(a);
      else {tickmin=S.T(0); offset=0;}
      if(!finite(tickmax)) 
	tickmax=S.T(arclength(g));
      real len=tickmax-tickmin;
      real norm=max(abs(a),abs(b));
      real lastpos=S.T(b)-S.T(a)-offset;
      real firstpos=-epsilon*lastpos-offset;
      if(Step == 0 && N == 0) {
	if(divisor.length > 0) {
	  real limit=axiscoverage*arclength(G);
	  for(int d=divisor.length-1; d >= 0; --d) {
	    N=divisor[d];
	    Step=len/N;
	    if(axiscoverage(N,T,g,Step,side,sign,Size,ticklabel,pTicklabel,
			    part,norm,limit,offset,firstpos,lastpos) <= limit)
	      {
	      if(N == 1 && !(S.automin() && S.automax()) 
		 && d < divisor.length-1) {
		// Try using 2 ticks (otherwise 1);
		int div=divisor[d+1];
		singletick=true; Step=quotient(div,2)*len/div;
		if(axiscoverage(2,T,g,Step,side,sign,Size,ticklabel,
				pTicklabel,part,norm,limit,offset,firstpos,
				lastpos) 
		   <= limit) N=2;
	      }
	      // Found a good divisor; now compute subtick divisor
	      if(n == 0) {
		n=quotient(divisor[-1],N);
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
	else {
	  Step=len/N;
	  if(cyclic(g) && len == arclength(g)) --N;
	}
      }

      if(n == 0) {
	if(step != 0) n=ceil(Step/step);
      } else step=Step/n;
      
      lastpos *= (1+epsilon);
      
      int count;
      if(cyclic(g)) {
	count=N;
	firstpos=-infinity;
	lastpos=infinity;
      } else count=Step > 0 ? floor(lastpos/Step)-ceil(firstpos/Step)+1 : 0;
      
      begingroup(f);
      if(opposite) draw(f,G,p);
      else draw(f,G,p,arrow);
      
      if(Size > 0) {
	int c=0;
	for(int i=0; i <= N; ++i) {
	  real pos=i*Step;
	  if(pos >= firstpos && pos <= lastpos) {
	    ++c;
	    if((begin || c > 1) && (end || c < count))
	      drawtick(f,T,g,g2,pos+offset,Size,sign,pTick,extend);
	  }
	}
	for(int i=0; i <= N; ++i) {
	  if(size > 0 && step > 0) {
	    real iStep=i*Step;
	    real jstop=(len-iStep)/step;
	    for(int j=1; j < n && j <= jstop; ++j) {
	      real pos=iStep+j*step;
	      if(pos >= firstpos && pos <= lastpos)
		drawtick(f,T,g,g2,pos+offset,size,sign,ptick,extend);
	    }
	  }
	}
      }
      endgroup(f);
    
      if(Size > 0 && !opposite) {
	int c=0;
	for(int i=0; i <= N; ++i) {
	  real pos=i*Step;
	  if(pos >= firstpos && pos <= lastpos) {
	    ++c;
	    if((beginlabel || c > 1) && (endlabel || c < count)) {
	      ticklabels=true;
	      labeltick(f,T,g,pos+offset,side,sign,Size,ticklabel,pTicklabel,
			part,norm);
	    }
	  }
	}
      }

    } else { // Logarithmic
      real initial=part(point(g,0));
      real final=part(point(g,length(g)));
      int first=ceil(initial-epsilon);
      int last=floor(final+epsilon);
      real len=arclength(g);
      real denom=final-initial;
      Step=denom != 0 ? len/denom : len;
      real offset=-initial*Step;
      real firstpos=-epsilon*len;
      real lastpos=len-firstpos-offset;
      firstpos -= offset;
    
      if(N == 0) {
	real limit=axiscoverage*arclength(G);
	N=1;
	while(N <= last-first) {
	  if(logaxiscoverage(N,T,g,Step,offset,side,sign,Size,pTicklabel,
			     part,limit,first,last,firstpos,lastpos) <= limit)
	    break;
	  ++N;
	}
      }
      
      if(N <= 2 && n == 0) n=10;
      
      int count=Step > 0 ? floor(lastpos/Step)-ceil(firstpos/Step)+1 : 0;
      
      begingroup(f);
      if(opposite) draw(f,G,p);
      else draw(f,G,p,arrow);

      if(N > 0) {
	int c=0;
	for(int i=first-1; i <= last+1; ++i) {
	  real pos=i*Step;
	  if(pos >= firstpos && pos <= lastpos) {
	    ++c;
	    if((begin || c > 1) && (end || c < count)) {
	      real Size0=((i-first) % N == 0 || n != 0) ? Size : size;
	      drawtick(f,T,g,g2,pos+offset,Size0,sign,pTick,extend);
	    }
	  }
	}
	for(int i=first-1; i <= last+1; ++i) {
	  if(n > 0) {
	    for(int j=2; j < n; ++j) {
	      real pos=(i+1+log10(j/n))*Step;
	      if(pos >= firstpos && pos <= lastpos) {
		drawtick(f,T,g,g2,pos+offset,size,sign,ptick,extend);
	      }
	    }
	  }
	}
      }
      endgroup(f);
      
      if(!opposite && N > 0) {
	int c=0;
	for(int i=first-1; i <= last+1; i += N) {
	  real pos=i*Step;
	  if(pos >= firstpos && pos <= lastpos) {
	    ++c;
	    if((beginlabel || c > 1) && (endlabel || c < count)) {
	      ticklabels=true;
	      labeltick(f,T,g,pos+offset,side,sign,Size,LogFormat,pTicklabel,
			part);
	    }
	  }
	}
      }
    }
    
    if(L.s != "" && !opposite) 
      labelaxis(f,L,G,ticklabels);
  };
}

ticks Ticks(int sign, real[] Ticks, real[] ticks=new real[],
	    real Size=Ticksize, real size=ticksize,
	    bool beginlabel=true, bool endlabel=true,
	    Label F=defaultformat, bool extend=false,
	    pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(frame f, transform T, Label L, pair side, path G, path G2, 
		  pen p, arrowbar arrow, autoscaleT S, part part,
		  bool opposite, int[] divisor, real tickmin, real tickmax) {
    // Use local copy of context variables:
    int sign=opposite ? -sign : sign;
    pen pTicklabel=F.p;
    pen pTick=pTick;
    pen ptick=ptick;
    
    Label L=L.copy();
    L.p(p);
    if(pTicklabel == nullpen) pTicklabel=p;
    if(pTick == nullpen) pTick=p;
    if(ptick == nullpen) ptick=p;
    
    ticklabel ticklabel=ticklabel(F.s);
    if(!F.align.side.alias(NoSide)) side=F.align.side.align((1,0));
    if(side == 0 && !F.align.side.alias(Center))
      side=(sign == 1) ? left : right;
    
    bool ticklabels=false;
    guide g=inverse(T)*G;
    guide g2=inverse(T)*G2;
    
    real a=part(point(g,0));
    real b=part(point(g,length(g)));
    real norm=max(abs(a),abs(b));
    if(logarithmic(S.scale)) ticklabel=new string(real x) {
      return format("$10^{%g}$",x);
    };

    begingroup(f);
    if(opposite) draw(f,G,p);
    else draw(f,G,p,arrow);
    for(int i=0; i < Ticks.length; ++i) {
      real pos=S.T(Ticks[i])-a;
      drawtick(f,T,g,g2,pos,Size,sign,pTick,extend);
    }
    for(int i=0; i < ticks.length; ++i) {
      real pos=S.T(ticks[i])-a;
      drawtick(f,T,g,g2,pos,size,sign,ptick,extend);
    }
    endgroup(f);
    
    if(Size > 0 && !opposite) {
      for(int i=(beginlabel ? 0 : 1);
	  i < (endlabel ? Ticks.length : Ticks.length-1); ++i) {
	real pos=S.T(Ticks[i])-a;
	ticklabels=true;
	labeltick(f,T,g,pos,side,sign,Size,ticklabel,pTicklabel,part,norm);
      }
    }
    if(L.s != "" && !opposite) 
      labelaxis(f,L,G,ticklabels);
  };
}

ticks NoTicks() {
  return new void(frame f, transform, Label L, pair, path G, path,
		  pen p, arrowbar arrow, autoscaleT, part, bool opposite,
		  int[], real, real) {
    if(opposite) draw(f,G,p);
    else {
      draw(f,G,p,arrow);
      if(L.s != "")
	labelaxis(f,L,G);
    }
  };
}

ticks LeftTicks(bool begin=true, bool end=true, int N=0, int n=0,
		real Step=0, real step=0,
		real Size=Ticksize, real size=ticksize,
		bool beginlabel=true, bool endlabel=true,
		Label format=defaultformat, bool extend=false,
		pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(begin,end,-1,N,n,Step,step,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

ticks RightTicks(bool begin=true, bool end=true, int N=0, int n=0,
		 real Step=0, real step=0,
		 real Size=Ticksize, real size=ticksize,
		 bool beginlabel=true, bool endlabel=true,
		 Label format=defaultformat, bool extend=false,
		 pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(begin,end,1,N,n,Step,step,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

ticks LeftTicks(real[] Ticks, real[] ticks=new real[],
		real Size=Ticksize, real size=ticksize,
		bool beginlabel=true, bool endlabel=true, 
		Label format=defaultformat, bool extend=false,
		pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(-1,Ticks,ticks,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

ticks RightTicks(real[] Ticks, real[] ticks=new real[],
		 real Size=Ticksize, real size=ticksize,
		 bool beginlabel=true, bool endlabel=true,
		 Label format=defaultformat, bool extend=false,
		 pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(1,Ticks,ticks,Size,size,beginlabel,endlabel,format,
	       extend,pTick,ptick);
}

public ticks
  NoTicks=NoTicks(),
  LeftTicks=LeftTicks(),
  RightTicks=RightTicks();

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
    axis.value2=infinity;
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
    axis.value2=infinity;
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
    axis.value2=infinity;
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
    axis.value2=infinity;
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
    axis.value2=infinity;
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
    axis.value2=infinity;
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

void axis(picture pic=currentpicture, guide g,
	  real tickmin=-infinity, real tickmax=infinity,
	  Label L="", pen p=currentpen, ticks ticks=NoTicks,
	  arrowbar arrow=None, int[] divisor=new int[],
	  bool logarithmic=false, scaleT scale=Linear, part part,
	  bool put=Above, bool opposite=false) 
{
  autoscaleT S;
  S.scale=scale;
  divisor=copy(divisor);
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    frame d;
    guide G=t*g;
    ticks(d,t,L,0,G,G,p,arrow,S,part,opposite,divisor,tickmin,tickmax);
    (put ? add : prepend)(f,t*T*inverse(t)*d);
  });
  
  pic.addPath(g,p);
  
  if(L.s != "") {
    frame f;
    Label L0=L.copy();
    L0.position(0);
    add(f,L0);
    pair pos=point(g,L.position.x*length(g));
    pic.addBox(pos,pos,min(f),max(f));
  }
}

void xaxisAt(picture pic=currentpicture,
	     real xmin=-infinity, real xmax=infinity, axis axis, Label L="",
	     pen p=currentpen, ticks ticks=NoTicks, arrowbar arrow=None,
	     bool put=Above, bool opposite=false)
{
  real y=opposite ? axis.value2.y : axis.value.y;
  real y2=axis.value2.y;
  Label L=L.copy();
  int[] divisor=copy(axis.xdivisor);
  pair side=axis.side;
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    pair a=xmin == -infinity ? (lb.x-min(p).x,ytrans(t,y)) : t*(xmin,y);
    pair b=xmax == infinity ? (rt.x-max(p).x,ytrans(t,y)) : t*(xmax,y);
    pair a2=xmin == -infinity ? (lb.x-min(p).x,ytrans(t,y2)) : t*(xmin,y2);
    pair b2=xmax == infinity ? (rt.x-max(p).x,ytrans(t,y2)) : t*(xmax,y2);
    frame d;
    ticks(d,t,L,side,a--b,a2--b2,p,arrow,
	  pic.scale.x,new real(pair z) {return pic.scale.x.Label(z.x);},
	  opposite,divisor,pic.scale.x.tickMin,pic.scale.x.tickMax);
    (put ? add : prepend)(f,t*T*inverse(t)*d);
  });

  pair a=(finite(xmin) ? xmin : pic.userMin.x,y);
  pair b=(finite(xmax) ? xmax : pic.userMax.x,y);
  pair a2=(finite(xmin) ? xmin : pic.userMin.x,y2);
  pair b2=(finite(xmax) ? xmax : pic.userMax.x,y2);
  
  pic.addPoint(a,min(p));
  pic.addPoint(a,max(p));
  pic.addPoint(b,min(p));
  pic.addPoint(b,max(p));
  
  if(finite(a) && finite(b)) {
    frame d;
    ticks(d,identity(),L,side,(a.x,0)--(b.x,0),(a2.x,0)--(b2.x,0),p,arrow,
	  pic.scale.x,new real(pair z) {return pic.scale.y.Label(z.x);},
	  opposite,divisor,pic.scale.x.tickMin,pic.scale.x.tickMax);
    frame f;
    if(L.s != "") {
      Label L0=L.copy();
      L0.position(0);
      add(f,L0);
    }
    pair pos=a+L.position.x*(b-a);
    pic.addBox(pos,pos,(min(f).x,min(d).y),(max(f).x,max(d).y));
  }
}

void yaxisAt(picture pic=currentpicture,
	     real ymin=-infinity, real ymax=infinity, axis axis, Label L="",
	     pen p=currentpen, ticks ticks=NoTicks, arrowbar arrow=None,
	     bool put=Above, bool opposite=false)
{
  real x=opposite ? axis.value2.x : axis.value.x;
  real x2=axis.value2.x;
  Label L=L.copy();
  int[] divisor=copy(axis.ydivisor);
  pair side=axis.side;
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
    pair a=ymin == -infinity ? (xtrans(t,x),lb.y-min(p).y) : t*(x,ymin);
    pair b=ymax == infinity ? (xtrans(t,x),rt.y-max(p).y) : t*(x,ymax);
    pair a2=ymin == -infinity ? (xtrans(t,x2),lb.y-min(p).y) : t*(x2,ymin);
    pair b2=ymax == infinity ? (xtrans(t,x2),rt.y-max(p).y) : t*(x2,ymax);
    frame d;
    ticks(d,t,L,side,a--b,a2--b2,p,arrow,pic.scale.y,
	  new real(pair z) {return pic.scale.y.Label(z.y);},
	  opposite,divisor,pic.scale.y.tickMin,pic.scale.y.tickMax);
    (put ? add : prepend)(f,t*T*inverse(t)*d);
  });
  
  pair a=(x,finite(ymin) ? ymin : pic.userMin.y);
  pair b=(x,finite(ymax) ? ymax : pic.userMax.y);
  pair a2=(x2,finite(ymin) ? ymin : pic.userMin.y);
  pair b2=(x2,finite(ymax) ? ymax : pic.userMax.y);
  
  pic.addPoint(a,min(p));
  pic.addPoint(a,max(p));
  pic.addPoint(b,min(p));
  pic.addPoint(b,max(p));
  
  if(finite(a) && finite(b)) {
    frame d;
    ticks(d,identity(),L,side,(0,a.y)--(0,b.y),(0,a2.y)--(0,b2.y),p,arrow,
	  pic.scale.y,new real(pair z) {return pic.scale.y.Label(z.y);},
	  opposite,divisor,pic.scale.y.tickMin,pic.scale.y.tickMax);
    frame f;
    if(L.s != "") {
      Label L0=L.copy();
      L0.position(0);
      add(f,L0);
    }
    pair pos=a+L.position.x*(b-a);
    pic.addBox(pos,pos,(min(d).x,min(f).y),(max(d).x,max(f).y));
  }
}

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

void crop(picture pic=currentpicture) 
{
  xlimits(pic,false);
  ylimits(pic,false);
  if(finite(pic.userMin) && finite(pic.userMax))
    clip(pic,box(pic.userMin,pic.userMax));
}

void limits(picture pic=currentpicture, pair min, pair max)
{
  xlimits(pic,min.x,max.x);
  ylimits(pic,min.y,max.y);
}
  
void autoscale(picture pic=currentpicture, axis axis)
{
  if(!pic.scale.set) {
    bounds mx,my;
    pic.scale.set=true;
    
    if(finite(pic.userMin.x) && finite(pic.userMax.x)) {
      mx=autoscale(pic.userMin.x,pic.userMax.x,pic.scale.x.scale);
      if(logarithmic(pic.scale.x.scale) && 
	 floor(pic.userMin.x) == floor(pic.userMax.x)) {
	pic.userMin=(floor(pic.userMin.x),pic.userMin.y);
	pic.userMax=(ceil(pic.userMax.x),pic.userMax.y);
      }
    } else {mx.min=mx.max=0; pic.scale.set=false;}
    
    if(finite(pic.userMin.y) && finite(pic.userMax.y)) {
      my=autoscale(pic.userMin.y,pic.userMax.y,pic.scale.y.scale);
      if(logarithmic(pic.scale.y.scale) && 
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

void checkaxis(picture pic, axis axis) 
{
  axis(pic,axis);
  if(!axis.extend && pic.empty()) abort("unextended axis called before draw");
}

void xaxis(picture pic=currentpicture, real xmin=-infinity, real xmax=infinity,
	   Label L="", pen p=currentpen, axis axis=YZero,
	   ticks ticks=NoTicks, arrowbar arrow=None, bool put=Below)
{
  Label L=L.copy();
  bool newticks=false;
  if(xmin != -infinity) {
    pic.userMin=(xmin,pic.userMin.y);
    newticks=true;
  }
  if(xmax != infinity) {
    pic.userMax=(xmax,pic.userMax.y);
    newticks=true;
  }
  
  if(pic.scale.set && newticks) {
    bounds mx=autoscale(xmin,xmax,pic.scale.x.scale);
    pic.scale.x.tickMin=mx.min;
    pic.scale.x.tickMax=mx.max;
    axis.xdivisor=mx.divisor;
  } else autoscale(pic,axis);
  
  checkaxis(pic,axis);
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
  
  xaxisAt(pic,xmin,xmax,axis,L,p,ticks,arrow,put);
  if(axis.value2 != infinity)
    xaxisAt(pic,xmin,xmax,axis,L,p,ticks,arrow,put,true);
}

void yaxis(picture pic=currentpicture, real ymin=-infinity, real ymax=infinity,
	   Label L="", pen p=currentpen, axis axis=XZero,
	   ticks ticks=NoTicks, arrowbar arrow=None, bool put=Below)
{
  Label L=L.copy();
  bool newticks=false;
  if(ymin != -infinity) {
    pic.userMin=(pic.userMin.x,ymin);
    newticks=true;
  }
  if(ymax != infinity) {
    pic.userMax=(pic.userMax.x,ymax);
    newticks=true;
  }
  
  if(pic.scale.set && newticks) {
    bounds my=autoscale(ymin,ymax,pic.scale.y.scale);
    pic.scale.y.tickMin=my.min;
    pic.scale.y.tickMax=my.max;
    axis.ydivisor=my.divisor;
  } else autoscale(pic,axis);
  
  checkaxis(pic,axis);
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
  
  yaxisAt(pic,ymin,ymax,axis,L,p,ticks,arrow,put);
  if(axis.value2 != infinity)
    yaxisAt(pic,ymin,ymax,axis,L,p,ticks,arrow,put,true);
}

void axes(picture pic=currentpicture, pen p=currentpen, bool put=Below)
{
  xaxis(pic,p,put);
  yaxis(pic,p,put);
}

void xequals(picture pic=currentpicture, real x, bool extend=false,
	     real ymin=-infinity, real ymax=infinity, Label L="",
	     pen p=currentpen, ticks ticks=NoTicks, bool put=Above,
	     arrowbar arrow=None)
{
  yaxis(pic,ymin,ymax,L,p,XEquals(x,extend),ticks,arrow,put);
}

void yequals(picture pic=currentpicture, real y, bool extend=false,
	     real xmin=-infinity, real xmax=infinity, Label L="",
	     pen p=currentpen, ticks ticks=NoTicks, bool put=Above,
	     arrowbar arrow=None)
{
  xaxis(pic,xmin,xmax,L,p,YEquals(y,extend),ticks,arrow,put);
}

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
	   string format=defaultformat, real size=Ticksize, pen p=currentpen)
{
  Label L=L.copy();
  if(L.defaultposition) L.position(z);
  if(L.align.default) {
    L.align(-dir);
  } else if(L.shift == 0) 
    L.shift(dot(dir,L.align.dir) > 0 ? dir*size :
	    ticklabelshift(L.align.dir,p));
  L.p(p);
  if(L.s == "") L.s=format(format,z.x);
  add(pic,L);
  tick(pic,z,dir,size,p);
}

void ytick(picture pic=currentpicture, Label L, explicit pair z, pair dir=E,
	   string format=defaultformat, real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,z,dir,format,size,p);
}

void ytick(picture pic=currentpicture, Label L, real y, pair dir=E,
	   string format=defaultformat, real size=Ticksize, pen p=currentpen)
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

void labelx(picture pic=currentpicture, Label L="", pair z, align align=S,
	    string format=defaultformat, pen p=nullpen)
{
  label(pic,L,z,z.x,align,format,p);
}

void labelx(picture pic=currentpicture, Label L,
	    string format=defaultformat, explicit pen p=currentpen)
{
  labelx(pic,L,L.position,format,p);
}

void labely(picture pic=currentpicture, Label L="", explicit pair z,
	    align align=W, string format=defaultformat, pen p=nullpen)
{
  label(pic,L,z,z.y,align,format,p);
}

void labely(picture pic=currentpicture, Label L="", real y,
	    align align=W, string format=defaultformat, pen p=nullpen)
{
  labely(pic,L,(0,y),align,format,p);
}

void labely(picture pic=currentpicture, Label L,
	    string format=defaultformat, explicit pen p=nullpen)
{
  labely(pic,L,L.position,format,p);
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
  return graph(join)(new pair (real x) {
    return (x,pic.scale.y.T(f(pic.scale.x.Tinv(x))));},
			 pic.scale.x.T(a),pic.scale.x.T(b),n);
}

guide graph(picture pic=currentpicture, real x(real), real y(real), real a,
	    real b, int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new pair (real t) {return Scale(pic,(x(t),y(t)));},
			   a,b,n);
}

guide graph(picture pic=currentpicture, pair z(real), real a, real b,
	    int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new pair (real t) {return Scale(pic,z(t));},
			   a,b,n);
}

private int next(int i, bool[] cond)
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
  return graph(join)(new pair (real) {
    i=next(i,cond);
    return Scale(pic,z[i]);},0,0,n);
}

private string differentlengths="attempt to graph arrays of different lengths";

guide graph(picture pic=currentpicture, real[] x, real[] y, bool[] cond={},
	    interpolate join=operator --)
{
  if(x.length != y.length) abort(differentlengths);
  int n=conditional(x,cond);
  int i=-1;
  return graph(join)(new pair (real) {
    i=next(i,cond);
    return Scale(pic,(x[i],y[i]));},0,0,n);
}

guide graph(real f(real), real a, real b, int n=ngraph,
	    real T(real), interpolate join=operator --)
{
  return graph(join)(new pair (real x) {return (T(x),f(T(x)));},a,b,n);
}

pair polar(real r, real theta)
{
  return r*expi(theta);
}

guide polargraph(real f(real), real a, real b, int n=ngraph,
		 interpolate join=operator --)
{
  return graph(join)(new pair (real theta) {
      return f(theta)*expi(theta);
    },a,b,n);
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

// True arc
guide Arc(pair c, real r, real angle1, real angle2)
{
  return shift(c)*polargraph(new real (real t){return r;},angle1,angle2,
  operator ..);
}

// True circle
guide Circle(pair c, real r)
{
  return Arc(c,r,0,2pi)--cycle;
}
