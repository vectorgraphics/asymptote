private import math;
import graph_splinetype;
import graph_settings;

scaleT Linear;

scaleT Log=scaleT(log10,pow10,logarithmic=true);
scaleT Logarithmic=Log;

string baselinetemplate="$10^4$";

// A linear scale, with optional autoscaling of minimum and maximum values,
// scaling factor s and intercept.
scaleT Linear(bool automin=false, bool automax=automin, real s=1,
              real intercept=0)
{
  real sinv=1/s;
  scalefcn T,Tinv;
  if(s == 1 && intercept == 0)
    T=Tinv=identity;
  else {
    T=new real(real x) {return (x-intercept)*s;};
    Tinv=new real(real x) {return x*sinv+intercept;};
  }
  return scaleT(T,Tinv,logarithmic=false,automin,automax);
}

// A logarithmic scale, with optional autoscaling of minimum and maximum
// values.
scaleT Log(bool automin=false, bool automax=automin)
{
  return scaleT(Log.T,Log.Tinv,logarithmic=true,automin,automax);
}

// A "broken" linear axis omitting the segment [a,b].
scaleT Broken(real a, real b, bool automin=false, bool automax=automin)
{
  real skip=b-a;
  real T(real x) {
    if(x <= a) return x;
    if(x <= b) return a;
    return x-skip;
  }
  real Tinv(real x) {
    if(x <= a) return x; 
    return x+skip; 
  }
  return scaleT(T,Tinv,logarithmic=false,automin,automax);
}

// A "broken" logarithmic axis omitting the segment [a,b], where a and b are
// automatically rounded to the nearest integral power of the base.  
scaleT BrokenLog(real a, real b, bool automin=false, bool automax=automin)
{
  real A=round(Log.T(a));
  real B=round(Log.T(b));
  a=Log.Tinv(A);
  b=Log.Tinv(B);
  real skip=B-A;
  real T(real x) {
    if(x <= a) return Log.T(x);
    if(x <= b) return A;
    return Log.T(x)-skip;
  }
  real Tinv(real x) {
    real X=Log.Tinv(x);
    if(X <= a) return X;
    return Log.Tinv(x+skip);
  }
  return scaleT(T,Tinv,logarithmic=true,automin,automax);
}

Label Break=Label("$\approx$",UnFill(0.2mm));

void scale(picture pic=currentpicture, scaleT x, scaleT y=x, scaleT z=y)
{
  pic.scale.x.scale=x;
  pic.scale.y.scale=y;
  pic.scale.z.scale=z;
  pic.scale.x.automin=x.automin;
  pic.scale.y.automin=y.automin;
  pic.scale.z.automin=z.automin;
  pic.scale.x.automax=x.automax;
  pic.scale.y.automax=y.automax;
  pic.scale.z.automax=z.automax;
}

void scale(picture pic=currentpicture, bool xautoscale=false,
           bool yautoscale=xautoscale, bool zautoscale=yautoscale)
{
  scale(pic,Linear(xautoscale,xautoscale),Linear(yautoscale,yautoscale),
        Linear(zautoscale,zautoscale));
}

struct scientific 
{
  int sign;
  real mantissa;
  int exponent;
  int ceil() {return sign*ceil(mantissa);}
  real scale(real x, real exp) {
    static real max=0.1*realMax;
    static real limit=-log10(max);
    return x*(exp > limit ? 10^-exp : max);
  }
  real ceil(real x, real exp) {return ceil(sign*scale(abs(x),exp));}
  real floor(real x, real exp) {return floor(sign*scale(abs(x),exp));}
}

// Convert x to scientific notation
scientific scientific(real x) 
{
  scientific s;
  s.sign=sgn(x);
  x=abs(x);
  if(x == 0) {s.mantissa=0; s.exponent=-intMax; return s;}
  real logx=log10(x);
  s.exponent=floor(logx);
  s.mantissa=s.scale(x,s.exponent);
  return s;
}

// Autoscale limits and tick divisor.
struct bounds {
  real min;
  real max;

  // Possible tick intervals:
  int[] divisor;

  void operator init(real min, real max, int[] divisor=new int[]) {
    this.min=min;
    this.max=max;
    this.divisor=divisor;
  }
}

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

  void zoom() {
    --exp;
    a=sa.floor(Min,exp);
    b=sb.ceil(Max,exp);
  }

  if(sb.mantissa <= 1.5)
    zoom();

  while((b-a)*10.0^exp > 10*(Max-Min))
    zoom();
  
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

ticklabel Format(string s=defaultformat)
{
  return new string(real x) {return format(s,x);};
}

ticklabel OmitFormat(string s=defaultformat ... real[] x)
{
  return new string(real v) {
    string V=format(s,v);
    for(real a : x)
      if(format(s,a) == V) return "";
    return V;
  };
}

string trailingzero="$%#$";
string signedtrailingzero="$%+#$";

ticklabel DefaultFormat=Format();
ticklabel NoZeroFormat=OmitFormat(0);

// Format tick values as integral powers of base; otherwise with DefaultFormat.
ticklabel DefaultLogFormat(int base) {
  return new string(real x) {
    string exponent=format("%.4f",log(x)/log(base));
    return find(exponent,".") == -1 ?
      "$"+(string) base+"^{"+exponent+"}$" : format(x);
  };
}

// Format all tick values as powers of base.
ticklabel LogFormat(int base) {
  return new string(real x) {
    return format("$"+(string) base+"^{%g}$",log(x)/log(base));
  };
}

ticklabel LogFormat=LogFormat(10);
ticklabel DefaultLogFormat=DefaultLogFormat(10);
  
// The default direction specifier.
pair zero(real) {return 0;}

struct ticklocate {
  real a,b;            // Tick values at point(g,0), point(g,length(g)).
  autoscaleT S;        // Autoscaling transformation.
  pair dir(real t);    // Absolute 2D tick direction.
  triple dir3(real t); // Absolute 3D tick direction.
  real time(real v);   // Returns the time corresponding to the value v. 
  ticklocate copy() {
    ticklocate T=new ticklocate;
    T.a=a;
    T.b=b;
    T.S=S.copy();
    T.dir=dir;
    T.dir3=dir3;
    T.time=time;
    return T;
  }
}

autoscaleT defaultS;
  
typedef real valuetime(real);

valuetime linear(scalefcn S=identity, real Min, real Max)
{
  real factor=Max == Min ? 0.0 : 1.0/(Max-Min);
  return new real(real v) {return (S(v)-Min)*factor;};
}

ticklocate ticklocate(real a, real b, autoscaleT S=defaultS,
                      real tickmin=-infinity, real tickmax=infinity,
                      real time(real)=null, pair dir(real)=zero) 
{
  if((valuetime) time == null) time=linear(S.T(),a,b);
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
  
  void dir(transform T, path g, ticklocate locate, real t) {
    pathdir=unit(shiftless(T)*dir(g,t));
    pair Dir=locate.dir(t);
    dir=Dir == 0 ? -I*pathdir : unit(Dir);
  }
  // Locate the desired position of a tick along a path.
  void calc(transform T, path g, ticklocate locate, real val) {
    t=locate.time(val);
    Z=T*point(g,t);
    dir(T,g,locate,t);
  }
}

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
    if(sign == 0) 
      draw(f,locate1.Z-Size*locate1.dir--locate1.Z+Size*locate1.dir,p);
    else
      draw(f,locate1.Z--locate1.Z+Size*sign*locate1.dir,p);
}

real zerotickfuzz=10*epsilon;

// Label a tick on a frame.
pair labeltick(frame d, transform T, path g, ticklocate locate, real val,
               pair side, int sign, real Size, ticklabel ticklabel,
               Label F, real norm=0)
{
  locateT locate1;
  locate1.calc(T,g,locate,val);
  pair align=side*locate1.dir;
  pair perp=I*locate1.pathdir;

  // Adjust tick label alignment
  pair adjust=unit(align+0.75perp*sgn(dot(align,perp)));
  // Project align onto adjusted direction.
  align=adjust*dot(align,adjust);
  pair shift=dot(align,-sign*locate1.dir) <= 0 ? align*Size :
    ticklabelshift(align,F.p);

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
  if(s != "")
    label(d,F.T*baseline(s,baselinetemplate),locate1.Z+shift,align,F.p,
          F.filltype);
  return locate1.pathdir;
}  

// Add axis label L to frame f.
void labelaxis(frame f, transform T, Label L, path g, 
               ticklocate locate=null, int sign=1, bool ticklabels=false)
{
  Label L0=L.copy();
  real t=L0.relative(g);
  pair z=point(g,t);
  pair dir=dir(g,t);
  pair perp=I*dir;
  if(locate != null) {
    locateT locate1;
    locate1.dir(T,g,locate,t);
    L0.align(L0.align,unit(-sgn(dot(sign*locate1.dir,perp))*perp));
  }                  
  pair align=L0.align.dir;
  if(L0.align.relative) align *= -perp;
  pair alignperp=dot(align,perp)*perp;
  pair offset;
  if(ticklabels) {
    if(piecewisestraight(g)) {
      real angle=degrees(dir,warn=false);
      transform S=rotate(-angle,z);
      frame F=S*f;
      pair Align=rotate(-angle)*alignperp;
      offset=unit(alignperp-sign*locate.dir(t))*
        abs((Align.y >= 0 ? max(F).y : (Align.y < 0 ? min(F).y : 0))-z.y);
    }
    z += offset;
  }

  L0.align(align);
  L0.position(z);
  frame d;
  add(d,L0);
  pair width=0.5*size(d);
  int n=length(g);
  real t=L.relative();
  pair s=realmult(width,dir(g,t));
  if(t <= 0) {
    if(L.align.default) s *= -axislabelfactor;
    d=shift(s)*d;
  } else if(t >= n) {
    if(L.align.default) s *= -axislabelfactor;
    d=shift(-s)*d;
  } else if(offset == 0 && L.align.default) {
    pair s=realmult(width,I*dir(g,t));
    s=axislabelfactor*s;
    d=shift(s)*d;
  }
  add(f,d);
}

// Check the tick coverage of a linear axis.
bool axiscoverage(int N, transform T, path g, ticklocate locate, real Step,
                  pair side, int sign, real Size, Label F, ticklabel ticklabel,
                  real norm, real limit)
{
  real coverage=0;
  bool loop=cyclic(g);
  real a=locate.S.Tinv(locate.a);
  real b=locate.S.Tinv(locate.b);
  real tickmin=finite(locate.S.tickMin) ? locate.S.Tinv(locate.S.tickMin) : a;
  if(Size > 0) {
    int count=0;
    if(loop) count=N+1;
    else {
      for(int i=0; i <= N; ++i) {
        real val=tickmin+i*Step;
        if(val >= a && val <= b)
          ++count;
      }
    }
    if(count > 0) limit /= count;
    for(int i=0; i <= N; ++i) {
      real val=tickmin+i*Step;
      if(loop || (val >= a && val <= b)) {
        frame d;
        pair dir=labeltick(d,T,g,locate,val,side,sign,Size,ticklabel,F,norm);
        if(abs(dot(size(d),dir)) > limit) return false;
      }
    }
  }
  return true;
}

// Check the tick coverage of a logarithmic axis.
bool logaxiscoverage(int N, transform T, path g, ticklocate locate, pair side,
                     int sign, real Size, Label F, ticklabel ticklabel, 
                     real limit, int first, int last)
{
  bool loop=cyclic(g);
  real coverage=0;
  real a=locate.a;
  real b=locate.b;
  int count=0;
  for(int i=first-1; i <= last+1; i += N) {
    if(loop || i >= a && i <= b)
      ++count;
  }
  if(count > 0) limit /= count;
  for(int i=first-1; i <= last+1; i += N) {
    if(loop || i >= a && i <= b) {
      frame d;
      pair dir=labeltick(d,T,g,locate,i,side,sign,Size,ticklabel,F);
      if(abs(dot(size(d),dir)) > limit) return false;
    }
  }
  return true;
}

struct tickvalues {
  real[] major;
  real[] minor;
  int N; // For logarithmic axes: number of decades between tick labels.
}

// Determine a format that distinguishes adjacent pairs of ticks, optionally
// adding trailing zeros.
string autoformat(string format="", real norm ... real[] a)
{
  bool trailingzero=(format == trailingzero);
  bool signedtrailingzero=(format == signedtrailingzero);
  if(!trailingzero && !signedtrailingzero && format != "") return format;

  real[] A=sort(a);
  real[] a=abs(A);

  bool signchange=(A.length > 1 && A[0] < 0 && A[A.length-1] >= 0);

  for(int i=0; i < A.length; ++i)
    if(a[i] < zerotickfuzz*norm) A[i]=a[i]=0;

  int n=0;

  bool Fixed=find(a >= 1e4-epsilon | (a > 0 & a <= 1e-4-epsilon)) < 0;
  
  string Format=defaultformat(4,fixed=Fixed);

  if(Fixed && n < 4) {
    for(int i=0; i < A.length; ++i) {
      real a=A[i];
      while(format(defaultformat(n,fixed=Fixed),a) != format(Format,a))
        ++n;
    }
  }

  string trailing=trailingzero ? (signchange ? "# " : "#") :
    signedtrailingzero ? "#+" : "";

  string format=defaultformat(n,trailing,Fixed);

  for(int i=0; i < A.length-1; ++i) {
    real a=A[i];
    real b=A[i+1];
    // Check if an extra digit of precision should be added.
    string fixedformat="%#."+string(n+1)+"f";
    string A=format(fixedformat,a);
    string B=format(fixedformat,b);
    if(substr(A,length(A)-1,1) != "0" || substr(B,length(B)-1,1) != "0") {
      a *= 0.1;
      b *= 0.1;
    }
    if(a != b) {
      while(format(format,a) == format(format,b))
        format=defaultformat(++n,trailing,Fixed);
    }
  }

  if(n == 0) return defaultformat;
  return format;
}

// Automatic tick generation routine.
tickvalues generateticks(int sign, Label F="", ticklabel ticklabel=null,
                         int N, int n=0, real Step=0, real step=0,
                         real Size=0, real size=0,
                         transform T, pair side, path g, real limit,
                         pen p, ticklocate locate, int[] divisor,
                         bool opposite)
{
  tickvalues tickvalues;
  sign=opposite ? -sign : sign;
  if(Size == 0) Size=Ticksize;
  if(size == 0) size=ticksize;
  F=F.copy();
  F.p(p);
    
  if(F.align.dir != 0) side=F.align.dir;
  else if(side == 0) side=((sign == 1) ? left : right);
    
  bool ticklabels=false;
  path G=T*g;
    
  if(!locate.S.scale.logarithmic) {
    real a=locate.S.Tinv(locate.a);
    real b=locate.S.Tinv(locate.b);
    real norm=max(abs(a),abs(b));
    string format=autoformat(F.s,norm,a,b);
    if(F.s == "%") F.s="";
    if(ticklabel == null) ticklabel=Format(format);

    if(a > b) {real temp=a; a=b; b=temp;}

    if(b-a < 100.0*epsilon*norm) b=a;
      
    bool autotick=Step == 0 && N == 0;
    
    real tickmin=finite(locate.S.tickMin) && (autotick || locate.S.automin) ? 
      locate.S.Tinv(locate.S.tickMin) : a;
    real tickmax=finite(locate.S.tickMax) && (autotick || locate.S.automax) ?
      locate.S.Tinv(locate.S.tickMax) : b;
    if(tickmin > tickmax) {real temp=tickmin; tickmin=tickmax; tickmax=temp;}
      
    real inStep=Step;

    bool calcStep=true;
    real len=tickmax-tickmin;
    if(autotick) {
      N=1;
      if(divisor.length > 0) {
        bool autoscale=locate.S.automin && locate.S.automax;
        real h=0.5*(b-a);
        if(h > 0) {
          for(int d=divisor.length-1; d >= 0; --d) {
            int N0=divisor[d];
            Step=len/N0;
            int N1=N0;
            int m=2;
            while(Step > h) {
              N0=m*N1;
              Step=len/N0;
              m *= 2;
            }
            if(axiscoverage(N0,T,g,locate,Step,side,sign,Size,F,ticklabel,norm,
                            limit)) {
              N=N0;
              if(N0 == 1 && !autoscale && d < divisor.length-1) {
                // Try using 2 ticks (otherwise 1);
                int div=divisor[d+1];
                Step=quotient(div,2)*len/div;
                calcStep=false; 
                if(axiscoverage(2,T,g,locate,Step,side,sign,Size,F,ticklabel,
                                norm,limit)) N=2;
                else Step=len;
              }
              // Found a good divisor; now compute subtick divisor
              if(n == 0) {
                if(step != 0) n=ceil(Step/step);
                else {
                  n=quotient(divisor[divisor.length-1],N);
                  if(N == 1) n=(a*b >= 0) ? 2 : 1;
                  if(n == 1) n=2;
                }
              }
              break;
            }
          }
        }
      }
    }
      
    if(inStep != 0 && !locate.S.automin) {
      tickmin=floor(tickmin/Step)*Step;
      len=tickmax-tickmin;
    }

    if(calcStep) {
      if(N == 1) N=2;
      if(N == 0) N=(int) (len/Step);
      else Step=len/N;
    }
    
    if(n == 0) {
      if(step != 0) n=ceil(Step/step);
    } else step=Step/n;
      
    b += epsilon*norm;
      
    if(Size > 0) {
      for(int i=0; i <= N; ++i) {
        real val=tickmin+i*Step;
        if(val >= a && val <= b)
          tickvalues.major.push(val);
        if(size > 0 && step > 0) {
          real iStep=i*Step;
          real jstop=(len-iStep)/step;
          for(int j=1; j < n && j <= jstop; ++j) {
            real val=tickmin+iStep+j*step;
            if(val >= a && val <= b)
              tickvalues.minor.push(val);
          }
        }
      }
    }
    
  } else { // Logarithmic
    string format=F.s;
    if(F.s == "%") F.s="";

    int base=round(locate.S.scale.Tinv(1));

    if(ticklabel == null) 
      ticklabel=format == "%" ? Format("") : DefaultLogFormat(base);
    real a=locate.S.postscale.Tinv(locate.a);
    real b=locate.S.postscale.Tinv(locate.b);
    if(a > b) {real temp=a; a=b; b=temp;}
      
    int first=floor(a-epsilon);
    int last=ceil(b+epsilon);
      
    if(N == 0) {
      N=1;
      while(N <= last-first) {
        if(logaxiscoverage(N,T,g,locate,side,sign,Size,F,ticklabel,limit,
                           first,last)) break;
        ++N;
      }
    }
      
    if(N <= 2 && n == 0) n=base;
    tickvalues.N=N;
      
    if(N > 0) {
      for(int i=first-1; i <= last+1; ++i) {
        if(i >= a && i <= b)
          tickvalues.major.push(locate.S.scale.Tinv(i));
        if(n > 0) {
          for(int j=2; j < n; ++j) {
            real val=(i+1+locate.S.scale.T(j/n));
            if(val >= a && val <= b)
              tickvalues.minor.push(locate.S.scale.Tinv(val));
          }
        }
      }
    }
  }     
  return tickvalues;
}

// Signature of routines that draw labelled paths with ticks and tick labels.
typedef void ticks(frame, transform, Label, pair, path, path, pen,
                   arrowbar, margin, ticklocate, int[], bool opposite=false);
                                          
// Tick construction routine for a user-specified array of tick values.
ticks Ticks(int sign, Label F="", ticklabel ticklabel=null,
            bool beginlabel=true, bool endlabel=true,
            real[] Ticks=new real[], real[] ticks=new real[], int N=1,
            bool begin=true, bool end=true,
            real Size=0, real size=0, bool extend=false,
            pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(frame f, transform t, Label L, pair side, path g, path g2, 
                  pen p, arrowbar arrow, margin margin, ticklocate locate,
                  int[] divisor, bool opposite) {
    // Use local copy of context variables:
    int sign=opposite ? -sign : sign;
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
    
    if(F.align.dir != 0) side=F.align.dir;
    else if(side == 0) side=F.T*((sign == 1) ? left : right);
    
    bool ticklabels=false;
    path G=t*g;
    path G2=t*g2;
    
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

    begingroup(f);
    if(opposite) draw(f,G,p);
    else draw(f,margin(G,p).g,p,arrow);
    for(int i=(begin ? 0 : 1); i < (end ? Ticks.length : Ticks.length-1); ++i) {
      real val=T(Ticks[i]);
      if(val >= a && val <= b)
        drawtick(f,t,g,g2,locate,val,Size,sign,pTick,extend);
    }
    for(int i=0; i < ticks.length; ++i) {
      real val=T(ticks[i]);
      if(val >= a && val <= b)
        drawtick(f,t,g,g2,locate,val,size,sign,ptick,extend);
    }
    endgroup(f);
    
    if(N == 0) N=1;
    if(Size > 0 && !opposite) {
      for(int i=(beginlabel ? 0 : 1);
          i < (endlabel ? Ticks.length : Ticks.length-1); i += N) {
        real val=T(Ticks[i]);
        if(val >= a && val <= b) {
          ticklabels=true;
          labeltick(f,t,g,locate,val,side,sign,Size,ticklabel,F,norm);
        }
      }
    }
    if(L.s != "" && !opposite) 
      labelaxis(f,t,L,G,locate,sign,ticklabels);
  };
}

// Optional routine to allow modification of auto-generated tick values.
typedef tickvalues tickmodifier(tickvalues);
tickvalues None(tickvalues v) {return v;}

// Tickmodifier that removes all ticks in the intervals [a[i],b[i]].
tickmodifier OmitTickIntervals(real[] a, real[] b) {
  return new tickvalues(tickvalues v) { 
    if(a.length != b.length) abort(differentlengths);
    void omit(real[] A) {
      if(A.length != 0) {
        real norm=max(abs(A));
        for(int i=0; i < a.length; ++i) {
          int j;
          while((j=find(A > a[i]-zerotickfuzz*norm
                        & A < b[i]+zerotickfuzz*norm)) >= 0) {
            A.delete(j);
          }
        }
      }
    }
    omit(v.major);
    omit(v.minor);
    return v;
  };
}

// Tickmodifier that removes all ticks in the interval [a,b].
tickmodifier OmitTickInterval(real a, real b) {
  return OmitTickIntervals(new real[] {a}, new real[] {b});
}

// Tickmodifier that removes the specified ticks.
tickmodifier OmitTick(... real[] x) {
  return OmitTickIntervals(x,x);
}

tickmodifier NoZero=OmitTick(0);

tickmodifier Break(real, real)=OmitTickInterval;

// Automatic tick construction routine.
ticks Ticks(int sign, Label F="", ticklabel ticklabel=null,
            bool beginlabel=true, bool endlabel=true,
            int N, int n=0, real Step=0, real step=0,
            bool begin=true, bool end=true, tickmodifier modify=None,
            real Size=0, real size=0, bool extend=false,
            pen pTick=nullpen, pen ptick=nullpen)
{
  return new void(frame f, transform T, Label L, pair side, path g, path g2,
                  pen p, arrowbar arrow, margin margin, ticklocate locate,
                  int[] divisor, bool opposite) {
    real limit=Step == 0 ? axiscoverage*arclength(T*g) : 0;
    tickvalues values=modify(generateticks(sign,F,ticklabel,N,n,Step,step,
                                           Size,size,T,side,g,
                                           limit,p,locate,divisor,opposite));

    Ticks(sign,F,ticklabel,beginlabel,endlabel,values.major,values.minor,
          values.N,begin,end,Size,size,extend,pTick,ptick)
      (f,T,L,side,g,g2,p,arrow,margin,locate,divisor,opposite);
  };
}

ticks NoTicks()
{
  return new void(frame f, transform T, Label L, pair, path g, path, pen p,
                  arrowbar arrow, margin margin, ticklocate,
                  int[], bool opposite) {
    path G=T*g;
    if(opposite) draw(f,G,p);
    else {
      draw(f,margin(G,p).g,p,arrow);
      if(L.s != "") {
        Label L=L.copy();
        L.p(p);
        labelaxis(f,T,L,G);
      }
    }
  };
}

ticks LeftTicks(Label format="", ticklabel ticklabel=null,
                bool beginlabel=true, bool endlabel=true,
                int N=0, int n=0, real Step=0, real step=0,
                bool begin=true, bool end=true, tickmodifier modify=None,
                real Size=0, real size=0, bool extend=false,
                pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(-1,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
               begin,end,modify,Size,size,extend,pTick,ptick);
}

ticks RightTicks(Label format="", ticklabel ticklabel=null,
                 bool beginlabel=true, bool endlabel=true,
                 int N=0, int n=0, real Step=0, real step=0,
                 bool begin=true, bool end=true, tickmodifier modify=None,
                 real Size=0, real size=0, bool extend=false,
                 pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(1,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
               begin,end,modify,Size,size,extend,pTick,ptick);
}

ticks Ticks(Label format="", ticklabel ticklabel=null,
            bool beginlabel=true, bool endlabel=true,
            int N=0, int n=0, real Step=0, real step=0,
            bool begin=true, bool end=true, tickmodifier modify=None,
            real Size=0, real size=0, bool extend=false,
            pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(0,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
               begin,end,modify,Size,size,extend,pTick,ptick);
}

ticks LeftTicks(Label format="", ticklabel ticklabel=null, 
                bool beginlabel=true, bool endlabel=true, 
                real[] Ticks, real[] ticks=new real[],
                real Size=0, real size=0, bool extend=false,
                pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(-1,format,ticklabel,beginlabel,endlabel,
               Ticks,ticks,Size,size,extend,pTick,ptick);
}

ticks RightTicks(Label format="", ticklabel ticklabel=null, 
                 bool beginlabel=true, bool endlabel=true, 
                 real[] Ticks, real[] ticks=new real[],
                 real Size=0, real size=0, bool extend=false,
                 pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(1,format,ticklabel,beginlabel,endlabel,
               Ticks,ticks,Size,size,extend,pTick,ptick);
}

ticks Ticks(Label format="", ticklabel ticklabel=null, 
            bool beginlabel=true, bool endlabel=true, 
            real[] Ticks, real[] ticks=new real[],
            real Size=0, real size=0, bool extend=false,
            pen pTick=nullpen, pen ptick=nullpen)
{
  return Ticks(0,format,ticklabel,beginlabel,endlabel,
               Ticks,ticks,Size,size,extend,pTick,ptick);
}

ticks NoTicks=NoTicks(),
LeftTicks=LeftTicks(),
RightTicks=RightTicks(),
Ticks=Ticks();

pair tickMin(picture pic)
{
  return minbound(pic.userMin(),(pic.scale.x.tickMin,pic.scale.y.tickMin));
}
  
pair tickMax(picture pic)
{
  return maxbound(pic.userMax(),(pic.scale.x.tickMax,pic.scale.y.tickMax));
}
                                               
int Min=-1;
int Value=0;
int Max=1;
int Both=2;

// Structure used to communicate axis and autoscale settings to tick routines. 
struct axisT {
  int type;        // -1 = min, 0 = given value, 1 = max, 2 = min/max
  int type2;       // for 3D axis
  real value;
  real value2;
  pair side;       // 2D tick label direction relative to path (left or right)
  real position;   // label position along axis
  align align;     // default axis label alignment and 3D tick label direction
  int[] xdivisor;
  int[] ydivisor;
  int[] zdivisor;
  bool extend;     // extend axis to graph boundary?
};

axisT axis;
typedef void axis(picture, axisT);

axis Bottom(bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Min;
    axis.position=0.5;
    axis.side=right;
    axis.align=S;
    axis.extend=extend;
  };
}

axis Top(bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Max;
    axis.position=0.5;
    axis.side=left;
    axis.align=N;
    axis.extend=extend;
  };
}

axis BottomTop(bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Both;
    axis.position=0.5;
    axis.side=right;
    axis.align=S;
    axis.extend=extend;
  };
}

axis Left(bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Min;
    axis.position=0.5;
    axis.side=left;
    axis.align=W;
    axis.extend=extend;
  };
}

axis Right(bool extend=false)
{
  return new void(picture pic, axisT axis) {
    axis.type=Max;
    axis.position=0.5;
    axis.side=right;
    axis.align=E;
    axis.extend=extend;
  };
}

axis LeftRight(bool extend=false) 
{
  return new void(picture pic, axisT axis) {
    axis.type=Both;
    axis.position=0.5;
    axis.side=left;
    axis.align=W;
    axis.extend=extend;
  };
}

axis XEquals(real x, bool extend=true)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.value=pic.scale.x.T(x);
    axis.position=1;
    axis.side=left;
    axis.align=W;
    axis.extend=extend;
  };
}

axis YEquals(real y, bool extend=true)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.value=pic.scale.y.T(y);
    axis.position=1;
    axis.side=right;
    axis.align=S;
    axis.extend=extend;
  };
}

axis XZero(bool extend=true)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.value=pic.scale.x.T(pic.scale.x.scale.logarithmic ? 1 : 0);
    axis.position=1;
    axis.side=left;
    axis.align=W;
    axis.extend=extend;
  };
}

axis YZero(bool extend=true)
{
  return new void(picture pic, axisT axis) {
    axis.type=Value;
    axis.value=pic.scale.y.T(pic.scale.y.scale.logarithmic ? 1 : 0);
    axis.position=1;
    axis.side=right;
    axis.align=S;
    axis.extend=extend;
  };
}

axis Bottom=Bottom(),
Top=Top(),
BottomTop=BottomTop(),
Left=Left(),
Right=Right(),
LeftRight=LeftRight(),
XZero=XZero(),
YZero=YZero();

// Draw a general axis.
void axis(picture pic=currentpicture, Label L="", path g, path g2=nullpath,
          pen p=currentpen, ticks ticks, ticklocate locate,
          arrowbar arrow=None, margin margin=NoMargin,
          int[] divisor=new int[], bool above=false, bool opposite=false) 
{
  Label L=L.copy();
  real t=reltime(g,0.5);
  if(L.defaultposition) L.position(t);
  divisor=copy(divisor);
  locate=locate.copy();
  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
      frame d;
      ticks(d,t,L,0,g,g2,p,arrow,margin,locate,divisor,opposite);
      (above ? add : prepend)(f,t*T*inverse(t)*d);
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

real xtrans(transform t, real x)
{
  return (t*(x,0)).x;
}

real ytrans(transform t, real y)
{
  return (t*(0,y)).y;
}

// An internal routine to draw an x axis at a particular y value.
void xaxisAt(picture pic=currentpicture, Label L="", axis axis,
             real xmin=-infinity, real xmax=infinity, pen p=currentpen,
             ticks ticks=NoTicks, arrowbar arrow=None, margin margin=NoMargin,
             bool above=true, bool opposite=false)
{
  real y=axis.value;
  real y2;
  Label L=L.copy();
  int[] divisor=copy(axis.xdivisor);
  pair side=axis.side;
  int type=axis.type;

  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
      transform tinv=inverse(t);
      pair a=xmin == -infinity ? tinv*(lb.x-min(p).x,ytrans(t,y)) : (xmin,y);
      pair b=xmax == infinity ? tinv*(rt.x-max(p).x,ytrans(t,y)) : (xmax,y);
      pair a2=xmin == -infinity ? tinv*(lb.x-min(p).x,ytrans(t,y2)) : (xmin,y2);
      pair b2=xmax == infinity ? tinv*(rt.x-max(p).x,ytrans(t,y2)) : (xmax,y2);

      if(xmin == -infinity || xmax == infinity) {
        bounds mx=autoscale(a.x,b.x,pic.scale.x.scale);
        pic.scale.x.tickMin=mx.min;
        pic.scale.x.tickMax=mx.max;
        divisor=mx.divisor;
      }
      
      real fuzz=epsilon*max(abs(a.x),abs(b.x));
      a -= (fuzz,0);
      b += (fuzz,0);

      frame d;
      ticks(d,t,L,side,a--b,finite(y2) ? a2--b2 : nullpath,p,arrow,margin,
            ticklocate(a.x,b.x,pic.scale.x),divisor,opposite);
      (above ? add : prepend)(f,t*T*tinv*d);
    });

  void bounds() {
    if(type == Both) {
      y2=pic.scale.y.automax() ? tickMax(pic).y : pic.userMax().y;
      y=opposite ? y2 :
        (pic.scale.y.automin() ? tickMin(pic).y : pic.userMin().y);
    } 
    else if(type == Min) 
      y=pic.scale.y.automin() ? tickMin(pic).y : pic.userMin().y;
    else if(type == Max)
      y=pic.scale.y.automax() ? tickMax(pic).y : pic.userMax().y;

    real Xmin=finite(xmin) ? xmin : pic.userMin().x;
    real Xmax=finite(xmax) ? xmax : pic.userMax().x;

    pair a=(Xmin,y);
    pair b=(Xmax,y);
    pair a2=(Xmin,y2);
    pair b2=(Xmax,y2);

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
      ticks(d,pic.scaling(warn=false),L,side,
            (a.x,0)--(b.x,0),(a2.x,0)--(b2.x,0),p,arrow,margin,
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

  // Process any queued y axis bound calculation requests.
  for(int i=0; i < pic.scale.y.bound.length; ++i)
    pic.scale.y.bound[i]();

  pic.scale.y.bound.delete();

  bounds();

  // Request another x bounds calculation before final picture scaling.
  pic.scale.x.bound.push(bounds);
}

// An internal routine to draw a y axis at a particular x value.
void yaxisAt(picture pic=currentpicture, Label L="", axis axis,
             real ymin=-infinity, real ymax=infinity, pen p=currentpen,
             ticks ticks=NoTicks, arrowbar arrow=None, margin margin=NoMargin,
             bool above=true, bool opposite=false)
{
  real x=axis.value;
  real x2;
  Label L=L.copy();
  int[] divisor=copy(axis.ydivisor);
  pair side=axis.side;
  int type=axis.type;

  pic.add(new void (frame f, transform t, transform T, pair lb, pair rt) {
      transform tinv=inverse(t);
      pair a=ymin == -infinity ? tinv*(xtrans(t,x),lb.y-min(p).y) : (x,ymin);
      pair b=ymax == infinity ? tinv*(xtrans(t,x),rt.y-max(p).y) : (x,ymax);
      pair a2=ymin == -infinity ? tinv*(xtrans(t,x2),lb.y-min(p).y) : (x2,ymin);
      pair b2=ymax == infinity ? tinv*(xtrans(t,x2),rt.y-max(p).y) : (x2,ymax);

      if(ymin == -infinity || ymax == infinity) {
        bounds my=autoscale(a.y,b.y,pic.scale.y.scale);
        pic.scale.y.tickMin=my.min;
        pic.scale.y.tickMax=my.max;
        divisor=my.divisor;
      }

      real fuzz=epsilon*max(abs(a.y),abs(b.y));
      a -= (0,fuzz);
      b += (0,fuzz);

      frame d;
      ticks(d,t,L,side,a--b,finite(x2) ? a2--b2 : nullpath,p,arrow,margin,
            ticklocate(a.y,b.y,pic.scale.y),divisor,opposite);
      (above ? add : prepend)(f,t*T*tinv*d);
    });
  
  void bounds() {
    if(type == Both) {
      x2=pic.scale.x.automax() ? tickMax(pic).x : pic.userMax().x;
      x=opposite ? x2 : 
        (pic.scale.x.automin() ? tickMin(pic).x : pic.userMin().x);
    } else if(type == Min) 
      x=pic.scale.x.automin() ? tickMin(pic).x : pic.userMin().x;
    else if(type == Max)
      x=pic.scale.x.automax() ? tickMax(pic).x : pic.userMax().x;

    real Ymin=finite(ymin) ? ymin : pic.userMin().y;
    real Ymax=finite(ymax) ? ymax : pic.userMax().y;

    pair a=(x,Ymin);
    pair b=(x,Ymax);
    pair a2=(x2,Ymin);
    pair b2=(x2,Ymax);
  
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
      ticks(d,pic.scaling(warn=false),L,side,
            (0,a.y)--(0,b.y),(0,a2.y)--(0,b2.y),p,arrow,margin,
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

  // Process any queued x axis bound calculation requests.
  for(int i=0; i < pic.scale.x.bound.length; ++i)
    pic.scale.x.bound[i]();

  pic.scale.x.bound.delete();

  bounds();

  // Request another y bounds calculation before final picture scaling.
  pic.scale.y.bound.push(bounds);
}

// Set the x limits of a picture.
void xlimits(picture pic=currentpicture, real min=-infinity, real max=infinity,
             bool crop=NoCrop)
{
  if(min > max) return;
  
  pic.scale.x.automin=min <= -infinity;
  pic.scale.x.automax=max >= infinity;
  
  bounds mx;
  if(pic.scale.x.automin() || pic.scale.x.automax())
    mx=autoscale(pic.userMin().x,pic.userMax().x,pic.scale.x.scale);
  
  if(pic.scale.x.automin) {
    if(pic.scale.x.automin()) pic.userMinx(mx.min);
  } else pic.userMinx(min(pic.scale.x.T(min),pic.scale.x.T(max)));
  
  if(pic.scale.x.automax) {
    if(pic.scale.x.automax()) pic.userMaxx(mx.max);
  } else pic.userMaxx(max(pic.scale.x.T(min),pic.scale.x.T(max)));
  
  if(crop) {
    pair userMin=pic.userMin();
    pair userMax=pic.userMax();
    pic.bounds.xclip(userMin.x,userMax.x);
    pic.clip(userMin, userMax,
      new void (frame f, transform t, transform T, pair, pair) {
        frame Tinvf=T == identity() ? f : t*inverse(T)*inverse(t)*f;
        clip(f,T*box(((t*userMin).x,(min(Tinvf)).y),
                     ((t*userMax).x,(max(Tinvf)).y)));
      });
  }
}

// Set the y limits of a picture.
void ylimits(picture pic=currentpicture, real min=-infinity, real max=infinity,
             bool crop=NoCrop)
{
  if(min > max) return;
  
  pic.scale.y.automin=min <= -infinity;
  pic.scale.y.automax=max >= infinity;
  
  bounds my;
  if(pic.scale.y.automin() || pic.scale.y.automax())
    my=autoscale(pic.userMin().y,pic.userMax().y,pic.scale.y.scale);
  
  if(pic.scale.y.automin) {
    if(pic.scale.y.automin()) pic.userMiny(my.min);
  } else pic.userMiny(min(pic.scale.y.T(min),pic.scale.y.T(max)));
  
  if(pic.scale.y.automax) {
    if(pic.scale.y.automax()) pic.userMaxy(my.max);
  } else pic.userMaxy(max(pic.scale.y.T(min),pic.scale.y.T(max)));
  
  if(crop) {
    pair userMin=pic.userMin();
    pair userMax=pic.userMax();
    pic.bounds.yclip(userMin.y,userMax.y);
    pic.clip(userMin, userMax, 
      new void (frame f, transform t, transform T, pair, pair) {
        frame Tinvf=T == identity() ? f : t*inverse(T)*inverse(t)*f;
        clip(f,T*box(((min(Tinvf)).x,(t*userMin).y),
                     ((max(Tinvf)).x,(t*userMax).y)));
      });
  }
}

// Crop a picture to the current user-space picture limits.
void crop(picture pic=currentpicture) 
{
  xlimits(pic,false);
  ylimits(pic,false);
  if(pic.userSetx() && pic.userSety())
    clip(pic,box(pic.userMin(),pic.userMax()));
}

// Restrict the x and y limits to box(min,max).
void limits(picture pic=currentpicture, pair min, pair max, bool crop=NoCrop)
{
  xlimits(pic,min.x,max.x);
  ylimits(pic,min.y,max.y);
  if(crop && pic.userSetx() && pic.userSety())
    clip(pic,box(pic.userMin(),pic.userMax()));
}
  
// Internal routine to autoscale the user limits of a picture.
void autoscale(picture pic=currentpicture, axis axis)
{
  if(!pic.scale.set) {
    bounds mx,my;
    pic.scale.set=true;
    
    if(pic.userSetx()) {
      mx=autoscale(pic.userMin().x,pic.userMax().x,pic.scale.x.scale);
      if(pic.scale.x.scale.logarithmic &&
         floor(pic.userMin().x) == floor(pic.userMax().x)) {
        if(pic.scale.x.automin())
          pic.userMinx2(floor(pic.userMin().x));
        if(pic.scale.x.automax())
          pic.userMaxx2(ceil(pic.userMax().x));
      }
    } else {mx.min=mx.max=0; pic.scale.set=false;}
    
    if(pic.userSety()) {
      my=autoscale(pic.userMin().y,pic.userMax().y,pic.scale.y.scale);
      if(pic.scale.y.scale.logarithmic &&
         floor(pic.userMin().y) == floor(pic.userMax().y)) {
        if(pic.scale.y.automin())
          pic.userMiny2(floor(pic.userMin().y));
        if(pic.scale.y.automax())
          pic.userMaxy2(ceil(pic.userMax().y));
      }
    } else {my.min=my.max=0; pic.scale.set=false;}
    
    pic.scale.x.tickMin=mx.min;
    pic.scale.x.tickMax=mx.max;
    pic.scale.y.tickMin=my.min;
    pic.scale.y.tickMax=my.max;
    axis.xdivisor=mx.divisor;
    axis.ydivisor=my.divisor;
  }
}

// Draw an x axis.
void xaxis(picture pic=currentpicture, Label L="", axis axis=YZero,
           real xmin=-infinity, real xmax=infinity, pen p=currentpen,
           ticks ticks=NoTicks, arrowbar arrow=None, margin margin=NoMargin,
           bool above=false)
{
  if(xmin > xmax) return;
  
  if(pic.scale.x.automin && xmin > -infinity) pic.scale.x.automin=false;
  if(pic.scale.x.automax && xmax < infinity) pic.scale.x.automax=false;

  if(!pic.scale.set) {
    axis(pic,axis);
    autoscale(pic,axis);
  }
  
  Label L=L.copy();
  bool newticks=false;
  
  if(xmin != -infinity) {
    xmin=pic.scale.x.T(xmin);
    newticks=true;
  }
  
  if(xmax != infinity) {
    xmax=pic.scale.x.T(xmax);
    newticks=true;
  }
  
  if(newticks && pic.userSetx() && ticks != NoTicks) {
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

  if(L.defaultposition) L.position(axis.position);
  L.align(L.align,axis.align);
  
  xaxisAt(pic,L,axis,xmin,xmax,p,ticks,arrow,margin,above);
  if(axis.type == Both)
    xaxisAt(pic,L,axis,xmin,xmax,p,ticks,arrow,margin,above,true);
}

// Draw a y axis.
void yaxis(picture pic=currentpicture, Label L="", axis axis=XZero,
           real ymin=-infinity, real ymax=infinity, pen p=currentpen,
           ticks ticks=NoTicks, arrowbar arrow=None, margin margin=NoMargin,
           bool above=false, bool autorotate=true)
{
  if(ymin > ymax) return;

  if(pic.scale.y.automin && ymin > -infinity) pic.scale.y.automin=false;
  if(pic.scale.y.automax && ymax < infinity) pic.scale.y.automax=false;
  
  if(!pic.scale.set) {
    axis(pic,axis);
    autoscale(pic,axis);
  }
  
  Label L=L.copy();
  bool newticks=false;
  
  if(ymin != -infinity) {
    ymin=pic.scale.y.T(ymin);
    newticks=true;
  }
  
  if(ymax != infinity) {
    ymax=pic.scale.y.T(ymax);
    newticks=true;
  }
  
  if(newticks && pic.userSety() && ticks != NoTicks) {
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

  if(L.defaultposition) L.position(axis.position);
  L.align(L.align,axis.align);
  
  if(autorotate && L.defaulttransform) {
    frame f;
    add(f,Label(L.s,(0,0),L.p));
    if(length(max(f)-min(f)) > ylabelwidth*fontsize(L.p)) 
      L.transform(rotate(90));
  }
  
  yaxisAt(pic,L,axis,ymin,ymax,p,ticks,arrow,margin,above);
  if(axis.type == Both)
    yaxisAt(pic,L,axis,ymin,ymax,p,ticks,arrow,margin,above,true);
}

// Draw x and y axes.
void axes(picture pic=currentpicture, Label xlabel="", Label ylabel="",
          bool extend=true,
          pair min=(-infinity,-infinity), pair max=(infinity,infinity),
          pen p=currentpen, arrowbar arrow=None, margin margin=NoMargin,
          bool above=false)
{
  xaxis(pic,xlabel,YZero(extend),min.x,max.x,p,arrow,margin,above);
  yaxis(pic,ylabel,XZero(extend),min.y,max.y,p,arrow,margin,above);
}

// Draw a yaxis at x.
void xequals(picture pic=currentpicture, Label L="", real x,
             bool extend=false, real ymin=-infinity, real ymax=infinity,
             pen p=currentpen, ticks ticks=NoTicks,
             arrowbar arrow=None, margin margin=NoMargin, bool above=true) 
{
  yaxis(pic,L,XEquals(x,extend),ymin,ymax,p,ticks,arrow,margin,above);
}

// Draw an xaxis at y.
void yequals(picture pic=currentpicture, Label L="", real y,
             bool extend=false, real xmin=-infinity, real xmax=infinity,
             pen p=currentpen, ticks ticks=NoTicks,
             arrowbar arrow=None, margin margin=NoMargin, bool above=true)
{
  xaxis(pic,L,YEquals(y,extend),xmin,xmax,p,ticks,arrow,margin,above);
}

pair Scale(picture pic=currentpicture, pair z)
{
  return (pic.scale.x.T(z.x),pic.scale.y.T(z.y));
}

real ScaleX(picture pic=currentpicture, real x)
{
  return pic.scale.x.T(x);
}

real ScaleY(picture pic=currentpicture, real y)
{
  return pic.scale.y.T(y);
}

// Draw a tick of length size at pair z in direction dir using pen p.
void tick(picture pic=currentpicture, pair z, pair dir, real size=Ticksize,
          pen p=currentpen)
{
  pair z=Scale(pic,z);
  pic.add(new void (frame f, transform t) {
      pair tz=t*z;
      draw(f,tz--tz+unit(dir)*size,p);
    });
  pic.addPoint(z,p);
  pic.addPoint(z,unit(dir)*size,p);
}

void xtick(picture pic=currentpicture, explicit pair z, pair dir=N,
           real size=Ticksize, pen p=currentpen)
{
  tick(pic,z,dir,size,p);
}

void xtick(picture pic=currentpicture, real x, pair dir=N,
           real size=Ticksize, pen p=currentpen)
{
  tick(pic,(x,pic.scale.y.scale.logarithmic ? 1 : 0),dir,size,p);
}

void ytick(picture pic=currentpicture, explicit pair z, pair dir=E,
           real size=Ticksize, pen p=currentpen) 
{
  tick(pic,z,dir,size,p);
}

void ytick(picture pic=currentpicture, real y, pair dir=E,
           real size=Ticksize, pen p=currentpen)
{
  tick(pic,(pic.scale.x.scale.logarithmic ? 1 : 0,y),dir,size,p);
}

void tick(picture pic=currentpicture, Label L, real value, explicit pair z,
          pair dir, string format="", real size=Ticksize, pen p=currentpen)
{
  Label L=L.copy();
  L.position(Scale(pic,z));
  L.align(L.align,-dir);
  if(shift(L.T)*0 == 0)
    L.T=shift(dot(dir,L.align.dir) > 0 ? dir*size :
              ticklabelshift(L.align.dir,p))*L.T;
  L.p(p);
  if(L.s == "") L.s=format(format == "" ? defaultformat : format,value);
  L.s=baseline(L.s,baselinetemplate);
  add(pic,L);
  tick(pic,z,dir,size,p);
}

void xtick(picture pic=currentpicture, Label L, explicit pair z, pair dir=N,
           string format="", real size=Ticksize, pen p=currentpen)
{
  tick(pic,L,z.x,z,dir,format,size,p);
}

void xtick(picture pic=currentpicture, Label L, real x, pair dir=N,
           string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,(x,pic.scale.y.scale.logarithmic ? 1 : 0),dir,size,p);
}

void ytick(picture pic=currentpicture, Label L, explicit pair z, pair dir=E,
           string format="", real size=Ticksize, pen p=currentpen)
{
  tick(pic,L,z.y,z,dir,format,size,p);
}

void ytick(picture pic=currentpicture, Label L, real y, pair dir=E,
           string format="", real size=Ticksize, pen p=currentpen)
{
  xtick(pic,L,(pic.scale.x.scale.logarithmic ? 1 : 0,y),dir,format,size,p);
}

private void label(picture pic, Label L, pair z, real x, align align,
                   string format, pen p)
{
  Label L=L.copy();
  L.position(z);
  L.align(align);
  L.p(p);
  if(shift(L.T)*0 == 0)
    L.T=shift(ticklabelshift(L.align.dir,L.p))*L.T;
  if(L.s == "") L.s=format(format == "" ? defaultformat : format,x);
  L.s=baseline(L.s,baselinetemplate);
  add(pic,L);
}

// Put a label on the x axis.
void labelx(picture pic=currentpicture, Label L="", explicit pair z,
            align align=S, string format="", pen p=currentpen)
{
  label(pic,L,Scale(pic,z),z.x,align,format,p);
}

void labelx(picture pic=currentpicture, Label L="", real x,
            align align=S, string format="", pen p=currentpen)
{
  labelx(pic,L,(x,pic.scale.y.scale.logarithmic ? 1 : 0),align,format,p);
}

void labelx(picture pic=currentpicture, Label L,
            string format="", explicit pen p=currentpen)
{
  labelx(pic,L,L.position.position,format,p);
}

// Put a label on the y axis.
void labely(picture pic=currentpicture, Label L="", explicit pair z,
            align align=W, string format="", pen p=currentpen)
{
  label(pic,L,Scale(pic,z),z.y,align,format,p);
}

void labely(picture pic=currentpicture, Label L="", real y,
            align align=W, string format="", pen p=currentpen)
{
  labely(pic,L,(pic.scale.x.scale.logarithmic ? 1 : 0,y),align,format,p);
}

void labely(picture pic=currentpicture, Label L,
            string format="", explicit pen p=currentpen)
{
  labely(pic,L,L.position.position,format,p);
}

private string noprimary="Primary axis must be drawn before secondary axis";

// Construct a secondary X axis
picture secondaryX(picture primary=currentpicture, void f(picture))
{
  if(!primary.scale.set) abort(noprimary);
  picture pic;
  size(pic,primary);
  if(primary.userMax().x == primary.userMin().x) return pic;
  f(pic);
  if(!pic.userSetx()) return pic;
  bounds a=autoscale(pic.userMin().x,pic.userMax().x,pic.scale.x.scale);
  real bmin=pic.scale.x.automin() ? a.min : pic.userMin().x;
  real bmax=pic.scale.x.automax() ? a.max : pic.userMax().x;
  
  real denom=bmax-bmin;
  if(denom != 0) {
    pic.erase();
    real m=(primary.userMax().x-primary.userMin().x)/denom;
    pic.scale.x.postscale=Linear(m,bmin-primary.userMin().x/m);
    pic.scale.set=true;
    pic.scale.x.tickMin=pic.scale.x.postscale.T(a.min);
    pic.scale.x.tickMax=pic.scale.x.postscale.T(a.max);
    pic.scale.y.tickMin=primary.userMin().y;
    pic.scale.y.tickMax=primary.userMax().y;
    axis.xdivisor=a.divisor;
    f(pic);
  }
  pic.userCopy(primary);
  return pic;
}

// Construct a secondary Y axis
picture secondaryY(picture primary=currentpicture, void f(picture))
{
  if(!primary.scale.set) abort(noprimary);
  picture pic;
  size(pic,primary);
  if(primary.userMax().y == primary.userMin().y) return pic;
  f(pic);
  if(!pic.userSety()) return pic;
  bounds a=autoscale(pic.userMin().y,pic.userMax().y,pic.scale.y.scale);
  real bmin=pic.scale.y.automin() ? a.min : pic.userMin().y;
  real bmax=pic.scale.y.automax() ? a.max : pic.userMax().y;

  real denom=bmax-bmin;
  if(denom != 0) {
    pic.erase();
    real m=(primary.userMax().y-primary.userMin().y)/denom;
    pic.scale.y.postscale=Linear(m,bmin-primary.userMin().y/m);
    pic.scale.set=true;
    pic.scale.x.tickMin=primary.userMin().x;
    pic.scale.x.tickMax=primary.userMax().x;
    pic.scale.y.tickMin=pic.scale.y.postscale.T(a.min);
    pic.scale.y.tickMax=pic.scale.y.postscale.T(a.max);
    axis.ydivisor=a.divisor;
    f(pic);
  }
  pic.userCopy(primary);
  return pic;
}

typedef guide graph(pair f(real), real, real, int);
typedef guide[] multigraph(pair f(real), real, real, int);
                       
graph graph(interpolate join)
{
  return new guide(pair f(real), real a, real b, int n) {
    real width=b-a;
    return n == 0 ? join(f(a)) :
      join(...sequence(new guide(int i) {return f(a+(i/n)*width);},n+1));
  };
}

multigraph graph(interpolate join, bool3 cond(real))
{
  return new guide[](pair f(real), real a, real b, int n) {
    real width=b-a;
    if(n == 0) return new guide[] {join(cond(a) ? f(a) : nullpath)};
    guide[] G;
    guide[] g;
    for(int i=0; i < n+1; ++i) {
      real t=a+(i/n)*width;
      bool3 b=cond(t);
      if(b)
        g.push(f(t));
      else {
        if(g.length > 0) {
          G.push(join(...g));
          g=new guide[] {};
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

guide Straight(... guide[])=operator --;
guide Spline(... guide[])=operator ..;

interpolate Hermite(splinetype splinetype)
{
  return new guide(... guide[] a) {
    int n=a.length;
    if(n == 0) return nullpath;
    real[] x,y;
    guide G;
    for(int i=0; i < n; ++i) {
      guide g=a[i];
      int m=size(g);
      if(m == 0) continue;
      pair z=point(g,0);
      x.push(z.x);
      y.push(z.y);
      if(m > 1) {
        G=G..hermite(x,y,splinetype) & g;
        pair z=point(g,m);
        x=new real[] {z.x};
        y=new real[] {z.y};
      }
    }
    return G & hermite(x,y,splinetype);
  };
}

interpolate Hermite=Hermite(Spline);

guide graph(picture pic=currentpicture, real f(real), real a, real b,
            int n=ngraph, real T(real)=identity, interpolate join=operator --)
{
  if(T == identity)
    return graph(join)(new pair(real x) {
        return (x,pic.scale.y.T(f(pic.scale.x.Tinv(x))));},
      pic.scale.x.T(a),pic.scale.x.T(b),n);
  else
    return graph(join)(new pair(real x) {
        return Scale(pic,(T(x),f(T(x))));},
      a,b,n);
}

guide[] graph(picture pic=currentpicture, real f(real), real a, real b,
              int n=ngraph, real T(real)=identity,
              bool3 cond(real), interpolate join=operator --)
{
  if(T == identity)
    return graph(join,cond)(new pair(real x) {
        return (x,pic.scale.y.T(f(pic.scale.x.Tinv(x))));},
      pic.scale.x.T(a),pic.scale.x.T(b),n);
  else
    return graph(join,cond)(new pair(real x) {
        return Scale(pic,(T(x),f(T(x))));},
      a,b,n);
}

guide graph(picture pic=currentpicture, real x(real), real y(real), real a,
            real b, int n=ngraph, real T(real)=identity,
            interpolate join=operator --)
{
  if(T == identity)
    return graph(join)(new pair(real t) {return Scale(pic,(x(t),y(t)));},a,b,n);
  else
    return graph(join)(new pair(real t) {
        return Scale(pic,(x(T(t)),y(T(t))));
      },a,b,n);
}

guide[] graph(picture pic=currentpicture, real x(real), real y(real), real a,
              real b, int n=ngraph, real T(real)=identity, bool3 cond(real),
              interpolate join=operator --)
{
  if(T == identity)
    return graph(join,cond)(new pair(real t) {return Scale(pic,(x(t),y(t)));},
                            a,b,n);
  else
    return graph(join,cond)(new pair(real t) {
        return Scale(pic,(x(T(t)),y(T(t))));},
      a,b,n);
}

guide graph(picture pic=currentpicture, pair z(real), real a, real b,
            int n=ngraph, real T(real)=identity, interpolate join=operator --)
{
  if(T == identity)
    return graph(join)(new pair(real t) {return Scale(pic,z(t));},a,b,n);
  else
    return graph(join)(new pair(real t) {
        return Scale(pic,z(T(t)));
      },a,b,n);
}

guide[] graph(picture pic=currentpicture, pair z(real), real a, real b,
              int n=ngraph, real T(real)=identity, bool3 cond(real),
              interpolate join=operator --)
{
  if(T == identity)
    return graph(join,cond)(new pair(real t) {return Scale(pic,z(t));},a,b,n);
  else
    return graph(join,cond)(new pair(real t) {
        return Scale(pic,z(T(t)));
      },a,b,n);
}

string conditionlength="condition array has different length than data";

void checkconditionlength(int x, int y)
{
  checklengths(x,y,conditionlength);
}

guide graph(picture pic=currentpicture, pair[] z, interpolate join=operator --)
{
  int i=0;
  return graph(join)(new pair(real) {
      pair w=Scale(pic,z[i]);
      ++i;
      return w;
    },0,0,z.length-1);
}

guide[] graph(picture pic=currentpicture, pair[] z, bool3[] cond,
              interpolate join=operator --)
{
  int n=z.length;
  int i=0;
  pair w;
  checkconditionlength(cond.length,n);
  bool3 condition(real) {
    bool3 b=cond[i];
    if(b != false) w=Scale(pic,z[i]);
    ++i;
    return b;
  }
  return graph(join,condition)(new pair(real) {return w;},0,0,n-1);
}

guide graph(picture pic=currentpicture, real[] x, real[] y,
            interpolate join=operator --)
{
  int n=x.length;
  checklengths(n,y.length);
  int i=0;
  return graph(join)(new pair(real) {
      pair w=Scale(pic,(x[i],y[i]));
      ++i;
      return w;
    },0,0,n-1);
}

guide[] graph(picture pic=currentpicture, real[] x, real[] y, bool3[] cond,
              interpolate join=operator --)
{
  int n=x.length;
  checklengths(n,y.length);
  int i=0;
  pair w;
  checkconditionlength(cond.length,n);
  bool3 condition(real) {
    bool3 b=cond[i];
    if(b != false) w=Scale(pic,(x[i],y[i]));
    ++i;
    return b;
  }
  return graph(join,condition)(new pair(real) {return w;},0,0,n-1);
}

// Connect points in z into segments corresponding to consecutive true elements
// of b using interpolation operator join. 
path[] segment(pair[] z, bool[] cond, interpolate join=operator --)
{
  checkconditionlength(cond.length,z.length);
  int[][] segment=segment(cond);
  return sequence(new path(int i) {return join(...z[segment[i]]);},
                  segment.length);
}

pair polar(real r, real theta)
{
  return r*expi(theta);
}

guide polargraph(picture pic=currentpicture, real r(real), real a, real b,
                 int n=ngraph, interpolate join=operator --)
{
  return graph(join)(new pair(real theta) {
      return Scale(pic,polar(r(theta),theta));
    },a,b,n);
}

guide polargraph(picture pic=currentpicture, real[] r, real[] theta,
                 interpolate join=operator--)
{
  int n=r.length;
  checklengths(n,theta.length);
  int i=0;
  return graph(join)(new pair(real) {
      pair w=Scale(pic,polar(r[i],theta[i]));
      ++i;
      return w;
    },0,0,n-1);
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
  int n=z.length;
  checklengths(n,dm.length);
  checklengths(n,dp.length);
  bool all=cond.length == 0;
  if(!all)
    checkconditionlength(cond.length,n);
  for(int i=0; i < n; ++i) {
    if(all || cond[i])
      errorbar(pic,z[i],dp[i],dm[i],p,size);
  }
}

void errorbars(picture pic=currentpicture, real[] x, real[] y,
               real[] dpx, real[] dpy, real[] dmx={}, real[] dmy={},
               bool[] cond={}, pen p=currentpen, real size=0)
{
  if(dmx.length == 0) dmx=dpx;
  if(dmy.length == 0) dmy=dpy;
  int n=x.length;
  checklengths(n,y.length);
  checklengths(n,dpx.length);
  checklengths(n,dpy.length);
  checklengths(n,dmx.length);
  checklengths(n,dmy.length);
  bool all=cond.length == 0;
  if(!all)
    checkconditionlength(cond.length,n);
  for(int i=0; i < n; ++i) {
    if(all || cond[i])
      errorbar(pic,(x[i],y[i]),(dpx[i],dpy[i]),(dmx[i],dmy[i]),p,size);
  }
}

void errorbars(picture pic=currentpicture, real[] x, real[] y,
               real[] dpy, bool[] cond={}, pen p=currentpen, real size=0)
{
  errorbars(pic,x,y,0*x,dpy,cond,p,size);
}

// Return a vector field on path g, specifying the vector as a function of the
// relative position along path g in [0,1].
picture vectorfield(path vector(real), path g, int n, bool truesize=false,
                    pen p=currentpen, arrowbar arrow=Arrow,
                    margin margin=PenMargin)
{
  picture pic;
  for(int i=0; i < n; ++i) {
    real x=(n == 1) ? 0.5 : i/(n-1);
    if(truesize)
      draw(relpoint(g,x),pic,vector(x),p,arrow);
    else 
      draw(pic,shift(relpoint(g,x))*vector(x),p,arrow,margin);
  }
  return pic;
}

real maxlength(pair a, pair b, int nx, int ny) 
{
  return min((b.x-a.x)/nx,(b.y-a.y)/ny);
}

// return a vector field over box(a,b).
picture vectorfield(path vector(pair), pair a, pair b,
                    int nx=nmesh, int ny=nx, bool truesize=false,
                    real maxlength=truesize ? 0 : maxlength(a,b,nx,ny),
                    bool cond(pair z)=null, pen p=currentpen,
                    arrowbar arrow=Arrow, margin margin=PenMargin)
{
  picture pic;
  real dx=1/nx;
  real dy=1/ny;
  bool all=cond == null;
  real scale;

  if(maxlength > 0) {
    real size(pair z) {
      path g=vector(z);
      return abs(point(g,size(g)-1)-point(g,0));
    }
    real max=size(a);
    for(int i=0; i <= nx; ++i) {
      real x=interp(a.x,b.x,i*dx);
      for(int j=0; j <= ny; ++j)
        max=max(max,size((x,interp(a.y,b.y,j*dy))));
    }
    scale=max > 0 ? maxlength/max : 1;
  } else scale=1;

  for(int i=0; i <= nx; ++i) {
    real x=interp(a.x,b.x,i*dx);
    for(int j=0; j <= ny; ++j) {
      real y=interp(a.y,b.y,j*dy);
      pair z=(x,y);
      if(all || cond(z)) {
        path g=scale(scale)*vector(z);
        if(truesize)
          draw(z,pic,g,p,arrow);
        else
          draw(pic,shift(z)*g,p,arrow,margin);
      }
    }
  }
  return pic;
}

// True arc
path Arc(pair c, real r, real angle1, real angle2, bool direction,
         int n=nCircle)
{
  angle1=radians(angle1);
  angle2=radians(angle2);
  if(direction) {
    if(angle1 >= angle2) angle1 -= 2pi;
  } else if(angle2 >= angle1) angle2 -= 2pi;
  return shift(c)*polargraph(new real(real t){return r;},angle1,angle2,n,
                             operator ..);
}

path Arc(pair c, real r, real angle1, real angle2, int n=nCircle)
{
  return Arc(c,r,angle1,angle2,angle2 >= angle1 ? CCW : CW,n);
}

path Arc(pair c, explicit pair z1, explicit pair z2, bool direction=CCW,
         int n=nCircle)
{
  return Arc(c,abs(z1-c),degrees(z1-c),degrees(z2-c),direction,n);
}

// True circle
path Circle(pair c, real r, int n=nCircle)
{
  return Arc(c,r,0,360,n)&cycle;
}
