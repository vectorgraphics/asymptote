/*****
 * plain.asy
 * Andy Hammerlindl and John Bowman 2004/08/19
 *
 * A package for general purpose drawing, with automatic sizing of pictures.
 *
 *****/

public bool shipped=false;
public bool uptodate=true;
static public pen currentpen;
static pen nullpen=linewidth(0);
static path nullpath;

static real inches=72;
static real inch=inches;
static real cm=inches/2.540005;
static real mm=0.1cm;
static real bp=1;	   // A PostScript point.
static real pt=72.0/72.27; // A TeX pt; slightly smaller than a PostScript bp.
static pair I=(0,1);
static real pi=pi();

static pair up=(0,1);
static pair down=(0,-1);
static pair right=(1,0);
static pair left=(-1,0);

static pair E=(1,0);
static pair N=(0,1);
static pair W=(-1,0);
static pair S=(0,-1);

static pair NE=unit(N+E);
static pair NW=unit(N+W);
static pair SW=unit(S+W);
static pair SE=unit(S+E);

static pair ENE=unit(E+NE);
static pair NNE=unit(N+NE);
static pair NNW=unit(N+NW);
static pair WNW=unit(W+NW);
static pair WSW=unit(W+SW);
static pair SSW=unit(S+SW);
static pair SSE=unit(S+SE);
static pair ESE=unit(E+SE);
  
static pen linetype(string s) 
{
  return linetype(s,true);
}

static pen solid=linetype("");
static pen dotted=linetype("0 4");
static pen dashed=linetype("8 8");
static pen longdashed=linetype("24 8");
static pen dashdotted=linetype("8 8 0 8");
static pen longdashdotted=linetype("24 8 0 8");

static pen Dotted=dotted+1.0;
static pen Dotted(pen p) {return dotted+2*linewidth(p);}

static pen squarecap=linecap(0);
static pen roundcap=linecap(1);
static pen extendcap=linecap(2);

static pen miterjoin=linejoin(0);
static pen roundjoin=linejoin(1);
static pen beveljoin=linejoin(2);

static pen zerowinding=fillrule(0);
static pen evenodd=fillrule(1);

static pen nobasealign=basealign(0);
static pen basealign=basealign(1);

static pen invisible=invisible();
static pen black=gray(0);
static pen lightgray=gray(0.9);
static pen lightgrey=lightgray;
static pen gray=gray(0.5);
static pen grey=gray;
static pen white=gray(1);

static pen red=rgb(1,0,0);
static pen green=rgb(0,1,0);
static pen blue=rgb(0,0,1);

static pen cmyk=cmyk(0,0,0,0);
static pen Cyan=cmyk(1,0,0,0);
static pen Magenta=cmyk(0,1,0,0);
static pen Yellow=cmyk(0,0,1,0);
static pen Black=cmyk(0,0,0,1);

static pen yellow=red+green;
static pen magenta=red+blue;
static pen cyan=blue+green;

static pen brown=red+black;
static pen darkgreen=green+black;
static pen darkblue=blue+black;

static pen orange=red+yellow;
static pen purple=magenta+blue;

static pen chartreuse=brown+green;
static pen fuchsia=red+darkblue;
static pen salmon=red+darkgreen+darkblue;
static pen lightblue=darkgreen+blue;
static pen lavender=brown+darkgreen+blue;
static pen pink=red+darkgreen+blue;

pen cmyk(pen p) {
  return p+cmyk;
}

// Options for handling label overwriting
static int Allow=0;
static int Suppress=1;
static int SuppressQuiet=2;
static int Move=3;
static int MoveQuiet=4;

// Global parameters:
static public real labelmargin=0.3;
static public real arrowlength=0.75cm;
static public real arrowfactor=15;
static public real arrowangle=15;
static public real arcarrowfactor=0.5*arrowfactor;
static public real arcarrowangle=2*arrowangle;
static public real barfactor=arrowfactor;
static public real dotfactor=6;

static public real legendlinelength=50;
static public real legendskip=1.5;
static public real legendmargin=10;

static public string defaultfilename;
static public string defaultformat="$%.4g$";

// Reduced for tension atleast infinity
static real infinity=sqrt(0.25*realMax());
static pair Infinity=(infinity,infinity);

static real epsilon=realEpsilon();

// Define a.. tension t ..b to be equivalent to
//        a.. tension t and t ..b
// and likewise with controls.
guide operator tension(real t, bool atLeast)
{
  return operator tension(t,t,atLeast);
}
guide operator controls(pair z)
{
  return operator controls(z,z);
}

real dotsize(pen p=currentpen) 
{
  return dotfactor*linewidth(p);
}

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

bool finite(real x)
{
  return abs(x) < infinity;
}

bool finite(pair z)
{
  return abs(z.x) < infinity && abs(z.y) < infinity;
}

// Avoid two parentheses.
transform shift(real x, real y)
{
  return shift((x,y));
}

// I/O operations

static public string commentchar="#";

file input(string name, bool check=true)
{
  return input(name,check,commentchar);
}
file input(string name, string comment) {return input(name,true,commentchar);}
file xinput(string name) {return xinput(name,true);}
file output(string name) {return output(name,false);}
file xoutput(string name) {return xoutput(name,false);}
file csv(file file) {return csv(file,true);}
file line(file file) {return line(file,true);}
file single(file file) {return single(file,true);}

file stdin=input("");
file stdout;

void none(file file) {}
void endl(file file) {_write(file,'\n'); flush(file);}
void tab(file file) {_write(file,'\t');}
typedef void suffix(file);

void write(file file=stdout, suffix e=endl) {e(file);}

void write(file file=stdout, string s="", bool x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", int x, suffix e) 
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", real x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", pair x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", triple x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", pen x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", transform x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=stdout, string s="", guide x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}

void write(file file=stdout, string x, suffix e)
{
  _write(file,x); e(file);
}

void write(file file=null, string x ... string[] a)
{
  if(file == null) {write(stdout,x...a); endl(stdout); return;}
  _write(file,x);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", bool x ... bool[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  _write(file,s); _write(file,x);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", int x ... int[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  write(file,s,x,none);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", real x ... real[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  write(file,s,x,none);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", pair x ... pair[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  write(file,s,x,none);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", triple x ... triple[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  write(file,s,x,none);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", pen x ... pen[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  write(file,s,x,none);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", transform x ... transform[] a)
{
  if(file == null) {write(stdout,s,x...a); endl(stdout); return;}
  write(file,s,x,none);
  for(int i=0; i < a.length; ++i) {tab(file); _write(file,a[i]);}
}

void write(file file=null, string s="", guide x)
{
  if(file == null) {write(stdout,s,x); endl(stdout); return;}
  write(file,s,x,none);
}

void _write(file file, path[] g)
{
  if(g.length > 0) _write(file,g[0]);
  for(int i=1; i < g.length; ++i) {
    write(file);
    _write(file," ^^");
    _write(file,g[i]);
  }
}
void write(file file=stdout, string s="", explicit path[] x, suffix e)
{
  _write(file,s); _write(file,x); e(file);
}
void write(file file=null, string s="", explicit path[] x)
{
  if(file == null) {write(stdout,s,x); endl(stdout); return;}
  write(file,s); _write(file,x);
}

string ask(string prompt)
{
  write(stdout,prompt);
  return stdin;
}

static public string getstringprefix=".asy_";

void savestring(string name, string value, string prefix=getstringprefix)
{
  file out=output(prefix+name);
  write(out,value);
  close(out);
}

string getstring(string name, string default="", string prompt="",
		 string prefix=getstringprefix, bool save=true)
{
  string value;
  file in=input(prefix+name,false);
  if(error(in)) value=default;
  else value=in;
  if(prompt == "") prompt=name+"? ["+value+"] ";
  string input=ask(prompt);
  if(input != "") value=input;
  if(save) savestring(name,value);
  return value;
}

real getreal(string name, real default=0, string prompt="",
	     string prefix=getstringprefix, bool save=true)
{
  string value=getstring(name,(string) default,prompt,getstringprefix,false);
  real x=(real) value;
  if(save) savestring(name,value);
  return x;
}

static private int GUIFilenum=0;
static public frame patterns;

private struct DELETET {}
static public DELETET DELETE=null;

struct GUIop
{
  public transform[] Transform;
  public bool[] Delete;
}

GUIop operator init() {return new GUIop;}
  
GUIop[] GUIlist;

// Delete item
void GUIop(int index, int filenum=0, DELETET)
{
  if(GUIlist.length <= filenum) GUIlist[filenum]=new GUIop;
  GUIop GUIobj=GUIlist[filenum];
  while(GUIobj.Transform.length <= index) {
    GUIobj.Transform.push(identity());
    GUIobj.Delete.push(false);
  }
  GUIobj.Delete[index]=true;
}

// Transform item
void GUIop(int index, int filenum=0, transform T)
{
  if(GUIlist.length <= filenum) GUIlist[filenum]=new GUIop;
  GUIop GUIobj=GUIlist[filenum];
  while(GUIobj.Transform.length <= index) {
    GUIobj.Transform.push(identity());
    GUIobj.Delete.push(false);
  }
  GUIobj.Transform[index]=T*GUIobj.Transform[index];
}

transform rotate(real angle) 
{
  return rotate(angle,0);
}

// A rotation in the direction dir limited to [-90,90]
// This is useful for rotating text along a line in the direction dir.
transform rotate(explicit pair dir)
{
  real angle=degrees(dir);
  if(angle > 90 && angle < 270) angle -= 180;
  return rotate(angle);
} 

transform shift(transform t)
{
  return (t.x,t.y,0,0,0,0);
}

transform shiftless(transform t)
{
  return (0,0,t.xx,t.xy,t.yx,t.yy);
}

pair intersect(path p1, path p2) 
{
  return intersect(p1,p2,0);
}

// A function that draws an object to frame pic, given that the transform
// from user coordinates to true-size coordinates is t.
typedef void drawer(frame f, transform t);

// A generalization of drawer that includes the final frame's bounds.
typedef void drawerBound(frame f, transform t, transform T, pair lb, pair rt);

// A coordinate in "flex space." A linear combination of user and true-size
// coordinates.
  
static struct coord {
  public real user,truesize;
  public bool finite=true;

  // Build a coord.
  static coord build(real user, real truesize) {
    coord c=new coord;
    c.user=user;
    c.truesize=truesize;
    return c;
  }

  // Deep copy of coordinate.  Users may add coords to the picture, but then
  // modify the struct. To prevent this from yielding unexpected results, deep
  // copying is used.
  coord copy() {
    return build(user, truesize);
  }
  
  void clip(real min, real max) {
    user=min(max(user,min),max);
  }
}

coord operator init() {return new coord;}
  
struct coords2 {
  coord[] x;
  coord[] y;
  void erase() {
    x=new coord[];
    y=new coord[];
  }
  // Only a shallow copy of the individual elements of x and y
  // is needed since, once entered, they are never modified.
  coords2 copy() {
    coords2 c=new coords2;
    c.x=copy(x);
    c.y=copy(y);
    return c;
  }
  void append(coords2 c) {
    x.append(c.x);
    y.append(c.y);
  }
  void push(pair user, pair truesize) {
    x.push(coord.build(user.x,truesize.x));
    y.push(coord.build(user.y,truesize.y));
  }
  void push(coord cx, coord cy) {
    x.push(cx);
    y.push(cy);
  }
  void push(transform t, coords2 c1, coords2 c2)
  {
    for(int i=0; i < c1.x.length; ++i) {
      coord cx=c1.x[i], cy=c2.y[i];
      pair tinf=shiftless(t)*((finite(cx.user) ? 0 : 1),
			      (finite(cy.user) ? 0 : 1));
      pair z=t*(cx.user,cy.user);
      pair w=(cx.truesize,cy.truesize);
      w=length(w)*unit(shiftless(t)*w);
      coord Cx,Cy;
      Cx.user=(tinf.x == 0 ? z.x : infinity);
      Cy.user=(tinf.y == 0 ? z.y : infinity);
      Cx.truesize=w.x;
      Cy.truesize=w.y;
      push(Cx,Cy);
    }
  }
  void xclip(real min, real max) {
    for(int i=0; i < x.length; ++i) 
      x[i].clip(min,max);
  }
  void yclip(real min, real max) {
    for(int i=0; i < y.length; ++i) 
      y[i].clip(min,max);
  }
}
  
coords2 operator init() {return new coords2;}

bool operator <= (coord a, coord b)
{
  return a.user <= b.user && a.truesize <= b.truesize;
}

bool operator >= (coord a, coord b)
{
  return a.user >= b.user && a.truesize >= b.truesize;
}

// Find the maximal elements of the input array, using the partial ordering
// given.
coord[] maxcoords(coord[] in, bool operator <= (coord,coord))
{
  // As operator <= is defined in the parameter list, it has a special
  // meaning in the body of the function.

  coord best;
  coord[] c;

  int n=in.length;
  
  // Find the first finite restriction.
  int first=0;
  for(first=0; first < n; ++first)
    if(finite(in[first].user)) break;
	
  if (first == n)
    return c;
  else {
    // Add the first coord without checking restrictions (as there are none).
    best=in[first];
    c.push(best);
  }

  static int NONE=-1;

  int dominator(coord x)
  {
    // This assumes it has already been checked against the best.
    for (int i=1; i < c.length; ++i)
      if (x <= c[i])
        return i;
    return NONE;
  }

  void promote(int i)
  {
    // Swap with the top
    coord x=c[i];
    c[i]=best;
    best=c[0]=x;
  }

  void addmaximal(coord x)
  {
    coord[] newc;

    // Check if it beats any others.
    for (int i=0; i < c.length; ++i) {
      coord y=c[i];
      if (!(y <= x))
        newc.push(y);
    }
    newc.push(x);
    c=newc;
    best=c[0];
  }

  void add(coord x)
  {
    if (x <= best || !finite(x.user))
      return;
    else {
      int i=dominator(x);
      if (i == NONE)
        addmaximal(x);
      else
        promote(i);
    }
  }

  for(int i=1; i < n; ++i)
    add(in[i]);

  return c;
}

typedef real scalefcn(real x);
					      
public struct scaleT {
  scalefcn T,Tinv;
  bool logarithmic;
  bool automin,automax;
  void init(scalefcn T, scalefcn Tinv, bool logarithmic=false,
	    bool automin=true, bool automax=true) {
    this.T=T;
    this.Tinv=Tinv;
    this.logarithmic=logarithmic;
    this.automin=automin;
    this.automax=automax;
  }
  scaleT copy() {
    scaleT dest=new scaleT;
    dest.init(T,Tinv,logarithmic,automin,automax);
    return dest;
  }
};

scaleT operator init()
{
  scaleT S=new scaleT;
  S.init(identity,identity);
  return S;
}
				  
public struct autoscaleT {
  public scaleT scale;
  public scaleT postscale;
  public real tickMin=-infinity, tickMax=infinity;
  public bool automin=true, automax=true;
  public bool automin() {return automin && scale.automin;}
  public bool automax() {return automax && scale.automax;}
  
  real T(real x) {return postscale.T(scale.T(x));}
  scalefcn T() {return scale.logarithmic ? postscale.T : T;}
  real Tinv(real x) {return scale.Tinv(postscale.Tinv(x));}
  
  autoscaleT copy() {
    autoscaleT dest=new autoscaleT;
    dest.scale=scale.copy();
    dest.postscale=postscale.copy();
    dest.tickMin=tickMin;
    dest.tickMax=tickMax;
    dest.automin=(bool) automin;
    dest.automax=(bool) automax;
    return dest;
  }
}

autoscaleT operator init() {return new autoscaleT;}
				  
public struct ScaleT {
  public bool set=false;
  public autoscaleT x;
  public autoscaleT y;
  public autoscaleT z;
  ScaleT copy() {
    ScaleT dest=new ScaleT;
    dest.set=set;
    dest.x=x.copy();
    dest.y=y.copy();
    return dest;
  }
};

ScaleT operator init() {return new ScaleT;}

struct Legend {
  public string label;
  public pen plabel;
  public pen p;
  public frame mark;
  public bool putmark;
  void init(string label, pen plabel=currentpen, pen p=nullpen,
	    frame mark=newframe, bool putmark=false) {
    this.label=label;
    this.plabel=plabel;
    this.p=(p == nullpen) ? plabel : p;
    this.mark=mark;
    this.putmark=putmark;
  }
}

Legend operator init() {return new Legend;}

pair minbound(pair z, pair w) 
{
  return (min(z.x,w.x),min(z.y,w.y));
}

pair maxbound(pair z, pair w) 
{
  return (max(z.x,w.x),max(z.y,w.y));
}

pair realmult(pair z, pair w) 
{
  return (z.x*w.x,z.y*w.y);
}

pair rectify(pair dir) 
{
  real scale=max(abs(dir.x),abs(dir.y));
  if(scale != 0) dir *= 0.5/scale;
  dir += (0.5,0.5);
  return dir;
}

pair point(frame f, pair dir)
{
  return min(f)+realmult(rectify(dir),max(f)-min(f));
}

guide[] operator cast(pair[] z)
{
  guide[] g=new guide[z.length];
  for(int i=0; i < z.length; ++i) g[i]=z[i];
  return g;
}

path[] operator cast(pair[] z)
{
  path[] g=new path[z.length];
  for(int i=0; i < z.length; ++i) g[i]=z[i];
  return g;
}

path[] operator cast(path g)
{
  return new path[] {g};
}

path[] operator cast(guide g)
{
  return new path[] {(path) g};
}

real min(... real[] a) {return min(a);}
real max(... real[] a) {return max(a);}

static bool Above=true;
static bool Below=false;

static bool Aspect=true;
static bool IgnoreAspect=false;

struct picture {
  // The functions to do the deferred drawing.
  drawerBound[] nodes;
  
  // The coordinates in flex space to be used in sizing the picture.
  struct bounds {
    coords2 point,min,max;
    void erase() {
      point.erase();
      min.erase();
      max.erase();
    }
    bounds copy() {
      bounds b=new bounds;
      b.point=point.copy();
      b.min=min.copy();
      b.max=max.copy();
      return b;
    }
    void xclip(real Min, real Max) {
      point.xclip(Min,Max);
      min.xclip(Min,Max);
      max.xclip(Min,Max);
    }
    void yclip(real Min, real Max) {
      point.yclip(Min,Max);
      min.yclip(Min,Max);
      max.yclip(Min,Max);
    }
    void clip(pair Min, pair Max) {
      xclip(Min.x,Max.x);
      yclip(Min.y,Max.y);
    }
  }
  
  bounds operator init() {return new bounds;}
  
  bounds bounds;
    
  // Transform to be applied to this picture.
  public transform T;
  
  // Cached user-space bounding box
  public pair userMin,userMax;
  
  public ScaleT scale; // Needed by graph
  public Legend[] legend;

  // The maximum sizes in the x and y directions; zero means no restriction.
  public real xsize=0, ysize=0;
  
  // If true, the x and y directions must be scaled by the same amount.
  public bool keepAspect=false;

  void init() {
    userMin=Infinity;
    userMax=-userMin;
  }
  init();
  
  // Erase the current picture, retaining any size specification.
  void erase() {
    nodes=new drawerBound[];
    bounds.erase();
    T=identity();
    scale=new ScaleT;
    legend=new Legend[];
    init();
  }
  
  bool empty() {
    return !(finite(userMin) && finite(userMax));
  }
	      
  // Cache the current user-space bounding box
  void userBox(pair min, pair max) {
    userMin=minbound(userMin,min);
    userMax=maxbound(userMax,max);
  }
  
  void add(drawerBound d) {
    if(interact()) uptodate=false;
    nodes.push(d);
  }

  void add(drawer d) {
    if(interact()) uptodate=false;
    nodes.push(new void (frame f, transform t, transform T, pair, pair) {
      d(f,t*T);
    });
  }

  void clip(drawer d) {
    if(interact()) uptodate=false;
    bounds.clip(userMin,userMax);
    nodes.push(new void (frame f, transform t, transform T, pair, pair) {
      d(f,t*T);
    });
  }

  // Add a point to the sizing.
  void addPoint(pair user, pair truesize=0) {
    bounds.point.push(user,truesize);
    userBox(user,user);
  }
  
  // Add a point to the sizing, accounting also for the size of the pen.
  void addPoint(pair user, pair truesize=0, pen p) {
    addPoint(user,truesize+min(p));
    addPoint(user,truesize+max(p));
  }
  
  // Add a box to the sizing.
  void addBox(pair userMin, pair userMax, pair trueMin=0, pair trueMax=0) {
    bounds.min.push(userMin,trueMin);
    bounds.max.push(userMax,trueMax);
    userBox(userMin,userMax);
  }

  // Add a (user space) path to the sizing.
  void addPath(path g) {
    addBox(min(g),max(g));
  }

  // Add a path to the sizing with the additional padding of a pen.
  void addPath(path g, pen p) {
    addBox(min(g),max(g),min(p),max(p));
  }

  void size(real x=0, real y=0, bool a=true) {
    xsize=x;
    ysize=y;
    keepAspect=a;
  }

  // The scaling in one dimension:  x --> a*x + b
  struct scaling {
    public real a,b;
    static scaling build(real a, real b) {
      scaling s=new scaling;
      s.a=a; s.b=b;
      return s;
    }
    real scale(real x) {
      return a*x+b;
    }
    real scale(coord c) {
      return scale(c.user) + c.truesize;
    }
  }

  // Calculate the minimum point in scaling the coords.
  real min(scaling s, coord[] c) {
    if (c.length > 0) {
      real m=infinity;
      for (int i=0; i < c.length; ++i)
	if (finite(c[i].user) && s.scale(c[i]) < m)
	  m=s.scale(c[i]);
      return m;
    }
    else return 0;
  }
 
  // Calculate the maximum point in scaling the coords.
  real max(scaling s, coord[] c) {
    if (c.length > 0) {
      real M=-infinity;
      for (int i=0; i < c.length; ++i)
        if (finite(c[i].user) && s.scale(c[i]) > M)
          M=s.scale(c[i]);
      return M;
    } else return 0;
  }

  // Calculate the min for the final picture, given the transform of coords.
  pair min(transform t) {
    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);
    return (min(min(xs,bounds.min.x),
		min(xs,bounds.max.x),
		min(xs,bounds.point.x)),
	    min(min(ys,bounds.min.y),
		min(ys,bounds.max.y),
		min(ys,bounds.point.y)));
  }

  // Calculate the max for the final picture, given the transform of coords.
  pair max(transform t) {
    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);
    return (max(max(xs,bounds.min.x),
		max(xs,bounds.max.x),
		max(xs,bounds.point.x)),
	    max(max(ys,bounds.min.y),
		max(ys,bounds.max.y),
		max(ys,bounds.point.y)));
  }

  // Calculate the sizing constants for the given array and maximum size.
  scaling calculateScaling(coord[] coords, real size) {
    import simplex;
    simplex.problem p;
   
    void addMinCoord(coord c) {
      // (a*user + b) + truesize >= 0:
      p.addRestriction(c.user,1,c.truesize);
    }
    void addMaxCoord(coord c) {
      // (a*user + b) + truesize <= size:
      p.addRestriction(-c.user,-1,size-c.truesize);
    }

    coord[] m=maxcoords(coords,operator >=);
    coord[] M=maxcoords(coords,operator <=);
    
    for(int i=0; i < m.length; ++i)
      addMinCoord(m[i]);
    for(int i=0; i < M.length; ++i)
      addMaxCoord(M[i]);

    int status=p.optimize();
    if (status == simplex.problem.OPTIMAL) {
      return scaling.build(p.a(),p.b());
    }
    else if (status == simplex.problem.UNBOUNDED) {
      write("warning: scaling in picture unbounded");
      return scaling.build(1,0);
    }
    else {
      write("warning: cannot fit picture to requested size...enlarging...");
      return calculateScaling(coords,sqrt(2)*size);
    }
  }

  void append(coords2 point, coords2 min, coords2 max, transform t,
	      bounds bounds) 
  {
    // Add the coord info to this picture.
    if(t == identity()) {
      point.append(bounds.point);
      min.append(bounds.min);
      max.append(bounds.max);
    } else {
      point.push(t,bounds.point,bounds.point);
      // Add in all 4 corner points, to properly size rectangular pictures.
      point.push(t,bounds.min,bounds.min);
      point.push(t,bounds.min,bounds.max);
      point.push(t,bounds.max,bounds.min);
      point.push(t,bounds.max,bounds.max);
    }
  }
  
  // Returns the transform for turning user-space pairs into true-space pairs.
  transform calculateTransform(real xsize, real ysize, bool keepAspect=true) {
    if (xsize == 0 && ysize == 0)
      return identity();
    
    coords2 Coords;
    
    append(Coords,Coords,Coords,T,bounds);
    
    if (ysize == 0) {
      scaling sx=calculateScaling(Coords.x,xsize);
      return scale(sx.a);
    }
    
    if (xsize == 0) {
      scaling sy=calculateScaling(Coords.y,ysize);
      return scale(sy.a);
    }
    
    scaling sx=calculateScaling(Coords.x,xsize);
    scaling sy=calculateScaling(Coords.y,ysize);
    if (keepAspect)
      return scale(min(sx.a,sy.a));
    else
      return xscale(sx.a)*yscale(sy.a);
  }

  transform calculateTransform() {
    return calculateTransform(xsize,ysize,keepAspect);
  }

  pair min() {
    return min(calculateTransform());
  }
  
  pair max() {
    return max(calculateTransform());
  }
  
  frame fit(transform t, transform T0=T, pair m, pair M) {
    frame f;
    for (int i=0; i < nodes.length; ++i)
      nodes[i](f,t,T0,m,M);
    return f;
  }

  // Returns a rigid version of the picture using t to transform user coords
  // into truesize coords.
  frame fit(transform t) {
    return fit(t,min(t),max(t));
  }

  // Returns the picture fit to the wanted size.
  frame fit(real xsize, real ysize, bool keepAspect=true) {
    return fit(calculateTransform(xsize,ysize,keepAspect));
  }

  frame fit() {
    return fit(xsize,ysize,keepAspect);
  }
  
  // Returns the picture fit to the wanted size, aligned in direction dir
  frame fit(real xsize, real ysize, bool keepAspect=true, pair dir) {
    frame f=fit(xsize,ysize,keepAspect);
    return shift(dir)*shift(-point(f,-dir))*f;
  }

  frame fit(pair dir) {
    frame f=fit();
    return shift(dir)*shift(-point(f,-dir))*f;
  }
  
  // Copies the drawing information, but not the sizing information into a new
  // picture. Warning: "fitting" this picture will not scale as a normal
  // picture would.
  picture drawcopy() {
    picture dest=new picture;
    dest.nodes=copy(nodes);
    dest.T=T;
    dest.userMin=userMin;
    dest.userMax=userMax;
    dest.scale=scale.copy();
    dest.legend=copy(legend);

    return dest;
  }

  // A deep copy of this picture.  Modifying the copied picture will not affect
  // the original.
  picture copy() {
    picture dest=drawcopy();

    dest.bounds=bounds.copy();
    
    dest.xsize=xsize; dest.ysize=ysize; dest.keepAspect=keepAspect;
    return dest;
  }

  // Add a picture to this picture, such that the user coordinates will be
  // scaled identically when fitted
  void add(picture src, bool group=true, bool put=Above) {
    // Copy the picture.  Only the drawing function closures are needed, so we
    // only copy them.  This needs to be a deep copy, as src could later have
    // objects added to it that should not be included in this picture.

    if(src == this) abort("cannot add picture to itself");
    
    picture srcCopy=src.drawcopy();
    // Draw by drawing the copied picture.
    nodes.push(new void (frame f, transform t, transform T, pair m, pair M) {
      frame d=srcCopy.fit(t,T*srcCopy.T,m,M);
      if(group) begingroup(f);
      (put ? add : prepend)(f,d);
      if(group) endgroup(f);
      legend.append(src.legend);
    });
    
    userBox(src.userMin,src.userMax);
    
    append(bounds.point,bounds.min,bounds.max,srcCopy.T,src.bounds);
  }
}

picture operator init() {return new picture;}

picture operator * (transform t, picture orig)
{
  picture pic=orig.copy();
  pic.T=t*pic.T;
  pair c00=t*pic.userMin;
  pair c01=t*(pic.userMin.x,pic.userMax.y);
  pair c10=t*(pic.userMax.x,pic.userMin.y);
  pair c11=t*pic.userMax;
  pic.userMin=(min(c00.x,c01.x,c10.x,c11.x),min(c00.y,c01.y,c10.y,c11.y));
  pic.userMax=(max(c00.x,c01.x,c10.x,c11.x),max(c00.y,c01.y,c10.y,c11.y));
  return pic;
}

public picture currentpicture;

public frame gui[];

frame gui(int index) {
  while(gui.length <= index) {
    frame f;
    gui.push(f);
  }
  return gui[index];
}

path[] operator ^^ (path p, path q) 
{
  return new path[] {p,q};
}

path[] operator ^^ (path p, explicit path[] q) 
{
  return concat(new path[] {p},q);
}

path[] operator ^^ (explicit path[] p, path q) 
{
  return concat(p,new path[] {q});
}

path[] operator ^^ (explicit path[] p, explicit path[] q) 
{
  return concat(p,q);
}

path[] operator * (transform t, explicit path[] p) 
{
  path[] P;
  for(int i=0; i < p.length; ++i) P[i]=t*p[i];
  return P;
}

pair[] operator * (transform t, pair[] z) 
{
  pair[] Z;
  for(int i=0; i < z.length; ++i) Z[i]=t*z[i];
  return Z;
}

pair min(explicit path[] g)
{
  pair ming=Infinity;
  for(int i=0; i < g.length; ++i)
    ming=minbound(ming,min(g[i]));
  return ming;
}

pair max(explicit path[] g)
{
  pair maxg=(-infinity,-infinity);
  for(int i=0; i < g.length; ++i)
    maxg=maxbound(maxg,max(g[i]));
  return maxg;
}

void size(picture pic=currentpicture,
          real xsize, real ysize, bool keepAspect=Aspect)
{
  pic.size(xsize,ysize,keepAspect);
}

// Ensure that each dimension is no more than size, preserving aspect ratio.
void size(picture pic=currentpicture, real Size)
{
  pic.size(Size,Size,Aspect);
}

pair size(frame f)
{
  return max(f)-min(f);
}
				     
void begingroup(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    begingroup(f);
  });
}

void endgroup(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    endgroup(f);
  });
}

// Add frame dest to frame src with optional grouping (default false)
void add(frame dest, frame src, bool group)
{
  if(group) begingroup(dest);
  add(dest,src);
  if(group) endgroup(dest);
}

// Add frame dest about origin to frame src with optional grouping
// (default false)
void add(pair origin, frame dest, frame src, bool group=false)
{
  if(group) begingroup(dest);
  add(dest,shift(origin)*src);
  if(group) endgroup(dest);
}

// Add frame src about origin to picture dest with optional grouping
// (default true)
void add(pair origin=0, picture dest=currentpicture, frame src,
	 bool group=true, bool put=Above)
{
  dest.add(new void (frame f, transform t) {
    if(group) begingroup(f);
    (put ? add : prepend)(f,shift(t*origin)*src);
    if(group) endgroup(f);
  });
  dest.addBox(origin,origin,min(src),max(src));
}

// Like add(pair,picture,frame,bool) but extend picture to accommodate frame
void attach(pair origin=0, picture dest=currentpicture, frame src,
	    bool group=true, bool put=Above)
{
  transform t=dest.calculateTransform(dest.xsize,dest.ysize,dest.keepAspect);
  add(origin,dest,src,group,put);
  pair s=size(dest.fit(t));
  size(dest,dest.xsize != 0 ? s.x : 0,dest.ysize != 0 ? s.y : 0,
       dest.keepAspect);
}

// Add a picture to another such that user coordinates in both will be scaled
// identically in the shipout.
void add(picture dest, picture src, bool group=true, bool put=Above)
{
  dest.add(src,group,put);
}

void add(picture src, bool group=true, bool put=Above)
{
  add(currentpicture,src,group,put);
}

// Fit the picture src using the identity transformation (so user
// coordinates and truesize coordinates agree) and add it about the point
// origin to picture dest.
void add(pair origin, picture dest, picture src, bool group=true,
	 bool put=Above)
{
  add(origin,dest,src.fit(identity()),group,put);
}

void add(pair origin, picture src, bool group=true, bool put=Above)
{
  add(origin,currentpicture,src,group,put);
}

guide box(pair a, pair b)
{
  return a--(a.x,b.y)--b--(b.x,a.y)--cycle;
}

guide unitsquare=box((0,0),(1,1));

guide square(pair z1, pair z2)
{
  pair v=z2-z1;
  pair z3=z2+I*v;
  pair z4=z3-v;
  return z1--z2--z3--z4--cycle;
}

guide unitcircle=E..N..W..S..cycle;

static public real circleprecision=0.0006;

guide circle(pair c, real r)
{
  return shift(c)*scale(r)*unitcircle;
}

guide ellipse(pair c, real a, real b)
{
  return shift(c)*xscale(a)*yscale(b)*unitcircle;
}

real labelmargin(pen p=currentpen)
{
  return labelmargin*fontsize(p);
}

private struct marginT {
  public path g;
  public real begin,end;
};

marginT operator init() {return new marginT;}

typedef marginT margin(path, pen);

path trim(path g, real begin, real end) {
  real a=arctime(g,begin);
  real b=arctime(g,arclength(g)-end);
  return a <= b ? subpath(g,a,b) : point(g,a);
}

margin NoMargin()
{ 
  return new marginT(path g, pen) {
    marginT margin;
    margin.begin=margin.end=0;
    margin.g=g;
    return margin;
  };
}
						      
margin Margin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    real factor=labelmargin(p);
    margin.begin=begin*factor;
    margin.end=end*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
							   
margin PenMargin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    real factor=linewidth(p);
    margin.begin=(begin+0.5)*factor;
    margin.end=(end+0.5)*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
					      
margin DotMargin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    real margindot(real x) {return x > 0 ? dotfactor*x : x;}
    real factor=linewidth(p);
    margin.begin=(margindot(begin)+0.5)*factor;
    margin.end=(margindot(end)+0.5)*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
						      
margin TrueMargin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    margin.begin=begin;
    margin.end=end;
    margin.g=trim(g,begin,end);
    return margin;
  };
}
						      
public margin
  NoMargin=NoMargin(),
  BeginMargin=Margin(1,0),
  Margin=Margin(0,1),
  EndMargin=Margin,
  Margins=Margin(1,1),
  BeginPenMargin=PenMargin(0.5,-0.5),
  PenMargin=PenMargin(-0.5,0.5),
  EndPenMargin=PenMargin,
  PenMargins=PenMargin(0.5,0.5),
  BeginDotMargin=DotMargin(0.5,-0.5),
  DotMargin=DotMargin(-0.5,0.5),
  EndDotMargin=DotMargin,
  DotMargins=DotMargin(0.5,0.5);

void draw(frame f, path g)
{
  draw(f,g,currentpen);
}

void draw(frame f, explicit path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) draw(f,g[i],p);
}

void Draw(picture pic=currentpicture, path g, pen p=currentpen)
{
  pic.add(new void (frame f, transform t) {
    draw(f,t*g,p);
  });
  pic.addPath(g,p);
}

void _draw(picture pic=currentpicture, path g, pen p=currentpen,
	   margin margin=NoMargin)
 
{
  pic.add(new void (frame f, transform t) {
    draw(f,margin(t*g,p).g,p);
  });
  pic.addPath(g,p);
}

void draw(picture pic=currentpicture, explicit path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) Draw(pic,g[i],p);
}

void fill(frame f, path[] g)
{
  fill(f,g,currentpen);
}

void filldraw(frame f, path[] g, pen p=currentpen)
{
  fill(f,g,p);
  draw(f,g,p);
}

void fill(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  g=copy(g);
  pic.add(new void (frame f, transform t) {
    fill(f,t*g,p);
  });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

// lattice shading
void fill(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
	  pen[][] p)
{
  g=copy(g);
  p=copy(p);
  pic.add(new void (frame f, transform t) {
    fill(f,t*g,fillrule,p);
  });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

// axial shading
void fill(picture pic=currentpicture, path[] g, pen pena, pair a,
	  pen penb, pair b)
{
  g=copy(g);
  pic.add(new void (frame f, transform t) {
    fill(f,t*g,pena,t*a,penb,t*b);
  });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

// radial shading
void fill(picture pic=currentpicture, path[] g, pen pena, pair a, real ra,
	  pen penb, pair b, real rb)
{
  g=copy(g);
  pic.add(new void (frame f, transform t) {
    pair A=t*a, B=t*b;
    real RA=abs(t*(a+ra)-A);
    real RB=abs(t*(b+rb)-B);
    fill(f,t*g,pena,A,RA,penb,B,RB);
  });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

// Gouraud shading
void fill(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
	  pen[] p, pair[] z, int[] edges)
{
  g=copy(g);
  p=copy(p);
  z=copy(z);
  edges=copy(edges);
  pic.add(new void (frame f, transform t) {
	    fill(f,t*g,fillrule,p,t*z,edges);
  });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void fill(pair origin, picture pic=currentpicture, path[] g, pen p=currentpen)
{
  picture opic;
  fill(opic,g,p);
  add(origin,pic,opic);
}
  
void filldraw(picture pic=currentpicture, path[] g, pen fillpen=currentpen,
	      pen drawpen=currentpen)
{
  fill(pic,g,fillpen);
  draw(pic,g,drawpen);
}

void clip(frame f, path[] g)
{
  clip(f,g,currentpen);
}

void clip(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  g=copy(g);
  pic.userMin=maxbound(pic.userMin,min(g));
  pic.userMax=minbound(pic.userMax,max(g));
  pic.clip(new void (frame f, transform t) {
    clip(f,t*g,p);
  });
}

void unfill(frame f, path[] g)
{
  clip(f,box(min(f),max(f))^^g,evenodd);
}

void unfill(picture pic=currentpicture, path[] g)
{
  g=copy(g);
  pic.clip(new void (frame f, transform t) {
    unfill(f,t*g);
  });
}

pair dir(path g)
{
  return dir(g,length(g));
}

pair dir(path g, path h)
{
  return 0.5*(dir(g)+dir(h));
}

// return the point on path g at arclength L
pair arcpoint(path g, real L)
{
    return point(g,arctime(g,L));
}

// return the direction on path g at arclength L
pair arcdir(path g, real L)
{
    return dir(g,arctime(g,L));
}

// return the time on path g at the given relative fraction of its arclength
real reltime(path g, real fraction)
{
  return arctime(g,fraction*arclength(g));
}

// return the point on path g at the given relative fraction of its arclength
pair relpoint(path g, real l)
{
  return point(g,reltime(g,l));
}

// return the direction of path g at the given relative fraction of its
// arclength
pair reldir(path g, real l)
{
  return dir(g,reltime(g,l));
}

// return the initial point of path g
pair beginpoint(path g)
{
    return point(g,0);
}

// return the point on path g at half of its arclength
pair midpoint(path g)
{
    return relpoint(g,0.5);
}

// return the final point of path g
pair endpoint(path g)
{
    return point(g,length(g));
}

struct align {
  public pair dir;
  public bool relative=false;
  bool default=true;
  void init(pair dir=0, bool relative=false, bool default=false) {
    this.dir=dir;
    this.relative=relative;
    this.default=default;
  }
  align copy() {
    align align=new align;
    align.init(dir,relative,default);
    return align;
  }
  void align(align align) {
    if(!align.default) init(align.dir,align.relative);
  }
  void align(align align, align default) {
    align(align);
    if(this.default) init(default.dir,default.relative,default.default);
  }
  void write(file file=stdout, suffix e=endl) {
    if(!default) {
      if(relative) {
	write(file,"Relative(");
	write(file,dir);
	write(file,")",e);
      } else write(file,dir,e);
    }
  }
  bool Center() {
    return relative && dir == 0;
  }
}

struct side {
  public pair align;
}

side operator init() {return new side;}
  
side Relative(explicit pair align)
{
  side s;
  s.align=align;
  return s;
}
  
side NoSide;
side LeftSide=Relative(W);
side Center=Relative((0,0));
side RightSide=Relative(E);

side operator * (real x, side s) 
{
  side S;
  S.align=x*s.align;
  return S;
}

align operator init() {return new align;}
align operator cast(pair dir) {align A; A.init(dir,false); return A;}
align operator cast(side side) {align A; A.init(side.align,true); return A;}
align NoAlign;

void write(file file=stdout, align align, suffix e=endl)
{
  align.write(file,e);
}

struct position {
  public pair position;
  public bool relative;
}

position operator init() {return new position;}
  
position Relative(real position)
{
  position p;
  p.position=position;
  p.relative=true;
  return p;
}
  
position BeginPoint=Relative(0);
position MidPoint=Relative(0.5);
position EndPoint=Relative(1);

position operator cast(pair x) {position P; P.position=x; return P;}
position operator cast(real x) {return (pair) x;}
position operator cast(int x) {return (pair) x;}

pair operator cast(position P) {return P.position;}

struct Label {
  public string s,size;
  position position;
  bool defaultposition=true;
  align align;
  pen p=nullpen;
  real angle;
  bool defaultangle=true;
  pair shift;
  
  void init(string s="", string size="", position position=0, 
	    bool defaultposition=true,
	    align align=NoAlign, pen p=nullpen, real angle=0,
	    bool defaultangle=true, pair shift=0) {
    this.s=s;
    this.size=size;
    this.position=position;
    this.defaultposition=defaultposition;
    this.align=align.copy();
    this.p=p;
    this.angle=angle;
    this.defaultangle=defaultangle;
    this.shift=shift;
  }
  
  void initalign(string s="", string size="", align align, pen p=nullpen) {
    init();
    this.s=s;
    this.size=size;
    this.align=align.copy();
    this.p=p;
  }
  
  Label copy() {
    Label L=new Label;
    L.init(s,size,position,defaultposition,align,p,angle,defaultangle,shift);
    return L;
  }
  
  void angle(real a) {
    this.angle=a;
    defaultangle=false;
  }
  
  void shift(pair a) {
    this.shift=a;
  }
  
  void position(position pos) {
    this.position=pos;
    defaultposition=false;
  }
  
  void align(align a) {
    align.align(a);
  }
  void align(align a, align default) {
    align.align(a,default);
  }
  
  void p(pen p0) {
    if(this.p == nullpen) this.p=p0;
  }
  
  void label(frame f, real angle=0, pair position,
	     pair align=0, pen p=currentpen)
  {
    _label(f,s,size,angle,position+align*labelmargin(p),align,p);
  }

  void out(frame f) {
    label(f,angle,position.position+shift,align.dir,p);
  }
  
  void label(picture pic=currentpicture, real angle=0, pair position,
	    pair align=0, pair shift=0, pen p=currentpen)
  {
    pic.add(new void (frame f, transform t) {
      transform t0=shiftless(t);
      _label(f,s,size,degrees(t0*dir(angle)),
	     t*position+align*labelmargin(p)+shift,
	     length(align)*unit(t0*align),p);
      });
    frame f;
    // Create a picture with label at the origin to extract its bbox truesize.
    label(f,angle,(0,0),align,p);
    pic.addBox(position,position,min(f),max(f));
  }

  void out(picture pic=currentpicture) {
    label(pic,angle,position.position,align.dir,shift,
	  p == nullpen ? currentpen : p);
  }
  
  void out(picture pic=currentpicture, path g) {
    bool relative=position.relative;
    real position=position.position.x;
    pair Align=align.dir;
    bool alignrelative=align.relative;
    if(defaultposition) {relative=true; position=0.5;}
    if(relative) position=reltime(g,position);
    if(align.default) {
      alignrelative=true;
      Align=position <= 0 ? S : position >= length(g) ? N : E;
    }
    label(pic,angle,point(g,position),
	  alignrelative ? Align*dir(g,position)/N : Align,shift,
	  p == nullpen ? currentpen : p);
  }
  
  void write(file file=stdout, suffix e=endl) {
    write(file,"s=\""+s+"\"");
    if(!defaultposition) write(file,", position=",position.position);
    if(!align.default) write(file,", align=");
    write(file,align);
    if(p != nullpen) write(file,", pen=",p);
    if(!defaultangle) write(file,", angle=",angle);
    if(shift != 0) write(file,", shift=",shift);
    plain.write(file,e);
  }
  
  real relative() {
    return defaultposition ? 0.5 : position.position.x;
  };
  
  real relative(path g) {
    return position.relative ? reltime(g,relative()) : relative();
  };
}

Label operator init() {return new Label;}

Label Label;

void add(frame f, Label L)
{
  L.out(f);
}
  
void add(picture pic=currentpicture, Label L)
{
  L.out(pic);
}
  
Label operator * (transform t, Label L)
{
  Label tL=L.copy();
  transform t0=shiftless(t);
  tL.align.dir=length(L.align.dir)*unit(t0*L.align.dir);
  tL.angle(degrees(t0*dir(L.angle)));
  tL.shift(shift(t)*L.shift);
  return tL;
}

Label Label(string s, string size="", explicit position position,
	    align align=NoAlign, pen p=nullpen)
{
  Label L;
  L.init(s,size,position,false,align,p);
  return L;
}

Label Label(string s, string size="", pair position, align align=NoAlign,
	    pen p=nullpen)
{
  return Label(s,size,(position) position,align,p);
}

Label Label(explicit pair position, align align=NoAlign, pen p=nullpen)
{
  return Label((string) position,position,align,p);
}

Label Label(string s="", string size="", align align=NoAlign,
	    explicit pen p=nullpen)
{
  Label L;
  L.initalign(s,size,align,p);
  return L;
}

Label Label(Label L, position position, align align=NoAlign, pen p=nullpen)
{
  Label L=L.copy();
  L.position(position);
  L.align(align);
  L.p(p);
  return L;
}

Label Label(Label L, align align=NoAlign, pen p=nullpen)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  return L;
}

void write(file file=stdout, Label L, suffix e=endl)
{
  L.write(file,e);
}

void label(frame f, Label L, pair position, align align=NoAlign,
	   pen p=currentpen)
{
  add(f,Label(L,position,align,p));
}
  
void label(picture pic=currentpicture, Label L, pair position,
	   align align=NoAlign, pen p=nullpen)
{
  Label L=L.copy();
  L.position(position);
  L.align(align);
  L.p(p);
  add(pic,L);
}
  
void label(picture pic=currentpicture, Label L, align align=NoAlign,
	   pen p=nullpen)
{
  label(pic,L,L.position,align,p);
}
  
void label(picture pic=currentpicture, Label L, explicit path g,
	   align align=NoAlign, pen p=currentpen)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  L.out(pic,g);
}

void label(picture pic=currentpicture, Label L, explicit guide g,
	   align align=NoAlign, pen p=currentpen)
{
  label(pic,L,(path) g,align,p);
}

Label operator cast(string s) {return Label(s);}

pair point(picture pic=currentpicture, pair dir)
{
  return pic.userMin+realmult(rectify(dir),pic.userMax-pic.userMin);
}

guide arrowhead(path g, real position, pen p=currentpen,
		real size=0, real angle=arrowangle)
{
  if(size == 0) size=arrowsize(p);
  path r=subpath(g,position,0.0);
  pair x=point(r,0);
  real t=arctime(r,size);
  pair y=point(r,t);
  path base=y+2*size*I*dir(r,t)--y-2*size*I*dir(r,t);
  path left=rotate(-angle,x)*r, right=rotate(angle,x)*r;
  real tl=intersect(left,base).x, tr=intersect(right,base).x;
  pair denom=point(right,tr)-y;
  real factor=denom != 0 ? length((point(left,tl)-y)/denom) : 1;
  left=rotate(-angle,x)*r; right=rotate(angle*factor,x)*r;
  tl=intersect(left,base).x; tr=intersect(right,base).x;
  return subpath(left,0,tl > 0 ? tl : t)--subpath(right,tr > 0 ? tr : t,0)
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

typedef void filltype(frame, path, pen);
void filltype(frame, path, pen) {}

filltype Fill(pen p)
{
  return new void(frame f, path g, pen drawpen) {
    drawpen += solid;
    fill(f,g,p == nullpen ? drawpen : p+solid);
    draw(f,g,drawpen);
  };
}

public filltype NoFill=new void(frame f, path g, pen p) {
  draw(f,g,p+solid);
};

public filltype Fill=Fill(nullpen);

void arrow(frame f, path G, pen p=currentpen, real size=0,
	   real angle=arrowangle, filltype filltype=Fill,
	   position position=EndPoint, bool forwards=true,
	   margin margin=NoMargin)
{
  if(size == 0) size=arrowsize(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(G,position);
  G=margin(G,p).g;
  if(!forwards) {
    G=reverse(G);
    position=length(G)-position;
  }
  path R=subpath(G,position,0.0);
  path S=subpath(G,position,length(G));
  size=min(arclength(G),size);
  draw(f,subpath(R,arctime(R,size),length(R)),p);
  draw(f,S,p);
  guide head=arrowhead(G,position,p,size,angle);
  filltype(f,head,p);
}

void arrow2(frame f, path G, pen p=currentpen, real size=0,
	    real angle=arrowangle, filltype filltype=Fill,
	    margin margin=NoMargin)
{
  if(size == 0) size=arrowsize(p);
  G=margin(G,p).g;
  path R=reverse(G);
  size=min(0.5*arclength(G),size);
  draw(f,subpath(R,arctime(R,size),length(R)-arctime(G,size)),p);
  guide head=arrowhead(G,length(G),p,size,angle);
  guide tail=arrowhead(R,length(R),p,size,angle);
  filltype(f,head,p);
  filltype(f,tail,p);
}

picture arrow(path g, pen p=currentpen, real size=0,
	      real angle=arrowangle, filltype filltype=Fill,
	      position position=EndPoint, bool forwards=true,
	      margin margin=NoMargin)
{
  picture pic;
  pic.add(new void (frame f, transform t) {
    arrow(f,t*g,p,size,angle,filltype,position,forwards,margin);
  });
  
  pic.addPath(g,p);
  arrowheadbbox(pic,forwards ? g : reverse(g),position,p,size,angle);
  return pic;
}

picture arrow2(path g, pen p=currentpen, real size=0,
	       real angle=arrowangle, filltype filltype=Fill,
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

void postscript(picture pic=currentpicture, string s)
{
  pic.add(new void (frame f, transform) {
    postscript(f,s);
    });
}

void tex(picture pic=currentpicture, string s)
{
  pic.add(new void (frame f, transform) {
    tex(f,s);
  });
}

void layer(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    layer(f);
  });
}

void newpage() 
{
  tex("\newpage");
  layer();
}

static bool Wait=true;				
static bool NoWait=false;

guide box(frame f, real xmargin=0, real ymargin=infinity,
	  pen p=currentpen, filltype filltype=NoFill)
{
  if(ymargin == infinity) ymargin=xmargin;
  pair z=(xmargin,ymargin);
  int sign=filltype == Fill ? -1 : 1;
  guide g=box(min(f)+0.5*sign*min(p)-z,max(f)+0.5*sign*max(p)+z);
  frame F;
  filltype(F,g,p);
  prepend(f,F);
  return g;
}

guide ellipse(frame f, real xmargin=0, real ymargin=infinity,
	      pen p=currentpen, filltype filltype=NoFill)
{
  if(ymargin == infinity) ymargin=xmargin;
  pair m=min(f);
  pair M=max(f);
  pair D=M-m;
  static real factor=0.5*sqrt(2);
  int sign=filltype == Fill ? -1 : 1;
  guide g=ellipse(0.5*(M+m),factor*D.x+0.5*sign*max(p).x+xmargin,
		  factor*D.y+0.5*sign*max(p).y+ymargin);
  frame F;
  filltype(F,g,p);
  prepend(f,F);
  return g;
}

frame bbox(picture pic=currentpicture, real xmargin=0, real ymargin=infinity,
	   pen p=currentpen, filltype filltype=NoFill)
{
  if(ymargin == infinity) ymargin=xmargin;
  frame f=pic.fit(max(pic.xsize-2*xmargin,0),max(pic.ysize-2*ymargin,0),
		  pic.keepAspect);
  box(f,xmargin,ymargin,p,filltype);
  return f;
}

guide box(frame f, Label L, real xmargin=0, real ymargin=infinity,
	  pen p=currentpen, filltype filltype=NoFill)
{
  add(f,L);
  return box(f,xmargin,ymargin,p,filltype);
}

guide ellipse(frame f, Label L, real xmargin=0, real ymargin=infinity,
	      pen p=currentpen, filltype filltype=NoFill)
{
  add(f,L);
  return ellipse(f,xmargin,ymargin,p,filltype);
}

void box(picture pic=currentpicture, Label L,
	 real xmargin=0, real ymargin=infinity, pen p=currentpen,
	 filltype filltype=NoFill)
{
  pic.add(new void (frame f, transform t) {
    transform t0=shiftless(t);
    frame d;
    _label(d,L.s,L.size,degrees(t0*dir(L.angle)),
	   t*L.position+L.align.dir*labelmargin(L.p)+L.shift,
	   length(L.align.dir)*unit(t0*L.align.dir),L.p);
    box(d,xmargin,ymargin,p,filltype);
    add(f,d);
  });
  Label L0=L.copy();
  L0.position(0);
  L0.p(p+overwrite(Allow));
  frame f;
  box(f,L0,xmargin,ymargin,p,filltype);
  pic.addBox(L.position,L.position,min(f),max(f));
}

real linewidth() 
{
  return linewidth(currentpen);
}

real interp(int a, int b, real c)
{
  return a+c*(b-a);
}

real interp(real a, real b, real c)
{
  return a+c*(b-a);
}

pair interp(pair a, pair b, real c)
{
  return a+c*(b-a);
}

triple interp(triple a, triple b, real c)
{
  return a+c*(b-a);
}

pen interp(pen a, pen b, real c) 
{
  return (1-c)*a+c*b;
}

void dot(frame f, pair z, pen p=currentpen)
{
  draw(f,z,dotsize(p)+p);
}

void dot(picture pic=currentpicture, pair z, pen p=currentpen)
{
  Draw(pic,z,dotsize(p)+p);
}

void dot(picture pic=currentpicture, pair[] z, pen p=currentpen)
{
  for(int i=0; i < z.length; ++i) dot(pic,z[i],p);
}

void dot(picture pic=currentpicture, explicit path g, pen p=currentpen)
{
  for(int i=0; i <= length(g); ++i) dot(pic,point(g,i),p);
}

void dot(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) dot(pic,g[i],p);
}

void dot(picture pic=currentpicture, Label L, pair z, align align=NoAlign,
	 string format=defaultformat, pen p=currentpen)
{
  Label L=L.copy();
  L.position(z);
  if(L.s == "") L.s="("+format(format,z.x)+","+format(format,z.y)+")";
  L.align(align,E);
  L.p(p);
  dot(pic,z,p);
  add(pic,L);
}

void dot(picture pic=currentpicture, Label L, pen p=currentpen)
{
  dot(pic,L,L.position,p);
}

// Return a unit polygon with n sides
guide polygon(int n) 
{
  guide g;
  for(int i=0; i < n; ++i) g=g--expi(2pi*(i+0.5)/n-0.5*pi);
  return g--cycle;
}

// Return an n-point unit cross
path[] cross(int n) 
{
  path[] g;
  for(int i=0; i < n; ++i) g=g^^(0,0)--expi(2pi*(i+0.5)/n-0.5*pi);
  return g;
}

path[] plus=(-1,0)--(1,0)^^(0,-1)--(0,1);

void mark(picture pic=currentpicture, path g, frame mark)
{
  for(int i=0; i <= length(g); ++i)
    add(point(g,i),pic,mark);
}

frame marker(path[] g, pen p=currentpen, filltype filltype=NoFill)
{
  frame f;
  if(filltype == Fill) fill(f,g,p);
  else draw(f,g,p);
  return f;
}

void shipout(string prefix=defaultfilename, frame f, frame preamble=patterns,
	     string format="", bool wait=NoWait)
{
  bool Transform=GUIFilenum < GUIlist.length;
  static transform[] noTransforms;
  static bool[] noDeletes;
  if(gui.length > 0) {
    frame F;
    add(F,f);
    for(int i=0; i < gui.length; ++i)
      add(F,gui(i));
    f=F;
  }
  shipout(prefix,f,preamble,format,wait,
  	  Transform ? GUIlist[GUIFilenum].Transform : noTransforms,
	  Transform ? GUIlist[GUIFilenum].Delete : noDeletes);
  ++GUIFilenum;
  shipped=true;
  uptodate=true;
}

picture legend(Legend[] legend)
{
  picture inset;
  size(inset,0,0,IgnoreAspect);
  if(legend.length > 0) {
    for(int i=0; i < legend.length; ++i) {
      Legend L=legend[i];
      pair z1=legendmargin-i*I*legendskip*fontsize(L.p);
      pair z2=z1+legendlinelength;
      if(!L.putmark && !empty(L.mark)) mark(inset,interp(z1,z2,0.5),L.mark);
      Draw(inset,z1--z2,L.p);
      label(inset,L.label,z2,E,L.plabel);
      if(L.putmark && !empty(L.mark)) mark(inset,interp(z1,z2,0.5),L.mark);
    }
  }
  return inset;
}
  
frame legend(picture pic=currentpicture, pair dir=0, 
	     real xmargin=legendmargin, real ymargin=infinity,
	     pen p=currentpen, filltype filltype=NoFill) 
{
  frame F;
  if(pic.legend.length == 0) return F;
  F=bbox(legend(pic.legend),xmargin,ymargin,p,filltype);
  return shift(dir-point(F,-dir))*F;
}

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);

void shipout(string prefix=defaultfilename, picture pic,
	     frame preamble=patterns, orientation orientation=Portrait,
	     string format="", bool wait=NoWait)
{
  shipout(prefix,orientation(pic.fit()),preamble,format,wait);
}

void shipout(string prefix=defaultfilename, orientation orientation=Portrait,
	     string format="", bool wait=NoWait)
{
  shipout(prefix,currentpicture,orientation,format,wait);
}

void erase(picture pic=currentpicture)
{
  pic.erase();
}

// A restore thunk is a function, that when called, restores the graphics state
// to what it was when the restore thunk was created.
typedef void restoreThunk();

// When save is called, this will be redefined to do the corresponding restore.
void restore()
{
  write("warning: restore called with no matching save");
}

restoreThunk buildRestoreThunk() {
  pen p=currentpen;
  picture pic=currentpicture.copy();
  restoreThunk r=restore;
  return new void () {
    currentpen=p;
    currentpicture=pic;
    restore=r;
    uptodate=false;
  };
}

// Save the current state, so that restore will put things back in that state.
restoreThunk save() 
{
  return restore=buildRestoreThunk();
}

real cap(real x, real m, real M, real bottom, real top)
{
  return x+top > M ? M-top : x+bottom < m ? m-bottom : x;
}

// Scales pair z, so that when drawn with pen p, it does not exceed box(lb,rt).
pair cap(pair z, pair lb, pair rt, pen p=currentpen)
{

  return (cap(z.x,lb.x,rt.x,min(p).x,max(p).x),
          cap(z.y,lb.y,rt.y,min(p).y,max(p).y));
}

real xtrans(transform t, real x)
{
  return (t*(x,0)).x;
}

real ytrans(transform t, real y)
{
  return (t*(0,y)).y;
}

real cap(transform t, real x, real m, real M, real bottom, real top,
	 real ct(transform,real))
{
  return x == infinity  ? M-top :
         x == -infinity ? m-bottom : cap(ct(t,x),m,M,bottom,top);
}

pair cap(transform t, pair z, pair lb, pair rt, pen p=currentpen)
{
  if (finite(z))
    return cap(t*z, lb, rt, p);
  else
    return (cap(t,z.x,lb.x,rt.x,min(p).x,max(p).x,xtrans),
            cap(t,z.y,lb.y,rt.y,min(p).y,max(p).y,ytrans));
}
  
static bool CCW=true;
static bool CW=false;						  

// return an arc centered at c with radius r from angle1 to angle2 in degrees,
// drawing in the given direction.
guide arc(pair c, real r, real angle1, real angle2, bool direction)
{
  real t1=intersect(unitcircle,(0,0)--2*dir(angle1)).x;
  real t2=intersect(unitcircle,(0,0)--2*dir(angle2)).x;
  static int n=length(unitcircle);
  if(t1 >= t2 && direction) t1 -= n;
  if(t2 >= t1 && !direction) t2 -= n;
  return shift(c)*scale(r)*subpath(unitcircle,t1,t2);
}
  
// return an arc centered at c with radius r > 0 from angle1 to angle2 in
// degrees, drawing counterclockwise if angle2 >= angle1 (otherwise clockwise).
// If r < 0, draw the complementary arc of radius |r|.
guide arc(pair c, real r, real angle1, real angle2)
{
  bool pos=angle2 >= angle1;
  if(r > 0) return arc(c,r,angle1,angle2,pos ? CCW : CW);
  else return arc(c,-r,angle1,angle2,pos ? CW : CCW);
}

// return an arc centered at c from pair z1 to z2 (assuming |z2-c|=|z1-c|),
// drawing in the given direction.
guide arc(pair c, explicit pair z1, explicit pair z2, bool direction=CCW)
{
  return arc(c,abs(z1-c),degrees(z1-c),degrees(z2-c),direction);
}

void bar(picture pic, pair a, pair d, pen p=currentpen)
{
  picture opic;
  Draw(opic,-0.5d--0.5d,p+solid);
  add(a,pic,opic);
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
		    filltype filltype=Fill, position position=BeginPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(g,p,size,angle,filltype,position,false,margin));
    return false;
  };
}

arrowbar Arrow(real size=0, real angle=arrowangle,
	       filltype filltype=Fill, position position=EndPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow(g,p,size,angle,filltype,position,margin));
    return false;
  };
}

arrowbar EndArrow(real size=0, real angle=arrowangle,
		  filltype filltype=Fill, position position=EndPoint)=Arrow;

arrowbar MidArrow(real size=0, real angle=arrowangle, filltype filltype=Fill)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,
		  arctime(g,(arclength(g)+size)/2),margin));
    return false;
  };
}
  
arrowbar Arrows(real size=0, real angle=arrowangle, filltype filltype=Fill)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    add(pic,arrow2(g,p,size,angle,filltype,margin));
    return false;
  };
}

arrowbar BeginArcArrow(real size=0, real angle=arcarrowangle,
		       filltype filltype=Fill, position position=BeginPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,position,false,margin));
    return false;
  };
}

arrowbar ArcArrow(real size=0, real angle=arcarrowangle,
		  filltype filltype=Fill, position position=EndPoint)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,position,margin));
    return false;
  };
}

arrowbar EndArcArrow(real size=0, real angle=arcarrowangle,
		     filltype filltype=Fill,
		     position position=EndPoint)=ArcArrow;
  
arrowbar MidArcArrow(real size=0, real angle=arcarrowangle,
		     filltype filltype=Fill)
{
  return new bool(picture pic, path g, pen p, margin margin) {
    real size=size == 0 ? arcarrowsize(p) : size;
    add(pic,arrow(g,p,size,angle,filltype,
		  arctime(g,(arclength(g)+size)/2),margin));
    return false;
  };
}
  
arrowbar ArcArrows(real size=0, real angle=arcarrowangle,
		   filltype filltype=Fill)
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

public arrowbar
  Blank=Blank(),
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
	  margin margin=NoMargin, string legend="", frame mark=newframe,
	  bool putmark=Above)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  if(!putmark && !empty(mark)) mark(pic,g,mark);
  if(L.s != "") L.out(pic,g);
  bool drawpath=arrow(pic,g,p,margin);
  if(bar(pic,g,p,margin) && drawpath) _draw(pic,g,p,margin);
  if(legend != "") {
    Legend l; l.init(legend,L.p,p,mark,putmark);
    pic.legend.push(l);
  }
  if(putmark && !empty(mark)) mark(pic,g,mark);
}

// Draw a fixed-size line about the user-coordinate 'origin'.
void draw(pair origin, picture pic=currentpicture, Label L="", path g,
	  align align=NoAlign, pen p=currentpen, arrowbar arrow=None,
	  arrowbar bar=None, margin margin=NoMargin, string legend="",
	  frame mark=newframe, bool putmark=Above)
{
  picture opic;
  draw(opic,L,g,align,p,arrow,bar,margin,legend,mark,putmark);
  add(origin,pic,opic);
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
  
string substr(string s, int pos)
{
  return substr(s,pos,-1);
}

int find(string s, string t)
{
  return find(s,t,0);
}

int rfind(string s, string t)
{
  return rfind(s,t,-1);
}

// returns a string with all occurrences of string 'from' in string 's'
// changed to string 'to'
string replace(string s, string from, string to) 
{
  return replace(s,new string[][] {{from,to}});
}

// Like texify but don't convert embedded TeX commands: \${}
string TeXify(string s) 
{
  static string[][] t={{"&","\&"},{"%","\%"},{"_","\_"},{"#","\#"},{"<","$<$"},
		       {">","$>$"},{"|","$|$"},{"^","$\hat{\ }$"},{". ",".\ "},
                       {"~","$\tilde{\ }$"}};
  return replace(s,t);
}

// Convert string to TeX
string texify(string s) 
{
  static string[][] t={{'\\',"\backslash "},{"$","\$"},{"{","\{"},{"}","\}"}};
  static string[][] u={{"\backslash ","$\backslash$"}};
  return TeXify(replace(replace(s,t),u));
}

string italic(string s)
{
  if(s == "") return s;
  return "{\it "+s+"}";
}

string baseline(string s, align align=S, string template="M") 
{
  if(s == "") return s;
  return align.dir.y <= -0.5*abs(align.dir.x) ? 
    "\ASYbase{"+template+"}{"+s+"}" : s;
}

string math(string s)
{
  if(s == "") return s;
  return "$"+s+"$";
}

string include(string name, string options="")
{
  if(options != "") options="["+options+"]";
  return "\includegraphics"+options+"{"+name+"}";
}

string minipage(string s, real width=100pt)
{
  return "\begin{minipage}{"+(string) (width*pt)+"pt}"+s+"\end{minipage}";
}

string math(real x)
{
  return math((string) x);
}

static bool Keep=true;
static bool Purge=false;			  

// delay is in units of 0.01s
int gifmerge(int loops=0, int delay=50, bool keep=Purge)
{
  return merge("-loop " +(string) loops+" -delay "+(string)delay,"gif",keep);
}

// Return the sequence 0,1,...n-1
int[] sequence(int n) {return sequence(new int(int x){return x;},n);}
// Return the sequence n,...m
int[] sequence(int n, int m) {
  return sequence(new int(int x){return x;},m-n+1)+n;
}
int[] reverse(int n) {return sequence(new int(int x){return n-1-x;},n);}

bool[] reverse(bool[] a) {return a[reverse(a.length)];}
int[] reverse(int[] a) {return a[reverse(a.length)];}
real[] reverse(real[] a) {return a[reverse(a.length)];}
pair[] reverse(pair[] a) {return a[reverse(a.length)];}
string[] reverse(string[] a) {return a[reverse(a.length)];}

int find(bool[] a) {return find(a,1);}

// Transform coordinate in [0,1]x[0,1] to current user coordinates.
pair relative(picture pic=currentpicture, pair z)
{
  pair w=(pic.userMax-pic.userMin);
  return pic.userMin+(z.x*w.x,z.y*w.y);
}

void pause(string w="Hit enter to continue") 
{
  write(w);
  w=stdin;
}

struct slice {
  public path before,after;
}
  
slice operator init() {return new slice;}

slice firstcut(path g, path knife) 
{
  slice s;
  real t=intersect(g,knife).x;
  if (t < 0) {s.before=g; s.after=nullpath; return s;}
  s.before=subpath(g,0,min(t,intersect(g,reverse(knife)).x));
  s.after=subpath(g,min(t,intersect(g,reverse(knife)).x),length(g));
  return s;
}

slice lastcut(path g, path knife) 
{
  slice s=firstcut(reverse(g),knife);
  path before=reverse(s.after);
  s.after=reverse(s.before);
  s.before=before;
  return s;
}

string format(real x)
{
  return format(defaultformat,x);
}

pen[] colorPen={red,blue,green,magenta,cyan,orange,purple,brown,darkblue,
		darkgreen,chartreuse,fuchsia,salmon,lightblue,black,lavender,
		pink,yellow,gray};
pen[] monoPen={solid,dashed,dotted,longdashed,dashdotted,longdashdotted};

transform invert=reflect((0,0),(1,0));
static public real circlescale=0.85;

frame[] Mark={
  scale(circlescale)*marker(unitcircle),
  marker(polygon(3)),marker(polygon(4)),
  marker(polygon(5)),marker(invert*polygon(3)),
  marker(cross(4)),marker(cross(6))
};

frame[] MarkFill={
  scale(circlescale)*marker(unitcircle,Fill),marker(polygon(3),Fill),
  marker(polygon(4),Fill),marker(polygon(5),Fill),
  marker(invert*polygon(3),Fill)
};

public bool mono=false;

pen monoPen(int n) 
{
  return monoPen[n % monoPen.length];
}

pen Pen(int n) 
{
  return mono ? monoPen(n) : colorPen[n % colorPen.length];
}

frame Mark(int n) 
{
  n=n % (Mark.length+MarkFill.length);
  if(n < Mark.length) return Mark[n];
  else return MarkFill[n-Mark.length];
}

pen fontsize(real size) 
{
  return fontsize(size,1.2*size);
}

pen font(string name) 
{
  return fontcommand("\font\ASYfont="+name+"\ASYfont");
}

pen font(string name, real size) 
{
  // Extract size of requested TeX font
  string basesize;
  for(int i=0; i < length(name); ++i) {
    string c=substr(name,i,1);
    if(c >= "0" && c <= "9") basesize += c;
    else if(basesize != "") break;
  }
  return basesize == "" ? font(name) :
    font(name+" scaled "+(string) (1000*size/(int) basesize)); 
}

pen font(string encoding, string family, string series="m", string shape="n") 
{
  return fontcommand("\usefont{"+encoding+"}{"+family+"}{"+series+"}{"+shape+
		     "}");
}

pen AvantGarde(string series="m", string shape="n")
{
  return font("OT1","pag",series,shape);
}
pen Bookman(string series="m", string shape="n")
{
  return font("OT1","pbk",series,shape);
}
pen Courier(string series="m", string shape="n")
{
  return font("OT1","pcr",series,shape);
}
pen Helvetica(string series="m", string shape="n")
{
  return font("OT1","phv",series,shape);
}
pen NewCenturySchoolBook(string series="m", string shape="n")
{
  return font("OT1","pnc",series,shape);
}
pen Palatino(string series="m", string shape="n")
{
  return font("OT1","ppl",series,shape);
}
pen TimesRoman(string series="m", string shape="n")
{
  return font("OT1","ptm",series,shape);
}
pen ZapfChancery(string series="m", string shape="n")
{
  return font("OT1","pzc",series,shape);
}
pen Symbol(string series="m", string shape="n")
{
  return font("OT1","psy",series,shape);
}
pen ZapfDingbats(string series="m", string shape="n")
{
  return font("OT1","pzd",series,shape);
}

real min(real[][] a) {
  real min=infinity;
  for(int i=0; i < a.length; ++i) min=min(min,min(a[i]));
  return min;
}

real max(real[][] a) {
  real max=-infinity;
  for(int i=0; i < a.length; ++i) max=max(max,max(a[i]));
  return max;
}

real min(real[][][] a) {
  real min=infinity;
  for(int i=0; i < a.length; ++i) min=min(min,min(a[i]));
  return min;
}

real max(real[][][] a) {
  real max=-infinity;
  for(int i=0; i < a.length; ++i) max=max(max,max(a[i]));
  return max;
}

void atexit()
{
  if(interact()) {
    if(!uptodate) shipout();
  } else if(!shipped) shipout();
}
atexit(atexit);

guide operator ::(... guide[] a)
{
  guide g=(a.length > 0) ? a[0] : nullpath;
  for(int i=1; i < a.length; ++i)
    g=g..operator tension(1,true)..a[i];
  return g;
}

guide operator ---(... guide[] a)
{
  guide g=(a.length > 0) ? a[0] : nullpath;
  for(int i=1; i < a.length; ++i)
    g=g..operator tension(infinity,true)..a[i];
  return g;
}

// Three-dimensional projections (move back to three.asy once new import
// scheme is functional):

typedef real[][] transform3;

static struct projection {
  public triple camera;
  public transform3 project;
  void init(triple camera, transform3 project) {
    this.camera=camera;
    this.project=project;
  }
}

projection operator init() {return new projection;}
  
public projection currentprojection;
