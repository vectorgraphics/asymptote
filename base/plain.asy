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

static real inches=72;
static real inch=inches;
static real cm=inches/2.540005;
static real mm=0.1cm;
static real bp=1;	    // A PostScript point.
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

// Global parameters:
static public real labelmargin=0.3;
static public real arrowlength=0.75cm;
static public real arrowfactor=15;
static public real arrowangle=15;
static public real arcarrowfactor=0.5*arrowfactor;
static public real arcarrowangle=2*arrowangle;
static public real barfactor=arrowfactor;
static public real dotfactor=6;

static public pair legendlocation=(1.0,0.8);
static public real legendlinelength=50;
static public real legendskip=1.5;
static public pen legendboxpen=black;
static public real legendmargin=10;

static public string defaultfilename;

// Reduced for tension atleast infinity
static real infinity=sqrt(0.5*realMax());
static real epsilon=realEpsilon();

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

// Cut two parentheses.
transform shift(real x, real y)
{
  return shift((x,y));
}

// I/O operations

file input(string name) {return input(name,true);}
file xinput(string name) {return xinput(name,true);}
file output(string name) {return output(name,false);}
file xoutput(string name) {return xoutput(name,false);}
file csv(file f) {return csv(f,true);}
file line(file f) {return line(f,true);}
file single(file f) {return single(f,true);}

file stdin=input("");
file stdout;

private struct endlT {};
static public endlT endl=null;

private struct tabT {};
static public tabT tab=null;

void write(file out, endlT endl=null) {write(out,'\n'); flush(out);}
void write(file out=stdout, bool x, endlT) {write(out,x); write(out);}
void write(file out=stdout, int x, endlT) {write(out,x); write(out);}
void write(file out=stdout, real x, endlT) {write(out,x); write(out);}
void write(file out=stdout, pair x, endlT) {write(out,x); write(out);}
void write(file out=stdout, string x, endlT) {write(out,x); write(out);}
void write(file out=stdout, guide x, endlT) {write(out,x); write(out);}
void write(file out=stdout, pen x, endlT) {write(out,x); write(out);}
void write(file out=stdout, transform x, endlT) {write(out,x); write(out);}

void write(file out=stdout, tabT) {write(out,'\t');}
void write(file out=stdout, bool x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, int x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, real x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, pair x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, string x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, guide x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, pen x, tabT) {write(out,x); write(out,tab);}
void write(file out=stdout, transform x, tabT) {write(out,x); write(out,tab);}

// write(x) with no file argument does a write(stdout,x,endl)
void write() {write(stdout);}
void write(guide x) {write(stdout,x,endl);}
void write(pen x) {write(stdout,x,endl);}
void write(transform x) {write(stdout,x,endl);}

void write(file out=stdout, string x, real y)
{
  write(out,x); write(out,y,endl);
}

void write(file out=stdout, string x, pair y)
{
  write(out,x); write(out,y,endl);
}

void write(file out=stdout, string x, real y, string x2, real y2)
{
  write(out,x); write(out,y,tab); write(out,x2,y2);
}

string ask(string prompt)
{
  write(stdout,prompt);
  return stdin;
}

static private int GUIFilenum=0;
static private int GUIObject=0;
static private string GUIPrefix;
static public frame patterns;

void deconstruct(frame d)
{
  if(deconstruct()) {
    string prefix=GUIPrefix == "" ? fileprefix() : GUIPrefix;
    shipout(prefix+"_"+(string) GUIObject,d,patterns,"tgif",false);
  }
  ++GUIObject;
}

private struct DELETET {}
static public DELETET DELETE=null;

struct GUIop
{
  public transform [] Transform=new transform [];
  public bool Delete=false;
}

GUIop [][] GUIlist;

// Delete item
void GUIop(int index, int filenum=0, DELETET)
{
  if(GUIlist.length <= filenum) GUIlist[filenum]=new GUIop [];
  GUIop [] GUIobj=GUIlist[filenum];
  while(GUIobj.length <= index) GUIobj.push(new GUIop);
  GUIobj[index].Delete=true;
}

// Transform item
void GUIop(int index, int filenum=0, transform g)
{
  if(GUIlist.length <= filenum) GUIlist[filenum]=new GUIop [];
  GUIop [] GUIobj=GUIlist[filenum];
  while(GUIobj.length <= index) GUIobj.push(new GUIop);
  GUIobj[index].Transform.push(g);
}

bool GUIDelete()
{
  if(GUIFilenum < GUIlist.length) {
    GUIop [] GUIobj=GUIlist[GUIFilenum];
    bool del=(GUIObject < GUIobj.length) ? GUIobj[GUIObject].Delete : false;
    if(del) deconstruct(nullframe);
    return del;
  }
  return false;
}

transform GUI(transform t=identity())
{
  if(GUIFilenum < GUIlist.length) {
    GUIop [] GUIobj=GUIlist[GUIFilenum];
    if(GUIObject < GUIobj.length) {
      transform [] G=GUIobj[GUIObject].Transform;
      for(int i=0; i < G.length; ++i) {
	t=G[i]*t;
      }
    }
  }
  return t;
}

// A function that draws an object to frame pic, given that the transform
// from user coordinates to true-size coordinates is t.
typedef void drawer(frame f, transform t);

// A generalization of drawer that includes the final frame's bounds.
typedef void drawerBound(frame f, transform t, transform T, pair lb, pair rt);

// A coordinate in "flex space." A linear combination of user and true-size
// coordinates.
static struct coord {
  public real user, truesize;

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

void append(coord[] dest, coord[] src)
{
  for(int i=0; i < src.length; ++i)
    dest.push(src[i]);
}

void append(coord[] x, coord[] y, transform T, coord[] srcx, coord[] srcy)
{
  for(int i=0; i < srcx.length; ++i) {
    pair z=T*(srcx[i].user,srcy[i].user);
    x.push(coord.build(z.x,srcx[i].truesize));
    y.push(coord.build(z.y,srcy[i].truesize));
  }
  return;
}

bool operator <= (coord a, coord b)
{
  return a.user <= b.user && a.truesize <= b.truesize;
}

bool operator >= (coord a, coord b)
{
  return a.user >= b.user && a.truesize >= b.truesize;
}

coord[] maxcoords(coord[] in, bool operator <= (coord,coord))
{
  // As operator <= is defined in the parameter list, it has a special
  // meaning in the body of the function.

  coord best;
  coord[] c;

  int n=in.length;
  if (n == 0)
    return c;
  else {
    // Add the first coord without checking restrictions (as there are none).
    best=in[0];
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
    if (x <= best)
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

public struct scaleT {
  typedef real scalefcn(real x);
  public scalefcn T,Tinv,Label;
  T=Tinv=Label=identity;
  public bool automin=true, automax=true;
};

public struct autoscaleT {
  public scaleT scale=new scaleT;
  public scaleT postscale=new scaleT;
  public real tickMin=infinity, tickMax=-infinity;
  public bool automin=true, automax=true;
  public bool automin() {return automin && scale.automin;}
  public bool automax() {return automax && scale.automax;}
  
  real T(real x) {return postscale.T(scale.T(x));}
  real Tinv(real x) {return scale.Tinv(postscale.Tinv(x));}
  real Label(real x) {return scale.Label(postscale.Tinv(x));}
}

public struct ScaleT {
  public bool set=false;
  public autoscaleT x=new autoscaleT;
  public autoscaleT y=new autoscaleT;
  public autoscaleT z=new autoscaleT;
};

struct Legend {
  public string label;
  public pen p;
  public frame mark;
  public bool topmark;
}

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

struct picture {
  // The functions to do the deferred drawing.
  drawerBound[] nodes;
  
  // The coordinates in flex space to be used in sizing the picture.
  coord[] xcoords,ycoords;

  // Transform to be applied to this picture.
  public transform T;
  
  public bool deconstruct=false;
  public pair userMin,userMax;
  
  public ScaleT scale; // Needed by graph
  public Legend legend[];

  // The maximum sizes in the x and y directions; zero means no restriction.
  public real xsize=0, ysize=0;
  
  // If true, the x and y must be scaled by the same amount.
  public bool keepAspect=false;

  void init() {
    userMin=(infinity,infinity);
    userMax=-userMin;
    scale=new ScaleT;
  }
  init();
  
  // Erase the current picture, retaining any size specification.
  void erase() {
    nodes=new drawerBound[];
    xcoords=new coord[];
    ycoords=new coord[];
    T=identity();
    legend=new Legend[];
    init();
  }
  
  void userBox(pair min, pair max) {
    userMin=minbound(userMin,min);
    userMax=maxbound(userMax,max);
  }
  
  void add(drawerBound d) {
    nodes.push(d);
  }

  void build(frame f, drawer d, transform t, bool deconstruct) {
      if(deconstruct) {
	if(GUIDelete()) return;
	t=GUI(t);
      }
      d(f,t);
      if(deconstruct) deconstruct(f);
      return;
  }
  
  void add(drawer d) {
    if(interact()) uptodate=false;
    bool deconstruct=this.deconstruct;
    add(new void (frame f, transform t, transform T, pair, pair) {
      frame F;
      build(F,d,t*T,deconstruct);
      add(f,F);
    });
  }

  static string clipwarn="warning: deconstructed pictures cannot be clipped";
  void clippingwarning()
  {
    if(deconstruct && deconstruct()) write(clipwarn);
  }
  
  void clip(drawer d) {
    if(interact()) uptodate=false;
    clippingwarning();
    for(int i=0; i < xcoords.length; ++i) {
      xcoords[i].clip(userMin.x,userMax.x);
      ycoords[i].clip(userMin.y,userMax.y);
    }
    add(new void (frame f, transform t, transform T, pair, pair) {
      d(f,t*T);
    });
  }

  // Add a point to the sizing.
  void addPoint(pair user, pair truesize=(0,0))
  {
    xcoords.push(coord.build(user.x,truesize.x));
    ycoords.push(coord.build(user.y,truesize.y));
    userBox(user,user);
  }
  
  // Add finite components of a pair to the sizing
  void addFinite(pair user, pair truesize=(0,0))
  {
    if(abs(user.x) != infinity) xcoords.push(coord.build(user.x,truesize.x));
    if(abs(user.y) != infinity) ycoords.push(coord.build(user.y,truesize.y));
    userBox(user,user);
  }
  
  // Add a box to the sizing.
  void addBox(pair userMin, pair userMax,
              pair trueMin=(0,0), pair trueMax=(0,0))
  {
    addPoint(userMin,trueMin);
    addPoint(userMax,trueMax);
  }

  // Add a point to the sizing, accounting also for the size of the pen.
  void addPoint(pair user, pair truesize=(0,0), pen p)
  {
    addPoint(user,truesize+min(p));
    addPoint(user,truesize+max(p));
  }
  
  // Add a (user space) path to the sizing.
  void addPath(path g)
  {
    addPoint(min(g));
    addPoint(max(g));
  }

  // Add a path to the sizing with the additional padding of a pen.
  void addPath(path g, pen p)
  {
    addPoint(min(g),min(p));
    addPoint(max(g),max(p));
  }

  void size(real x=0, real y=0, bool a=true)
  {
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
  real min(scaling s, coord[] c)
  {
    if (c.length > 0) {
      real m=infinity;
      for (int i=0; i < c.length; ++i)
	if(s.scale(c[i]) < m) m=s.scale(c[i]);
      return m;
    }
    else
      // I don't know...
      return 0;
  }
 
  // Calculate the maximum point in scaling the coords.
  real max(scaling s, coord[] c)
  {
    if (c.length > 0) {
      real M=-infinity;
      for (int i=0; i < c.length; ++i)
        if (s.scale(c[i]) > M)
          M=s.scale(c[i]);
      return M;
    }
    else
      // I don't know...
      return 0;
  }

  // Calculate the min for the final picture, given the transform of coords.
  pair min(transform t)
  {
    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x, b.x);
    scaling ys=scaling.build(a.y, b.y);
    return (min(xs,xcoords),min(ys,ycoords));
  }

  // Calculate the max for the final picture, given the transform of coords.
  pair max(transform t)
  {
    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x, b.x);
    scaling ys=scaling.build(a.y, b.y);
    return (max(xs,xcoords),max(ys,ycoords));
  }

  // Calculate the sizing constants for the given array and maximum size.
  scaling calculateScaling(coord[] coords, real max) {
    import simplex;
    simplex.problem p=new simplex.problem;
   
    void addMinCoord(coord c) {
      // (a*user + b) + truesize >= 0:
      p.addRestriction(c.user,1,c.truesize);
    }
    void addMaxCoord(coord c) {
      // (a*user + b) + truesize <= max:
      p.addRestriction(-c.user,-1,max-c.truesize);
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
      return calculateScaling(coords,sqrt(2)*max);
    }
  }

  // Returns the transform for turning user-space pairs into true-space pairs.
  transform calculateTransform(real xsize, real ysize, bool keepAspect=true)
  {
    if (xsize == 0 && ysize == 0)
      return identity();
    else if (ysize == 0) {
      scaling sx=calculateScaling(xcoords,xsize);
      return scale(sx.a);
    }
    else if (xsize == 0) {
      scaling sy=calculateScaling(ycoords,ysize);
      return scale(sy.a);
    }
    else {
      scaling sx=calculateScaling(xcoords,xsize);
      scaling sy=calculateScaling(ycoords,ysize);
      if (keepAspect)
        return scale(min(sx.a,sy.a));
      else
        return xscale(sx.a)*yscale(sy.a);
    }
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
  // picture. Warning: Shipping out this picture will not scale as a normal
  // picture would.
  picture drawcopy()
  {
    picture dest=new picture;
    for (int i=0; i < nodes.length; ++i)
      dest.add(nodes[i]);
    
    dest.T=T;
    dest.deconstruct=deconstruct;
    dest.userMin=userMin;
    dest.userMax=userMax;
    dest.scale=scale;
    dest.legend=legend;

    return dest;
  }

  // A deep copy of this picture.  Modifying the copied picture will not affect
  // the original.
  picture copy()
  {
    picture dest=drawcopy();

    append(dest.xcoords,xcoords);
    append(dest.ycoords,ycoords);

    dest.xsize=xsize; dest.ysize=ysize; dest.keepAspect=keepAspect;

    return dest;
  }

  // Add a picture to this picture, such that the user coordinates will be
  // scaled identically in the shipout.
  void add(picture src)
  {
    // Copy the picture.  Only the drawing function closures are needed, so we
    // only copy them.  This needs to be a deep copy, as src could later have
    // objects added to it that should not be included in this picture.

    picture srcCopy=src.drawcopy();

    // Draw by drawing the copied picture.
    add(new void (frame f, transform t, transform T, pair m, pair M) {
      if(deconstruct && !src.deconstruct) {
	if(GUIDelete()) return;
	T=GUI(T);
      }
     frame d=srcCopy.fit(t,T*srcCopy.T,m,M);
     if(deconstruct && !src.deconstruct) deconstruct(d);
     add(f,d);
     for(int i=0; i < src.legend.length; ++i)
       legend.push(src.legend[i]);
    });
    
    userBox(src.userMin,src.userMax);

    // Add the coord info to this picture.
    append(xcoords,ycoords,srcCopy.T,src.xcoords,src.ycoords);
  }
}

picture operator * (transform t, picture orig)
{
  picture pic=orig.copy();
  pic.T=t*pic.T;
  pic.userMin=t*pic.userMin;
  pic.userMax=t*pic.userMax;
  return pic;
}

public picture currentpicture=new picture;
currentpicture.deconstruct=true;

public picture gui[];

picture gui(int index) {
  while(gui.length <= index) {
    picture g=new picture;
    g.deconstruct=true;
    gui.push(g);
  }
  return gui[index];
}

path[] operator ^^ (path p, path q) 
{
  return new path[] {p,q};
}

path[] operator ^^ (path p, path[] q) 
{
  path[] P=new path[] {p};
  for(int i=0; i < q.length; ++i) P.push(q[i]);
  return P;
}

path[] operator ^^ (path[] p, path q) 
{
  path[] P=copy(p);
  P.push(q);
  return P;
}

path[] operator ^^ (path[] p, path[] q) 
{
  path[] P=copy(p);
  for(int i=0; i < q.length; ++i) P.push(q[i]);
  return P;
}

path[] operator * (transform t, path[] p) 
{
  path[] P;
  for(int i=0; i < p.length; ++i) P[i]=t*p[i];
  return P;
}

pair min(path[] g)
{
  pair ming=(infinity,infinity);
  for(int i=0; i < g.length; ++i)
    ming=minbound(ming,min(g[i]));
  return ming;
}

pair max(path[] g)
{
  pair maxg=(-infinity,-infinity);
  for(int i=0; i < g.length; ++i)
    maxg=maxbound(maxg,max(g[i]));
  return maxg;
}

// Add frame f about origin to picture pic
void add(pair origin=(0,0), picture pic=currentpicture, frame src)
{
  pic.add(new void (frame dest, transform t) {
    add(dest,shift(t*origin)*src);
  });
  pic.addBox(origin,origin,min(src),max(src));
}

// Add a picture to another such that user coordinates in both will be scaled
// identically in the shipout.
void add(picture dest, picture src)
{
  dest.add(src);
}

void add(picture src)
{
  add(currentpicture,src);
}

// Fit the picture src using the identity transformation (so user
// coordinates and truesize coordinates agree) and add it about the point
// origin to picture dest.
void add(pair origin, picture dest, picture src)
{
  add(origin,dest,src.fit(identity()));
}

void add(pair origin, picture src)
{
  add(origin,currentpicture,src);
}

guide box(pair a, pair b)
{
  return a--(a.x,b.y)--b--(b.x,a.y)--cycle;
}

real labelmargin(pen p=currentpen)
{
  return labelmargin*fontsize(p);
}

private struct marginT {
  public real begin,end;
};
public marginT margin=new marginT;

typedef path margin(path, pen, marginT);
private path margin(path, pen, marginT) {return nullpath;}

path trim(path g, real begin, real end) {
  real a=arctime(g,begin);
  real b=arctime(g,arclength(g)-end);
  return a <= b ? subpath(g,a,b) : point(g,a);
}

margin NoMargin()
{ 
  return new path(path g, pen, marginT margin) {
    margin.begin=margin.end=0;
    return g;
  };
}
						      
margin Margin(real begin, real end)
{ 
  return new path(path g, pen p, marginT margin) {
    real factor=labelmargin(p);
    margin.begin=begin*factor;
    margin.end=end*factor;
    return trim(g,margin.begin,margin.end);
  };
}
							   
margin PenMargin(real begin, real end)
{ 
  return new path(path g, pen p, marginT margin) {
    real factor=linewidth(p);
    margin.begin=(begin+0.5)*factor;
    margin.end=(end+0.5)*factor;
    return trim(g,margin.begin,margin.end);
  };
}
					      
margin DotMargin(real begin, real end)
{ 
  return new path(path g, pen p, marginT margin) {
    real margindot(real x) {return x > 0 ? dotfactor*x : x;}
    real factor=linewidth(p);
    margin.begin=(margindot(begin)+0.5)*factor;
    margin.end=(margindot(end)+0.5)*factor;
    return trim(g,margin.begin,margin.end);
  };
}
						      
margin TrueMargin(real begin, real end)
{ 
  margin.begin=begin;
  margin.end=end;
  return new path(path g, pen p, marginT margin) {
    return trim(g,begin,end);
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

void draw(frame f, path[] g, pen p=currentpen)
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
    draw(f,margin(t*g,p,margin),p);
  });
  pic.addPath(g,p);
}

void draw(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) Draw(pic,g[i],p);
}

void fill(frame f, path g, pen p)
{
  fill(f,g,p,0,0,p,0,0);
}

void fill(frame f, path g)
{
  fill(f,g,currentpen);
}

void fill(frame f, path[] g, pen p)
{
  fill(f,g,p,0,0,p,0,0);
}

void fill(frame f, path[] g)
{
  fill(f,g,currentpen);
}

void filldraw(frame f, path g, pen p=currentpen)
{
  fill(f,g,p);
  draw(f,g,p);
}

void fill(picture pic=currentpicture, path g,
	  pen pena=currentpen, pair a=0, real ra=0,
	  pen penb=currentpen, pair b=0, real rb=0)
{
  pic.add(new void (frame f, transform t) {
    pair A=t*a, B=t*b;
    fill(f,t*g,pena,A,abs(t*(a+ra)-A),penb,B,abs(t*(b+rb)-B));
  });
  pic.addPath(g);
}

void fill(picture pic=currentpicture, path[] g,
	  pen pena=currentpen, pair a=0, real ra=0,
	  pen penb=currentpen, pair b=0, real rb=0)
{
  g=copy(g);
  pic.add(new void (frame f, transform t) {
    pair A=t*a, B=t*b;
    fill(f,t*g,pena,A,abs(t*(a+ra)-A),penb,B,abs(t*(b+rb)-B));
  });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void fill(pair origin, picture pic=currentpicture, path g,
	       pen pena=currentpen, pair a=0, real ra=0,
	       pen penb=currentpen, pair b=0, real rb=0)
{
  picture opic=new picture;
  fill(opic,g,pena,a,ra,penb,b,rb);
  add(origin,pic,opic);
}
  
void filldraw(picture pic=currentpicture, path g,
	      pen pena=currentpen, pen drawpen=currentpen, pair a=0, real ra=0,
	      pen penb=currentpen, pair b=0, real rb=0)
{
  fill(pic,g,pena,a,ra,penb,b,rb);
  Draw(pic,g,drawpen);
}

void filldraw(picture pic=currentpicture, path[] g,
	      pen pena=currentpen, pen drawpen=currentpen, pair a=0, real ra=0,
	      pen penb=currentpen, pair b=0, real rb=0)
{
  fill(pic,g,pena,a,ra,penb,b,rb);
  draw(pic,g,drawpen);
}

void filldraw(pair origin, picture pic=currentpicture, path g,
	      pen pena=currentpen, pen drawpen=currentpen,
	      pair a=0, real ra=0,
	      pen penb=currentpen, pair b=0, real rb=0)
{
  picture opic=new picture;
  filldraw(opic,g,pena,drawpen,a,ra,penb,b,rb);
  add(origin,pic,opic);
}
  
void clip(frame f, path g)
{
  clip(f,g,currentpen);
}

void clip(frame f, path[] g)
{
  clip(f,g,currentpen);
}

void clip(picture pic=currentpicture, path g, pen p=currentpen)
{
  pic.userMin=maxbound(pic.userMin,min(g));
  pic.userMax=minbound(pic.userMax,max(g));
  pic.clip(new void (frame f, transform t) {
    clip(f,t*g,p);
  });
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

void unfill(frame f, path g)
{
  clip(f,box(min(f),max(f))^^g,evenodd);
}

void unfill(frame f, path[] g)
{
  clip(f,box(min(f),max(f))^^g,evenodd);
}

void unfill(picture pic=currentpicture, path g)
{
  pic.clip(new void (frame f, transform t) {
    unfill(f,t*g);
  });
}

void unfill(picture pic=currentpicture, path[] g)
{
  g=copy(g);
  pic.clip(new void (frame f, transform t) {
    unfill(f,t*g);
  });
}

void label(frame f, string s, real angle=0, pair position,
	   pair align=0, pen p=currentpen)
{
  _label(f,s,angle,position+align*labelmargin(p),align,p);
}

void label(picture pic=currentpicture, string s, real angle=0, pair position,
	   pair align=0, pair shift=0, pen p=currentpen)
{
  pic.add(new void (frame f, transform t) {
    pair offset=t*0;
    label(f,s,Angle(t*dir(angle)-offset),t*position+shift,
	  length(align)*unit(t*align-offset),p);
  });
  frame f;
  // Create a picture with label at the origin to extract its bbox truesize.
  label(f,s,angle,(0,0),align,p);
  pic.addBox(position,position,min(f),max(f));
}

pair point(picture pic=currentpicture, pair dir)
{
  return pic.userMin+realmult(rectify(dir),pic.userMax-pic.userMin);
}

transform rotate(real a) 
{
  return rotate(a,0);
}

guide arrowhead(path g, real position=infinity, pen p=currentpen,
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

void arrowheadbbox(picture pic=currentpicture, path g, real position=infinity,
		   pen p=currentpen, real size=0,
		   real angle=arrowangle)
{
  // Estimate the bounding box contribution using the local slope at endpoint
  // and ignoring margin.
  if(size == 0) size=arrowsize(p);
  path r=subpath(g,position,0.0);
  pair x=point(r,0);
  pair y=point(r,arctime(r,size))-x;
  pair dz1=rotate(-angle)*y;
  pair dz2=rotate(angle)*y;
  pic.addPoint(x,p);
  pic.addPoint(x,dz1,p);
  pic.addPoint(x,dz2,p);
}

private struct fillT {};
public fillT filltype=null;
typedef void filltype(frame, path, pen, fillT);
void filltype(frame, path, pen, fillT) {}
public filltype
  Fill=new void(frame f, path g, pen p, fillT) {
    p += solid;
    fill(f,g,p);
    draw(f,g,p);
  },
  NoFill=new void(frame f, path g, pen p, fillT) {
    draw(f,g,p+solid);
  };

typedef void Filltype(picture, path, pen, fillT);
public Filltype
  Fill=new void(picture pic, path g, pen p, fillT) {
    p += solid;
    fill(pic,g,p);
    Draw(pic,g,p);
  },
  NoFill=new void(picture pic, path g, pen p, fillT) {
    Draw(pic,g,p);
};

void arrow(frame f, path G, pen p=currentpen, real size=0,
	   real angle=arrowangle, filltype filltype=Fill,
	   real position=infinity, bool forwards=true, margin margin=NoMargin)
{
  if(size == 0) size=arrowsize(p);
  G=margin(G,p,margin);
  if(!forwards) G=reverse(G);
  path R=subpath(G,position,0.0);
  path S=subpath(G,position,length(G));
  size=min(arclength(G),size);
  draw(f,subpath(R,arctime(R,size),length(R)),p);
  draw(f,S,p);
  guide head=arrowhead(G,position,p,size,angle);
  filltype(f,head,p,filltype);
}

void arrow2(frame f, path G, pen p=currentpen, real size=0,
	    real angle=arrowangle, filltype filltype=Fill,
	    margin margin=NoMargin)
{
  if(size == 0) size=arrowsize(p);
  G=margin(G,p,margin);
  path R=reverse(G);
  size=min(0.5*arclength(G),size);
  draw(f,subpath(R,arctime(R,size),length(R)-arctime(G,size)),p);
  guide head=arrowhead(G,p,size,angle);
  guide tail=arrowhead(R,p,size,angle);
  filltype(f,head,p,filltype);
  filltype(f,tail,p,filltype);
}

picture arrow(path g, pen p=currentpen, real size=0,
	      real angle=arrowangle, filltype filltype=Fill,
	      real position=infinity, bool forwards=true,
	      margin margin=NoMargin)
{
  picture pic=new picture;
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
  picture pic=new picture;
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

bool Aspect=true;
bool IgnoreAspect=false;

private struct waitT {};
public waitT wait=null;
typedef bool wait(waitT);
public wait
  Wait=new bool(waitT) {return true;},
  NoWait=new bool(waitT) {return false;};

void size(picture pic=currentpicture,
          real xsize, real ysize=0, bool keepAspect=Aspect)
{
  pic.size(xsize,ysize,keepAspect);
}

private struct bboxT {};
public bboxT bbox=null;
typedef bool bbox(bboxT);
public bbox
  Background=new bool(bboxT) {return true;},
  Boundary=new bool(bboxT) {return false;};

frame _bbox(frame f, real xmargin=0, real ymargin=infinity, pen p=currentpen,
	   bbox bbox=Boundary)
{
  if(ymargin == infinity) ymargin=xmargin;
  pair z=(xmargin,ymargin);
  frame d;
  if(bbox(bbox)) fill(d,box(min(f)-0.5*min(p)-z,max(f)-0.5*max(p)+z),p);
  else draw(d,box(min(f)+0.5*min(p)-z,max(f)+0.5*max(p)+z),p);
  return d;
}

frame bbox(frame f, real xmargin=0, real ymargin=infinity, pen p=currentpen,
	   bbox bbox=Boundary)
{
  frame d=_bbox(f,xmargin,ymargin,p,bbox);
  add(d,f);
  return d;
}

frame bbox(picture pic=currentpicture, real xmargin=0, real ymargin=infinity,
	   real xsize=infinity, real ysize=infinity, bool keepAspect,
	   pen p=currentpen, bbox bbox=Boundary)
{
  if(ymargin == infinity) ymargin=xmargin;
  if(xsize == infinity) xsize=pic.xsize;
  if(ysize == infinity) ysize=pic.ysize;
  frame f=pic.fit(max(xsize-2*xmargin,0),max(ysize-2*ymargin,0),keepAspect);
  if(pic.deconstruct && GUIDelete()) return f;
  frame d=_bbox(f,xmargin,ymargin,p,bbox);
  if(pic.deconstruct) {
    d=GUI()*d;
    deconstruct(d);
  }
  add(d,f);
  return d;
}

frame bbox(picture pic=currentpicture, real xmargin=0, real ymargin=infinity,
	   real xsize=infinity, real ysize=infinity, pen p=currentpen,
	   bbox bbox=Boundary)
{
  return bbox(pic,xmargin,ymargin,xsize,ysize,
	      pic.keepAspect ? Aspect : IgnoreAspect,p,bbox);
}

frame labelBox(real xmargin=0, real ymargin=infinity,
	       string s, real angle=0, pair position,
	       pair align=0, pen p=currentpen, pen pbox=currentpen)
{
  if(ymargin == infinity) ymargin=xmargin;
  pair margin=(xmargin,ymargin);
  frame b;
  label(b,s,angle,position,align,p);
  draw(b,box(min(b)-margin,max(b)+margin),pbox);
  return b;
}

void labelbox(frame f, real xmargin=0, real ymargin=infinity,
	      string s, real angle=0, pair position,
	      pair align=0, pen p=currentpen, pen pbox=currentpen)
{
  frame b=labelBox(xmargin,ymargin,s,angle,position,align,p,pbox);
  add(f,b);
}

void labelbox(picture pic=currentpicture, real xmargin=0,
	      real ymargin=infinity, string s, real angle=0, pair position,
	      pair align=0, pair shift=0, pen p=currentpen,
	      pen pbox=currentpen)
{
  pic.add(new void (frame f, transform t) {
	    pair offset=t*0;
	    labelbox(f,xmargin,ymargin,s,Angle(t*dir(angle)-offset),
		       t*position+shift,
		       length(align)*unit(t*align-offset),p,pbox);	
  });
  frame f;
  labelbox(f,xmargin,ymargin,s,angle,(0,0),align,p,pbox);
  pic.addBox(position,position,min(f),max(f));
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

pen interp(pen a, pen b, real c) 
{
  return (1-c)*a+c*b;
}

void dot(picture pic=currentpicture, pair c)
{
  Draw(pic,c,currentpen+dotsize());
}

void dot(picture pic=currentpicture, pair c, pen p)
{
  Draw(pic,c,dotsize(p)+p);
}

void dot(picture pic=currentpicture, pair[] c, pen p=currentpen)
{
  for(int i=0; i < c.length; ++i) dot(pic,c[i],p);
}

void dot(picture pic=currentpicture, guide g, pen p=currentpen)
{
  for(int i=0; i <= length(g); ++i) dot(pic,point(g,i),p);
}

void dot(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) dot(pic,g[i],p);
}

void labeldot(picture pic=currentpicture, string s="", real angle=0,
	      pair c, pair align=E, pair shift=0, pen p=currentpen)
{
  if(s == "") s=(string) c;
  dot(pic,c,p);
  label(pic,s,angle,c,align,shift,p);
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

guide unitsquare=box((0,0),(1,1));

guide square(pair z1, pair z2)
{
  pair v=z2-z1;
  pair z3=z2+I*v;
  pair z4=z3-v;
  return z1--z2--z3--z4--cycle;
}

guide unitcircle=E..N..W..S..cycle;

guide circle(pair c, real r)
{
  return shift(c)*scale(r)*unitcircle;
}

guide ellipse(pair c, real a, real b)
{
  return shift(c)*xscale(a)*yscale(b)*unitcircle;
}

void mark(picture pic=currentpicture, guide g, frame mark)
{
  for(int i=0; i <= length(g); ++i)
    add(point(g,i),pic,mark);
}

frame marker(path g, pen p=currentpen, filltype filltype=Fill)
{
  frame f;
  filltype(f,g,p,filltype);
  return f;
}

frame marker(path[] g, pen p=currentpen, filltype filltype=Fill)
{
  frame f;
  if(filltype == Fill) fill(f,g,p);
  else draw(f,g,p);
  return f;
}

void legend(frame f, Legend[] legend, bool placement=true)
{
  if(legend.length > 0 && !GUIDelete()) {
    picture inset=new picture;
    for(int i=0; i < legend.length; ++i) {
      Legend L=legend[i];
      pen p=L.p;
      pair z1=-i*I*legendskip*fontsize(p);
      pair z2=z1+legendlinelength;
      if(!L.topmark && !empty(L.mark)) mark(inset,interp(z1,z2,0.5),L.mark);
      Draw(inset,z1--z2,p);
      label(inset,L.label,z2,E,p);
      if(L.topmark && !empty(L.mark)) mark(inset,interp(z1,z2,0.5),L.mark);
    }
    frame d;
    // Place legend with top left corner at legendlocation;
    add(d,bbox(inset,legendmargin,legendmargin,0,0,IgnoreAspect,legendboxpen));
    if(placement) {
      pair topleft=min(f)+realmult(legendlocation,max(f)-min(f))+legendmargin;
      d=GUI()*shift(topleft-(min(d).x,max(d).y))*d;
      deconstruct(d);
    }
    add(f,d);
  }
}
  
void shipout(string prefix=defaultfilename, frame f, frame preamble=patterns,
	     Legend[] legend={}, string format="", wait wait=NoWait)
{
  GUIPrefix=prefix;
  add(f,gui(GUIFilenum).fit(identity()));
  legend(f,legend);
  shipout(prefix,f,preamble,format,wait(wait));
  ++GUIFilenum;
  GUIObject=0;
  shipped=true;
  uptodate=true;
}

// Useful for compensating for space taken up by legend. For example:
// shipout(currentpicture.xsize-legendsize().x);
pair legendsize(picture pic=currentpicture)
{
  frame f;
  legend(f,pic.legend,false);
  return legendmargin+max(f)-min(f);
}

private struct orientationT {};
public orientationT orientation=null;
typedef frame orientation(frame, orientationT);
public orientation
  Portrait=new frame(frame f, orientationT) {return f;},
  Landscape=new frame(frame f, orientationT) {return rotate(90)*f;},
  Seascape=new frame(frame f, orientationT) {return rotate(-90)*f;};

void shipout(string prefix=defaultfilename, picture pic,
	     frame preamble=patterns, real xsize=infinity, real ysize=infinity,
	     bool keepAspect, orientation orientation=Portrait,
	     string format="", wait wait=NoWait)
{
  if(xsize == infinity) xsize=pic.xsize;
  if(ysize == infinity) ysize=pic.ysize;
  GUIPrefix=prefix;
  pic.deconstruct=true;
  frame f=pic.fit(xsize,ysize,keepAspect);
  shipout(prefix,orientation(f,orientation),preamble,pic.legend,format,wait);
}

void shipout(string prefix=defaultfilename,
	     real xsize=infinity, real ysize=infinity,
	     bool keepAspect, orientation orientation=Portrait,
	     string format="", wait wait=NoWait)
{
  shipout(prefix,currentpicture,ysize,keepAspect,orientation,format,wait);
}

void shipout(string prefix=defaultfilename, picture pic,
	     frame preamble=patterns, real xsize=infinity, real ysize=infinity,
	     orientation orientation=Portrait, string format="",
	     wait wait=NoWait)
{
  shipout(prefix,pic,preamble,xsize,ysize,
	  pic.keepAspect ? Aspect : IgnoreAspect,orientation,format,wait);
}

void shipout(string prefix=defaultfilename,
	     real xsize=infinity, real ysize=infinity,
	     orientation orientation=Portrait, string format="",
	     wait wait=NoWait)
{
  shipout(prefix,currentpicture,xsize,ysize,
	  currentpicture.keepAspect ? Aspect : IgnoreAspect,
	  orientation,format,wait);
}

void erase(picture pic=currentpicture)
{
  pic.erase();
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
  if (finite(z.x) && finite(z.y))
    return cap(t*z, lb, rt, p);
  else
    return (cap(t,z.x,lb.x,rt.x,min(p).x,max(p).x,xtrans),
            cap(t,z.y,lb.y,rt.y,min(p).y,max(p).y,ytrans));
}
  
void label(picture pic=currentpicture, real angle=0, pair position,
	   pair align=0, pair shift=0, pen p=currentpen)
{
  label(pic,(string) position,angle,position,align,shift,p);
}

private struct sideT {};
public sideT side=null;
typedef pair side(pair, sideT);
public side
  LeftSide=new pair(pair align, sideT) {return -align;},
  Center=new pair(pair align, sideT) {return 0;},
  RightSide=new pair(pair align, sideT) {return align;};

void label(picture pic=currentpicture, string s, real angle=0,
	   path g, real position=infinity, pair align=0, pair shift=0,
	   side side=RightSide, pen p=currentpen)
{
  real L=length(g);
  if(position == infinity) position=0.5L;
  if(align == 0) {
    if(position <= 0) align=-dir(g,0);
    else if(position >= L) align=dir(g,L);
    else align=side(-dir(g,position)*I,side);
  }
  label(pic,s,angle,point(g,position),align,shift,p);
}

void arrow(picture pic=currentpicture, string s, real angle=0, pair shift=0,
	   path g, pen p=currentpen, real size=0,
	   real Angle=arrowangle, filltype filltype=Fill,
	   margin margin=NoMargin)
{
  add(pic,arrow(g,p,size,Angle,filltype,margin));
  pair a=point(g,0);
  pair b=point(g,1);
  label(pic,s,angle,a,unit(a-b),shift,p);
}

private struct directionT {};
public directionT direction=null;
typedef bool direction(directionT);
public direction
  CCW=new bool(directionT) {return true;},
  CW=new bool(directionT) {return false;};

// return an arc centered at c with radius r from angle1 to angle2 in degrees,
// drawing in the given direction.
guide arc(pair c, real r, real angle1, real angle2, direction direction)
{
  real t1=intersect(unitcircle,(0,0)--2*dir(angle1)).x;
  real t2=intersect(unitcircle,(0,0)--2*dir(angle2)).x;
  static int n=length(unitcircle);
  if(t1 >= t2 && direction(direction)) t1 -= n;
  if(t2 >= t1 && !direction(direction)) t2 -= n;
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
guide arc(pair c, explicit pair z1, explicit pair z2, direction direction=CCW)
{
  return arc(c,abs(z1-c),Angle(z1-c),Angle(z2-c),direction);
}

void bar(picture pic, pair a, pair d, pen p=currentpen)
{
  picture opic=new picture;
  Draw(opic,-0.5d--0.5d,p+solid);
  add(a,pic,opic);
}
						      
picture bar(pair a, pair d, pen p=currentpen)
{
  picture pic=new picture;
  bar(pic,a,d,p);
  return pic;
}

private struct arrowbarT {
  public bool drawpath=true;
};
public arrowbarT arrowbar=new arrowbarT;

typedef void arrowbar(picture, path, pen, margin, arrowbarT);
private void arrowbar(picture, path, pen, margin, arrowbarT) {}

arrowbar Blank()
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
  };	
}

arrowbar None()
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {};	
}

arrowbar BeginArrow(real size=0, real angle=arrowangle,
		    filltype filltype=Fill, real position=infinity)
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
    add(pic,arrow(g,p,size,angle,filltype,position,false,margin));
  };
}

arrowbar Arrow(real size=0, real angle=arrowangle,
	       filltype filltype=Fill, real position=infinity)
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
    add(pic,arrow(g,p,size,angle,filltype,position,margin));
  };
}

arrowbar EndArrow(real size=0, real angle=arrowangle,
		  filltype filltype=Fill, real position=infinity)
{
  return Arrow(size,angle);
}

arrowbar Arrows(real size=0, real angle=arrowangle,
		filltype filltype=Fill, real position=infinity)
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
    add(pic,arrow2(g,p,size,angle,filltype,margin));
  };
}

arrowbar BeginArcArrow(real size=0, real angle=arcarrowangle,
		       filltype filltype=Fill, real position=infinity)
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
    if(size == 0) size=arcarrowsize(p);
    add(pic,arrow(g,p,size,angle,filltype,position,false,margin));
  };
}

arrowbar ArcArrow(real size=0, real angle=arcarrowangle,
		  filltype filltype=Fill, real position=infinity)
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
    if(size == 0) size=arcarrowsize(p);
    add(pic,arrow(g,p,size,angle,filltype,position,margin));
  };
}

arrowbar EndArcArrow(real size=0, real angle=arcarrowangle,
		     filltype filltype=Fill, real position=infinity)
{
  return ArcArrow(size,angle,filltype,position);
}
  
arrowbar ArcArrows(real size=0, real angle=arcarrowangle,
		   filltype filltype=Fill, real position=infinity)
{
  return new void(picture pic, path g, pen p, margin margin,
		  arrowbarT arrowbar) {
    arrowbar.drawpath=false;
    if(size == 0) size=arcarrowsize(p);
    add(pic,arrow2(g,p,size,angle,filltype,margin));
  };
}
  
arrowbar BeginBar(real size=0) 
{
  return new void(picture pic, path g, pen p, margin margin, arrowbarT) {
    if(size == 0) size=barsize(p);
    bar(pic,point(g,0),size*dir(g,0)*I,p);
  };
}

arrowbar Bar(real size=0) 
{
  return new void(picture pic, path g, pen p, margin margin, arrowbarT) {
    int L=length(g);
    if(size == 0) size=barsize(p);
    bar(pic,point(g,L),size*dir(g,L)*I,p);
  };
}

arrowbar EndBar(real size=0) 
{
  return Bar(size);
}

arrowbar Bars(real size=0) 
{
  return new void(picture pic, path g, pen p, margin margin, arrowbarT) {
    if(size == 0) size=barsize(p);
    BeginBar(size)(pic,g,p,margin,arrowbar);
    EndBar(size)(pic,g,p,margin,arrowbar);
  };
}

public arrowbar
  Blank=Blank(),
  None=None(),
  BeginArrow=BeginArrow(),
  Arrow=Arrow(),
  EndArrow=Arrow(),
  Arrows=Arrows(),
  BeginArcArrow=BeginArcArrow(),
  ArcArrow=ArcArrow(),
  EndArcArrow=ArcArrow(),
  ArcArrows=ArcArrows(),
  BeginBar=BeginBar(),
  Bar=Bar(),
  EndBar=Bar(),
  Bars=Bars();

void draw(picture pic=currentpicture, string s="", real angle=0,
	  path g, real position=infinity, pair align=0, pair shift=0,
	  side side=RightSide, pen p=currentpen,
	  arrowbar arrow=None, arrowbar bar=None,
	  margin margin=NoMargin, string legend="", frame mark=nullframe,
	  bool topmark=true)
{
  if(!topmark && !empty(mark)) mark(pic,g,mark);
  arrowbarT arrowbar=new arrowbarT;
  if(s != "") label(pic,s,angle,g,position,align,shift,side,p);
  bar(pic,g,p,margin,arrowbar);
  arrow(pic,g,p,margin,arrowbar);
  if(arrowbar.drawpath) _draw(pic,g,p,margin);
  if(legend != "") {
    Legend L=new Legend; L.label=legend; L.p=p; L.mark=mark; L.topmark=topmark;
    pic.legend.push(L);
  }
  if(topmark && !empty(mark)) mark(pic,g,mark);
}

// Draw a fixed-size object about the user-coordinate 'origin'.
void draw(pair origin, picture pic=currentpicture, string s="",
	       real angle=0, path g, real position=infinity, pair align=0,
	       pair shift=0, side side=RightSide, pen p=currentpen,
	       arrowbar arrow=None, arrowbar bar=None, margin margin=NoMargin,
	       string legend="", frame mark=nullframe, bool topmark=true)
{
  picture opic=new picture;
  draw(opic,s,angle,g,position,align,shift,side,p,arrow,bar,margin,legend,
       mark,topmark);
  add(origin,pic,opic);
}

void arrow(picture pic=currentpicture, string s="", real angle=0,
	   pair b, pair align, real length=arrowlength, real position=infinity,
	   pair shift=0, pen p=currentpen, real size=0,
	   real Angle=arrowangle, filltype filltype=Fill,
	   margin margin=EndMargin)
{
  if(position == infinity) position=0;
  marginT margin=new marginT;
  margin(b--b,p,margin); // Extract margin.begin and margin.end  
  pair a=(margin.begin+length+margin.end)*unit(align);
  draw(b,pic,s,angle,a--(0,0),0.0,align,shift,p,
	    Arrow(size,Angle,filltype),margin);
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

string baseline(string s, pair align=S, string template="M") 
{
  if(s == "") return s;
  return align.y <= -0.5*abs(align.x) ? 
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

private struct keepT {};
public keepT keep=null;
typedef bool keep(keepT);
public keep
  Keep=new bool(keepT) {return true;},
  Purge=new bool(keepT) {return false;};

// delay is in units of 0.01s
int gifmerge(int loops=0, int delay=50, keep keep=Purge)
{
  return merge("-loop " +(string) loops+" -delay "+(string)delay,"gif",
	       keep(keep));
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

// Options for handling label overwriting
private struct OverwriteT {};
public OverwriteT Overwrite=null;
typedef int Overwrite(OverwriteT);
public Overwrite
  Allow=new int(OverwriteT) {return 0;},
  Suppress=new int(OverwriteT) {return 1;},
  SuppressQuiet=new int(OverwriteT) {return 2;},
  Move=new int(OverwriteT) {return 3;},
  MoveQuiet=new int(OverwriteT) {return 4;};
    
pen overwrite(Overwrite Overwrite) 
{
  return overwrite(Overwrite(Overwrite));
}

struct slice {
  public path before,after;
}

slice firstcut(path p, path knife) 
{
  slice s=new slice;
  real t=intersect(p,knife).x;
  if (t < 0) {s.before=p; s.after=nullpath; return s;}
  s.before=subpath(p,0,min(t,intersect(p,reverse(knife)).x));
  s.after=subpath(p,min(t,intersect(p,reverse(knife)).x),length(p));
  return s;
}

slice lastcut(path p, path knife) 
{
  slice s=firstcut(reverse(p),knife);
  path before=reverse(s.after);
  s.after=reverse(s.before);
  s.before=before;
  return s;
}

string format(real x) {
  return format("%.9g",x);
}

pair dir(path g)
{
  return dir(g,length(g));
}

pair dir(path g, path h)
{
  return 0.5*(dir(g)+dir(h));
}

// return the point on path p at arclength L
pair arcpoint(path p, real L)
{
    return point(p,arctime(p,L));
}

// return the point on path p at the given relative fraction of its arclength
pair relpoint(path p, real l)
{
  return arcpoint(p,l*arclength(p));
}

// return the direction on path p at arclength L
pair arcdir(path p, real L)
{
    return dir(p,arctime(p,L));
}

// return the direction of path p at the given relative fraction of its
// arclength
pair reldir(path p, real l)
{
  return arcdir(p,l*arclength(p));
}

// return the initial point of path p
pair beginpoint(path p)
{
    return point(p,0);
}

// return the point on path p at half of its arclength
pair midpoint(path p)
{
    return relpoint(p,0.5);
}

// return the final point of path p
pair endpoint(path p)
{
    return point(p,length(p));
}

pen[] colorPens={red,blue,green,magenta,cyan,orange,purple,brown,darkblue,
		 darkgreen,chartreuse,fuchsia,salmon,lightblue,black,lavender,
		 pink,yellow,gray};
pen[] monoPens={solid,dashed,dotted,longdashed,dashdotted,longdashdotted};

public bool mono=false;
pen Pen(int n) 
{
  return mono ? monoPens[n % monoPens.length] : 
    colorPens[n % colorPens.length];
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

real min(real [][] a) {
  real min=infinity;
  for(int i=0; i < a.length; ++i) min=min(min,min(a[i]));
  return min;
}

real max(real [][] a) {
  real max=-infinity;
  for(int i=0; i < a.length; ++i) max=max(max,max(a[i]));
  return max;
}

real min(real [][][] a) {
  real min=infinity;
  for(int i=0; i < a.length; ++i) min=min(min,min(a[i]));
  return min;
}

real max(real [][][] a) {
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
