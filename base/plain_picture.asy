restricted bool Aspect=true;
restricted bool IgnoreAspect=false;

bool shipped;

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
  
// A function that draws an object to frame pic, given that the transform
// from user coordinates to true-size coordinates is t.
typedef void drawer(frame f, transform t);

// A generalization of drawer that includes the final frame's bounds.
typedef void drawerBound(frame f, transform t, transform T, pair lb, pair rt);

// A coordinate in "flex space." A linear combination of user and true-size
// coordinates.
  
struct coord {
  real user,truesize;
  bool finite=true;

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
    truesize=0;
  }
}

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
                                              
struct scaleT {
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
                                  
typedef void boundRoutine();

struct autoscaleT {
  scaleT scale;
  scaleT postscale;
  real tickMin=-infinity, tickMax=infinity;
  boundRoutine[] bound; // Optional routines to recompute the bounding box.
  bool automin=true, automax=true;
  bool automin() {return automin && scale.automin;}
  bool automax() {return automax && scale.automax;}
  
  real T(real x) {return postscale.T(scale.T(x));}
  scalefcn T() {return scale.logarithmic ? postscale.T : T;}
  real Tinv(real x) {return scale.Tinv(postscale.Tinv(x));}
  
  autoscaleT copy() {
    autoscaleT dest=new autoscaleT;
    dest.scale=scale.copy();
    dest.postscale=postscale.copy();
    dest.tickMin=tickMin;
    dest.tickMax=tickMax;
    dest.bound=copy(bound);
    dest.automin=(bool) automin;
    dest.automax=(bool) automax;
    return dest;
  }
}

struct ScaleT {
  bool set;
  autoscaleT x;
  autoscaleT y;
  autoscaleT z;
  
  ScaleT copy() {
    ScaleT dest=new ScaleT;
    dest.set=set;
    dest.x=x.copy();
    dest.y=y.copy();
    dest.z=z.copy();
    return dest;
  }
};

struct Legend {
  string label;
  pen plabel;
  pen p;
  frame mark;
  bool put;
  void init(string label, pen plabel=currentpen, pen p=nullpen,
            frame mark=newframe, bool put=Above) {
    this.label=label;
    this.plabel=plabel;
    this.p=(p == nullpen) ? plabel : p;
    this.mark=mark;
    this.put=put;
  }
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

// Returns a copy of frame f aligned in the direction align
frame align(frame f, pair align) 
{
  return shift(align-point(f,-align))*f;
}

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
  
  bounds bounds;
    
  // Transform to be applied to this picture.
  transform T;
  
  // Cached user-space bounding box
  pair userMin,userMax;
  bool userSetx,userSety;
  
  ScaleT scale; // Needed by graph
  Legend[] legend;

  // The maximum sizes in the x and y directions; zero means no restriction.
  real xsize=0, ysize=0;

  // Fixed unitsizes in the x and y directions; zero means use xsize, ysize.
  real xunitsize=0, yunitsize=0;
  
  // If true, the x and y directions must be scaled by the same amount.
  bool keepAspect=true;

  // A fixed scaling transform.
  bool fixed;
  transform fixedscaling;
  
  void init() {
    userMin=userMax=0;
    userSetx=userSety=false;
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
    return nodes.length == 0;
  }
  
  void userMinx(real x) {
    userMin=(x,userMin.y);
    userSetx=true;
  }
  
  void userMiny(real y) {
    userMin=(userMin.x,y);
    userSety=true;
  }
  
  void userMaxx(real x) {
    userMax=(x,userMax.y);
    userSetx=true;
  }
  
  void userMaxy(real y) {
    userMax=(userMax.x,y);
    userSety=true;
  }
  
  void userCorners(pair c00, pair c01, pair c10, pair c11) {
    userMin=(min(c00.x,c01.x,c10.x,c11.x),min(c00.y,c01.y,c10.y,c11.y));
    userMax=(max(c00.x,c01.x,c10.x,c11.x),max(c00.y,c01.y,c10.y,c11.y));
  }
  
  void userCopy(picture pic) {
    userMin=pic.userMin;
    userMax=pic.userMax;
    userSetx=pic.userSetx;
    userSety=pic.userSety;
  }
  
  typedef real binop(real, real);

  // Cache the current user-space bounding box x coodinates
  void userBoxX(real min, real max, binop m=min, binop M=max) {
    if(userSetx) {
      userMin=(m(userMin.x,min),userMin.y);
      userMax=(M(userMax.x,max),userMax.y);
    } else {
      userMin=(min,userMin.y);
      userMax=(max,userMax.y);
      userSetx=true;
    }
  }
  
  // Cache the current user-space bounding box y coodinates
  void userBoxY(real min, real max, binop m=min, binop M=max) {
    if(userSety) {
      userMin=(userMin.x,m(userMin.y,min));
      userMax=(userMax.x,M(userMax.y,max));
    } else {
      userMin=(userMin.x,min);
      userMax=(userMax.x,max);
      userSety=true;
    }
  }
  
  // Cache the current user-space bounding box
  void userBox(pair min, pair max) {
    userBoxX(min.x,max.x);
    userBoxY(min.y,max.y);
  }
  
  // Clip the current user-space bounding box
  void userClip(pair min, pair max) {
    userBoxX(min.x,max.x,max,min);
    userBoxY(min.y,max.y,max,min);
  }
  
  void add(drawerBound d) {
    uptodate(false);
    nodes.push(d);
  }

  void add(drawer d) {
    uptodate(false);
    nodes.push(new void(frame f, transform t, transform T, pair, pair) {
        d(f,t*T);
      });
  }

  void clip(drawer d) {
    bounds.clip(userMin,userMax);
    this.add(d);
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

  void size(real x, real y=x, bool keepAspect=this.keepAspect) {
    xsize=x;
    ysize=y;
    this.keepAspect=keepAspect;
  }

  void unitsize(real x, real y=x) {
    xunitsize=x;
    yunitsize=y;
  }

  // The scaling in one dimension:  x --> a*x + b
  struct scaling {
    real a,b;
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

  // Calculate the min for the final frame, given the coordinate transform.
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

  // Calculate the max for the final frame, given the coordinate transform.
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
  real calculateScaling(string dir, coord[] coords, real size,
			bool warn=true) {
    access simplex;
    simplex.problem p=new simplex.problem;
   
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
      return scaling.build(p.a(),p.b()).a;
    } else if (status == simplex.problem.UNBOUNDED) {
      if(warn) write("warning: "+dir+" scaling in picture unbounded");
      return 1;
    } else {
      if(!warn) return 1;
      bool userzero=true;
      for(int i=0; i < coords.length; ++i) {
        if(coords[i].user != 0) userzero=false;
        if(!finite(coords[i].user) || !finite(coords[i].truesize))
          abort("unbounded picture");
      }
      if(userzero) return 1;
      write("warning: cannot fit picture to "+dir+"size "+(string) size
            +"...enlarging...");
      return calculateScaling(dir,coords,sqrt(2)*size,warn);
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
  transform scaling(real xsize, real ysize, bool keepAspect=true,
		    bool warn=true) {
    if(xsize == 0 && xunitsize == 0 && ysize == 0 && yunitsize == 0)
      return identity();

    coords2 Coords;
    
    append(Coords,Coords,Coords,T,bounds);
    
    real sx;
    if(xunitsize == 0) {
      if(xsize != 0) sx=calculateScaling("x",Coords.x,xsize,warn);
    } else sx=xunitsize;

    real sy;
    if(yunitsize == 0) {
      if(ysize != 0) sy=calculateScaling("y",Coords.y,ysize,warn);
    } else sy=yunitsize;

    if(sx == 0) sx=sy;
    else if(sy == 0) sy=sx;

    if(keepAspect && (xunitsize == 0 || yunitsize == 0))
      return scale(min(sx,sy));
    else
      return xscale(sx)*yscale(sy);
  }

  transform scaling(bool warn=true) {
    return scaling(xsize,ysize,keepAspect,warn);
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

  frame scaled() {
    frame f=fit(fixedscaling);
    pair d=max(f)-min(f);
    static real epsilon=100*realEpsilon;
    if(d.x > xsize*(1+epsilon)) 
      write("warning: frame exceeds xlimit: "+(string) d.x+" > "+
            (string) xsize);
    if(d.y > ysize*(1+epsilon))
      write("warning: frame exceeds ylimit: "+(string) d.y+" > "+
            (string) ysize);
    return f;
  }
  
  // Calculate additional scaling required if only an approximate picture
  // size estimate is available.
  transform scale(frame f, bool keepaspect=this.keepAspect) {
    pair m=min(f);
    pair M=max(f);
    real width=M.x-m.x;
    real height=M.y-m.y;
    real xgrow=xsize == 0 || width == 0 ? 1 : xsize/width;
    real ygrow=ysize == 0 || height == 0 ? 1 : ysize/height;
    return keepAspect ? 
      scale(min(xsize > 0 ? xgrow : ygrow, ysize > 0 ? ygrow : xgrow)) :
      xscale(xgrow)*yscale(ygrow);
  }

  // Return the transform that would be used to fit the picture to a frame
  transform calculateTransform(real xsize, real ysize, bool keepAspect=true,
			       bool warn=true) {
    transform t=scaling(xsize,ysize,keepAspect,warn);
    return scale(fit(t),keepAspect)*t;
  }

  transform calculateTransform(bool warn=true) {
    return calculateTransform(xsize,ysize,keepAspect,warn);
  }

  pair min(real xsize=this.xsize, real ysize=this.ysize,
           bool keepAspect=this.keepAspect) {
    return min(calculateTransform(xsize,ysize,keepAspect));
  }
  
  pair max(real xsize=this.xsize, real ysize=this.ysize,
           bool keepAspect=this.keepAspect) {
    return max(calculateTransform(xsize,ysize,keepAspect));
  }
  
  // Returns the picture fit to the requested size.
  frame fit(real xsize=this.xsize, real ysize=this.ysize,
            bool keepAspect=this.keepAspect) {
    if(fixed) return scaled();
    if(empty()) return newframe;
    transform t=scaling(xsize,ysize,keepAspect);
    frame f=fit(t);
    transform s=scale(f,keepAspect);
    if(s == identity()) return f;
    return fit(s*t);
  }

  // In case only an approximate picture size estimate is available, return the
  // fitted frame slightly scaled (including labels and true size distances)
  // so that it precisely meets the given size specification. 
  frame scale(real xsize=this.xsize, real ysize=this.ysize,
              bool keepAspect=this.keepAspect) {
    frame f=fit(xsize,ysize,keepAspect);
    return scale(f,keepAspect)*f;
  }

  // Copies the drawing information, but not the sizing information into a new
  // picture. Fitting this picture will not scale as the original picture would.
  picture drawcopy() {
    picture dest=new picture;
    dest.nodes=copy(nodes);
    dest.T=T;
    dest.userCopy(this);
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
    dest.xunitsize=xunitsize; dest.yunitsize=yunitsize;
    dest.fixed=fixed; dest.fixedscaling=fixedscaling;
    
    return dest;
  }

  // Add a picture to this picture, such that the user coordinates will be
  // scaled identically when fitted
  void add(picture src, bool group=true, filltype filltype=NoFill,
           bool put=Above) {
    // Copy the picture.  Only the drawing function closures are needed, so we
    // only copy them.  This needs to be a deep copy, as src could later have
    // objects added to it that should not be included in this picture.

    if(src == this) abort("cannot add picture to itself");
    
    picture srcCopy=src.drawcopy();
    // Draw by drawing the copied picture.
    nodes.push(new void(frame f, transform t, transform T, pair m, pair M) {
        frame d=srcCopy.fit(t,T*srcCopy.T,m,M);
        add(f,d,put,filltype,group);
      });
    
    legend.append(src.legend);
    
    if(src.userSetx) userBoxX(src.userMin.x,src.userMax.x);
    if(src.userSety) userBoxY(src.userMin.y,src.userMax.y);
    
    append(bounds.point,bounds.min,bounds.max,srcCopy.T,src.bounds);
  }
}

picture operator * (transform t, picture orig)
{
  picture pic=orig.copy();
  pic.T=t*pic.T;
  pic.userCorners(t*pic.userMin,
                  t*(pic.userMin.x,pic.userMax.y),
                  t*(pic.userMax.x,pic.userMin.y),
                  t*pic.userMax);
  return pic;
}

picture currentpicture;

void size(picture pic=currentpicture, real x, real y=x, 
          bool keepAspect=pic.keepAspect)
{
  pic.size(x,y,keepAspect);
}

void unitsize(picture pic=currentpicture, real x, real y=x) 
{
  pic.unitsize(x,y);
}

void size(picture dest, picture src)
{
  dest.size(src.xsize,src.ysize,src.keepAspect);
  dest.unitsize(src.xunitsize,src.yunitsize);
}

void size(picture src)
{
  size(currentpicture,src);
}

pair size(frame f)
{
  return max(f)-min(f);
}
                                     
pair min(picture pic)
{
  return pic.min();
}
  
pair max(picture pic)
{
  return pic.max();
}
  
void add(picture pic=currentpicture, drawer d)
{
  pic.add(d);
}

void begingroup(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      begingroup(f);
    });
}

void endgroup(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      endgroup(f);
    });
}

void Draw(picture pic=currentpicture, path g, pen p=currentpen)
{
  pic.add(new void(frame f, transform t) {
      draw(f,t*g,p);
    });
  pic.addPath(g,p);
}

void _draw(picture pic=currentpicture, path g, pen p=currentpen,
           margin margin=NoMargin)
{
  pic.add(new void(frame f, transform t) {
      draw(f,margin(t*g,p).g,p);
    });
  pic.addPath(g,p);
}

void Draw(picture pic=currentpicture, explicit path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) Draw(pic,g[i],p);
}

void fill(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  g=copy(g);
  pic.add(new void(frame f, transform t) {
      fill(f,t*g,p);
    });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void latticeshade(picture pic=currentpicture, path[] g,
                  pen fillrule=currentpen, pen[][] p)
{
  g=copy(g);
  p=copy(p);
  pic.add(new void(frame f, transform t) {
      latticeshade(f,t*g,fillrule,p);
    });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void axialshade(picture pic=currentpicture, path[] g, pen pena, pair a,
                pen penb, pair b)
{
  g=copy(g);
  pic.add(new void(frame f, transform t) {
      axialshade(f,t*g,pena,t*a,penb,t*b);
    });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void radialshade(picture pic=currentpicture, path[] g, pen pena, pair a,
                 real ra, pen penb, pair b, real rb)
{
  g=copy(g);
  pic.add(new void(frame f, transform t) {
      pair A=t*a, B=t*b;
      real RA=abs(t*(a+ra)-A);
      real RB=abs(t*(b+rb)-B);
      radialshade(f,t*g,pena,A,RA,penb,B,RB);
    });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void gouraudshade(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
		  pen[] p, pair[] z, int[] edges)
{
  g=copy(g);
  p=copy(p);
  z=copy(z);
  edges=copy(edges);
  pic.add(new void(frame f, transform t) {
      gouraudshade(f,t*g,fillrule,p,t*z,edges);
    });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void tensorshade(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
		 pen[][] p, path[] b=g, pair[][] z=new pair[][])
{
  g=copy(g);
  p=copy(p);
  b=copy(b);
  z=copy(z);
  pic.add(new void(frame f, transform t) {
      pair[][] Z=new pair[z.length][0];
      for(int i=0; i < z.length; ++i)
	Z[i]=t*z[i];
      tensorshade(f,t*g,fillrule,p,t*b,Z);
    });
  for(int i=0; i < g.length; ++i) 
    pic.addPath(g[i]);
}

void tensorshade(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
		 pen[] p, path b=g.length > 0 ? g[0] : nullpath)
{
  tensorshade(pic,g,fillrule,new pen[][] {p},b);
}

void tensorshade(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
		 pen[] p, path b=g.length > 0 ? g[0] : nullpath, pair[] z)
{
  tensorshade(pic,g,fillrule,new pen[][] {p},b,new pair[][] {z});
}

void filldraw(picture pic=currentpicture, path[] g, pen fillpen=currentpen,
              pen drawpen=currentpen)
{
  begingroup(pic);
  fill(pic,g,fillpen);
  Draw(pic,g,drawpen);
  endgroup(pic);
}

void clip(frame f, path[] g)
{
  clip(f,g,currentpen);
}

void clip(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  g=copy(g);
  pic.userClip(min(g),max(g));
  pic.clip(new void(frame f, transform t) {
      clip(f,t*g,p);
    });
}

void unfill(picture pic=currentpicture, path[] g)
{
  g=copy(g);
  pic.add(new void(frame f, transform t) {
      unfill(f,t*g);
    });
}

void filloutside(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  g=copy(g);
  pic.add(new void(frame f, transform t) {
      filloutside(f,t*g,p);
    });
}

bool inside(path[] g, pair z) 
{
  return inside(g,z,currentpen);
}

// Use a fixed scaling to map user coordinates in box(min,max) to the 
// desired picture size.
transform fixedscaling(picture pic=currentpicture, pair min, pair max,
                       pen p=nullpen)
{
  Draw(pic,min,p+invisible);
  Draw(pic,max,p+invisible);
  pic.fixed=true;
  return pic.fixedscaling=pic.calculateTransform();
}

// Add frame dest about position to frame src with optional grouping
void add(frame dest, frame src, pair position, bool group=false,
         filltype filltype=NoFill, bool put=Above)
{
  add(dest,shift(position)*src,group,filltype,put);
}

// Add frame src about position to picture dest with optional grouping
void add(picture dest=currentpicture, frame src, pair position=0,
         bool group=true, filltype filltype=NoFill, bool put=Above)
{
  dest.add(new void(frame f, transform t) {
      add(f,shift(t*position)*src,group,filltype,put);
    });
  dest.addBox(position,position,min(src),max(src));
}

// Like add(pair,picture,frame) but extend picture to accommodate frame
void attach(picture dest=currentpicture, frame src, pair position=0,
            bool group=true, filltype filltype=NoFill, bool put=Above)
{
  transform t=dest.calculateTransform();
  add(dest,src,position,group,filltype,put);
  pair s=size(dest.fit(t));
  size(dest,dest.xsize != 0 ? s.x : 0,dest.ysize != 0 ? s.y : 0);
}

// Like add(picture,frame,pair) but align frame in direction align.
void add(picture dest=currentpicture, frame src, pair position, pair align,
         bool group=true, filltype filltype=NoFill, bool put=Above)
{
  add(dest,align(src,align),position,group,filltype,put);
}

// Like attach(picture,frame,pair) but align frame in direction align.
void attach(picture dest=currentpicture, frame src, pair position, pair align,
            bool group=true, filltype filltype=NoFill, bool put=Above)
{
  attach(dest,align(src,align),position,group,filltype,put);
}

// Add a picture to another such that user coordinates in both will be scaled
// identically in the shipout.
void add(picture dest, picture src, bool group=true, filltype filltype=NoFill,
         bool put=Above)
{
  dest.add(src,group,filltype,put);
}

void add(picture src, bool group=true, filltype filltype=NoFill,
         bool put=Above)
{
  add(currentpicture,src,group,filltype,put);
}

// Fit the picture src using the identity transformation (so user
// coordinates and truesize coordinates agree) and add it about the point
// position to picture dest.
void add(picture dest, picture src, pair position, bool group=true,
         filltype filltype=NoFill, bool put=Above)
{
  add(dest,src.fit(identity()),position,group,filltype,put);
}

void add(picture src, pair position, bool group=true, filltype filltype=NoFill,
         bool put=Above)
{
  add(currentpicture,src,position,group,filltype,put);
}

// Fill a region about the user-coordinate 'origin'.
void fill(pair origin, picture pic=currentpicture, path[] g, pen p=currentpen)
{
  picture opic;
  fill(opic,g,p);
  add(pic,opic,origin);
}

void postscript(picture pic=currentpicture, string s)
{
  pic.add(new void(frame f, transform) {
      postscript(f,s);
    });
}

void tex(picture pic=currentpicture, string s)
{
  pic.add(new void(frame f, transform) {
      tex(f,s);
    });
}

void postscript(picture pic=currentpicture, string s, pair min, pair max)
{
  pic.add(new void(frame f, transform t) {
      postscript(f,s,t*min,t*max);
    });
}

void tex(picture pic=currentpicture, string s, pair min, pair max)
{
  pic.add(new void(frame f, transform t) {
      tex(f,s,t*min,t*max);
    });
}

void layer(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      layer(f);
    });
}

pair point(picture pic=currentpicture, pair dir)
{
  return pic.userMin+realmult(rectify(dir),pic.userMax-pic.userMin);
}

pair framepoint(picture pic=currentpicture, pair dir,
		transform t=pic.calculateTransform())
{
  pair m=pic.min(t);
  pair M=pic.max(t);
  return m+realmult(rectify(dir),M-m);
}

pair truepoint(picture pic=currentpicture, pair dir)
{
  transform t=pic.calculateTransform();
  return inverse(t)*framepoint(pic,dir,t);
}

// Transform coordinate in [0,1]x[0,1] to current user coordinates.
pair relative(picture pic=currentpicture, pair z)
{
  pair w=pic.userMax-pic.userMin;
  return pic.userMin+(z.x*w.x,z.y*w.y);
}

void erase(picture pic=currentpicture)
{
  uptodate(false);
  pic.erase();
}
