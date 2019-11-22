// Pre picture <<<1
import plain_scaling;
import plain_bounds;

include plain_prethree;

// This variable is required by asymptote.sty.
pair viewportsize=0;       // Horizontal and vertical viewport limits.

restricted bool Aspect=true;
restricted bool IgnoreAspect=false;

struct coords3 {
  coord[] x,y,z;
  void erase() {
    x.delete();
    y.delete();
    z.delete();
  }
  // Only a shallow copy of the individual elements of x and y
  // is needed since, once entered, they are never modified.
  coords3 copy() {
    coords3 c=new coords3;
    c.x=copy(x);
    c.y=copy(y);
    c.z=copy(z);
    return c;
  }
  void append(coords3 c) {
    x.append(c.x);
    y.append(c.y);
    z.append(c.z);
  }
  void push(triple user, triple truesize) {
    x.push(coord.build(user.x,truesize.x));
    y.push(coord.build(user.y,truesize.y));
    z.push(coord.build(user.z,truesize.z));
  }
  void push(coord cx, coord cy, coord cz) {
    x.push(cx);
    y.push(cy);
    z.push(cz);
  }
  void push(transform3 t, coords3 c1, coords3 c2, coords3 c3) {
    for(int i=0; i < c1.x.length; ++i) {
      coord cx=c1.x[i], cy=c2.y[i], cz=c3.z[i];
      triple tinf=shiftless(t)*(0,0,0);
      triple z=t*(cx.user,cy.user,cz.user);
      triple w=(cx.truesize,cy.truesize,cz.truesize);
      w=length(w)*unit(shiftless(t)*w);
      coord Cx,Cy,Cz;
      Cx.user=z.x;
      Cy.user=z.y;
      Cz.user=z.z;
      Cx.truesize=w.x;
      Cy.truesize=w.y;
      Cz.truesize=w.z;
      push(Cx,Cy,Cz);
    }
  }
}
  
// scaleT and Legend <<<
typedef real scalefcn(real x);
                                              
struct scaleT {
  scalefcn T,Tinv;
  bool logarithmic;
  bool automin,automax;
  void operator init(scalefcn T, scalefcn Tinv, bool logarithmic=false,
                     bool automin=false, bool automax=false) {
    this.T=T;
    this.Tinv=Tinv;
    this.logarithmic=logarithmic;
    this.automin=automin;
    this.automax=automax;
  }
  scaleT copy() {
    scaleT dest=scaleT(T,Tinv,logarithmic,automin,automax);
    return dest;
  }
};

scaleT operator init()
{
  scaleT S=scaleT(identity,identity);
  return S;
}
                                  
typedef void boundRoutine();

struct autoscaleT {
  scaleT scale;
  scaleT postscale;
  real tickMin=-infinity, tickMax=infinity;
  boundRoutine[] bound; // Optional routines to recompute the bounding box.
  bool automin=false, automax=false;
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
  bool above;
  void operator init(string label, pen plabel=currentpen, pen p=nullpen,
                     frame mark=newframe, bool above=true) {
    this.label=label;
    this.plabel=plabel;
    this.p=(p == nullpen) ? plabel : p;
    this.mark=mark;
    this.above=above;
  }
}

// >>>

// Frame Alignment was here

triple min3(pen p)
{
  return linewidth(p)*(-0.5,-0.5,-0.5);
}

triple max3(pen p)
{
  return linewidth(p)*(0.5,0.5,0.5);
}

// A function that draws an object to frame pic, given that the transform
// from user coordinates to true-size coordinates is t.
typedef void drawer(frame f, transform t);

// A generalization of drawer that includes the final frame's bounds.
// TODO: Add documentation as to what T is.
typedef void drawerBound(frame f, transform t, transform T, pair lb, pair rt);

struct node {
  drawerBound d;
  string key;
  void operator init(drawerBound d, string key=xasyKEY()) {
    this.d=d;
    this.key=key;
  }
}

// PairOrTriple <<<1
// This struct is used to represent a userMin/userMax which serves as both a
// pair and a triple depending on the context.
struct pairOrTriple {
  real x,y,z;
  void init() { x = y = z = 0; }
};
void copyPairOrTriple(pairOrTriple dest, pairOrTriple src)
{
  dest.x = src.x;
  dest.y = src.y;
  dest.z = src.z;
}
pair operator cast (pairOrTriple a) {
  return (a.x, a.y);
};
triple operator cast (pairOrTriple a) {
  return (a.x, a.y, a.z);
}
void write(pairOrTriple a) {
  write((triple) a);
}

struct picture { // <<<1
  // Nodes <<<2
  // Three-dimensional version of drawer and drawerBound:
  typedef void drawer3(frame f, transform3 t, picture pic, projection P);
  typedef void drawerBound3(frame f, transform3 t, transform3 T,
                            picture pic, projection P, triple lb, triple rt);

  struct node3 {
    drawerBound3 d;
    string key;
    void operator init(drawerBound3 d, string key=xasyKEY()) {
      this.d=d;
      this.key=key;
    }
  }

  // The functions to do the deferred drawing.
  node[] nodes;
  node3[] nodes3;

  bool uptodate=true;

  struct bounds3 {
    coords3 point,min,max;
    bool exact=true; // An accurate picture bounds is provided by the user.
    void erase() {
      point.erase();
      min.erase();
      max.erase();
    }
    bounds3 copy() {
      bounds3 b=new bounds3;
      b.point=point.copy();
      b.min=min.copy();
      b.max=max.copy();
      b.exact=exact;
      return b;
    }
  }
  
  bounds bounds;
  bounds3 bounds3;
    
  // Other Fields <<<2
  // Transform to be applied to this picture.
  transform T;
  transform3 T3;
  
  // The internal representation of the 3D user bounds.
  private pairOrTriple umin, umax;
  private bool usetx, usety, usetz;

  ScaleT scale; // Needed by graph
  Legend[] legend;

  pair[] clipmax; // Used by beginclip/endclip
  pair[] clipmin;

  // The maximum sizes in the x, y, and z directions; zero means no restriction.
  real xsize=0, ysize=0;

  real xsize3=0, ysize3=0, zsize3=0;

  // Fixed unitsizes in the x y, and z directions; zero means use
  // xsize, ysize, and zsize.
  real xunitsize=0, yunitsize=0, zunitsize=0;
  
  // If true, the x and y directions must be scaled by the same amount.
  bool keepAspect=true;

  // A fixed scaling transform.
  bool fixed;
  transform fixedscaling;
  
  // Init and erase <<<2
  void init() {
    umin.init();
    umax.init();
    usetx=usety=usetz=false;
    T3=identity(4);
  }
  init();
  
  // Erase the current picture, retaining bounds.
  void clear() {
    nodes.delete();
    nodes3.delete();
    legend.delete();
  }

  // Erase the current picture, retaining any size specification.
  void erase() {
    clear();
    bounds.erase();
    bounds3.erase();
    T=identity();
    scale=new ScaleT;
    init();
  }
  
  // Empty <<<2
  bool empty2() {
    return nodes.length == 0;
  }

  bool empty3() {
    return nodes3.length == 0;
  }

  bool empty() {
    return empty2() && empty3();
  }
  
  // User min/max <<<2
  pair userMin2() {return bounds.userMin(); }
  pair userMax2() {return bounds.userMax(); }

  bool userSetx2() { return bounds.userBoundsAreSet(); }
  bool userSety2() { return bounds.userBoundsAreSet(); }

  triple userMin3() { return umin; }
  triple userMax3() { return umax; }

  bool userSetx3() { return usetx; }
  bool userSety3() { return usety; }
  bool userSetz3() { return usetz; }

  private typedef real binop(real, real);

  // Helper functions for finding the minimum/maximum of two data, one of
  // which may not be defined.
  private static real merge(real x1, bool set1, real x2, bool set2, binop m)
  {
    return set1 ? (set2 ? m(x1,x2) : x1) : x2;
  }
  private pairOrTriple userExtreme(pair u2(), triple u3(), binop m)
  {
    bool setx2 = userSetx2();
    bool sety2 = userSety2();
    bool setx3 = userSetx3();
    bool sety3 = userSety3();

    pair p;
    if (setx2 || sety2)
      p = u2();
    triple t = u3();

    pairOrTriple r;
    r.x = merge(p.x, setx2, t.x, setx3, m);
    r.y = merge(p.y, sety2, t.y, sety3, m);
    r.z = t.z;

    return r;
  }

  // The combination of 2D and 3D data.
  pairOrTriple userMin() {
    return userExtreme(userMin2, userMin3, min);
  }
  pairOrTriple userMax() {
    return userExtreme(userMax2, userMax3, max);
  }

  bool userSetx() { return userSetx2() || userSetx3(); }
  bool userSety() { return userSety2() || userSety3(); }
  bool userSetz() = userSetz3;

  // Functions for setting the user bounds.
  void userMinx3(real x) {
    umin.x=x;
    usetx=true;
  }
  
  void userMiny3(real y) {
    umin.y=y;
    usety=true;
  }
  
  void userMinz3(real z) {
    umin.z=z;
    usetz=true;
  }
  
  void userMaxx3(real x) {
    umax.x=x;
    usetx=true;
  }
  
  void userMaxy3(real y) {
    umax.y=y;
    usety=true;
  }
  
  void userMaxz3(real z) {
    umax.z=z;
    usetz=true;
  }

  void userMinx2(real x) { bounds.alterUserBound("minx", x); }
  void userMinx(real x) { userMinx2(x); userMinx3(x); }
  void userMiny2(real y) { bounds.alterUserBound("miny", y); }
  void userMiny(real y) { userMiny2(y); userMiny3(y); }
  void userMaxx2(real x) { bounds.alterUserBound("maxx", x); }
  void userMaxx(real x) { userMaxx2(x); userMaxx3(x); }
  void userMaxy2(real y) { bounds.alterUserBound("maxy", y); }
  void userMaxy(real y) { userMaxy2(y); userMaxy3(y); }
  void userMinz(real z) = userMinz3;
  void userMaxz(real z) = userMaxz3;
  
  void userCorners3(triple c000, triple c001, triple c010, triple c011,
                    triple c100, triple c101, triple c110, triple c111) {
    umin.x = min(c000.x,c001.x,c010.x,c011.x,c100.x,c101.x,c110.x,c111.x);
    umin.y = min(c000.y,c001.y,c010.y,c011.y,c100.y,c101.y,c110.y,c111.y);
    umin.z = min(c000.z,c001.z,c010.z,c011.z,c100.z,c101.z,c110.z,c111.z);
    umax.x = max(c000.x,c001.x,c010.x,c011.x,c100.x,c101.x,c110.x,c111.x);
    umax.y = max(c000.y,c001.y,c010.y,c011.y,c100.y,c101.y,c110.y,c111.y);
    umax.z = max(c000.z,c001.z,c010.z,c011.z,c100.z,c101.z,c110.z,c111.z);
  }
  
  // Cache the current user-space bounding box x coodinates
  void userBoxX3(real min, real max, binop m=min, binop M=max) {
    if (usetx) {
      umin.x=m(umin.x,min);
      umax.x=M(umax.x,max);
    } else {
      umin.x=min;
      umax.x=max;
      usetx=true;
    }
  }

  // Cache the current user-space bounding box y coodinates
  void userBoxY3(real min, real max, binop m=min, binop M=max) {
    if (usety) {
      umin.y=m(umin.y,min);
      umax.y=M(umax.y,max);
    } else {
      umin.y=min;
      umax.y=max;
      usety=true;
    }
  }

  // Cache the current user-space bounding box z coodinates
  void userBoxZ3(real min, real max, binop m=min, binop M=max) {
    if (usetz) {
      umin.z=m(umin.z,min);
      umax.z=M(umax.z,max);
    } else {
      umin.z=min;
      umax.z=max;
      usetz=true;
    }
  }

  // Cache the current user-space bounding box
  void userBox3(triple min, triple max) {
    userBoxX3(min.x,max.x);
    userBoxY3(min.y,max.y);
    userBoxZ3(min.z,max.z);
  }
  
  // Add drawer <<<2
  void add(drawerBound d, bool exact=false, bool above=true) {
    uptodate=false;
    if(!exact) bounds.exact=false;
    if(above)
      nodes.push(node(d));
    else
      nodes.insert(0,node(d));
  }
  
  // Faster implementation of most common case.
  void addExactAbove(drawerBound d) {
    uptodate=false;
    nodes.push(node(d));
  }

  void add(drawer d, bool exact=false, bool above=true) {
    add(new void(frame f, transform t, transform T, pair, pair) {
        d(f,t*T);
      },exact,above);
  }

  void add(drawerBound3 d, bool exact=false, bool above=true) {
    uptodate=false;
    if(!exact) bounds.exact=false;
    if(above)
      nodes3.push(node3(d));
    else
      nodes3.insert(0,node3(d));
  }

  void add(drawer3 d, bool exact=false, bool above=true) {
    add(new void(frame f, transform3 t, transform3 T, picture pic,
                 projection P, triple, triple) {
          d(f,t*T,pic,P);
        },exact,above);
  }

  // Clip <<<2
  void clip(pair min, pair max, drawer d, bool exact=false) {
    bounds.clip(min, max);
    this.add(d,exact);
  }

  void clip(pair min, pair max, drawerBound d, bool exact=false) {
    bounds.clip(min, max);
    this.add(d,exact);
  }

  // Add sizing <<<2
  // Add a point to the sizing.
  void addPoint(pair user, pair truesize=0) {
    bounds.addPoint(user,truesize);
    //userBox(user,user);
  }
  
  // Add a point to the sizing, accounting also for the size of the pen.
  void addPoint(pair user, pair truesize=0, pen p) {
    addPoint(user,truesize+min(p));
    addPoint(user,truesize+max(p));
  }
  
  void addPoint(triple user, triple truesize=(0,0,0)) {
    bounds3.point.push(user,truesize);
    userBox3(user,user);
  }

  void addPoint(triple user, triple truesize=(0,0,0), pen p) {
    addPoint(user,truesize+min3(p));
    addPoint(user,truesize+max3(p));
  }

  // Add a box to the sizing.
  void addBox(pair userMin, pair userMax, pair trueMin=0, pair trueMax=0) {
    bounds.addBox(userMin, userMax, trueMin, trueMax);
  }

  void addBox(triple userMin, triple userMax, triple trueMin=(0,0,0),
              triple trueMax=(0,0,0)) {
    bounds3.min.push(userMin,trueMin);
    bounds3.max.push(userMax,trueMax);
    userBox3(userMin,userMax);
  }

  // For speed reason, we unravel the addPath routines from bounds.  This
  // avoids an extra function call.
  from bounds unravel addPath;

  // Size commands <<<2
  void size(real x, real y=x, bool keepAspect=this.keepAspect) {
    if(!empty()) uptodate=false;
    xsize=x;
    ysize=y;
    this.keepAspect=keepAspect;
  }

  void size3(real x, real y=x, real z=y, bool keepAspect=this.keepAspect) {
    if(!empty3()) uptodate=false;
    xsize3=x;
    ysize3=y;
    zsize3=z;
    this.keepAspect=keepAspect;
  }

  void unitsize(real x, real y=x, real z=y) {
    uptodate=false;
    xunitsize=x;
    yunitsize=y;
    zunitsize=z;
  }

  // min/max of picture <<<2
  // Calculate the min for the final frame, given the coordinate transform.
  pair min(transform t) {
    return bounds.min(t);
  }

  // Calculate the max for the final frame, given the coordinate transform.
  pair max(transform t) {
    return bounds.max(t);
  }

  // Calculate the min for the final frame, given the coordinate transform.
  triple min(transform3 t) {
    if(bounds3.min.x.length == 0 && bounds3.point.x.length == 0 &&
       bounds3.max.x.length == 0) return (0,0,0);
    triple a=t*(1,1,1)-t*(0,0,0), b=t*(0,0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);
    scaling zs=scaling.build(a.z,b.z);
    return (min(min(min(infinity,xs,bounds3.point.x),xs,bounds3.min.x),
                xs,bounds3.max.x),
            min(min(min(infinity,ys,bounds3.point.y),ys,bounds3.min.y),
                ys,bounds3.max.y),
            min(min(min(infinity,zs,bounds3.point.z),zs,bounds3.min.z),
                zs,bounds3.max.z));
  }

  // Calculate the max for the final frame, given the coordinate transform.
  triple max(transform3 t) {
    if(bounds3.min.x.length == 0 && bounds3.point.x.length == 0 &&
       bounds3.max.x.length == 0) return (0,0,0);
    triple a=t*(1,1,1)-t*(0,0,0), b=t*(0,0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);
    scaling zs=scaling.build(a.z,b.z);
    return (max(max(max(-infinity,xs,bounds3.point.x),xs,bounds3.min.x),
                xs,bounds3.max.x),
            max(max(max(-infinity,ys,bounds3.point.y),ys,bounds3.min.y),
                ys,bounds3.max.y),
            max(max(max(-infinity,zs,bounds3.point.z),zs,bounds3.min.z),
                zs,bounds3.max.z));
  }

  void append(coords3 point, coords3 min, coords3 max, transform3 t,
              bounds3 bounds) 
  {
    // Add the coord info to this picture.
    if(t == identity4) {
      point.append(bounds.point);
      min.append(bounds.min);
      max.append(bounds.max);
    } else {
      point.push(t,bounds.point,bounds.point,bounds.point);
      // Add in all 8 corner points, to properly size cuboid pictures.
      point.push(t,bounds.min,bounds.min,bounds.min);
      point.push(t,bounds.min,bounds.min,bounds.max);
      point.push(t,bounds.min,bounds.max,bounds.min);
      point.push(t,bounds.min,bounds.max,bounds.max);
      point.push(t,bounds.max,bounds.min,bounds.min);
      point.push(t,bounds.max,bounds.min,bounds.max);
      point.push(t,bounds.max,bounds.max,bounds.min);
      point.push(t,bounds.max,bounds.max,bounds.max);
    }
  }
  
  // Scaling and Fit <<<2
  // Returns the transform for turning user-space pairs into true-space pairs.
  transform scaling(real xsize, real ysize, bool keepAspect=true,
                    bool warn=true) {
    bounds b = (T == identity()) ? this.bounds : T * this.bounds;

    return b.scaling(xsize, ysize, xunitsize, yunitsize, keepAspect, warn);
  }

  transform scaling(bool warn=true) {
    return scaling(xsize,ysize,keepAspect,warn);
  }

  // Returns the transform for turning user-space pairs into true-space triples.
  transform3 scaling(real xsize, real ysize, real zsize, bool keepAspect=true,
                     bool warn=true) {
    if(xsize == 0 && xunitsize == 0 && ysize == 0 && yunitsize == 0
       && zsize == 0 && zunitsize == 0)
      return identity(4);

    coords3 Coords;
    
    append(Coords,Coords,Coords,T3,bounds3);
    
    real sx;
    if(xunitsize == 0) {
      if(xsize != 0) sx=calculateScaling("x",Coords.x,xsize,warn);
    } else sx=xunitsize;

    real sy;
    if(yunitsize == 0) {
      if(ysize != 0) sy=calculateScaling("y",Coords.y,ysize,warn);
    } else sy=yunitsize;

    real sz;
    if(zunitsize == 0) {
      if(zsize != 0) sz=calculateScaling("z",Coords.z,zsize,warn);
    } else sz=zunitsize;

    if(sx == 0) {
      sx=max(sy,sz);
      if(sx == 0)
        return identity(4);
    }
    if(sy == 0) sy=max(sz,sx);
    if(sz == 0) sz=max(sx,sy);

    if(keepAspect && (xunitsize == 0 || yunitsize == 0 || zunitsize == 0))
      return scale3(min(sx,sy,sz));
    else
      return scale(sx,sy,sz);
  }

  transform3 scaling3(bool warn=true) {
    return scaling(xsize3,ysize3,zsize3,keepAspect,warn);
  }

  frame fit(transform t, transform T0=T, pair m, pair M) {
    frame f;
    for(node n : nodes) {
      xasyKEY(n.key);
      n.d(f,t,T0,m,M);
    }
    return f;
  }

  frame fit3(transform3 t, transform3 T0=T3, picture pic, projection P,
             triple m, triple M) {
    frame f;
    for(node3 n : nodes3) {
      xasyKEY(n.key);
      n.d(f,t,T0,pic,P,m,M);
    }
    return f;
  }

  // Returns a rigid version of the picture using t to transform user coords
  // into truesize coords.
  frame fit(transform t) {
    return fit(t,min(t),max(t));
  }

  frame fit3(transform3 t, picture pic, projection P) {
    return fit3(t,pic,P,min(t),max(t));
  }

  // Add drawer wrappers <<<2
  void add(void d(picture, transform), bool exact=false) {
    add(new void(frame f, transform t) {
        picture opic=new picture;
        d(opic,t);
        add(f,opic.fit(identity()));
      },exact);
  }

  void add(void d(picture, transform3), bool exact=false, bool above=true) {
    add(new void(frame f, transform3 t, picture pic2, projection P) {
        picture opic=new picture;
        d(opic,t);
        add(f,opic.fit3(identity4,pic2,P));
      },exact,above);
  }

  void add(void d(picture, transform3, transform3, triple, triple),
           bool exact=false, bool above=true) {
    add(new void(frame f, transform3 t, transform3 T, picture pic2,
                 projection P, triple lb, triple rt) {
          picture opic=new picture;
          d(opic,t,T,lb,rt);
          add(f,opic.fit3(identity4,pic2,P));
        },exact,above);
  }

  // More scaling <<<2
  frame scaled() {
    frame f=fit(fixedscaling);
    pair d=size(f);
    static real epsilon=100*realEpsilon;
    if(d.x > xsize*(1+epsilon)) 
      warning("xlimit","frame exceeds xlimit: "+(string) d.x+" > "+
              (string) xsize);
    if(d.y > ysize*(1+epsilon))
      warning("ylimit","frame exceeds ylimit: "+(string) d.y+" > "+
              (string) ysize);
    return f;
  }
  
  // Calculate additional scaling required if only an approximate picture
  // size estimate is available.
  transform scale(frame f, real xsize=this.xsize, real ysize=this.ysize,
                  bool keepaspect=this.keepAspect) {
    if(bounds.exact) return identity();
    pair m=min(f);
    pair M=max(f);
    real width=M.x-m.x;
    real height=M.y-m.y;
    real xgrow=xsize == 0 || width == 0 ? 1 : xsize/width;
    real ygrow=ysize == 0 || height == 0 ? 1 : ysize/height;
    if(keepAspect) {
      real[] grow;
      if(xsize > 0) grow.push(xgrow);
      if(ysize > 0) grow.push(ygrow);
      return scale(grow.length == 0 ? 1 : min(grow));
    } else return scale(xgrow,ygrow);

  }

  // Calculate additional scaling required if only an approximate picture
  // size estimate is available.
  transform3 scale3(frame f, real xsize3=this.xsize3,
                    real ysize3=this.ysize3, real zsize3=this.zsize3,
                    bool keepaspect=this.keepAspect) {
    if(bounds3.exact) return identity(4);
    triple m=min3(f);
    triple M=max3(f);
    real width=M.x-m.x;
    real height=M.y-m.y;
    real depth=M.z-m.z;
    real xgrow=xsize3 == 0 || width == 0 ? 1 : xsize3/width;
    real ygrow=ysize3 == 0 || height == 0 ? 1 : ysize3/height;
    real zgrow=zsize3 == 0 || depth == 0 ? 1 : zsize3/depth;
    if(keepAspect) {
      real[] grow;
      if(xsize3 > 0) grow.push(xgrow);
      if(ysize3 > 0) grow.push(ygrow);
      if(zsize3 > 0) grow.push(zgrow);
      return scale3(grow.length == 0 ? 1 : min(grow));
    } else return scale(xgrow,ygrow,zgrow);
  }

  // calculateTransform with scaling <<<2
  // Return the transform that would be used to fit the picture to a frame
  transform calculateTransform(real xsize, real ysize, bool keepAspect=true,
                               bool warn=true) {
    transform t=scaling(xsize,ysize,keepAspect,warn);
    return scale(fit(t),xsize,ysize,keepAspect)*t;
  }

  transform calculateTransform(bool warn=true) {
    if(fixed) return fixedscaling;
    return calculateTransform(xsize,ysize,keepAspect,warn);
  }

  transform3 calculateTransform3(real xsize=xsize3, real ysize=ysize3,
                                 real zsize=zsize3,
                                 bool keepAspect=true, bool warn=true,
                                 projection P=currentprojection) {
    transform3 t=scaling(xsize,ysize,zsize,keepAspect,warn);
    return scale3(fit3(t,null,P),keepAspect)*t;
  }

  // min/max with xsize and ysize <<<2
  // NOTE: These are probably very slow as implemented.
  pair min(real xsize=this.xsize, real ysize=this.ysize,
           bool keepAspect=this.keepAspect, bool warn=true) {
    return min(calculateTransform(xsize,ysize,keepAspect,warn));
  }
  
  pair max(real xsize=this.xsize, real ysize=this.ysize,
           bool keepAspect=this.keepAspect, bool warn=true) {
    return max(calculateTransform(xsize,ysize,keepAspect,warn));
  }
  
  triple min3(real xsize=this.xsize3, real ysize=this.ysize3,
              real zsize=this.zsize3, bool keepAspect=this.keepAspect,
              bool warn=true, projection P) {
    return min(calculateTransform3(xsize,ysize,zsize,keepAspect,warn,P));
  }
  
  triple max3(real xsize=this.xsize3, real ysize=this.ysize3,
              real zsize=this.zsize3, bool keepAspect=this.keepAspect,
              bool warn=true, projection P) {
    return max(calculateTransform3(xsize,ysize,zsize,keepAspect,warn,P));
  }
  
  // More Fitting <<<2
  // Returns the 2D picture fit to the requested size.
  frame fit2(real xsize=this.xsize, real ysize=this.ysize,
             bool keepAspect=this.keepAspect) {
    if(fixed) return scaled();
    if(empty2())
      return newframe;

    transform t=scaling(xsize,ysize,keepAspect);
    frame f=fit(t);
    transform s=scale(f,xsize,ysize,keepAspect);
    if(s == identity()) return f;
    return fit(s*t);
  }

  static frame fitter(string,picture,string,real,real,bool,bool,string,string,
                      light,projection);
  frame fit(string prefix="", string format="",
            real xsize=this.xsize, real ysize=this.ysize,
            bool keepAspect=this.keepAspect, bool view=false,
            string options="", string script="", light light=currentlight,
            projection P=currentprojection) {
    return fitter == null ? fit2(xsize,ysize,keepAspect) :
      fitter(prefix,this,format,xsize,ysize,keepAspect,view,options,script,
             light,P);
  }
  
  // Fit a 3D picture.
  frame fit3(projection P=currentprojection) {
    if(settings.render == 0) return fit(P);
    if(fixed) return scaled();
    if(empty3()) return newframe;
    transform3 t=scaling(xsize3,ysize3,zsize3,keepAspect);
    frame f=fit3(t,null,P);
    transform3 s=scale3(f,xsize3,ysize3,zsize3,keepAspect);
    if(s == identity4) return f;
    return fit3(s*t,null,P);
  }

  // In case only an approximate picture size estimate is available, return the
  // fitted frame slightly scaled (including labels and true size distances)
  // so that it precisely meets the given size specification. 
  frame scale(real xsize=this.xsize, real ysize=this.ysize,
              bool keepAspect=this.keepAspect) {
    frame f=fit(xsize,ysize,keepAspect);
    transform s=scale(f,xsize,ysize,keepAspect);
    if(s == identity()) return f;
    return s*f;
  }

  // Copying <<<2
  
  // Copies enough information to yield the same userMin/userMax.
  void userCopy2(picture pic) {
    userMinx2(pic.userMin2().x);
    userMiny2(pic.userMin2().y);
    userMaxx2(pic.userMax2().x);
    userMaxy2(pic.userMax2().y);
  }

  void userCopy3(picture pic) {
    copyPairOrTriple(umin, pic.umin);
    copyPairOrTriple(umax, pic.umax);
    usetx=pic.usetx;
    usety=pic.usety;
    usetz=pic.usetz;
  }
  
  void userCopy(picture pic) {
    userCopy2(pic);
    userCopy3(pic);
  }
  
  // Copies the drawing information, but not the sizing information into a new
  // picture. Fitting this picture will not scale as the original picture would.
  picture drawcopy() {
    picture dest=new picture;
    dest.nodes=copy(nodes);
    dest.nodes3=copy(nodes3);
    dest.T=T;
    dest.T3=T3;

    // TODO: User bounds are sizing info, which probably shouldn't be part of
    // a draw copy.  Should we move this down to copy()?
    dest.userCopy3(this);

    dest.scale=scale.copy();
    dest.legend=copy(legend);

    return dest;
  }

  // A deep copy of this picture.  Modifying the copied picture will not affect
  // the original.
  picture copy() {
    picture dest=drawcopy();

    dest.uptodate=uptodate;
    dest.bounds=bounds.copy();
    dest.bounds3=bounds3.copy();
    
    dest.xsize=xsize; dest.ysize=ysize;
    dest.xsize3=xsize; dest.ysize3=ysize3; dest.zsize3=zsize3;
    dest.keepAspect=keepAspect;
    dest.xunitsize=xunitsize; dest.yunitsize=yunitsize;
    dest.zunitsize=zunitsize;
    dest.fixed=fixed; dest.fixedscaling=fixedscaling;
    
    return dest;
  }

  // Helper function for defining transformed pictures.  Do not call it
  // directly.
  picture transformed(transform t) {
    picture dest=drawcopy();

    // Replace nodes with a single drawer that realizes the transform.
    node[] oldnodes = dest.nodes;
    void drawAll(frame f, transform tt, transform T, pair lb, pair rt) {
      transform Tt = T*t;
      for (node n : oldnodes) {
        xasyKEY(n.key);
        n.d(f,tt,Tt,lb,rt);
      }
    }
    dest.nodes = new node[] {node(drawAll)};

    dest.uptodate=uptodate;
    dest.bounds=bounds.transformed(t);
    dest.bounds3=bounds3.copy();
    
    dest.bounds.exact=false;

    dest.xsize=xsize; dest.ysize=ysize;
    dest.xsize3=xsize; dest.ysize3=ysize3; dest.zsize3=zsize3;
    dest.keepAspect=keepAspect;
    dest.xunitsize=xunitsize; dest.yunitsize=yunitsize;
    dest.zunitsize=zunitsize;
    dest.fixed=fixed; dest.fixedscaling=fixedscaling;
    
    return dest;
  }

  // Add Picture <<<2
  // Add a picture to this picture, such that the user coordinates will be
  // scaled identically when fitted
  void add(picture src, bool group=true, filltype filltype=NoFill,
           bool above=true) {
    // Copy the picture.  Only the drawing function closures are needed, so we
    // only copy them.  This needs to be a deep copy, as src could later have
    // objects added to it that should not be included in this picture.

    if(src == this) abort("cannot add picture to itself");
    
    uptodate=false;

    picture srcCopy=src.drawcopy();
    // Draw by drawing the copied picture.
    if(srcCopy.nodes.length > 0) {
      nodes.push(node(new void(frame f, transform t, transform T,
                               pair m, pair M) {
          add(f,srcCopy.fit(t,T*srcCopy.T,m,M),group,filltype,above);
          }));
    }
    
    if(srcCopy.nodes3.length > 0) {
      nodes3.push(node3(new void(frame f, transform3 t, transform3 T3,
                                 picture pic, projection P, triple m, triple M)
                        {
                    add(f,srcCopy.fit3(t,T3*srcCopy.T3,pic,P,m,M),group,above);
                        }));
    }
    
    legend.append(src.legend);
    
    if(src.usetx) userBoxX3(src.umin.x,src.umax.x);
    if(src.usety) userBoxY3(src.umin.y,src.umax.y);
    if(src.usetz) userBoxZ3(src.umin.z,src.umax.z);
    
    bounds.append(srcCopy.T, src.bounds);
    //append(bounds.point,bounds.min,bounds.max,srcCopy.T,src.bounds);
    append(bounds3.point,bounds3.min,bounds3.max,srcCopy.T3,src.bounds3);

    //if(!src.bounds.exact) bounds.exact=false;
    if(!src.bounds3.exact) bounds3.exact=false;
  }
}

// Post Struct <<<1
picture operator * (transform t, picture orig)
{
  return orig.transformed(t);
}

picture operator * (transform3 t, picture orig)
{
  picture pic=orig.copy();
  pic.T3=t*pic.T3;
  triple umin=pic.userMin3(), umax=pic.userMax3();
  pic.userCorners3(t*umin,
                   t*(umin.x,umin.y,umax.z),
                   t*(umin.x,umax.y,umin.z),
                   t*(umin.x,umax.y,umax.z),
                   t*(umax.x,umin.y,umin.z),
                   t*(umax.x,umin.y,umax.z),
                   t*(umax.x,umax.y,umin.z),
                   t*umax);
  pic.bounds3.exact=false;
  return pic;
}

picture currentpicture;

void size(picture pic=currentpicture, real x, real y=x,
          bool keepAspect=pic.keepAspect)
{
  pic.size(x,y,keepAspect);
}

void size(picture pic=currentpicture, transform t)
{
  if(pic.empty3()) {
    pair z=size(pic.fit(t));
    pic.size(z.x,z.y);
  }
}

void size3(picture pic=currentpicture, real x, real y=x, real z=y,
           bool keepAspect=pic.keepAspect)
{
  pic.size3(x,y,z,keepAspect);
}

void unitsize(picture pic=currentpicture, real x, real y=x, real z=y) 
{
  pic.unitsize(x,y,z);
}

void size(picture pic=currentpicture, real xsize, real ysize,
          pair min, pair max)
{
  pair size=max-min;
  pic.unitsize(size.x != 0 ? xsize/size.x : 0,
               size.y != 0 ? ysize/size.y : 0);
}

void size(picture dest, picture src)
{
  dest.size(src.xsize,src.ysize,src.keepAspect);
  dest.size3(src.xsize3,src.ysize3,src.zsize3,src.keepAspect);
  dest.unitsize(src.xunitsize,src.yunitsize,src.zunitsize);
}

pair min(picture pic, bool user=false)
{
  transform t=pic.calculateTransform();
  pair z=pic.min(t);
  return user ? inverse(t)*z : z;
}
  
pair max(picture pic, bool user=false)
{
  transform t=pic.calculateTransform();
  pair z=pic.max(t);
  return user ? inverse(t)*z : z;
}
  
pair size(picture pic, bool user=false)
{
  transform t=pic.calculateTransform();
  pair M=pic.max(t);
  pair m=pic.min(t);
  if(!user) return M-m;
  t=inverse(t);
  return t*M-t*m;
}

// Frame Alignment <<<
pair rectify(pair dir) 
{
  real scale=max(abs(dir.x),abs(dir.y));
  if(scale != 0) dir *= 0.5/scale;
  dir += (0.5,0.5);
  return dir;
}

pair point(frame f, pair dir)
{
  pair m=min(f);
  pair M=max(f);
  return m+realmult(rectify(dir),M-m);
}

path[] align(path[] g, transform t=identity(), pair position,
             pair align, pen p=currentpen)
{
  if(g.length == 0) return g;
  pair m=min(g);
  pair M=max(g);
  pair dir=rectify(inverse(t)*-align);
  if(basealign(p) == 1)
    dir -= (0,m.y/(M.y-m.y));
  pair a=m+realmult(dir,M-m);
  return shift(position+align*labelmargin(p))*t*shift(-a)*g;
}

// Returns a transform for aligning frame f in the direction align
transform shift(frame f, pair align) 
{
  return shift(align-point(f,-align));
}

// Returns a copy of frame f aligned in the direction align
frame align(frame f, pair align) 
{
  return shift(f,align)*f;
}
// >>>

pair point(picture pic=currentpicture, pair dir, bool user=true)
{
  pair umin = pic.userMin2();
  pair umax = pic.userMax2();

  pair z=umin+realmult(rectify(dir),umax-umin);
  return user ? z : pic.calculateTransform()*z;
}

pair truepoint(picture pic=currentpicture, pair dir, bool user=true)
{
  transform t=pic.calculateTransform();
  pair m=pic.min(t);
  pair M=pic.max(t);
  pair z=m+realmult(rectify(dir),M-m);
  return user ? inverse(t)*z : z;
}

// Transform coordinate in [0,1]x[0,1] to current user coordinates.
pair relative(picture pic=currentpicture, pair z)
{
  return pic.userMin2()+realmult(z,pic.userMax2()-pic.userMin2());
}

void add(picture pic=currentpicture, drawer d, bool exact=false)
{
  pic.add(d,exact);
}

typedef void drawer3(frame f, transform3 t, picture pic, projection P);
void add(picture pic=currentpicture, drawer3 d, bool exact=false)
{
  pic.add(d,exact);
}

void add(picture pic=currentpicture, void d(picture,transform),
         bool exact=false)
{
  pic.add(d,exact);
}

void add(picture pic=currentpicture, void d(picture,transform3),
         bool exact=false)
{
  pic.add(d,exact);
}

void begingroup(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      begingroup(f);
    },true);
}

void endgroup(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      endgroup(f);
    },true);
}

void Draw(picture pic=currentpicture, path g, pen p=currentpen)
{
  pic.add(new void(frame f, transform t) {
      draw(f,t*g,p);
    },true);
  pic.addPath(g,p);
}

// Default arguments have been removed to increase speed.
void _draw(picture pic, path g, pen p, margin margin)
{
  if (size(nib(p)) == 0 && margin==NoMargin) {
    // Inline the drawerBound wrapper for speed.
    pic.addExactAbove(new void(frame f, transform t, transform T, pair, pair) {
        _draw(f,t*T*g,p);
      });
  } else {
    pic.add(new void(frame f, transform t) {
        draw(f,margin(t*g,p).g,p);
      },true);
  }
  pic.addPath(g,p);
}

void Draw(picture pic=currentpicture, explicit path[] g, pen p=currentpen)
{
  // Could optimize this by adding one drawer.
  for(int i=0; i < g.length; ++i) Draw(pic,g[i],p);
}

void fill(picture pic=currentpicture, path[] g, pen p=currentpen,
          bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      fill(f,t*g,p,false);
    },true);
  pic.addPath(g);
}

void drawstrokepath(picture pic=currentpicture, path g, pen strokepen,
                    pen p=currentpen)
{
  pic.add(new void(frame f, transform t) {
      draw(f,strokepath(t*g,strokepen),p);
    },true);
  pic.addPath(g,p);
}

void latticeshade(picture pic=currentpicture, path[] g, bool stroke=false,
                  pen fillrule=currentpen, pen[][] p, bool copy=true)
{
  if(copy) {
    g=copy(g);
    p=copy(p);
  }
  pic.add(new void(frame f, transform t) {
      latticeshade(f,t*g,stroke,fillrule,p,t,false);
    },true);
  pic.addPath(g);
}

void axialshade(picture pic=currentpicture, path[] g, bool stroke=false,
                pen pena, pair a, bool extenda=true,
                pen penb, pair b, bool extendb=true, bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      axialshade(f,t*g,stroke,pena,t*a,extenda,penb,t*b,extendb,false);
    },true);
  pic.addPath(g);
}

void radialshade(picture pic=currentpicture, path[] g, bool stroke=false,
                 pen pena, pair a, real ra, bool extenda=true,
                 pen penb, pair b, real rb, bool extendb=true, bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      pair A=t*a, B=t*b;
      real RA=abs(t*(a+ra)-A);
      real RB=abs(t*(b+rb)-B);
      radialshade(f,t*g,stroke,pena,A,RA,extenda,penb,B,RB,extendb,false);
    },true);
  pic.addPath(g);
}

void gouraudshade(picture pic=currentpicture, path[] g, bool stroke=false,
                  pen fillrule=currentpen, pen[] p, pair[] z, int[] edges,
                  bool copy=true)
{
  if(copy) {
    g=copy(g);
    p=copy(p);
    z=copy(z);
    edges=copy(edges);
  }
  pic.add(new void(frame f, transform t) {
      gouraudshade(f,t*g,stroke,fillrule,p,t*z,edges,false);
    },true);
  pic.addPath(g);
}

void gouraudshade(picture pic=currentpicture, path[] g, bool stroke=false,
                  pen fillrule=currentpen, pen[] p, int[] edges, bool copy=true)
{
  if(copy) {
    g=copy(g);
    p=copy(p);
    edges=copy(edges);
  }
  pic.add(new void(frame f, transform t) {
      gouraudshade(f,t*g,stroke,fillrule,p,edges,false);
    },true);
  pic.addPath(g);
}

void tensorshade(picture pic=currentpicture, path[] g, bool stroke=false,
                 pen fillrule=currentpen, pen[][] p, path[] b=g,
                 pair[][] z=new pair[][], bool copy=true)
{
  if(copy) {
    g=copy(g);
    p=copy(p);
    b=copy(b);
    z=copy(z);
  }
  pic.add(new void(frame f, transform t) {
      pair[][] Z=new pair[z.length][];
      for(int i=0; i < z.length; ++i)
        Z[i]=t*z[i];
      tensorshade(f,t*g,stroke,fillrule,p,t*b,Z,false);
    },true);
  pic.addPath(g);
}

void tensorshade(frame f, path[] g, bool stroke=false,
                 pen fillrule=currentpen, pen[] p,
                 path b=g.length > 0 ? g[0] : nullpath)
{
  tensorshade(f,g,stroke,fillrule,new pen[][] {p},b);
}

void tensorshade(frame f, path[] g, bool stroke=false,
                 pen fillrule=currentpen, pen[] p,
                 path b=g.length > 0 ? g[0] : nullpath, pair[] z)
{
  tensorshade(f,g,stroke,fillrule,new pen[][] {p},b,new pair[][] {z});
}

void tensorshade(picture pic=currentpicture, path[] g, bool stroke=false,
                 pen fillrule=currentpen, pen[] p,
                 path b=g.length > 0 ? g[0] : nullpath)
{
  tensorshade(pic,g,stroke,fillrule,new pen[][] {p},b);
}

void tensorshade(picture pic=currentpicture, path[] g, bool stroke=false,
                 pen fillrule=currentpen, pen[] p,
                 path b=g.length > 0 ? g[0] : nullpath, pair[] z)
{
  tensorshade(pic,g,stroke,fillrule,new pen[][] {p},b,new pair[][] {z});
}

// Smoothly shade the regions between consecutive paths of a sequence using a
// given array of pens:
void draw(picture pic=currentpicture, path[] g, pen fillrule=currentpen,
          pen[] p)
{
  path[] G;
  pen[][] P;
  string differentlengths="arrays have different lengths";
  if(g.length != p.length) abort(differentlengths);
  for(int i=0; i < g.length-1; ++i) {
    path g0=g[i];
    path g1=g[i+1];
    if(length(g0) != length(g1)) abort(differentlengths);
    for(int j=0; j < length(g0); ++j) {
      G.push(subpath(g0,j,j+1)--reverse(subpath(g1,j,j+1))--cycle);
      P.push(new pen[] {p[i],p[i],p[i+1],p[i+1]});
    }
  }
  tensorshade(pic,G,fillrule,P);
}

void functionshade(picture pic=currentpicture, path[] g, bool stroke=false,
                   pen fillrule=currentpen, string shader, bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      functionshade(f,t*g,stroke,fillrule,shader);
    },true);
  pic.addPath(g);
}

void filldraw(picture pic=currentpicture, path[] g, pen fillpen=currentpen,
              pen drawpen=currentpen)
{
  begingroup(pic);
  fill(pic,g,fillpen);
  Draw(pic,g,drawpen);
  endgroup(pic);
}

void clip(picture pic=currentpicture, path[] g, bool stroke=false,
          pen fillrule=currentpen, bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.clip(min(g), max(g),
           new void(frame f, transform t) {
             clip(f,t*g,stroke,fillrule,false);
           },
           true);
}

void beginclip(picture pic=currentpicture, path[] g, bool stroke=false,
               pen fillrule=currentpen, bool copy=true) 
{
  if(copy)
    g=copy(g);

  pic.clipmin.push(min(g));
  pic.clipmax.push(max(g));

  pic.add(new void(frame f, transform t) {
      beginclip(f,t*g,stroke,fillrule,false);
    },true);
}

void endclip(picture pic=currentpicture)
{
  pair min,max;
  if (pic.clipmin.length > 0 && pic.clipmax.length > 0)
    {
      min = pic.clipmin.pop();
      max = pic.clipmax.pop();
    }
  else
    {
      // We should probably abort here, since the PostScript output will be
      // garbage.
      warning("endclip", "endclip without beginclip");
      min = pic.userMin2();
      max = pic.userMax2();
    }

  pic.clip(min, max,
           new void(frame f, transform) {
             endclip(f);
           },
           true);
}

void unfill(picture pic=currentpicture, path[] g, bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      unfill(f,t*g,false);
    },true);
}

void filloutside(picture pic=currentpicture, path[] g, pen p=currentpen,
                 bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      filloutside(f,t*g,p,false);
    },true);
  pic.addPath(g);
}

// Use a fixed scaling to map user coordinates in box(min,max) to the 
// desired picture size.
transform fixedscaling(picture pic=currentpicture, pair min, pair max,
                       pen p=nullpen, bool warn=false)
{
  Draw(pic,min,p+invisible);
  Draw(pic,max,p+invisible);
  pic.fixed=true;
  return pic.fixedscaling=pic.calculateTransform(pic.xsize,pic.ysize,
                                                 pic.keepAspect);
}

// Add frame src to frame dest about position with optional grouping.
void add(frame dest, frame src, pair position, bool group=false,
         filltype filltype=NoFill, bool above=true)
{
  add(dest,shift(position)*src,group,filltype,above);
}

// Add frame src to picture dest about position with optional grouping.
void add(picture dest=currentpicture, frame src, pair position=0,
         bool group=true, filltype filltype=NoFill, bool above=true)
{
  if(is3D(src)) {
    dest.add(new void(frame f, transform3, picture, projection) {
        add(f,src); // always add about 3D origin (ignore position)
      },true);
    dest.addBox((0,0,0),(0,0,0),min3(src),max3(src));
  } else {
    dest.add(new void(frame f, transform t) {
        add(f,shift(t*position)*src,group,filltype,above);
      },true);
    dest.addBox(position,position,min(src),max(src));
  }
}

// Like add(picture,frame,pair) but extend picture to accommodate frame.
void attach(picture dest=currentpicture, frame src, pair position=0,
            bool group=true, filltype filltype=NoFill, bool above=true)
{
  transform t=dest.calculateTransform();
  add(dest,src,position,group,filltype,above);
  pair s=size(dest.fit(t));
  size(dest,dest.xsize != 0 ? s.x : 0,dest.ysize != 0 ? s.y : 0);
}

// Like add(picture,frame,pair) but align frame in direction align.
void add(picture dest=currentpicture, frame src, pair position, pair align,
         bool group=true, filltype filltype=NoFill, bool above=true)
{
  add(dest,align(src,align),position,group,filltype,above);
}

// Like add(frame,frame,pair) but align frame in direction align.
void add(frame dest, frame src, pair position, pair align,
         bool group=true, filltype filltype=NoFill, bool above=true)
{
  add(dest,align(src,align),position,group,filltype,above);
}

// Like add(picture,frame,pair,pair) but extend picture to accommodate frame;
void attach(picture dest=currentpicture, frame src, pair position,
            pair align, bool group=true, filltype filltype=NoFill,
            bool above=true)
{
  attach(dest,align(src,align),position,group,filltype,above);
}

// Add a picture to another such that user coordinates in both will be scaled
// identically in the shipout.
void add(picture dest, picture src, bool group=true, filltype filltype=NoFill,
         bool above=true)
{
  dest.add(src,group,filltype,above);
}

void add(picture src, bool group=true, filltype filltype=NoFill,
         bool above=true)
{
  currentpicture.add(src,group,filltype,above);
}

// Fit the picture src using the identity transformation (so user
// coordinates and truesize coordinates agree) and add it about the point
// position to picture dest.
void add(picture dest, picture src, pair position, bool group=true,
         filltype filltype=NoFill, bool above=true)
{
  add(dest,src.fit(identity()),position,group,filltype,above);
}

void add(picture src, pair position, bool group=true, filltype filltype=NoFill,
         bool above=true)
{
  add(currentpicture,src,position,group,filltype,above);
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
    },true);
}

void postscript(picture pic=currentpicture, string s, pair min, pair max)
{
  pic.add(new void(frame f, transform t) {
      postscript(f,s,t*min,t*max);
    },true);
}

void tex(picture pic=currentpicture, string s)
{
  // Force TeX string s to be evaluated immediately (in case it is a macro).
  frame g;
  tex(g,s);
  size(g);
  pic.add(new void(frame f, transform) {
      tex(f,s);
    },true);
}

void tex(picture pic=currentpicture, string s, pair min, pair max)
{
  frame g;
  tex(g,s);
  size(g);
  pic.add(new void(frame f, transform t) {
      tex(f,s,t*min,t*max);
    },true);
}

void layer(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      layer(f);
    },true);
}

void erase(picture pic=currentpicture)
{
  pic.uptodate=false;
  pic.erase();
}

void begin(picture pic=currentpicture, string name, string id="",
           bool visible=true)
{
  if(!latex() || !pdf()) return;
  settings.twice=true;
  if(id == "") id=string(++ocgindex);
  tex(pic,"\begin{ocg}{"+name+"}{"+id+"}{"+(visible ? "1" : "0")+"}");
  layer(pic);
}

void end(picture pic=currentpicture)
{
  if(!latex() || !pdf()) return;
  tex(pic,"\end{ocg}");
  layer(pic);
}

// For users of the LaTeX babel package.
void deactivatequote(picture pic=currentpicture)
{
  tex(pic,"\catcode`\"=12");
}

void activatequote(picture pic=currentpicture)
{
  tex(pic,"\catcode`\"=13");
}
