private import math;

if(inXasyMode) settings.render=0;

if(prc0()) {
  if(settings.tex == "context") settings.prc=false;
  else {
    access embed;
    Embed=embed.embed;
    Link=embed.link;
  }
}

real defaultshininess=0.25;
real defaultgranularity=0;
real linegranularity=0.01;
real tubegranularity=0.003;
real dotgranularity=0.0001;
real viewportfactor=1.002;   // Factor used to expand orthographic viewport.
real angleprecision=1e-3;    // Precision for centering perspective projections.
real anglefactor=max(1.01,1+angleprecision);
// Factor used to expand perspective viewport.

string defaultembed3Doptions;
string defaultembed3Dscript;

triple O=(0,0,0);
triple X=(1,0,0), Y=(0,1,0), Z=(0,0,1);

// A translation in 3D space.
transform3 shift(triple v)
{
  transform3 t=identity(4);
  t[0][3]=v.x;
  t[1][3]=v.y;
  t[2][3]=v.z;
  return t;
}

// Avoid two parentheses.
transform3 shift(real x, real y, real z)
{
  return shift((x,y,z));
}

transform3 shift(transform3 t)
{
  transform3 T=identity(4);
  T[0][3]=t[0][3];
  T[1][3]=t[1][3];
  T[2][3]=t[2][3];
  return T;
}

// A 3D scaling in the x direction.
transform3 xscale3(real x)
{
  transform3 t=identity(4);
  t[0][0]=x;
  return t;
}

// A 3D scaling in the y direction.
transform3 yscale3(real y)
{
  transform3 t=identity(4);
  t[1][1]=y;
  return t;
}

// A 3D scaling in the z direction.
transform3 zscale3(real z)
{
  transform3 t=identity(4);
  t[2][2]=z;
  return t;
}

// A 3D scaling by s in the v direction.
transform3 scale(triple v, real s)
{
  v=unit(v);
  s -= 1;
  return new real[][] {
    {1+s*v.x^2, s*v.x*v.y, s*v.x*v.z, 0}, 
      {s*v.x*v.y, 1+s*v.y^2, s*v.y*v.z, 0}, 
        {s*v.x*v.z, s*v.y*v.z, 1+s*v.z^2, 0},
          {0, 0, 0, 1}};
}

// A transformation representing rotation by an angle in degrees about
// an axis v through the origin (in the right-handed direction).
transform3 rotate(real angle, triple v)
{
  if(v == O) abort("cannot rotate about the zero vector");
  v=unit(v);
  real x=v.x, y=v.y, z=v.z;
  real s=Sin(angle), c=Cos(angle), t=1-c;

  return new real[][] {
    {t*x^2+c,   t*x*y-s*z, t*x*z+s*y, 0},
      {t*x*y+s*z, t*y^2+c,   t*y*z-s*x, 0},
        {t*x*z-s*y, t*y*z+s*x, t*z^2+c,   0},
          {0,         0,         0,         1}};
}

// A transformation representing rotation by an angle in degrees about
// the line u--v (in the right-handed direction).
transform3 rotate(real angle, triple u, triple v)
{
  return shift(u)*rotate(angle,v-u)*shift(-u);
}

// Reflects about the plane through u, v, and w.
transform3 reflect(triple u, triple v, triple w)
{
  triple n=unit(cross(v-u,w-u));
  if(n == O)
    abort("points determining reflection plane cannot be colinear");

  return new real[][] {
    {1-2*n.x^2, -2*n.x*n.y, -2*n.x*n.z, u.x},
      {-2*n.x*n.y, 1-2*n.y^2, -2*n.y*n.z, u.y},
        {-2*n.x*n.z, -2*n.y*n.z, 1-2*n.z^2, u.z},
          {0, 0, 0, 1}
  }*shift(-u);
}

bool operator != (real[][] a, real[][] b) {
  return !(a == b);
}

// Project u onto v.
triple project(triple u, triple v)
{
  v=unit(v);
  return dot(u,v)*v;
}

// Return a unit vector perpendicular to a given unit vector v.
triple perp(triple v)
{
  triple u=cross(v,Y);
  return (abs(u) > sqrtEpsilon) ? unit(u) : unit(cross(v,Z));
}

// Return the transformation corresponding to moving the camera from the target
// (looking in the negative z direction) to the point 'eye' (looking at target),
// orienting the camera so that direction 'up' points upwards.
// Since, in actuality, we are transforming the points instead of the camera,
// we calculate the inverse matrix.
// Based on the gluLookAt implementation in the OpenGL manual.
transform3 look(triple eye, triple up=Z, triple target=O)
{
  triple f=unit(target-eye);
  if(f == O)
    f=-Z; // The eye is already at the origin: look down.

  triple s=cross(f,up);

  // If the eye is pointing either directly up or down, there is no
  // preferred "up" direction.  Pick one arbitrarily.
  s=s != O ? unit(s) : perp(f);

  triple u=cross(s,f);

  transform3 M={{ s.x,  s.y,  s.z, 0},
                { u.x,  u.y,  u.z, 0},
                {-f.x, -f.y, -f.z, 0},
                {   0,    0,    0, 1}};

  return M*shift(-eye);
}

// Return a matrix to do perspective distortion based on a triple v.
transform3 distort(triple v) 
{
  transform3 t=identity(4);
  real d=length(v);
  if(d == 0) return t;
  t[3][2]=-1/d;
  t[3][3]=0;
  return t;
}

projection operator * (transform3 t, projection P)
{
  projection P=P.copy();
  if(!P.absolute) {
    P.camera=t*P.camera;
    P.target=t*P.target;
    P.calculate();
  }
  return P;
}

// With this, save() and restore() in plain also save and restore the
// currentprojection.
addSaveFunction(new restoreThunk() {
    projection P=currentprojection.copy();
    return new void() {
      currentprojection=P;
    };
  });

pair project(triple v, projection P=currentprojection)
{
  return project(v,P.t);
}

pair dir(triple v, triple dir, projection P)
{
  return unit(project(v+0.5dir,P)-project(v-0.5*dir,P));
}

// Uses the homogenous coordinate to perform perspective distortion.
// When combined with a projection to the XY plane, this effectively maps
// points in three space to a plane through target and
// perpendicular to the vector camera-target.
projection perspective(triple camera, triple up=Z, triple target=O,
                       real zoom=1, real angle=0, pair viewportshift=0,
                       bool showtarget=true, bool autoadjust=true,
                       bool center=autoadjust)
{
  if(camera == target)
    abort("camera cannot be at target");
  return projection(camera,up,target,zoom,angle,viewportshift,
                    showtarget,autoadjust,center,
                    new transformation(triple camera, triple up, triple target)
                    {return transformation(look(camera,up,target),
                                           distort(camera-target));});
}

projection perspective(real x, real y, real z, triple up=Z, triple target=O,
                       real zoom=1, real angle=0, pair viewportshift=0,
                       bool showtarget=true, bool autoadjust=true,
                       bool center=autoadjust)
{
  return perspective((x,y,z),up,target,zoom,angle,viewportshift,showtarget,
                     autoadjust,center);
}

projection orthographic(triple camera, triple up=Z, triple target=O,
                        real zoom=1, bool showtarget=true, bool center=false)
{
  return projection(camera,up,target,zoom,showtarget,
                    center=center,
                    new transformation(triple camera, triple up,
                                       triple target) {
                      return transformation(look(camera,up,target));});
}

projection orthographic(real x, real y, real z, triple up=Z,
                        triple target=O, real zoom=1,
                        bool showtarget=true, bool center=false)
{
  return orthographic((x,y,z),up,target,zoom,showtarget,center=center);
}

projection oblique(real angle=45)
{
  transform3 t=identity(4);
  real c2=Cos(angle)^2;
  real s2=1-c2;
  t[0][2]=-c2;
  t[1][2]=-s2;
  t[2][2]=1;
  return projection((c2,s2,1),up=Y,
                    new transformation(triple,triple,triple) {
                      return transformation(t,oblique=true);});
}

projection obliqueZ(real angle=45) {return oblique(angle);}

projection obliqueX(real angle=45)
{
  transform3 t=identity(4);
  real c2=Cos(angle)^2;
  real s2=1-c2;
  t[0][0]=-c2;
  t[1][0]=-s2;
  t[1][1]=0;
  t[0][1]=1;
  t[1][2]=1;
  t[2][2]=0;
  t[2][0]=1;
  return projection((1,c2,s2),
                    new transformation(triple,triple,triple) {
                      return transformation(t,oblique=true);});
}

projection obliqueY(real angle=45)
{
  transform3 t=identity(4);
  real c2=Cos(angle)^2;
  real s2=1-c2;
  t[0][1]=c2;
  t[1][1]=s2;
  t[1][2]=1;
  t[2][1]=-1;
  t[2][2]=0;
  return projection((c2,-1,s2),
                    new transformation(triple,triple,triple) {
                      return transformation(t,oblique=true);});
}

projection oblique=oblique();
projection obliqueX=obliqueX(), obliqueY=obliqueY(), obliqueZ=obliqueZ();

projection LeftView=orthographic(-X,showtarget=true);
projection RightView=orthographic(X,showtarget=true);
projection FrontView=orthographic(-Y,showtarget=true);
projection BackView=orthographic(Y,showtarget=true);
projection BottomView=orthographic(-Z,showtarget=true);
projection TopView=orthographic(Z,showtarget=true);

currentprojection=perspective(5,4,2);

// Map pair z to a triple by inverting the projection P onto the
// plane perpendicular to normal and passing through point.
triple invert(pair z, triple normal, triple point,
              projection P=currentprojection)
{
  transform3 t=P.t;
  real[][] A={{t[0][0]-z.x*t[3][0],t[0][1]-z.x*t[3][1],t[0][2]-z.x*t[3][2]},
              {t[1][0]-z.y*t[3][0],t[1][1]-z.y*t[3][1],t[1][2]-z.y*t[3][2]},
              {normal.x,normal.y,normal.z}};
  real[] b={z.x*t[3][3]-t[0][3],z.y*t[3][3]-t[1][3],dot(normal,point)};
  real[] x=solve(A,b,warn=false);
  return x.length > 0 ? (x[0],x[1],x[2]) : P.camera;
}

// Map pair to a triple on the projection plane.
triple invert(pair z, projection P=currentprojection)
{
  return invert(z,P.vector(),P.target,P);
}

// Map pair dir to a triple direction at point v on the projection plane.
triple invert(pair dir, triple v, projection P=currentprojection)
{
  return invert(project(v,P)+dir,P.vector(),v,P)-v;
}

pair xypart(triple v)
{
  return (v.x,v.y);
}

struct control {
  triple post,pre;
  bool active=false;
  bool straight=true;
  void operator init(triple post, triple pre, bool straight=false) {
    this.post=post;
    this.pre=pre;
    active=true;
    this.straight=straight;
  }
}

control nocontrol;
  
control operator * (transform3 t, control c) 
{
  control C;
  C.post=t*c.post;
  C.pre=t*c.pre;
  C.active=c.active;
  C.straight=c.straight;
  return C;
}

void write(file file, control c)
{
  write(file,".. controls ");
  write(file,c.post);
  write(file," and ");
  write(file,c.pre);
}
  
struct Tension {
  real out,in;
  bool atLeast;
  bool active;
  void operator init(real out=1, real in=1, bool atLeast=false,
                     bool active=true) {
    real check(real val) {
      if(val < 0.75) abort("tension cannot be less than 3/4");
      return val;
    }
    this.out=check(out);
    this.in=check(in);
    this.atLeast=atLeast;
    this.active=active;
  }
}

Tension operator init()
{
  return Tension();
}

Tension noTension;
noTension.active=false;
  
void write(file file, Tension t)
{
  write(file,"..tension ");
  if(t.atLeast) write(file,"atleast ");
  write(file,t.out);
  write(file," and ");
  write(file,t.in);
}
  
struct dir {
  triple dir;
  real gamma=1; // endpoint curl
  bool Curl;    // curl specified
  bool active() {
    return dir != O || Curl;
  }
  void init(triple v) {
    this.dir=v;
  }
  void init(real gamma) {
    if(gamma < 0) abort("curl cannot be less than 0");
    this.gamma=gamma;
    this.Curl=true;
  }
  void init(dir d) {
    dir=d.dir;
    gamma=d.gamma;
    Curl=d.Curl;
  }
  void default(triple v) {
    if(!active()) init(v);
  }
  void default(dir d) {
    if(!active()) init(d);
  }
  dir copy() {
    dir d=new dir;
    d.init(this);
    return d;
  }
}

void write(file file, dir d)
{
  if(d.dir != O) {
    write(file,"{"); write(file,unit(d.dir)); write(file,"}");
  } else if(d.Curl) {
    write(file,"{curl "); write(file,d.gamma); write(file,"}");
  }
}
  
dir operator * (transform3 t, dir d) 
{
  dir D=d.copy();
  D.init(unit(shiftless(t)*d.dir));
  return D;
}

void checkEmpty(int n) {
  if(n == 0)
    abort("nullpath3 has no points");
}

int adjustedIndex(int i, int n, bool cycles)
{
  checkEmpty(n);
  if(cycles)
    return i % n;
  else if(i < 0)
    return 0;
  else if(i >= n)
    return n-1;
  else
    return i;
}

struct flatguide3 {
  triple[] nodes;
  bool[] cyclic;     // true if node is really a cycle
  control[] control; // control points for segment starting at node
  Tension[] Tension; // Tension parameters for segment starting at node
  dir[] in,out;      // in and out directions for segment starting at node

  bool cyclic() {int n=cyclic.length; return n > 0 ? cyclic[n-1] : false;}
  bool precyclic() {int i=find(cyclic); return i >= 0 && i < cyclic.length-1;}
  
  int size() {
    return cyclic() ? nodes.length-1 : nodes.length;
  }
  
  void node(triple v, bool b=false) {
    nodes.push(v);
    control.push(nocontrol);
    Tension.push(noTension);
    in.push(new dir);
    out.push(new dir);
    cyclic.push(b);
  }

  void control(triple post, triple pre) {
    if(control.length > 0) {
      control c=control(post,pre,false);
      control[control.length-1]=c;
    }
  }

  void Tension(real out, real in, bool atLeast) {
    if(Tension.length > 0)
      Tension[Tension.length-1]=Tension(out,in,atLeast,true);
  }

  void in(triple v) {
    if(in.length > 0) {
      in[in.length-1].init(v);
    }
  }

  void out(triple v) {
    if(out.length > 0) {
      out[out.length-1].init(v);
    }
  }

  void in(real gamma) {
    if(in.length > 0) {
      in[in.length-1].init(gamma);
    }
  }

  void out(real gamma) {
    if(out.length > 0) {
      out[out.length-1].init(gamma);
    }
  }

  void cycleToken() {
    if(nodes.length > 0)
      node(nodes[0],true);
  }
  
  // Return true if outgoing direction at node i is known.
  bool solved(int i) {
    return out[i].active() || control[i].active;
  }
}

void write(file file, string s="", explicit flatguide3 x, suffix suffix=none)
{
  write(file,s);
  if(x.size() == 0) write(file,"<nullpath3>");
  else for(int i=0; i < x.nodes.length; ++i) {
      if(i > 0) write(file,endl);
      if(x.cyclic[i]) write(file,"cycle");
      else write(file,x.nodes[i]);
      if(i < x.nodes.length-1) {
        // Explicit control points trump other specifiers
        if(x.control[i].active)
          write(file,x.control[i]);
        else {
          write(file,x.out[i]);
          if(x.Tension[i].active) write(file,x.Tension[i]);
        }
        write(file,"..");
        if(!x.control[i].active) write(file,x.in[i]);
      }
    }
  write(file,suffix);
}

void write(string s="", flatguide3 x, suffix suffix=endl)
{
  write(stdout,s,x,suffix);
}

// A guide3 is most easily represented as something that modifies a flatguide3.
typedef void guide3(flatguide3);

restricted void nullpath3(flatguide3) {};

guide3 operator init() {return nullpath3;}

guide3 operator cast(triple v)
{
  return new void(flatguide3 f) {
    f.node(v);
  };
}

guide3 operator cast(cycleToken) {
  return new void(flatguide3 f) {
    f.cycleToken();
  };
}

guide3 operator controls(triple post, triple pre) 
{
  return new void(flatguide3 f) {
    f.control(post,pre);
  };
};
  
guide3 operator controls(triple v)
{
  return operator controls(v,v);
}

guide3 operator cast(tensionSpecifier t)
{
  return new void(flatguide3 f) {
    f.Tension(t.out, t.in, t.atLeast);
  };
}

guide3 operator cast(curlSpecifier spec)
{
  return new void(flatguide3 f) {
    if(spec.side == JOIN_OUT) f.out(spec.value);
    else if(spec.side == JOIN_IN) f.in(spec.value);
    else
      abort("invalid curl specifier");
  };
}

guide3 operator spec(triple v, int side)
{
  return new void(flatguide3 f) {
    if(side == JOIN_OUT) f.out(v);
    else if(side == JOIN_IN) f.in(v);
    else
      abort("invalid direction specifier");
  };
}
  
guide3 operator -- (... guide3[] g)
{
  return new void(flatguide3 f) {
    if(g.length > 0) {
      for(int i=0; i < g.length-1; ++i) {
        g[i](f);
        f.out(1);
        f.in(1);
      }
      g[g.length-1](f);
    }
  };
}

guide3 operator .. (... guide3[] g)
{
  return new void(flatguide3 f) {
    for(int i=0; i < g.length; ++i)
      g[i](f);
  };
}

guide3 operator ::(... guide3[] a)
{
  if(a.length == 0) return nullpath3;
  guide3 g=a[0];
  for(int i=1; i < a.length; ++i)
    g=g.. tension atleast 1 ..a[i];
  return g;
}

guide3 operator ---(... guide3[] a)
{
  if(a.length == 0) return nullpath3;
  guide3 g=a[0];
  for(int i=1; i < a.length; ++i)
    g=g.. tension atleast infinity ..a[i];
  return g;
}

flatguide3 operator cast(guide3 g)
{
  flatguide3 f;
  g(f);
  return f;
}

flatguide3[] operator cast(guide3[] g)
{
  flatguide3[] p=new flatguide3[g.length];
  for(int i=0; i < g.length; ++i) {
    flatguide3 f;
    g[i](f);
    p[i]=f;
  }
  return p;
}

// A version of asin that tolerates numerical imprecision
real asin1(real x)
{
  return asin(min(max(x,-1),1));
}
  
// A version of acos that tolerates numerical imprecision
real acos1(real x)
{
  return acos(min(max(x,-1),1));
}
  
struct Controls {
  triple c0,c1;

  // 3D extension of John Hobby's control point formula
  // (cf. The MetaFont Book, page 131),
  // as described in John C. Bowman and A. Hammerlindl,
  // TUGBOAT: The Communications of th TeX Users Group 29:2 (2008).

  void operator init(triple v0, triple v1, triple d0, triple d1, real tout,
                     real tin, bool atLeast) {
    triple v=v1-v0;
    triple u=unit(v);
    real L=length(v);
    d0=unit(d0);
    d1=unit(d1);
    real theta=acos1(dot(d0,u));
    real phi=acos1(dot(d1,u));
    if(dot(cross(d0,v),cross(v,d1)) < 0) phi=-phi;
    c0=v0+d0*L*relativedistance(theta,phi,tout,atLeast);
    c1=v1-d1*L*relativedistance(phi,theta,tin,atLeast);
  }
}

private triple cross(triple d0, triple d1, triple reference)
{
  triple normal=cross(d0,d1);
  return normal == O ? reference : normal;
}
                                        
private triple dir(real theta, triple d0, triple d1, triple reference)
{
  triple normal=cross(d0,d1,reference);
  if(normal == O) return d1;
  return rotate(degrees(theta),dot(normal,reference) >= 0 ? normal : -normal)*
    d1;
}

private real angle(triple d0, triple d1, triple reference)
{
  real theta=acos1(dot(unit(d0),unit(d1)));
  return dot(cross(d0,d1,reference),reference) >= 0 ? theta : -theta;
}

// 3D extension of John Hobby's angle formula (The MetaFont Book, page 131).
// Notational differences: here psi[i] is the turning angle at z[i+1],
// beta[i] is the tension for segment i, and in[i] is the incoming
// direction for segment i (where segment i begins at node i).

real[] theta(triple[] v, real[] alpha, real[] beta, 
             triple dir0, triple dirn, real g0, real gn, triple reference)
{
  real[] a,b,c,f,l,psi;
  int n=alpha.length;
  bool cyclic=v.cyclicflag;
  for(int i=0; i < n; ++i)
    l[i]=1/length(v[i+1]-v[i]);
  int i0,in;
  if(cyclic) {i0=0; in=n;}
  else {i0=1; in=n-1;}
  for(int i=0; i < in; ++i)
    psi[i]=angle(v[i+1]-v[i],v[i+2]-v[i+1],reference);
  if(cyclic) {
    l.cyclic(true);
    psi.cyclic(true);
  } else {
    psi[n-1]=0;
    if(dir0 == O) {
      real a0=alpha[0];
      real b0=beta[0];
      real chi=g0*(b0/a0)^2;
      a[0]=0;
      b[0]=3a0-a0/b0+chi;
      real C=chi*(3a0-1)+a0/b0;
      c[0]=C;
      f[0]=-C*psi[0];
    } else {
      a[0]=c[0]=0;
      b[0]=1;
      f[0]=angle(v[1]-v[0],dir0,reference);
    }
    if(dirn == O) {
      real an=alpha[n-1];
      real bn=beta[n-1];
      real chi=gn*(an/bn)^2;
      a[n]=chi*(3bn-1)+bn/an;
      b[n]=3bn-bn/an+chi;
      c[n]=f[n]=0;
    } else {
      a[n]=c[n]=0;
      b[n]=1;
      f[n]=angle(v[n]-v[n-1],dirn,reference);
    }
  }
  
  for(int i=i0; i < n; ++i) {
    real in=beta[i-1]^2*l[i-1];
    real A=in/alpha[i-1];
    a[i]=A;
    real B=3*in-A;
    real out=alpha[i]^2*l[i];
    real C=out/beta[i];
    b[i]=B+3*out-C;
    c[i]=C;
    f[i]=-B*psi[i-1]-C*psi[i];
  }
  
  return tridiagonal(a,b,c,f);
}

triple reference(triple[] v, int n, triple d0, triple d1)
{
  triple[] V=sequence(new triple(int i) {
      return cross(v[i+1]-v[i],v[i+2]-v[i+1]); 
    },n-1);
  if(n > 0) {
    V.push(cross(d0,v[1]-v[0]));
    V.push(cross(v[n]-v[n-1],d1));
  }

  triple max=V[0];
  real M=abs(max);
  for(int i=1; i < V.length; ++i) {
    triple vi=V[i];
    real a=abs(vi);
    if(a > M) {
      M=a;
      max=vi;
    }
  }

  triple reference;
  for(int i=0; i < V.length; ++i) {
    triple u=unit(V[i]);
    reference += dot(u,max) < 0 ? -u : u;
  }

  return reference;
}

// Fill in missing directions for n cyclic nodes.
void aim(flatguide3 g, int N) 
{
  bool cyclic=true;
  int start=0, end=0;
  
  // If the cycle contains one or more direction specifiers, break the loop.
  for(int k=0; k < N; ++k)
    if(g.solved(k)) {cyclic=false; end=k; break;}
  for(int k=N-1; k >= 0; --k)
    if(g.solved(k)) {cyclic=false; start=k; break;}
  while(start < N && g.control[start].active) ++start;
  
  int n=N-(start-end);
  if(n <= 1 || (cyclic && n <= 2)) return;

  triple[] v=new triple[cyclic ? n : n+1];
  real[] alpha=new real[n];
  real[] beta=new real[n];
  for(int k=0; k < n; ++k) {
    int K=(start+k) % N;
    v[k]=g.nodes[K];
    alpha[k]=g.Tension[K].out;
    beta[k]=g.Tension[K].in;
  }
  if(cyclic) {
    v.cyclic(true);
    alpha.cyclic(true);
    beta.cyclic(true);
  } else v[n]=g.nodes[(start+n) % N];
  int final=(end-1) % N;

  triple d0=g.out[start].dir;
  triple d1=g.in[final].dir;

  triple reference=reference(v,n,d0,d1);

  real[] theta=theta(v,alpha,beta,d0,d1,g.out[start].gamma,g.in[final].gamma,
                     reference);

  v.cyclic(true);
  theta.cyclic(true);
    
  for(int k=1; k < (cyclic ? n+1 : n); ++k) {
    triple w=dir(theta[k],v[k]-v[k-1],v[k+1]-v[k],reference);
    g.in[(start+k-1) % N].init(w);
    g.out[(start+k) % N].init(w);
  }

  if(g.out[start].dir == O)
    g.out[start].init(dir(theta[0],v[0]-g.nodes[(start-1) % N],v[1]-v[0],
                          reference));
  if(g.in[final].dir == O)
    g.in[final].init(dir(theta[n],v[n-1]-v[n-2],v[n]-v[n-1],reference));
}

// Fill in missing directions for the sequence of nodes i...n.
void aim(flatguide3 g, int i, int n) 
{
  int j=n-i;
  if(j > 1 || g.out[i].dir != O || g.in[i].dir != O) {
    triple[] v=new triple[j+1];
    real[] alpha=new real[j];
    real[] beta=new real[j];
    for(int k=0; k < j; ++k) {
      v[k]=g.nodes[i+k];
      alpha[k]=g.Tension[i+k].out;
      beta[k]=g.Tension[i+k].in;
    }
    v[j]=g.nodes[n];
    
    triple d0=g.out[i].dir;
    triple d1=g.in[n-1].dir;

    triple reference=reference(v,j,d0,d1);

    real[] theta=theta(v,alpha,beta,d0,d1,g.out[i].gamma,g.in[n-1].gamma,
                       reference);
    
    for(int k=1; k < j; ++k) {
      triple w=dir(theta[k],v[k]-v[k-1],v[k+1]-v[k],reference);
      g.in[i+k-1].init(w);
      g.out[i+k].init(w);
    }
    if(g.out[i].dir == O) {
      triple w=dir(theta[0],g.in[i].dir,v[1]-v[0],reference);
      if(i > 0) g.in[i-1].init(w);
      g.out[i].init(w);
    }
    if(g.in[n-1].dir == O) {
      triple w=dir(theta[j],g.out[n-1].dir,v[j]-v[j-1],reference);
      g.in[n-1].init(w);
      g.out[n].init(w);
    }
  }
}

private real Fuzz=10*realEpsilon;

triple XYplane(pair z) {return (z.x,z.y,0);}
triple YZplane(pair z) {return (0,z.x,z.y);}
triple ZXplane(pair z) {return (z.y,0,z.x);}

bool cyclic(guide3 g) {flatguide3 f; g(f); return f.cyclic();}
int size(guide3 g) {flatguide3 f; g(f); return f.size();}
int length(guide3 g) {flatguide3 f; g(f); return f.nodes.length-1;}

path3 path3(triple v)
{
  triple[] point={v};
  return path3(point,point,point,new bool[] {false},false);
}

path3 path3(path p, triple plane(pair)=XYplane)
{
  int n=size(p);
  return path3(sequence(new triple(int i) {return plane(precontrol(p,i));},n),
               sequence(new triple(int i) {return plane(point(p,i));},n),
               sequence(new triple(int i) {return plane(postcontrol(p,i));},n),
               sequence(new bool(int i) {return straight(p,i);},n),
               cyclic(p));
}

path3[] path3(explicit path[] g, triple plane(pair)=XYplane)
{
  return sequence(new path3(int i) {return path3(g[i],plane);},g.length);
}

// Construct a path from a path3 by applying P to each control point.
path path(path3 p, pair P(triple)=xypart)
{
  int n=length(p);
  if(n < 0) return nullpath;
  guide g=P(point(p,0));
  if(n == 0) return g;
  for(int i=1; i < n; ++i)
    g=straight(p,i-1) ? g--P(point(p,i)) :
      g..controls P(postcontrol(p,i-1)) and P(precontrol(p,i))..P(point(p,i));
  
  if(straight(p,n-1))
    return cyclic(p) ? g--cycle : g--P(point(p,n));

  pair post=P(postcontrol(p,n-1));
  pair pre=P(precontrol(p,n));
  return cyclic(p) ? g..controls post and pre..cycle :
    g..controls post and pre..P(point(p,n));
}

void write(file file, string s="", explicit path3 x, suffix suffix=none)
{
  write(file,s);
  int n=length(x);
  if(n < 0) write("<nullpath3>");
  else {
    for(int i=0; i < n; ++i) {
      write(file,point(x,i));
      if(i < length(x)) {
        if(straight(x,i)) write(file,"--");
        else {
          write(file,".. controls ");
          write(file,postcontrol(x,i));
          write(file," and ");
          write(file,precontrol(x,i+1),newl);
          write(file," ..");
        }
      }
    }
    if(cyclic(x))
      write(file,"cycle",suffix);
    else
      write(file,point(x,n),suffix);
  }
}

void write(string s="", explicit path3 x, suffix suffix=endl)
{
  write(stdout,s,x,suffix);
}

void write(file file, string s="", explicit path3[] x, suffix suffix=none)
{
  write(file,s);
  if(x.length > 0) write(file,x[0]);
  for(int i=1; i < x.length; ++i) {
    write(file,endl);
    write(file," ^^");
    write(file,x[i]);
  }
  write(file,suffix);
}

void write(string s="", explicit path3[] x, suffix suffix=endl)
{
  write(stdout,s,x,suffix);
}

path3 solve(flatguide3 g)
{
  int n=g.nodes.length-1;

  // If duplicate points occur consecutively, add dummy controls (if absent).
  for(int i=0; i < n; ++i) {
    if(g.nodes[i] == g.nodes[i+1] && !g.control[i].active)
      g.control[i]=control(g.nodes[i],g.nodes[i],straight=true);
  }  
  
  // Fill in empty direction specifiers inherited from explicit control points.
  for(int i=0; i < n; ++i) {
    if(g.control[i].active) {
      g.out[i].init(g.control[i].post-g.nodes[i]);
      g.in[i].init(g.nodes[i+1]-g.control[i].pre);
    }
  }  
  
  // Propagate directions across nodes.
  for(int i=0; i < n; ++i) {
    int next=g.cyclic[i+1] ? 0 : i+1;
    if(g.out[next].active())
      g.in[i].default(g.out[next]);
    if(g.in[i].active()) {
      g.out[next].default(g.in[i]);
      g.out[i+1].default(g.in[i]);
    }
  }  
    
  // Compute missing 3D directions.
  // First, resolve cycles
  int i=find(g.cyclic);
  if(i > 0) {
    aim(g,i);
    // All other cycles can now be reduced to sequences.
    triple v=g.out[0].dir;
    for(int j=i; j <= n; ++j) {
      if(g.cyclic[j]) {
        g.in[j-1].default(v);
        g.out[j].default(v);
        if(g.nodes[j-1] == g.nodes[j] && !g.control[j-1].active)
          g.control[j-1]=control(g.nodes[j-1],g.nodes[j-1]);
      }
    }
  }
    
  // Next, resolve sequences.
  int i=0;
  int start=0;
  while(i < n) {
    // Look for a missing outgoing direction.
    while(i <= n && g.solved(i)) {start=i; ++i;}
    if(i > n) break;
    // Look for the end of the sequence.
    while(i < n && !g.solved(i)) ++i;
    
    while(start < i && g.control[start].active) ++start;
    
    if(start < i) 
      aim(g,start,i);
  }
  
  // Compute missing 3D control points.
  for(int i=0; i < n; ++i) {
    int next=g.cyclic[i+1] ? 0 : i+1;
    if(!g.control[i].active) {
      control c;
      if((g.out[i].Curl && g.in[i].Curl) ||
         (g.out[i].dir == O && g.in[i].dir == O)) {
        // fill in straight control points for path3 functions
        triple delta=(g.nodes[i+1]-g.nodes[i])/3;
        c=control(g.nodes[i]+delta,g.nodes[i+1]-delta,straight=true);
      } else {
        Controls C=Controls(g.nodes[i],g.nodes[next],g.out[i].dir,g.in[i].dir,
                            g.Tension[i].out,g.Tension[i].in,
                            g.Tension[i].atLeast);
        c=control(C.c0,C.c1);
      }
      g.control[i]=c;
    }
  }

  // Convert to Knuth's format (control points stored with nodes)
  int n=g.nodes.length;
  bool cyclic;
  if(n > 0) {
    cyclic=g.cyclic[n-1];
    if(cyclic) --n;
  }
  triple[] pre=new triple[n];
  triple[] point=new triple[n];
  triple[] post=new triple[n];
  bool[] straight=new bool[n];
  if(n > 0) {
    for(int i=0; i < n-1; ++i) {
      point[i]=g.nodes[i];
      post[i]=g.control[i].post;
      pre[i+1]=g.control[i].pre;
      straight[i]=g.control[i].straight;
    }
    point[n-1]=g.nodes[n-1];
    if(cyclic) {
      pre[0]=g.control[n-1].pre;
      post[n-1]=g.control[n-1].post;
      straight[n-1]=g.control[n-1].straight;
    } else {
      pre[0]=point[0];
      post[n-1]=point[n-1];
      straight[n-1]=false;
    }
  }
  
  return path3(pre,point,post,straight,cyclic);
}

path nurb(path3 p, projection P, int ninterpolate=P.ninterpolate)
{
  triple f=P.camera;
  triple u=unit(P.vector());
  transform3 t=P.t;

  path nurb(triple v0, triple v1, triple v2, triple v3) {
    return nurb(project(v0,t),project(v1,t),project(v2,t),project(v3,t),
                dot(u,f-v0),dot(u,f-v1),dot(u,f-v2),dot(u,f-v3),ninterpolate);
  }

  path g;

  if(straight(p,0))
    g=project(point(p,0),t);

  int last=length(p);
  for(int i=0; i < last; ++i) {
    if(straight(p,i))
      g=g--project(point(p,i+1),t);
    else
      g=g&nurb(point(p,i),postcontrol(p,i),precontrol(p,i+1),point(p,i+1));
  }

  int n=length(g);
  if(cyclic(p)) g=g&cycle;

  return g;
}

path project(path3 p, projection P=currentprojection,
             int ninterpolate=P.ninterpolate)
{
  guide g;

  int last=length(p);
  if(last < 0) return g;
  
  transform3 t=P.t;

  if(ninterpolate == 1 || piecewisestraight(p)) {
    g=project(point(p,0),t);
    // Construct the path.
    int stop=cyclic(p) ? last-1 : last;
    for(int i=0; i < stop; ++i) {
      if(straight(p,i))
        g=g--project(point(p,i+1),t);
      else {
        g=g..controls project(postcontrol(p,i),t) and
          project(precontrol(p,i+1),t)..project(point(p,i+1),t);
      }
    }
  } else return nurb(p,P);
  
  if(cyclic(p))
    g=straight(p,last-1) ? g--cycle :
      g..controls project(postcontrol(p,last-1),t) and
      project(precontrol(p,last),t)..cycle;
  return g;
}

pair[] project(triple[] v, projection P=currentprojection)
{
  return sequence(new pair(int i) {return project(v[i],P.t);},v.length);
}

path[] project(explicit path3[] g, projection P=currentprojection)
{
  return sequence(new path(int i) {return project(g[i],P);},g.length);
}
  
guide3 operator cast(path3 p)
{
  int last=length(p);
  
  bool cyclic=cyclic(p);
  int stop=cyclic ? last-1 : last;
  return new void(flatguide3 f) {
    if(last >= 0) {
      f.node(point(p,0));
      for(int i=0; i < stop; ++i) {
        if(straight(p,i)) {
          f.out(1);
          f.in(1);
        } else
          f.control(postcontrol(p,i),precontrol(p,i+1));
        f.node(point(p,i+1));
      }
      if(cyclic) {
        if(straight(p,stop)) {
          f.out(1);
          f.in(1);
        } else
          f.control(postcontrol(p,stop),precontrol(p,last));
        f.cycleToken();
      }
    }
  };
}

// Return a unit normal vector to a planar path p (or O if the path is
// nonplanar).
triple normal(path3 p)
{
  triple normal;
  real fuzz=sqrtEpsilon*abs(max(p)-min(p));
  real absnormal;
  real theta;
  
  bool Cross(triple a, triple b) {
    if(abs(a) >= fuzz && abs(b) >= fuzz) {
      triple n=cross(unit(a),unit(b));
      real absn=abs(n);
      n=unit(n);
      if(absnormal > 0 && absn > sqrtEpsilon &&
         abs(normal-n) > sqrtEpsilon && abs(normal+n) > sqrtEpsilon)
        return true;
      else {
        int sign=dot(n,normal) >= 0 ? 1 : -1;
        theta += sign*asin1(absn);
        if(absn > absnormal) {
          absnormal=absn;
          normal=n;
          theta=sign*theta;
        }
      }
    }
    return false;
  }
  
  int L=length(p);
  if(L <= 0) return O;

  triple zi=point(p,0);
  triple v0=zi-precontrol(p,0);
  for(int i=0; i < L; ++i) {
    triple c0=postcontrol(p,i);
    triple c1=precontrol(p,i+1);
    triple zp=point(p,i+1);
    triple v1=c0-zi;
    triple v2=c1-c0;
    triple v3=zp-c1;
    if(Cross(v0,v1) || Cross(v1,v2) || Cross(v2,v3)) return O;
    v0=v3;
    zi=zp;
  }
  return theta >= 0 ? normal : -normal;
}

// Return a unit normal vector to a polygon with vertices in p.
triple normal(triple[] p)
{
  triple normal;
  real fuzz=sqrtEpsilon*abs(maxbound(p)-minbound(p));
  real absnormal;
  real theta;
  
  bool Cross(triple a, triple b) {
    if(abs(a) >= fuzz && abs(b) >= fuzz) {
      triple n=cross(unit(a),unit(b));
      real absn=abs(n);
      n=unit(n);
      if(absnormal > 0 && absn > sqrtEpsilon &&
         abs(normal-n) > sqrtEpsilon && abs(normal+n) > sqrtEpsilon)
        return true;
      else {
        int sign=dot(n,normal) >= 0 ? 1 : -1;
        theta += sign*asin1(absn);
        if(absn > absnormal) {
          absnormal=absn;
          normal=n;
          theta=sign*theta;
        }
      }
    }
    return false;
  }
  
  if(p.length <= 0) return O;

  triple zi=p[0];
  triple v0=zi-p[p.length-1];
  for(int i=0; i < p.length-1; ++i) {
    triple zp=p[i+1];
    triple v1=zp-zi;
    if(Cross(v0,v1)) return O;
    v0=v1;
    zi=zp;
  }
  return theta >= 0 ? normal : -normal;
}

// Transforms that map XY plane to YX, YZ, ZY, ZX, and XZ planes.
restricted transform3 XY=identity4;
restricted transform3 YX=rotate(-90,O,Z);
restricted transform3 YZ=rotate(90,O,Z)*rotate(90,O,X);
restricted transform3 ZY=rotate(-90,O,X)*YZ;
restricted transform3 ZX=rotate(-90,O,Z)*rotate(-90,O,Y);
restricted transform3 XZ=rotate(-90,O,Y)*ZX;

private transform3 flip(transform3 t, triple X, triple Y, triple Z,
                        projection P)
{
  static transform3 flip(triple v) {
    static real s(real x) {return x > 0 ? -1 : 1;}
    return scale(s(v.x),s(v.y),s(v.z));
  }

  triple u=unit(P.vector());
  triple up=unit(perp(P.up,u));
  bool upright=dot(Z,u) >= 0;
  if(dot(Y,up) < 0) {
    t=flip(Y)*t;
    upright=!upright;
  }
  return upright ? t : flip(X)*t;
}

restricted transform3 XY(projection P=currentprojection)
{
  return flip(XY,X,Y,Z,P);
}

restricted transform3 YX(projection P=currentprojection)
{
  return flip(YX,Y,X,Z,P);
}

restricted transform3 YZ(projection P=currentprojection)
{
  return flip(YZ,Y,Z,X,P);
}

restricted transform3 ZY(projection P=currentprojection)
{
  return flip(ZY,Z,Y,X,P);
}

restricted transform3 ZX(projection P=currentprojection)
{
  return flip(ZX,Z,X,Y,P);
}

restricted transform3 XZ(projection P=currentprojection)
{
  return flip(XZ,X,Z,Y,P);
}

// Transform3 that projects in direction dir onto plane with normal n
// through point O.
transform3 planeproject(triple n, triple O=O, triple dir=n)
{
  real a=n.x, b=n.y, c=n.z;
  real u=dir.x, v=dir.y, w=dir.z;
  real delta=1.0/(a*u+b*v+c*w);
  real d=-(a*O.x+b*O.y+c*O.z)*delta;
  return new real[][] {
    {(b*v+c*w)*delta,-b*u*delta,-c*u*delta,-d*u},
      {-a*v*delta,(a*u+c*w)*delta,-c*v*delta,-d*v},
        {-a*w*delta,-b*w*delta,(a*u+b*v)*delta,-d*w},
          {0,0,0,1}
  };
}

// Transform3 that projects in direction dir onto plane defined by p.
transform3 planeproject(path3 p, triple dir=O)
{
  triple n=normal(p);
  return planeproject(n,point(p,0),dir == O ? n : dir);
}

// Transform for projecting onto plane through point O with normal cross(u,v).
transform transform(triple u, triple v, triple O=O,
                    projection P=currentprojection)
{
  transform3 t=P.t;
  static real[] O={0,0,0,1};
  real[] tO=t*O;
  real tO3=tO[3];
  real factor=1/tO3^2;
  real[] x=(tO3*t[0]-tO[0]*t[3])*factor;
  real[] y=(tO3*t[1]-tO[1]*t[3])*factor;
  triple x=(x[0],x[1],x[2]);
  triple y=(y[0],y[1],y[2]);
  u=unit(u);
  v=unit(v);
  return (0,0,dot(u,x),dot(v,x),dot(u,y),dot(v,y));
}

// Project Label onto plane through point O with normal cross(u,v).
Label project(Label L, triple u, triple v, triple O=O,
              projection P=currentprojection) {
  Label L=L.copy();
  L.position=project(O,P.t);
  L.transform(transform(u,v,O,P)); 
  return L;
}

path3 operator cast(guide3 g) {return solve(g);}
path3 operator cast(triple v) {return path3(v);}

guide3[] operator cast(triple[] v)
{
  return sequence(new guide3(int i) {return v[i];},v.length);
}

path3[] operator cast(triple[] v)
{
  return sequence(new path3(int i) {return v[i];},v.length);
}

path3[] operator cast(guide3[] g)
{
  return sequence(new path3(int i) {return solve(g[i]);},g.length);
}

guide3[] operator cast(path3[] g)
{
  return sequence(new guide3(int i) {return g[i];},g.length);
}

void write(file file, string s="", explicit guide3[] x, suffix suffix=none)
{
  write(file,s,(path3[]) x,suffix);
}

void write(string s="", explicit guide3[] x, suffix suffix=endl) 
{
  write(stdout,s,(path3[]) x,suffix);
}

triple point(explicit guide3 g, int t) {
  flatguide3 f;
  g(f);
  int n=f.size();
  return f.nodes[adjustedIndex(t,n,f.cyclic())];
}

triple[] dirSpecifier(guide3 g, int t)
{
  flatguide3 f;
  g(f);
  int n=f.size();
  checkEmpty(n);
  if(f.cyclic()) t=t % n;
  else if(t < 0 || t >= n-1) return new triple[];
  return new triple[] {f.out[t].dir,f.in[t].dir};
}

triple[] controlSpecifier(guide3 g, int t) {
  flatguide3 f;
  g(f);
  int n=f.size();
  checkEmpty(n);
  if(f.cyclic()) t=t % n;
  else if(t < 0 || t >= n-1) return new triple[];
  control c=f.control[t];
  if(c.active) return new triple[] {c.post,c.pre};
  else return new triple[];
}

tensionSpecifier tensionSpecifier(guide3 g, int t)
{
  flatguide3 f;
  g(f);
  int n=f.size();
  checkEmpty(n);
  if(f.cyclic()) t=t % n;
  else if(t < 0 || t >= n-1) return operator tension(1,1,false);
  Tension T=f.Tension[t];
  return operator tension(T.out,T.in,T.atLeast);
}

real[] curlSpecifier(guide3 g, int t)
{
  flatguide3 f;
  g(f);
  int n=f.size();
  checkEmpty(n);
  if(f.cyclic()) t=t % n;
  else if(t < 0 || t >= n-1) return new real[];
  return new real[] {f.out[t].gamma,f.in[t].gamma};
}

guide3 reverse(guide3 g)
{
  flatguide3 f;
  bool cyclic=cyclic(g);
  g(f);

  if(f.precyclic())
    return reverse(solve(g));

  int n=f.size();
  checkEmpty(n);
  guide3 G;
  if(n >= 0) {
    int start=cyclic ? n : n-1;
    for(int i=start; i > 0; --i) {
      G=G..f.nodes[i];
      control c=f.control[i-1];
      if(c.active)
        G=G..operator controls(c.pre,c.post);
      else {
        dir in=f.in[i-1];
        triple d=in.dir;
        if(d != O) G=G..operator spec(-d,JOIN_OUT);
        else if(in.Curl) G=G..operator curl(in.gamma,JOIN_OUT);
        dir out=f.out[i-1];
        triple d=out.dir;
        if(d != O) G=G..operator spec(-d,JOIN_IN);
        else if(out.Curl) G=G..operator curl(out.gamma,JOIN_IN);
      }
    }
    if(cyclic) G=G..cycle;
    else G=G..f.nodes[0];
  }
  return G;
}

triple intersectionpoint(path3 p, path3 q, real fuzz=-1)
{
  real[] t=intersect(p,q,fuzz);
  if(t.length == 0) abort("paths do not intersect");
  return point(p,t[0]);
}

// return an array containing all intersection points of p and q
triple[] intersectionpoints(path3 p, path3 q, real fuzz=-1)
{
  real[][] t=intersections(p,q,fuzz);
  return sequence(new triple(int i) {return point(p,t[i][0]);},t.length);
}

triple[] intersectionpoints(explicit path3[] p, explicit path3[] q,
                            real fuzz=-1)
{
  triple[] v;
  for(int i=0; i < p.length; ++i)
    for(int j=0; j < q.length; ++j)
      v.append(intersectionpoints(p[i],q[j],fuzz));
  return v;
}

path3 operator &(path3 p, cycleToken tok)
{
  int n=length(p);
  if(n < 0) return nullpath3;
  triple a=point(p,0);
  triple b=point(p,n);
  return subpath(p,0,n-1)..controls postcontrol(p,n-1) and precontrol(p,n)..
    cycle;
}

// return the point on path3 p at arclength L
triple arcpoint(path3 p, real L)
{
  return point(p,arctime(p,L));
}

// return the point on path3 p at arclength L
triple arcpoint(path3 p, real L)
{
  return point(p,arctime(p,L));
}

// return the direction on path3 p at arclength L
triple arcdir(path3 p, real L)
{
  return dir(p,arctime(p,L));
}

// return the time on path3 p at the relative fraction l of its arclength
real reltime(path3 p, real l)
{
  return arctime(p,l*arclength(p));
}

// return the point on path3 p at the relative fraction l of its arclength
triple relpoint(path3 p, real l)
{
  return point(p,reltime(p,l));
}

// return the direction of path3 p at the relative fraction l of its arclength
triple reldir(path3 p, real l)
{
  return dir(p,reltime(p,l));
}

// return the point on path3 p at half of its arclength
triple midpoint(path3 p)
{
  return relpoint(p,0.5);
}

real relative(Label L, path3 g)
{
  return L.position.relative ? reltime(g,L.relative()) : L.relative();
}

// return the linear transformation that maps X,Y,Z to u,v,w.
transform3 transform3(triple u, triple v, triple w=cross(u,v)) 
{
  return new real[][] {
    {u.x,v.x,w.x,0},
      {u.y,v.y,w.y,0},
        {u.z,v.z,w.z,0},
          {0,0,0,1}
  };
}

// return the rotation that maps Z to a unit vector u about cross(u,Z),
transform3 align(triple u)
{
  real a=u.x;
  real b=u.y;
  real c=u.z;
  real d=a^2+b^2;

  if(d != 0) {
    d=sqrt(d);
    real e=1/d;
    return new real[][] {
      {-b*e,-a*c*e,a,0},
        {a*e,-b*c*e,b,0},
          {0,d,c,0},
            {0,0,0,1}};
  }
  return c >= 0 ? identity(4) : diagonal(1,-1,-1,1);
}

// return a rotation that maps X,Y to the projection plane.
transform3 transform3(projection P)
{
  triple v=unit(P.oblique ? P.camera : P.vector());
  triple u=unit(perp(P.up,v));
  if(u == O) u=cross(perp(v),v);
  v=cross(u,v);
  return v != O ? transform3(v,u) : identity(4);
}

triple[] triples(real[] x, real[] y, real[] z)
{
  if(x.length != y.length || x.length != z.length)
    abort("arrays have different lengths");
  return sequence(new triple(int i) {return (x[i],y[i],z[i]);},x.length);
}

path3[] operator cast(path3 p)
{
  return new path3[] {p};
}

path3[] operator cast(guide3 g)
{
  return new path3[] {(path3) g};
}

path3[] operator ^^ (path3 p, path3  q) 
{
  return new path3[] {p,q};
}

path3[] operator ^^ (path3 p, explicit path3[] q) 
{
  return concat(new path3[] {p},q);
}

path3[] operator ^^ (explicit path3[] p, path3 q) 
{
  return concat(p,new path3[] {q});
}

path3[] operator ^^ (explicit path3[] p, explicit path3[] q) 
{
  return concat(p,q);
}

path3[] operator * (transform3 t, explicit path3[] p) 
{
  return sequence(new path3(int i) {return t*p[i];},p.length);
}

triple[] operator * (transform3 t, triple[] v) 
{
  return sequence(new triple(int i) {return t*v[i];},v.length);
}

triple min(explicit path3[] p)
{
  checkEmpty(p.length);
  triple minp=min(p[0]);
  for(int i=1; i < p.length; ++i)
    minp=minbound(minp,min(p[i]));
  return minp;
}

triple max(explicit path3[] p)
{
  checkEmpty(p.length);
  triple maxp=max(p[0]);
  for(int i=1; i < p.length; ++i)
    maxp=maxbound(maxp,max(p[i]));
  return maxp;
}

typedef guide3 interpolate3(... guide3[]);

path3 randompath3(int n, bool cumulate=true, interpolate3 join=operator ..)
{
  guide3 g;
  triple w;
  for(int i=0; i <= n; ++i) {
    triple z=(unitrand()-0.5,unitrand()-0.5,unitrand()-0.5);
    if(cumulate) w += z; 
    else w=z;
    g=join(g,w);
  }
  return g;
}

path3[] box(triple v1, triple v2)
{
  return
    (v1.x,v1.y,v1.z)--
    (v1.x,v1.y,v2.z)--
    (v1.x,v2.y,v2.z)--
    (v1.x,v2.y,v1.z)--
    (v1.x,v1.y,v1.z)--
    (v2.x,v1.y,v1.z)--
    (v2.x,v1.y,v2.z)--
    (v2.x,v2.y,v2.z)--
    (v2.x,v2.y,v1.z)--
    (v2.x,v1.y,v1.z)^^
    (v2.x,v2.y,v1.z)--
    (v1.x,v2.y,v1.z)^^
    (v1.x,v2.y,v2.z)--
    (v2.x,v2.y,v2.z)^^
    (v2.x,v1.y,v2.z)--
    (v1.x,v1.y,v2.z);
}

restricted path3[] unitbox=box(O,(1,1,1));
restricted path3 unitcircle3=X..Y..-X..-Y..cycle;
restricted path3 unitsquare3=O--X--X+Y--Y--cycle;

path3 circle(triple c, real r, triple normal=Z)
{
  path3 p=normal == Z ? unitcircle3 : align(unit(normal))*unitcircle3;
  return shift(c)*scale3(r)*p;
}

// return an arc centered at c from triple v1 to v2 (assuming |v2-c|=|v1-c|),
// drawing in the given direction.
// The normal must be explicitly specified if c and the endpoints are colinear.
path3 arc(triple c, triple v1, triple v2, triple normal=O, bool direction=CCW)
{
  v1 -= c;
  real r=abs(v1);
  v1=unit(v1);
  v2=unit(v2-c);

  if(normal == O) {
    normal=cross(v1,v2);
    if(normal == O) abort("explicit normal required for these endpoints");
  }

  transform3 T;
  bool align=normal != Z;
  if(align) {
    T=align(unit(normal));
    transform3 Tinv=transpose(T);
    v1=Tinv*v1;
    v2=Tinv*v2;
  }
  
  string invalidnormal="invalid normal vector";
  real fuzz=sqrtEpsilon*max(abs(v1),abs(v2));
  if(abs(v1.z) > fuzz || abs(v2.z) > fuzz)
    abort(invalidnormal);
  
  real[] t1=intersect(unitcircle3,O--2*(v1.x,v1.y,0));
  real[] t2=intersect(unitcircle3,O--2*(v2.x,v2.y,0));
  
  if(t1.length == 0 || t2.length == 0)
    abort(invalidnormal);

  real t1=t1[0];
  real t2=t2[0];
  int n=length(unitcircle3);
  if(t1 >= t2 && direction) t1 -= n;
  if(t2 >= t1 && !direction) t2 -= n;

  path3 p=subpath(unitcircle3,t1,t2);
  if(align) p=T*p;
  return shift(c)*scale3(r)*p;
}

// return an arc centered at c with radius r from c+r*dir(theta1,phi1) to
// c+r*dir(theta2,phi2) in degrees, drawing in the given direction
// relative to the normal vector cross(dir(theta1,phi1),dir(theta2,phi2)).
// The normal must be explicitly specified if c and the endpoints are colinear.
path3 arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
          triple normal=O, bool direction)
{
  return arc(c,c+r*dir(theta1,phi1),c+r*dir(theta2,phi2),normal,direction);
}

// return an arc centered at c with radius r from c+r*dir(theta1,phi1) to
// c+r*dir(theta2,phi2) in degrees, drawing drawing counterclockwise
// relative to the normal vector cross(dir(theta1,phi1),dir(theta2,phi2))
// iff theta2 > theta1 or (theta2 == theta1 and phi2 >= phi1).
// The normal must be explicitly specified if c and the endpoints are colinear.
path3 arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
          triple normal=O)
{
  return arc(c,r,theta1,phi1,theta2,phi2,normal,
             theta2 > theta1 || (theta2 == theta1 && phi2 >= phi1) ? CCW : CW);
}

private real epsilon=1000*realEpsilon;

// Return a representation of the plane through point O with normal cross(u,v).
path3 plane(triple u, triple v, triple O=O)
{
  return O--O+u--O+u+v--O+v--cycle;
}

triple size3(frame f)
{
  return max3(f)-min3(f);
}

// PRC/OpenGL support

include three_light;

void draw(frame f, path3 g, material p=currentpen, light light=nolight,
          projection P=currentprojection);

void begingroup3(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform3, picture opic, projection) {
      if(opic != null)
        begingroup(opic);
    },true);
}

void endgroup3(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform3, picture opic, projection) {
      if(opic != null)
        endgroup(opic);
    },true);
}

void addPath(picture pic, path3 g, pen p)
{
  if(size(g) > 0)
    pic.addBox(min(g),max(g),min3(p),max3(p));
}

include three_surface;
include three_margins;

void draw(picture pic=currentpicture, Label L="", path3 g,
          align align=NoAlign, material p=currentpen, margin3 margin=NoMargin3,
          light light=nolight)
{
  pen q=(pen) p;
  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      path3 G=margin(t*g,q).g;
      if(is3D()) {
        draw(f,G,p,light,null);
        if(pic != null && size(G) > 0) {
          pic.addPoint(min(G,P.t));
          pic.addPoint(max(G,P.t));
        }
      }
      if(pic != null)
        draw(pic,project(G,P),q);
    },true);
  Label L=L.copy();
  L.align(align);
  if(L.s != "") {
    L.p(q);
    label(pic,L,g);
  }
  addPath(pic,g,q);
}

include three_arrows;

draw=new void(frame f, path3 g, material p=currentpen,
              light light=nolight, projection P=currentprojection) {
  pen q=(pen) p;
  if(is3D()) {
    p=material(p,(p.granularity >= 0) ? p.granularity : linegranularity);
    void drawthick(path3 g) {
      if(settings.thick) {
        real width=linewidth(q);
        if(width > 0) {
          surface s=tube(g,width,p.granularity);
          int L=length(g);
          if(L >= 0) {
            if(!cyclic(g)) {
              real r=0.5*width;
              real linecap=linecap(q);
              transform3 scale3r=scale3(r);
              surface cap;
              triple dirL=dir(g,L);
              triple dir0=dir(g,0);
              if(linecap == 0)
                cap=scale(r,r,1)*unitdisk;
              else if(linecap == 1)
                cap=scale3r*((dir0 == O || dirL == O) ?
                             unitsphere : unithemisphere);
              else if(linecap == 2) {
                cap=scale3r*unitcylinder;
                cap.append(scale3r*shift(Z)*unitdisk);
              }
              s.append(shift(point(g,0))*align(-dir0)*cap);
              s.append(shift(point(g,L))*align(dirL)*cap);
            }
            if(opacity(q) == 1) _draw(f,g,q);
          }
          for(int i=0; i < s.s.length; ++i)
            draw3D(f,s.s[i],p,light);
        } else _draw(f,g,q);
      } else _draw(f,g,q);
    }
    string type=linetype(adjust(q,arclength(g),cyclic(g)));
    if(length(type) == 0) drawthick(g);
    else {
      real[] dash=(real[]) split(type," ");
      if(sum(dash) > 0) {
        dash.cyclic(true);
        real offset=offset(q);
        real L=arclength(g);
        int i=0;
        real l=offset;
        while(l <= L) {
          real t1=arctime(g,l);
          l += dash[i];
          real t2=arctime(g,min(l,L));
          drawthick(subpath(g,t1,t2));
          ++i;
          l += dash[i];
          ++i;
        }
      }
    }
  } else draw(f,project(g,P),q);
};

void draw(frame f, explicit path3[] g, material p=currentpen,
          light light=nolight, projection P=currentprojection)
{
  for(int i=0; i < g.length; ++i) draw(f,g[i],p,light,P);
}

void draw(picture pic=currentpicture, explicit path3[] g,
          material p=currentpen, margin3 margin=NoMargin3, light light=nolight)
{
  for(int i=0; i < g.length; ++i) draw(pic,g[i],p,margin,light);
}

void draw(picture pic=currentpicture, Label L="", path3 g, 
          align align=NoAlign, material p=currentpen, arrowbar3 arrow,
          arrowbar3 bar=None, margin3 margin=NoMargin3, light light=nolight,
          light arrowheadlight=currentlight)
{
  begingroup3(pic);
  bool drawpath=arrow(pic,g,p,margin,light,arrowheadlight);
  if(bar(pic,g,p,margin,light,arrowheadlight) && drawpath)
    draw(pic,L,g,align,p,margin,light);
  endgroup3(pic);
  label(pic,L,g,align,(pen) p);
}

void draw(frame f, path3 g, material p=currentpen, arrowbar3 arrow,
          light light=nolight, light arrowheadlight=currentlight,
          projection P=currentprojection)
{
  picture pic;
  if(arrow(pic,g,p,NoMargin3,light,arrowheadlight))
    draw(f,g,p,light,P);
  add(f,pic.fit());
}

void add(picture pic=currentpicture, void d(picture,transform3),
         bool exact=false)
{
  pic.add(d,exact);
}

// Fit the picture src using the identity transformation (so user
// coordinates and truesize coordinates agree) and add it about the point
// position to picture dest.
void add(picture dest, picture src, triple position, bool group=true,
         bool above=true)
{
  dest.add(new void(picture f, transform3 t) {
      f.add(shift(t*position)*src,group,above);
    });
}

void add(picture src, triple position, bool group=true, bool above=true)
{
  add(currentpicture,src,position,group,above);
}

// Align an arrow pointing to b from the direction dir. The arrow is
// 'length' PostScript units long.
void arrow(picture pic=currentpicture, Label L="", triple b, triple dir,
           real length=arrowlength, align align=NoAlign,
           pen p=currentpen, arrowbar3 arrow=Arrow3, margin3 margin=EndMargin3,
           light light=nolight, light arrowheadlight=currentlight)
{
  Label L=L.copy();
  if(L.defaultposition) L.position(0);
  L.align(L.align,dir);
  L.p(p);
  picture opic;
  marginT3 margin=margin(b--b,p); // Extract margin.begin and margin.end
  triple a=(margin.begin+length+margin.end)*unit(dir);
  draw(opic,L,a--O,align,p,arrow,margin,light,arrowheadlight);
  add(pic,opic,b);
}

void arrow(picture pic=currentpicture, Label L="", triple b, pair dir,
           real length=arrowlength, align align=NoAlign,
           pen p=currentpen, arrowbar3 arrow=Arrow3, margin3 margin=EndMargin3,
           light light=nolight, light arrowheadlight=currentlight,
           projection P=currentprojection)
{
  arrow(pic,L,b,invert(dir,b,P),length,align,p,arrow,margin,light,
        arrowheadlight);
}

triple min3(picture pic, projection P=currentprojection)
{
  return pic.min3(P);
}
  
triple max3(picture pic, projection P=currentprojection)
{
  return pic.max3(P);
}
  
triple size3(picture pic, bool user=false, projection P=currentprojection)
{
  transform3 t=pic.calculateTransform3(P);
  triple M=pic.max(t);
  triple m=pic.min(t);
  if(!user) return M-m;
  t=inverse(t);
  return t*M-t*m;
}

triple point(frame f, triple dir)
{
  triple m=min3(f);
  triple M=max3(f);
  return m+realmult(rectify(dir),M-m);
}

triple point(picture pic=currentpicture, triple dir, bool user=true,
             projection P=currentprojection)
{
  triple v=pic.userMin+realmult(rectify(dir),pic.userMax-pic.userMin);
  return user ? v : pic.calculateTransform3(P)*v;
}

triple truepoint(picture pic=currentpicture, triple dir, bool user=true,
                 projection P=currentprojection)
{
  transform3 t=pic.calculateTransform3(P);
  triple m=pic.min(t);
  triple M=pic.max(t);
  triple v=m+realmult(rectify(dir),M-m);
  return user ? inverse(t)*v : v;
}

void add(picture dest=currentpicture, object src, pair position=0, pair align=0,
         bool group=true, filltype filltype=NoFill, bool above=true)
{
  if(prc())
    label(dest,src,position,align);
  else if(settings.render == 0)
    plain.add(dest,src,position,align,group,filltype,above);
}

string cameralink(string label, string text="View Parameters")
{
  if(!prc() || Link == null) return "";
  return Link(label,text,"3Dgetview");
}

private struct viewpoint {
  triple target,camera,up;
  real angle;
  void operator init(string s) {
    s=replace(s,new string[][] {{" ",","},{"}{",","},{"{",""},{"}",""},});
    string[] S=split(s,",");
    target=((real) S[0],(real) S[1],(real) S[2])*cm;
    camera=target+(real) S[6]*((real) S[3],(real) S[4],(real) S[5])*cm;
    triple u=unit(target-camera);
    triple w=unit(Z-u.z*u);
    up=rotate((real) S[7],O,u)*w;
    angle=S[8] == "" ? 30 : (real) S[8];
  }
}

projection perspective(string s)
{
  viewpoint v=viewpoint(s);
  projection P=perspective(v.camera,v.up,v.target);
  P.angle=v.angle;
  P.absolute=true;
  return P;
}

private string Format(real x)
{
  // Work around movie15.sty division by zero bug;
  // e.g. u=unit((1e-10,1e-10,0.9));
  if(abs(x) < 1e-9) x=0; 
  assert(abs(x) < 1e18,"Number too large: "+string(x));
  return format("%.18f",x,"C");
}

private string Format(triple v, string sep=" ")
{
  return Format(v.x)+sep+Format(v.y)+sep+Format(v.z);
}

private string Format(real[] c)
{
  return Format((c[0],c[1],c[2]));
}

private string[] file3;

private string projection(bool infinity, real viewplanesize)
{
  return "activeCamera=scene.cameras.getByIndex(0);
function asyProjection() {"+
    (infinity ? "activeCamera.projectionType=activeCamera.TYPE_ORTHOGRAPHIC;" :
     "activeCamera.projectionType=activeCamera.TYPE_PERSPECTIVE;")+"
activeCamera.viewPlaneSize="+string(viewplanesize)+";
activeCamera.binding=activeCamera.BINDING_"+(infinity ? "MAX" : "VERTICAL")+";
}

asyProjection();

handler=new CameraEventHandler();
runtime.addEventHandler(handler);
handler.onEvent=function(event) 
{
  asyProjection();
  scene.update();
}";
}

string lightscript(light light) {
  string script="for(var i=scene.lights.count-1; i >= 0; i--)
  scene.lights.removeByIndex(i);"+'\n\n';
  for(int i=0; i < light.position.length; ++i) {
    string Li="L"+string(i);
    real[] diffuse=light.diffuse[i];
    script += Li+"=scene.createLight();"+'\n'+
      Li+".direction.set("+Format(-light.position[i],",")+");"+'\n'+
      Li+".color.set("+Format((diffuse[0],diffuse[1],diffuse[2]),",")+");"+'\n';
  }
  // Work around initialization bug in Adobe Reader 8.0:
  return script +"
scene.lightScheme=scene.LIGHT_MODE_HEADLAMP;
scene.lightScheme=scene.LIGHT_MODE_FILE;
";
}

void writeJavaScript(string name, string preamble, string script) 
{
  file out=output(name);
  write(out,preamble);
  if(script != "") {
    file in=input(script);
    while(true) {
      string line=in;
      if(eof(in)) break;
      write(out,line,endl);
    }
  }
  close(out);
  if(settings.verbose > 1) write("Wrote "+name);
  if(!settings.inlinetex)
    file3.push(name);
}

pair viewportmargin(pair lambda)
{
  return maxbound(0.5*(viewportsize-lambda),viewportmargin);
}

string embed3D(string label="", string text=label, string prefix,
               frame f, string format="",
               real width=0, real height=0, real angle=30,
               string options="", string script="",
               light light=currentlight, projection P=currentprojection)
{
  if(!prc(format) || Embed == null) return "";

  if(width == 0) width=settings.paperwidth;
  if(height == 0) height=settings.paperheight;

  if(script == "") script=defaultembed3Dscript;

  // Adobe Reader doesn't appear to support user-specified viewport lights.
  string lightscript=light.on() && !light.viewport ? lightscript(light) : "";

  real viewplanesize;
  if(P.infinity) {
    triple lambda=max3(f)-min3(f);
    pair margin=viewportmargin((lambda.x,lambda.y));
    viewplanesize=(max(lambda.x+2*margin.x,lambda.y+2*margin.y))/(cm*P.zoom);
  } else
    if(!P.absolute) angle=2*aTan(Tan(0.5*angle));

  string name=prefix+".js";
  writeJavaScript(name,lightscript+projection(P.infinity,viewplanesize),script);

  shipout3(prefix,f);

  prefix += ".prc";
  if(!settings.inlinetex)
    file3.push(prefix);

  triple v=P.vector()/cm;
  triple u=unit(v);
  triple w=Z-u.z*u;
  real roll;
  if(abs(w) > sqrtEpsilon) {
    w=unit(w);
    triple up=unit(perp(P.up,u));
    roll=degrees(acos1(dot(up,w)))*sgn(dot(cross(up,w),u));
  } else roll=0;
  
  string options3=light.viewport ? "3Dlights=Headlamp" : "3Dlights=File";
  if(defaultembed3Doptions != "") options3 += ","+defaultembed3Doptions;
  if((settings.render < 0 || !settings.embed) && settings.auto3D)
    options3 += ",poster";
  options3 += ",text={"+text+"},label="+label+
    ",toolbar="+(settings.toolbar ? "true" : "false")+
    ",3Daac="+Format(P.absolute ? P.angle : angle)+
    ",3Dc2c="+Format(u)+
    ",3Dcoo="+Format(P.target/cm)+
    ",3Droll="+Format(roll)+
    ",3Droo="+Format(abs(v))+
    ",3Dbg="+Format(light.background());
  if(options != "") options3 += ","+options;
  if(name != "") options3 += ",3Djscript="+stripdirectory(name);

  return Embed(stripdirectory(prefix),options3,width,height);
}

object embed(string label="", string text=label, 
             string prefix=defaultfilename, 
             frame f, string format="",
             real width=0, real height=0, real angle=30,
             string options="", string script="", 
             light light=currentlight, projection P=currentprojection)
{
  object F;

  if(is3D(format))
    F.L=embed3D(label,text,prefix,f,format,width,height,angle,options,script,
                light,P);
  else
    F.f=f;
  return F;
}

object embed(string label="", string text=label,
             string prefix=defaultfilename,
             picture pic, string format="",
             real xsize=pic.xsize, real ysize=pic.ysize,
             bool keepAspect=pic.keepAspect, bool view=true, string options="",
             string script="", real angle=0,
             light light=currentlight, projection P=currentprojection)
{
  object F;
  real xsize3=pic.xsize3, ysize3=pic.ysize3, zsize3=pic.zsize3;
  bool warn=true;
  transform3 modelview;
        
  if(xsize3 == 0 && ysize3 == 0 && zsize3 == 0) {
    xsize3=ysize3=zsize3=max(xsize,ysize);
    warn=false;
  }

  projection P=P.copy();

  if(!P.absolute && P.showtarget)
    draw(pic,P.target,nullpen);

  transform3 t=pic.scaling(xsize3,ysize3,zsize3,keepAspect,warn);
  bool adjusted=false;
  transform3 tinv=inverse(t);
  triple m=pic.min(t);
  triple M=pic.max(t);

  if(!P.absolute) {
    P=t*P;
    if(P.center) {
      bool recalculate=false;
      triple target=0.5*(m+M);
      P.target=target;
      recalculate=true;
      if(recalculate) P.calculate();
    }
    if(P.autoadjust || P.infinity) 
      adjusted=adjusted | P.adjust(m,M);
  }

  picture pic2;
  
  frame f=pic.fit3(t,pic.bounds3.exact ? pic2 : null,P);

  if(!pic.bounds3.exact) {
    transform3 s=pic.scale3(f,xsize3,ysize3,zsize3,keepAspect);
    t=s*t;
    tinv=inverse(t);
    P=s*P;
    f=pic.fit3(t,pic2,P);
  }

  bool is3D=is3D(format);
  bool scale=xsize != 0 || ysize != 0;

  if(is3D || scale) {
    pic2.bounds.exact=true;
    transform s=pic2.scaling(xsize,ysize,keepAspect);
    pair m2=pic2.min(s);
    pair M2=pic2.max(s);
    pair lambda=M2-m2;
    pair viewportmargin=viewportmargin(lambda);
    real width=ceil(lambda.x+2*viewportmargin.x);
    real height=ceil(lambda.y+2*viewportmargin.y);

    projection Q;
    if(!P.absolute) {
      if(scale) {
        pair v=(s.xx,s.yy);
        transform3 T=P.t;
        pair x=project(X,T);
        pair y=project(Y,T);
        pair z=project(Z,T);
        real f(pair a, pair b) {
          return b == 0 ? (0.5*(a.x+a.y)) : (b.x^2*a.x+b.y^2*a.y)/(b.x^2+b.y^2);
        }
        pic2.erase();
        t=xscale3(f(v,x))*yscale3(f(v,y))*zscale3(f(v,z))*t;
        f=pic.fit3(t,is3D ? null : pic2,P);
      }

      if(P.autoadjust || P.infinity)
        adjusted=adjusted | P.adjust(min3(f),max3(f));

      triple target=P.target;
      modelview=P.modelview();
      f=modelview*f;
      P=modelview*P;
      Q=P.copy();
      light=modelview*light;

      if(P.infinity) {
        triple m=min3(f);
        triple M=max3(f);
        triple lambda=M-m;
        viewportmargin=viewportmargin((lambda.x,lambda.y));
        width=lambda.x+2*viewportmargin.x;
        height=lambda.y+2*viewportmargin.y;

        triple s=(-0.5(m.x+M.x),-0.5*(m.y+M.y),0);
        f=shift(s-target)*f;
      } else {
        transform3 T=identity4;
        // Choose the angle to be just large enough to view the entire image:
        if(angle == 0) angle=P.angle;
        int maxiterations=100;
        if(is3D && angle == 0) {
          real h=-0.5*P.target.z;
          pair r,R;
          real diff=realMax;
          pair s;
          int i;
          do {
            r=minratio(f);
            R=maxratio(f);
            pair lasts=s;
            if(P.autoadjust) {
              s=r+R;
              if(s != 0) {
                transform3 t=shift(h*s.x,h*s.y,0);
                f=t*f;
                T=t*T;
                adjusted=true;
              }
            }
            diff=abs(s-lasts);
            ++i;
          } while (diff > angleprecision && i < maxiterations);
          real aspect=width > 0 ? height/width : 1;
          real rx=-r.x*aspect;
          real Rx=R.x*aspect;
          real ry=-r.y;
          real Ry=R.y;
          if(!P.autoadjust) {
            if(rx > Rx) Rx=rx;
            else rx=Rx;
            if(ry > Ry) Ry=ry;
            else ry=Ry;
          }
          
          angle=anglefactor*max(aTan(rx)+aTan(Rx),aTan(ry)+aTan(Ry));
          if(viewportmargin.y != 0)
            angle=2*aTan(Tan(0.5*angle)-viewportmargin.y/P.target.z);
          
          modelview=T*modelview;
        }
        if(settings.verbose > 0) {
          transform3 inv=inverse(modelview);
          if(adjusted) 
            write("adjusting camera to ",tinv*inv*P.camera);
          target=inv*P.target;
        }
        P=T*P;
      }
      if(settings.verbose > 0) {
        if(P.center || (!P.infinity && P.autoadjust))
          write("adjusting target to ",tinv*target);
      }
    }
    
    if(prefix == "") prefix=outprefix();
    bool prc=prc(format);
    bool preview=settings.render > 0;
    if(prc) {
      // The movie15.sty package cannot handle spaces or dots in filenames.
      prefix=replace(prefix,new string[][]{{" ","_"},{".","_"}});
      if(settings.embed || nativeformat() == "pdf")
        prefix += "+"+(string) file3.length;
    } else
      preview=false;
    if(preview || (!prc && settings.render != 0)) {
      frame f=f;
      triple m,M;
      real zcenter;
      if(P.absolute) {
        modelview=P.modelview();
        f=modelview*f;
        Q=modelview*P;
        m=min3(f);
        M=max3(f);
        real r=0.5*abs(M-m);
        zcenter=0.5*(M.z+m.z);
        M=(M.x,M.y,zcenter+r);
        m=(m.x,m.y,zcenter-r);
        angle=P.angle;
      } else {
        m=min3(f);
        M=max3(f);
        zcenter=P.target.z;
        real d=P.distance(m,M);
        M=(M.x,M.y,zcenter+d);
        m=(m.x,m.y,zcenter-d);
      }

      if(P.infinity) {
        triple margin=(viewportfactor-1.0)*(abs(M.x-m.x),abs(M.y-m.y),0)
          +(viewportmargin.x,viewportmargin.y,0);
        M += margin; 
        m -= margin;
      } else if(M.z >= 0) abort("camera too close");

      shipout3(prefix,f,preview ? nativeformat() : format,
               width,height,P.infinity ? 0 : angle,P.zoom,m,M,
               prc ? 0 : P.viewportshift,
               tinv*inverse(modelview)*shift(0,0,zcenter),light.background(),
               P.absolute ? (modelview*light).position : light.position,
               light.diffuse,light.ambient,light.specular,
               light.viewport,view && !preview);
      if(!preview) return F;
    }

    string image;
    if(preview && settings.embed) {
      image=prefix;
      if(settings.inlinetex) image += "_0";
      image += "."+nativeformat();
      if(!settings.inlinetex) file3.push(image);
      image=graphic(image,"hiresbb");
    }
    if(prc) {
      if(P.viewportshift != 0)
        write("warning: PRC output ignores viewportshift");
      F.L=embed3D(label,text=image,prefix,f,format,
                        width,height,angle,options,script,light,Q);
    }
    
  }

  if(!is3D) {
    transform T=pic2.scaling(xsize,ysize,keepAspect);
    F.f=pic.fit(scale(t[0][0])*T);
    add(F.f,pic2.fit(T));
  }
      
  return F;
}

embed3=new object(string prefix, frame f, string format, string options,
                  string script, projection P) {
  return embed(prefix=prefix,f,format,options,script,P);
};

currentpicture.fitter=new frame(string prefix, picture pic, string format,
                                real xsize, real ysize,
                                bool keepAspect, bool view,
                                string options, string script, projection P) {
  frame f;
  bool empty3=pic.empty3();
  if(is3D(format) || empty3) add(f,pic.fit2(xsize,ysize,keepAspect));
  if(!empty3) {
    bool prc=prc(format);
    if(!prc && settings.render != 0 && !view) {
      static int previewcount=0;
      bool keep=prefix != "";
      prefix=outprefix(prefix)+"+"+(string) previewcount;
      ++previewcount;
      format=nativeformat();
      if(!keep) file3.push(prefix+"."+format);
    }
    object F=embed(prefix=prefix,pic,format,xsize,ysize,keepAspect,view,
                   options,script,currentlight,P);
    if(prc)
      label(f,F.L);
    else {
      if(settings.render == 0) {
        add(f,F.f);
        if(currentlight.background != nullpen)
          box(f,currentlight.background,Fill,above=false);
      } else if(!view)
        label(f,graphic(prefix,"hiresbb"));
    }
  }
  return f;
};

void addViews(picture dest, picture src, bool group=true,
              filltype filltype=NoFill)
{
  if(group) begingroup(dest);
  frame Front=src.fit(FrontView);
  add(dest,Front,filltype);
  frame Top=src.fit(TopView);
  add(dest,shift(0,min(Front).y-max(Top).y)*Top,filltype);
  frame Right=src.fit(RightView);
  add(dest,shift(min(Front).x-max(Right).x)*Right,filltype);
  if(group) endgroup(dest);
}

void addViews(picture src, bool group=true, filltype filltype=NoFill)
{
  addViews(currentpicture,src,group,filltype);
}

void addAllViews(picture dest, picture src,
                 real xmargin=0, real ymargin=xmargin,
                 bool group=true,
                 filltype filltype=NoFill)
{
  picture picL,picM,picR,picLM;
  if(xmargin == 0) xmargin=sqrtEpsilon;
  if(ymargin == 0) ymargin=sqrtEpsilon;

  add(picL,src.fit(FrontView),(0,0),ymargin*N);
  add(picL,src.fit(BackView),(0,0),ymargin*S);

  add(picM,src.fit(TopView),(0,0),ymargin*N);
  add(picM,src.fit(BottomView),(0,0),ymargin*S);

  add(picR,src.fit(RightView),(0,0),ymargin*N);
  add(picR,src.fit(LeftView),(0,0),ymargin*S);

  add(picLM,picL.fit(),(0,0),xmargin*W);
  add(picLM,picM.fit(),(0,0),xmargin*E);

  if(group) begingroup(dest);
  add(dest,picLM.fit(),(0,0),xmargin*W,filltype);
  add(dest,picR.fit(),(0,0),xmargin*E,filltype);
  if(group) endgroup(dest);
}

void addAllViews(picture src, bool group=true, filltype filltype=NoFill)
{
  addAllViews(currentpicture,src,group,filltype);
}

// Force an array of 3D pictures to be as least as large as picture all.
void rescale3(picture[] pictures, picture all, projection P=currentprojection)
{
  if(!all.empty3()) {
    transform3 t=inverse(all.calculateTransform3(P)*pictures[0].T3);
    triple m=t*min3(all);
    triple M=t*max3(all);
    for(int i=0; i < pictures.length; ++i) {
      draw(pictures[i],m,nullpen);
      draw(pictures[i],M,nullpen);
    }
  }
}

// Force an array of pictures to have a uniform scaling using currenprojection.
rescale=new void(picture[] pictures) {
  if(pictures.length == 0) return;
  picture all;
  size(all,pictures[0]);
  for(picture pic : pictures)
    add(all,pic);
  rescale2(pictures,all);
  rescale3(pictures,all);
};

exitfcn currentexitfunction=atexit();

void exitfunction()
{
  if(currentexitfunction != null) currentexitfunction();
  if(!settings.keep)
    for(int i=0; i < file3.length; ++i)
      delete(file3[i]);
  file3=new string[];
}

atexit(exitfunction);
