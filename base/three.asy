import math;

triple O=(0,0,0);
triple X=(1,0,0), Y=(0,1,0), Z=(0,0,1);

real[] operator ecast(triple v)
{
  return new real[] {v.x, v.y, v.z, 1};
}

triple operator ecast(real[] a)
{
  if(a.length != 4) abort("vector length of "+(string) a.length+" != 4");
  if(a[3] == 0) abort("camera is too close to object");
  return (a[0],a[1],a[2])/a[3];
}

typedef real[][] transform3;

triple operator * (transform3 t, triple v)
{
  return (triple) (t*(real[]) v);
}

// Return the longitude of v, ignoring errors if v.x=v.y=0.
real Longitude(triple v) {
  if(v.x == 0 && v.y == 0) return 0;
  return longitude(v);
}

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

// A uniform scaling in 3D space.
transform3 scale3(real s)
{
  transform3 t=identity(4);
  t[0][0]=t[1][1]=t[2][2]=s;
  return t;
}

// A scaling in the x direction in 3D space.
transform3 xscale3(real s)
{
  transform3 t=identity(4);
  t[0][0]=s;
  return t;
}

// A scaling in the y direction in 3D space.
transform3 yscale3(real s)
{
  transform3 t=identity(4);
  t[1][1]=s;
  return t;
}

// A scaling in the z direction in 3D space.
transform3 zscale3(real s)
{
  transform3 t=identity(4);
  t[2][2]=s;
  return t;
}

// A transformation representing rotation by an angle in degrees about
// an axis v through the origin (in the right-handed direction).
transform3 rotate(real angle, triple v)
{
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
  return shift(u)*rotate(angle,v)*shift(-u);
}

transform3 reflect(triple u, triple v, triple w)
{
  triple normal=cross(v-u,w-u);
  if(normal == O)
    abort("points determining plane to reflect about cannot be colinear");
  transform3 basis=shift(u);
  if(normal.x != 0 || normal.y != 0)
    basis *= rotate(longitude(normal),Z)*rotate(colatitude(normal),Y);
  
  return basis*zscale3(-1)*inverse(basis);
}

// Transformation corresponding to moving the camera from the origin (looking
// down the negative z axis) to the point 'from' (looking at the origin).
// Since, in actuality, we are transforming the points instead of
// the camera, we calculate the inverse matrix.
transform3 lookAtOrigin(triple from)
{
  transform3 t=(from.x == 0 && from.y == 0) ? shift(-from) : 
    shift((0,0,-length(from)))*
    rotate(-90,Z)*
    rotate(-colatitude(from),Y)*
    rotate(-longitude(from),Z);
  return t;
}

transform3 lookAt(triple from, triple to)
{
  return lookAtOrigin(from-to)*shift(-to);
}

typedef pair project(triple v);

struct projection {
  public triple camera;
  public transform3 project;
  void init(triple camera, transform3 project) {
    this.camera=camera;
    this.project=project;
  }
}

projection operator init() {return new projection;}
  
// Uses the homogenous coordinate to perform perspective distortion.  When
// combined with a projection to the XY plane, this effectively maps
// points in three space to a plane at a distance d from the camera.
projection perspective(triple camera)
{
  transform3 t=identity(4);
  real d=length(camera);
  t[3][2]=-1/d;
  t[3][3]=0;
  projection P;
  P.init(camera,t*lookAtOrigin(camera));
  return P;
}

projection perspective(real x, real y, real z)
{
  return perspective((x,y,z));
}

projection orthographic(triple camera)
{
  projection P;
  P.init(camera,lookAtOrigin(camera));
  return P;
}

projection orthographic(real x, real y, real z)
{
  return orthographic((x,y,z));
}

projection oblique(real angle=45)
{
  transform3 t=identity(4);
  real c2=Cos(angle)^2;
  real s2=1-c2;
  t[0][2]=-c2;
  t[1][2]=-s2;
  t[2][2]=0;
  projection P;
  P.init((c2,s2,1),t);
  return P;
}

projection oblique=oblique();

public projection currentprojection=perspective(5,4,2);

pair projectXY(triple v)
{
  return (v.x,v.y);
}

project operator cast(transform3 t)
{
  return new pair(triple v) {
    return projectXY(t*v);
  };
}

struct control {
  public triple post,pre;
  public bool active=false;
  void init(triple post, triple pre) {
    this.post=post;
    this.pre=pre;
    active=true;
  }
}

control operator init() {return new control;}
control nocontrol;
  
control operator * (transform3 t, control c) 
{
  control C;
  C.post=t*c.post;
  C.pre=t*c.pre;
  C.active=c.active;
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
  public real out,in;
  public bool atLeast;
  public bool active=false;
  void init(real out, real in, bool atLeast) {
    this.out=out;
    this.in=in;
    this.atLeast=atLeast;
    active=true;
  }
}

Tension operator init() {return new Tension;}
Tension noTension;
noTension.in=noTension.out=1;
  
void write(file file, Tension t)
{
  write(file,"..tension ");
  if(t.atLeast) write(file,"atleast ");
  write(file,t.out);
  write(file," and ");
  write(file,t.in);
}
  
struct dir {
  public triple dir;
  public real gamma=1; // endpoint curl
  public bool Curl;    // curl specified
  bool active() {
    return dir != O || Curl;
  }
  void init(triple v) {
    this.dir=v;
  }
  void init(real gamma) {
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
    write(file,"{curl3 "); write(file,d.gamma); write(file,"}");
  }
}
  
dir operator init() {return new dir;}

transform3 shiftless(transform3 t)
{
  transform3 T=copy(t);
  T[0][3]=T[1][3]=T[2][3]=0;
  return T;
}

dir operator * (transform3 t, dir d) 
{
  dir D=d.copy();
  D.init(unit(shiftless(t)*d.dir));
  return D;
}

struct flatguide3 {
  public triple[] nodes;
  public bool[] cyclic;     // true if node is really a cycle
  public control[] control; // control points for segment starting at node
  public Tension[] Tension; // Tension parameters for segment starting at node
  public dir[] in,out;    // in and out directions for segment starting at node

  bool cyclic() {return cyclic[cyclic.length-1];}
  
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
      control c;
      c.init(post,pre);
      control[control.length-1]=c;
    }
  }

  void Tension(real out, real in, bool atLeast) {
    if(Tension.length > 0) {
      Tension t;
      t.init(out,in,atLeast);
      Tension[Tension.length-1]=t;
    }
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

  void cyclic() {
    node(nodes[0],true);
  }
  
  // Return true if outgoing direction at node i is known.
  bool solved(int i) {
    return out[i].active() || control[i].active;
  }
}

flatguide3 operator init() {return new flatguide3;}
  
void write(file file, explicit flatguide3 g)
{
  if(g.size() == 0) write("<nullpath>");
  else for(int i=0; i < g.nodes.length; ++i) {
    if(i > 0) write(file);
    if(g.cyclic[i]) write(file,"cycle3");
    else write(file,g.nodes[i]);

    if(g.control[i].active) // Explicit control points trump other specifiers
      write(file,g.control[i]);
    else {
      write(file,g.out[i]);
      if(g.Tension[i].active) write(file,g.Tension[i]);
    }
    if(i < g.nodes.length-1) write(file,"..");
    if(!g.control[i].active) write(file,g.in[i]);
  }
}
  
void write(file file=stdout, explicit flatguide3 x,
	   suffix s) {write(file,x); s(file);}
void write(explicit flatguide3 g) {write(stdout,g,endl);}

void write(file file, flatguide3[] g)
{
  if(g.length > 0) write(file,g[0]);
  for(int i=1; i < g.length; ++i) {
    write(file);
    write(file," ^^");
    write(file,g[i]);
  }
}

void write(file file=stdout, flatguide3[] x, suffix s)
{
  write(file,x); s(file);
}

// A guide3 is most easily represented as something that modifies a flatguide3.
typedef void guide3(flatguide3);

void nullpath(flatguide3) {};

guide3 operator init() {return nullpath;}

guide3 operator cast(triple v)
{
  return new void(flatguide3 f) {
    f.node(v);
  };
}

guide3[] operator cast(triple[] v)
{
  guide3[] g=new guide3[v.length];
  for(int i=0; i < v.length; ++i)
    g[i]=v[i];
  return g;
}

void cycle3(flatguide3 f)
{
  f.cyclic();
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

guide3 operator tension3(real out, real in, bool atLeast)
{
  return new void(flatguide3 f) {
    f.Tension(out,in,atLeast);
  };
};
  
guide3 operator tension3(real t, bool atLeast)
{
  return operator tension3(t,t,atLeast);
}

guide3 operator curl3(real gamma, int p)
{
  return new void(flatguide3 f) {
    if(p == 0) f.out(gamma);
    else if(p == 1) f.in(gamma);
  };
}
  
guide3 operator spec(triple v, int p)
{
  return new void(flatguide3 f) {
    if(p == 0) f.out(v);
    else if(p == 1) f.in(v);
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
  guide3 g;
  for(int i=0; i < a.length; ++i)
    g=g..operator tension3(1,true)..a[i];
  return g;
}

guide3 operator ---(... guide3[] a)
{
  guide3 g;
  for(int i=0; i < a.length; ++i)
    g=g..operator tension3(infinity,true)..a[i];
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

guide3 operator * (transform3 t, guide3 g) 
{
  return new void(flatguide3 f) {
    g(f);
    for(int i=0; i < f.nodes.length; ++i) {
      f.nodes[i]=t*f.nodes[i];
      f.cyclic[i]=f.cyclic[i];
      f.control[i]=t*f.control[i];
      f.Tension[i]=f.Tension[i];
      f.in[i]=t*f.in[i];
      f.out[i]=t*f.out[i];
    }
  };
}

guide3[] operator * (transform3 t, guide3[] g) 
{
  guide3[] G=new guide3[g.length];
  for(int i=0; i < g.length; ++i)
    G[i]=t*g[i];
  return G;
}

// A version of acos that tolerates numerical imprecision
real acos1(real x) {
  if(x < -1) x=-1;
  if(x > 1) x=1;
  return acos(x);
}
  
struct Controls {
  triple c0,c1;

// 3D extension of John Hobby's control point formula
//  (The MetaFont Book, page 131).
  void init(triple v0, triple v1, triple d0, triple d1, real tout, real tin,
	    bool atLeast) {
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

Controls operator init() {return new Controls;}
  
private triple cross(triple d0, triple d1, triple camera)
{
  triple normal=cross(d0,d1);
  return normal == O ? camera : normal;
}
					
private triple dir(real theta, triple d0, triple d1, triple camera)
{
  triple normal=cross(d0,d1,camera);
  return rotate(degrees(theta),dot(normal,camera) >= 0 ? normal : -normal)*d1;
}

private real angle(triple d0, triple d1, triple camera)
{
  real theta=acos1(dot(unit(d0),unit(d1)));
  return dot(cross(d0,d1,camera),camera) >= 0 ? theta : -theta;
}

// 3D extension of John Hobby's angle formula (The MetaFont Book, page 131).
// Notational differences: here psi[i] is the turning angle at z[i+1],
// beta[i] is the tension for segment i, and in[i] is the incoming
// direction for segment i (where segment i begins at node i).

real[] theta(triple[] v, real[] alpha, real[] beta, 
	     triple dir0, triple dirn, real g0, real gn, triple camera)
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
    psi[i]=angle(v[i+1]-v[i],v[i+2]-v[i+1],camera);
  if(cyclic) {
    l.cyclic(true);
    psi.cyclic(true);
  } else {
    psi[n-1]=0;
    if(dir0 == O) {
      real a0=alpha[0];
      real b0=beta[0];
      a[0]=0;
      b[0]=g0*b0^3+a0^3*(3b0-1);
      real C=g0*b0^3*(3a0-1)+a0^3;
      c[0]=C;
      f[0]=-C*psi[0];
    } else {
      a[0]=c[0]=0;
      b[0]=1;
      f[0]=angle(v[1]-v[0],dir0,camera);
    }
    if(dirn == O) {
      real an=alpha[n-1];
      real bn=beta[n-1];
      a[n]=bn^3+gn*an^3*(3bn-1);
      real C=bn^3*(3an-1)+gn*an^3;
      b[n]=C;
      c[n]=f[n]=0;
    } else {
      a[n]=c[n]=0;
      b[n]=1;
      f[n]=angle(v[n]-v[n-1],dirn,camera);
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

// Fill in missing directions for n cyclic nodes.
void aim(flatguide3 g, int N, triple camera) 
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
  real[] theta=theta(v,alpha,beta,g.out[start].dir,g.in[final].dir,
		     g.out[start].gamma,g.in[final].gamma,camera);
  v.cyclic(true);
  theta.cyclic(true);
    
  for(int k=1; k < (cyclic ? n+1 : n); ++k) {
    triple w=dir(theta[k],v[k]-v[k-1],v[k+1]-v[k],camera);
    g.in[(start+k-1) % N].init(w);
    g.out[(start+k) % N].init(w);
  }
  if(g.out[start].dir == O)
    g.out[start].init(dir(theta[0],v[0]-g.nodes[(start-1) % N],v[1]-v[0],
			  camera));
  if(g.in[final].dir == O)
    g.in[final].init(dir(theta[n],v[n-1]-v[n-2],v[n]-v[n-1],camera));
}

// Fill in missing directions for the sequence of nodes i...n.
void aim(flatguide3 g, int i, int n, triple camera) 
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
    real[] theta=theta(v,alpha,beta,g.out[i].dir,g.in[n-1].dir,
		       g.out[i].gamma,g.in[n-1].gamma,camera);
    
    for(int k=1; k < j; ++k) {
      triple w=dir(theta[k],v[k]-v[k-1],v[k+1]-v[k],camera);
      g.in[i+k-1].init(w);
      g.out[i+k].init(w);
    }
    if(g.out[i].dir == O) {
      triple w=dir(theta[0],g.in[i].dir,v[1]-v[0],camera);
      if(i > 0) g.in[i-1].init(w);
      g.out[i].init(w);
     }
    if(g.in[n-1].dir == O) {
      triple w=dir(theta[j],g.out[n-1].dir,v[j]-v[j-1],camera);
      g.in[n-1].init(w);
      g.out[n].init(w);
    }
  }
}

struct node {
  public triple pre,point,post;
  public bool straight;
  node copy() {
    node n=new node;
    n.pre=pre;
    n.point=point;
    n.post=post;
    n.straight=straight;
    return n;
  }
}
  
triple split(real t, triple x, triple y) { return x+(y-x)*t; }

void splitCubic(node[] sn, real t, node left_, node right_)
{
  node left=sn[0]=left_.copy(), mid=sn[1], right=sn[2]=right_.copy();
  triple x=split(t,left.post,right.pre);
  left.post=split(t,left.point,left.post);
  right.pre=split(t,right.pre,right.point);
  mid.pre=split(t,left.post,x);
  mid.post=split(t,x,right.pre);
  mid.point=split(t,mid.pre,mid.post);
}

node[] nodes(int n)
{
  node[] nodes=new node[n];
  for(int i=0; i < n; ++i)
    nodes[i]=new node;
  return nodes;
}

struct bbox3 {
  bool empty=true;
  real left,bottom,lower;
  real right,top,upper;
  
  void add(triple v) {
    real x=v.x; 
    real y=v.y;
    real z=v.z;
    
    if(empty) {
      left=right=x;
      bottom=top=y;
      lower=upper=z;
      empty=false;
    } else {
      if (x < left)
	left = x;  
      if (x > right)
	right = x;  
      if (y < bottom)
	bottom = y;
      if (y > top)
	top = y;
      if (z < lower)
	lower = z;
      if (z > upper)
	upper = z;
    }
  }

  triple Min() {
    return (left,bottom,lower);
  }
  
  triple Max() {
    return (right,top,upper);
  }
}

bbox3 operator init() {return new bbox3;}
  
struct path3 {
  node[] nodes;
  bool cycles;
  int n;
  real cached_length=-1;
  bbox3 box;
  
  static path3 path3(node[] nodes, bool cycles=false, real cached_length=-1) {
    path3 p=new path3;
    for(int i=0; i < nodes.length; ++i)
      p.nodes[i]=nodes[i].copy();
    p.cycles=cycles;
    p.cached_length=cached_length;
    p.n=cycles ? nodes.length-1 : nodes.length;
    return p;
  }
  
  static path3 path3(triple v) {
    node node;
    node.pre=node.point=node.post=v;
    node.straight=false;
    return path3(new node[] {node});
  }
  
  static path3 path3(node n0, node n1) {
    node N0,N1;
    N0 = n0.copy();
    N1 = n1.copy();
    N0.pre = N0.point;
    N1.post = N1.point;
    return path3(new node[] {N0,N1});
  }
  
  int size() {return n;}
  int length() {return nodes.length-1;}
  bool empty() {return n == 0;}
  bool cyclic() {return cycles;}
  
  void emptyError() {
    if(empty())
      abort("nullpath has no points");
  }
  
  bool straight(int i) {
    if (cycles) return nodes[i % n].straight;
    return (i < n) ? nodes[i].straight : false;
  }
  
  triple point(int i) {
    emptyError();
    
    if (cycles)
      return nodes[i % n].point;
    else if (i < 0)
      return nodes[0].point;
    else if (i >= n)
      return nodes[n-1].point;
    else
      return nodes[i].point;
  }

  triple precontrol(int i) {
    emptyError();
		       
    if (cycles)
      return nodes[i % n].pre;
    else if (i < 0)
      return nodes[0].pre;
    else if (i >= n)
      return nodes[n-1].pre;
    else
      return nodes[i].pre;
  }

  triple postcontrol(int i)
  {
    emptyError();
		       
    if (cycles)
      return nodes[i % n].post;
    else if (i < 0)
      return nodes[0].post;
    else if (i >= n)
      return nodes[n-1].post;
    else
      return nodes[i].post;
  }

  triple point(real t)
  {
    emptyError();
    
    int i = floor(t);
    int iplus;
    t = fmod(t,1);
    if (t < 0) t += 1;

    if (cycles) {
      i = i % n;
      iplus = (i+1) % n;
    }
    else if (i < 0)
      return nodes[0].point;
    else if (i >= n-1)
      return nodes[n-1].point;
    else
      iplus = i+1;

    real one_t = 1.0-t;

    triple a = nodes[i].point,
      b = nodes[i].post,
      c = nodes[iplus].pre,
      d = nodes[iplus].point,
      ab   = one_t*a   + t*b,
      bc   = one_t*b   + t*c,
      cd   = one_t*c   + t*d,
      abc  = one_t*ab  + t*bc,
      bcd  = one_t*bc  + t*cd,
      abcd = one_t*abc + t*bcd;

    return abcd;
  }
  
  triple precontrol(real t) {
    emptyError();
		     
    int i = floor(t);
    int iplus;
    t = fmod(t,1);
    if (t < 0) t += 1;

    if (cycles) {
      i = i % n;
      iplus = (i+1) % n;
    }
    else if (i < 0)
      return nodes[0].pre;
    else if (i >= n-1)
      return nodes[n-1].pre;
    else
      iplus = i+1;

    real one_t = 1.0-t;

    triple a = nodes[i].point,
      b = nodes[i].post,
      c = nodes[iplus].pre,
      ab   = one_t*a   + t*b,
      bc   = one_t*b   + t*c,
      abc  = one_t*ab  + t*bc;

    return abc;
  }
        
 
  triple postcontrol(real t) {
    emptyError();
  
    // NOTE: may be better methods, but let's not split hairs, yet.
    int i = floor(t);
    int iplus;
    t = fmod(t,1);
    if (t < 0) t += 1;

    if (cycles) {
      i = i % n;
      iplus = (i+1) % n;
    }
    else if (i < 0)
      return nodes[0].post;
    else if (i >= n-1)
      return nodes[n-1].post;
    else
      iplus = i+1;

    real one_t = 1.0-t;

    triple b = nodes[i].post,
      c = nodes[iplus].pre,
      d = nodes[iplus].point,
      bc   = one_t*b   + t*c,
      cd   = one_t*c   + t*d,
      bcd  = one_t*bc  + t*cd;

    return bcd;
  }

  triple dir(int i)
  {
    return unit(postcontrol(i) - precontrol(i));
  }

  triple dir(real t)
  {
    return unit(postcontrol(t) - precontrol(t));
  }

  path3 concat(path3 p1, path3 p2)
  {
    int n1 = p1.length(), n2 = p2.length();

    if (n1 == 0) return p2;
    if (n2 == 0) return p1;
    if (p1.point(n1) != p2.point(0))
      abort("path3 arguments in concatenation do not meet");

    node[] nodes = nodes(n1+n2+1);

    int i = 0;
    nodes[0].pre = p1.point(0);
    for (int j = 0; j < n1; ++j) {
      nodes[i].point = p1.point(j);
      nodes[i].straight = p1.straight(j);
      nodes[i].post = p1.postcontrol(j);
      nodes[i+1].pre = p1.precontrol(j+1);
      ++i;
    }
    for (int j = 0; j < n2; ++j) {
      nodes[i].point = p2.point(j);
      nodes[i].straight = p2.straight(j);
      nodes[i].post = p2.postcontrol(j);
      nodes[i+1].pre = p2.precontrol(j+1);
      ++i;
    }
    nodes[i].point = nodes[i].post = p2.point(n2);

    return path3(nodes);
  }

  real arclength() {
    if(cached_length != -1) return cached_length;
    
    real L=0.0;
    for(int i = 0; i < n-1; ++i)
      L += cubiclength(nodes[i].point,nodes[i].post,nodes[i+1].pre,
		       nodes[i+1].point,-1);

    if(cycles) L += cubiclength(nodes[n-1].point,nodes[n-1].post,
				nodes[n].pre,nodes[n].point,-1);
    cached_length = L;
    return L;
  }
  
  path3 reverse() {
    node[] nodes=nodes(nodes.length);
    for(int i=0, j=length(); i < nodes.length; ++i, --j) {
      nodes[i].pre = postcontrol(j);
      nodes[i].point = point(j);
      nodes[i].post = precontrol(j);
      nodes[i].straight = straight(j);
    }
    return path3(nodes,cycles,cached_length);
  }
  
  real arctime(real goal) {
    if(cycles) {
      if(goal == 0) return 0;
      if(goal < 0)  {
	path3 rp = reverse();
	return -rp.arctime(-goal);
      }
      if(cached_length > 0 && goal >= cached_length) {
	int loops = (int)(goal / cached_length);
	goal -= loops*cached_length;
	return loops*n+arctime(goal);
      }      
    } else {
      if(goal <= 0)
	return 0;
      if(cached_length > 0 && goal >= cached_length)
	return n-1;
    }
    
    real l,L=0;
    for(int i = 0; i < n-1; ++i) {
      l = cubiclength(nodes[i].point,nodes[i].post,nodes[i+1].pre,
		      nodes[i+1].point,goal);
      if(l < 0)
	return (-l+i);
      else {
	L += l;
	goal -= l;
	if (goal <= 0)
	  return i+1;
      }
    }
    if(cycles) {
      l = cubiclength(nodes[n-1].point,nodes[n-1].post,nodes[n].pre,
		      nodes[n].point,goal);
      if(l < 0)
	return -l+n-1;
      if(cached_length > 0 && cached_length != L+l) {
	abort("arclength != length");
      }
      cached_length = L += l;
      goal -= l;
      return arctime(goal)+n;
    }
    else {
      cached_length = L;
      return nodes.length-1;
    }
  }
  
  path3 subpath(int start, int end) {
    if(empty()) return new path3;

    if (start > end) {
      path3 rp = reverse();
      path3 result = rp.subpath(length()-start, length()-end);
      return result;
    }

    if (!cycles) {
      if (start < 0)
	start = 0;
      if (end > n-1)
	end = n-1;
    }

    int sn = end-start+1;
    node[] nodes=nodes(sn);
    for (int i = 0, j = start; j <= end; ++i, ++j) {
      nodes[i].pre = precontrol(j);
      nodes[i].point = point(j);
      nodes[i].post = postcontrol(j);
      nodes[i].straight = straight(j);
    }
    nodes[0].pre = nodes[0].point;
    nodes[sn-1].post = nodes[sn-1].point;

    return path3(nodes);
  }
  
  path3 subpath(real start, real end)
  {
    if(empty()) return new path3;
  
    if (start > end)
      return reverse().subpath(length()-start, length()-end);

    node startL, startR, endL, endR;
    if (!cycles) {
      if (start < 0)
	start = 0;
      if (end > n-1)
	end = n-1;
      startL = nodes[floor(start)];
      startR = nodes[ceil(start)];
      endL = nodes[floor(end)];
      endR = nodes[ceil(end)];
    } else {
      startL = nodes[floor(start) % n];
      startR = nodes[ceil(start) % n];
      endL = nodes[floor(end) % n];
      endR = nodes[ceil(end) % n];
    }

    if (start == end) {
      return path3(point(start));
    }
    
    node[] sn=nodes(3);
    path3 p = subpath(ceil(start), floor(end));
    if (start > floor(start)) {
      if (end < ceil(start)) {
	splitCubic(sn,start-floor(start),startL,startR);
	splitCubic(sn,(end-start)/(ceil(end)-start),sn[1],sn[2]);
	return path3(sn[0],sn[1]);
      }
      splitCubic(sn,start-floor(start),startL,startR);
      p=concat(path3(sn[1],sn[2]),p);
    }
    if (ceil(end) > end) {
      splitCubic(sn,end-floor(end),endL,endR);
      p=concat(p,path3(sn[0],sn[1]));
    }
    return p;
  }
  
  static pair intersectcubics(node left1, node right1,
			      node left2, node right2,
			      int depth = 48) {
    pair F=(-1,-1);

    node left1=left1.copy();
    node right1=right1.copy();
    node left2=left2.copy();
    node right2=right2.copy();
    
    bbox3 box1, box2;
    box1.add(left1.point); box1.add(left1.post);
    box1.add(right1.pre);  box1.add(right1.point);
    box2.add(left2.point); box2.add(left2.post);
    box2.add(right2.pre);  box2.add(right2.point);
    if (box1.Max().x >= box2.Min().x &&
	box1.Max().y >= box2.Min().y &&
	box1.Max().z >= box2.Min().z &&
	box2.Max().x >= box1.Min().x &&
	box2.Max().y >= box1.Min().y &&
	box2.Max().z >= box1.Min().z
	) {
      if (depth == 0) return (0,0);
      node[] sn1=nodes(3), sn2=nodes(3);
      splitCubic(sn1,0.5,left1,right1);
      splitCubic(sn2,0.5,left2,right2);
      pair t;
      --depth;
      if ((t=intersectcubics(sn1[0],sn1[1],sn2[0],sn2[1],depth)) != F)
	return t*0.5;
      if ((t=intersectcubics(sn1[0],sn1[1],sn2[1],sn2[2],depth)) != F)
	return t*0.5+(0,1);
      if ((t=intersectcubics(sn1[1],sn1[2],sn2[0],sn2[1],depth)) != F)
	return t*0.5+(1,0);
      if ((t=intersectcubics(sn1[1],sn1[2],sn2[1],sn2[2],depth)) != F)
	return t*0.5+(1,1);
    }
    return F;
  }

  static pair intersect(path3 p1, path3 p2) {
    pair F=(-1,-1);
    for (int i = 0; i < p1.length(); ++i) {
      for (int j = 0; j < p2.length(); ++j) {
	pair t=intersectcubics(p1.nodes[i],(p1.cycles && i == p1.n-1) ?
			       p1.nodes[0] : p1.nodes[i+1],
			       p2.nodes[j],(p2.cycles && j == p2.n-1) ?
			       p2.nodes[0] : p2.nodes[j+1]);
	if (t != F) return t*0.5 + (i,j);
      }
    }
    return F;  
  }

  bbox3 bounds() {
    if (empty()) {
      // No bounds
      return new bbox3;
    }

    if(!box.empty) return box;
    
    for (int i = 0; i < length(); ++i) {
      box.add(point(i));
      if(straight(i)) continue;
    
      triple z0=point(i);
      triple z0p=postcontrol(i);
      triple z1m=precontrol(i+1);
      triple z1=point(i+1);
      
      triple a=z1-z0+3.0*(z0p-z1m);
      triple b=2.0*(z0+z1m)-4.0*z0p;
      triple c=z0p-z0;
      
      quad ret;
    
      // Check x coordinate
      ret=solveQuadratic(a.x,b.x,c.x);
      if(ret.roots != quad.NONE) box.add(point(i+ret.x1));
      if(ret.roots == quad.DOUBLE) box.add(point(i+ret.x2));
    
      // Check y coordinate
      ret=solveQuadratic(a.y,b.y,c.y);
      if(ret.roots != quad.NONE) box.add(point(i+ret.x1));
      if(ret.roots == quad.DOUBLE) box.add(point(i+ret.x2));
    }
    box.add(point(length()));
    return box;
  }
  
  triple max() {
    return bounds().Max();
  }
  triple min() {
    return bounds().Min();
  }
  
}

path3 operator init() {return new path3;}
  
bool cyclic(explicit path3 p) {return p.cyclic();}
int size(explicit path3 p) {return p.size();}
int length(explicit path3 p) {return p.length();}

path3 operator * (transform3 t, path3 p) 
{
  int m=p.nodes.length;
  node[] nodes=nodes(m);
  for(int i=0; i < m; ++i) {
    nodes[i].pre=t*p.nodes[i].pre;
    nodes[i].point=t*p.nodes[i].point;
    nodes[i].post=t*p.nodes[i].post;
  }
  return path3.path3(nodes,p.cycles);
}

void write(file file, path3 p)
{
  if(size(p) == 0) write("<nullpath>");
  else for(int i=0; i < p.nodes.length; ++i) {
    if(i == p.nodes.length-1 && p.cycles) write(file,"cycle3");
    else write(file,p.nodes[i].point,endl);
    if(i < length(p)) {
      if(p.nodes[i].straight) write(file,"--");
      else {
	write(file,".. controls ");
	write(file,p.nodes[i].post);
	write(file," and ");
	write(file,p.nodes[i+1].pre);
	write(file,"..",endl);
      }
    }
  }
}
  
void write(file file=stdout, path3 x, suffix s) {write(file,x); s(file);}
void write(path3 g) {write(stdout,g,endl);}

path3 solve(flatguide3 g, projection Q=currentprojection)
{
  project P=Q.project;
  int n=g.nodes.length-1;
  path3 p;

  // If duplicate points occur consecutively, add dummy controls (if absent).
  for(int i=1; i < n; ++i) {
    if(g.nodes[i] == g.nodes[i+1] && !g.control[i].active) {
      control c;
      c.init(g.nodes[i],g.nodes[i]);
      g.control[i]=c;
    }
  }  
  
  // Fill in empty direction specifiers inherited from explicit control points.
  for(int i=0; i < n; ++i) {
    if(g.control[i].active) {
      g.out[i].default(g.control[i].post-g.nodes[i]);
      g.in[i].default(g.nodes[i+1]-g.control[i].pre);
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
    aim(g,i,Q.camera);
    // All other cycles can now be reduced to sequences.
    triple v=g.out[0].dir;
    for(int j=i; j <= n; ++j) {
      if(g.cyclic[j]) {
	g.in[j-1].default(v);
	g.out[j].default(v);
	if(g.nodes[j-1] == g.nodes[j] && !g.control[j-1].active) {
	  control c;
	  c.init(g.nodes[j-1],g.nodes[j-1]);
	  g.control[j-1]=c;
	}
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
      aim(g,start,i,Q.camera);
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
	c.init(g.nodes[i]+delta,g.nodes[i+1]-delta);
	c.active=false;
      } else {
	Controls C;
	C.init(g.nodes[i],g.nodes[next],g.out[i].dir,g.in[i].dir,
	       g.Tension[i].out,g.Tension[i].in,g.Tension[i].atLeast);
	c.init(C.c0,C.c1);
      }
      g.control[i]=c;
    }
  }
  
  // Convert to Knuth's format (control points stored with nodes)
  node[] nodes=nodes(g.nodes.length);
  bool cyclic=g.cyclic[g.cyclic.length-1];
  if(cyclic) nodes[0].pre=g.control[nodes.length-2].pre;
  for(int i=0; i < g.nodes.length-1; ++i) {
    nodes[i].point=g.nodes[i];
    nodes[i].post=g.control[i].post;
    nodes[i+1].pre=g.control[i].pre;
    nodes[i].straight=!g.control[i].active; // TODO: test control points here
  }
  nodes[g.nodes.length-1].point=g.nodes[g.nodes.length-1];
  nodes[g.nodes.length-1].post=g.control[g.nodes.length-1].post;
  
  return path3.path3(nodes,cyclic);
}

bool cyclic(explicit flatguide3 g) {return g.cyclic[g.cyclic.length-1];}
int size(explicit flatguide3 g) {
  return cyclic(g) ? g.nodes.length-1 : g.nodes.length;
}
int length(explicit flatguide3 g) {return g.nodes.length-1;}

path project(explicit path3 p, projection Q=currentprojection)
{
  guide g;
  project P=Q.project;
  
  int last=p.nodes.length-1;
  if(last < 0) return g;
  
  g=P(p.nodes[0].point);
  // Construct the path.
  for(int i=0; i < last; ++i) {
    if(p.nodes[i].straight)
      g=g--P(p.nodes[i+1].point);
    else 
      g=g..controls P(p.nodes[i].post) and P(p.nodes[i+1].pre)..
      P(p.nodes[i+1].point);
  }
  
  if(p.cycles)
    g=p.nodes[last].straight ? g--cycle : g..cycle;

  return g;
}

path project(flatguide3 g, projection P=currentprojection)
{
  return project(solve(g,P),P);
}

pair project(triple v, projection P=currentprojection)
{
  project P=P.project;
  return P(v);
}

path[] project(flatguide3[] g, projection P=currentprojection)
{
  path[] p=new path[g.length];
  for(int i=0; i < g.length; ++i) 
    p[i]=project(g[i],P);
  return p;
}
  
guide3 operator cast(path3 p) {
  guide3 g;
  int last=p.nodes.length-1;
  if(last < 0) return g;
  
  int i,stop=(p.cycles ? last-1 : last);
  // Construct the path.
  g=p.nodes[0].point;
  for(i=0; i < stop; ++i) {
    if(p.nodes[i].straight) g=g--p.nodes[i+1].point;
    else g=g..controls p.nodes[i].post and p.nodes[i+1].pre..
      p.nodes[i+1].point;
  }
  
  if(p.cycles) {
    if(p.nodes[i].straight) g=g--cycle3;
    else g=g..controls p.nodes[i].post and p.nodes[i+1].pre..cycle3;
  }
  
  return g;
}

pair operator cast(triple v) {return project(v);}
pair[] operator cast(triple[] v) {
  int n=v.length;
  pair[] z=new pair[n];
  for(int i=0; i < n; ++i)
    z[i]=project(v[i]);
  return z;
}

path3 operator cast(guide3 g) {return solve(g);}
path operator cast(path3 p) {return project(p);}
path operator cast(triple v) {return project(v);}
path operator cast(guide3 g) {return project(solve(g));}

path[] operator cast(path3 g) {return new path[] {(path) g};}
path[] operator cast(guide3 g) {return new path[] {(path) g};}
path[] operator cast(guide3[] g) {return project(g);}

bool straight(path3 p, int i) {return p.straight(i);}
bool straight(explicit guide3 g, int i) {return ((path3) g).straight(i);}

triple point(path3 p, int i) {return p.point(i);}
triple point(explicit guide3 g, int i) {return ((path3) g).point(i);}
triple point(path3 p, real t) {return p.point(t);}
triple point(explicit guide3 g, real t) {return ((path3) g).point(t);}

triple postcontrol(path3 p, int i) {return p.postcontrol(i);}
triple postcontrol(explicit guide3 g, int i) {
  return ((path3) g).postcontrol(i);
}
triple postcontrol(path3 p, real t) {return p.postcontrol(t);}
triple postcontrol(explicit guide3 g, real t) {
  return ((path3) g).postcontrol(t);
}

triple precontrol(path3 p, int i) {return p.precontrol(i);}
triple precontrol(explicit guide3 g, int i) {
  return ((path3) g).precontrol(i);
}
triple precontrol(path3 p, real t) {return p.precontrol(t);}
triple precontrol(explicit guide3 g, real t) {
  return ((path3) g).precontrol(t);
}

triple dir(path3 p, int n) {return p.dir(n);}
triple dir(explicit guide3 g, int n) {return ((path3) g).dir(n);}
triple dir(path3 p, real t) {return p.dir(t);}
triple dir(explicit guide3 g, real t) {return ((path3) g).dir(t);}

path3 reverse(path3 p) {return p.reverse();}
path3 reverse(explicit guide3 g) {return ((path3) g).reverse();}

real arclength(path3 p) {return p.arclength();}
real arclength(explicit guide3 g) {return ((path3) g).arclength();}

real arctime(path3 p, real l) {return p.arctime(l);}
real arctime(explicit guide3 g, real l) {return ((path3) g).arctime(l);}

triple max(path3 p) {return p.max();}
triple max(explicit guide3 g) {return ((path3) g).max();}

triple min(path3 p) {return p.min();}
triple min(explicit guide3 g) {return ((path3) g).min();}

path3 subpath(path3 p, int start, int end) {return p.subpath(start,end);}
path3 subpath(explicit guide3 g, int start, int end)
{
  return ((path3) g).subpath(start,end);
}

path3 subpath(path3 p, real start, real end) {return p.subpath(start,end);}
path3 subpath(explicit guide3 g, real start, real end) 
{
  return ((path3) g).subpath(start,end);
}

pair intersect(path3 p, path3 q) {return path3.intersect(p,q);}
pair intersect(explicit guide3 p, explicit guide3 q)
{
  return path3.intersect((path3) p,(path3) q);
}

path3 operator & (path3 p, path3 q) {return p.concat(p,q);}
path3 operator & (explicit guide3 p, explicit guide3 q)
{
  return ((path3) p).concat(p,q);
}

void draw(frame f, guide3[] g, pen p=currentpen)
{
  draw(f,(path[]) g,p);
}

void draw(picture pic=currentpicture, guide3[] g, pen p=currentpen)
{
  draw(pic,(path[]) g,p);
}

guide3[] operator ^^ (guide3 p, guide3 q) 
{
  return new guide3[] {p,q};
}

guide3[] operator ^^ (guide3 p, guide3[] q) 
{
  return concat(new guide3[] {p},q);
}

guide3[] operator ^^ (guide3[] p, guide3 q) 
{
  return concat(p,new guide3[] {q});
}

guide3[] operator ^^ (guide3[] p, guide3[] q) 
{
  return concat(p,q);
}

guide3[] box(triple v1, triple v2)
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

guide3[] unitcube=box((0,0,0),(1,1,1));

path3 unitcircle3=X..Y..-X..-Y..cycle3;

path3 circle(triple c, real r, triple normal=Z)
{
  path3 p=shift(c)*scale3(r)*unitcircle3;
  if(normal != Z) p=rotate(longitude(normal),Z)*rotate(colatitude(normal),Y)*p;
  return p;
}

// return an arc centered at c with radius r from c+r*dir(theta1,phi1) to
// c+r*dir(theta2,phi2) in degrees, drawing in the given direction
// relative to the normal vector cross(dir(theta1,phi1),dir(theta2,phi2)).
// The normal must be explicitly specified if c and the endpoints are colinear.
path3 arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
	  triple normal=O, bool direction)
{
  if(normal == O) {
    normal=cross(dir(theta1,phi1),dir(theta2,phi2));
    if(normal == O) abort("explicit normal required for these endpoints");
  }
  transform3 T=identity(4);
  if(normal.x != 0 || normal.y != 0)
    T=rotate(colatitude(normal),cross(Z,normal));
  transform3 Tinv=inverse(T);
  triple v1=Tinv*dir(theta1,phi1);
  triple v2=Tinv*dir(theta2,phi2);
  real t1=intersect(unitcircle3,O--2*(v1.x,v1.y,0)).x;
  real t2=intersect(unitcircle3,O--2*(v2.x,v2.y,0)).x;
  int n=length(unitcircle3);
  if(t1 >= t2 && direction) t1 -= n;
  if(t2 >= t1 && !direction) t2 -= n;
  return shift(c)*scale3(r)*T*subpath(unitcircle3,t1,t2);
}

// return an arc centered at c with radius r from c+r*dir(theta1,phi1) to
// c+r*dir(theta2,phi2) in degrees, drawing drawing counterclockwise
// relative to the normal vector cross(dir(theta1,phi1),dir(theta2,phi2))
// iff theta2 > theta1 or (theta2 == theta1 and phi2 >= phi1).
// The normal must be explicitly specified if c and the endpoints are colinear.
// If r < 0, draw the complementary arc of radius |r|.
path3 arc(triple c, real r, real theta1, real phi1, real theta2, real phi2,
	  triple normal=O)
{
  bool pos=theta2 > theta1 || (theta2 == theta1 && phi2 >= phi1);
  if(r > 0) return arc(c,r,theta1,phi1,theta2,phi2,normal,pos ? CCW : CW);
  else return arc(c,-r,theta1,phi1,theta2,phi2,normal,pos ? CW : CCW);
}

// return an arc centered at c from triple v1 to v2 (assuming |v2-c|=|v1-c|),
// drawing in the given direction.
path3 arc(triple c, triple v1, triple v2, triple normal=O, bool direction=CCW)
{
  v1 -= c; v2 -= c;
  return arc(c,abs(v1),colatitude(v1),Longitude(v1),
	     colatitude(v2),Longitude(v2),direction);
}

static public real epsilon=1000*realEpsilon();

// Return a representation of the plane passing through v1, v2, and v3.
path3 plane(triple v1, triple v2, triple v3)
{
  return v1--v2--v3--(v3+v1-v2)--cycle3;
}

triple normal(path3 p, triple f(path3, int), triple normal=O) {
  int n=size(p);
  for(int i=0; i < size(p)-1; ++i) {
    triple point=point(p,i);
    triple v1=f(p,i)-point;
    triple v2=f(p,i+1)-point;
    triple n=cross(unit(v1),unit(v2));
    if(abs(n) > epsilon) {
      n=unit(n);
      if(normal != O && abs(normal-n) > epsilon) abort("path is not planar");
      normal=n;
    }
  }
  return normal;
}
  
// Return the unit normal vector to a planar path p.
triple normal(path3 p) {
  triple normal=normal(p,precontrol);
  normal=normal(p,postcontrol,-normal);
  if(normal == O) abort("path is straight");
  return normal;
}

// Routines for hidden surface removal (via binary space partition):
// Structure face is derived from picture.
struct face {
  picture pic;
  public transform t;
  public frame fit;
  public triple normal,point;
  static face face(path3 p) {
    face f=new face;
    f.normal=normal(p);
    f.point=point(p,0);
    return f;
  }
  face copy() {
    face f=new face;
    f.pic=pic.copy();
    f.t=t;
    f.normal=normal;
    f.point=point;
    add(f.fit,fit);
    return f;
  }
}

face operator init() {return new face;}

picture operator cast(face f) {return f.pic;}
face operator cast(path3 p) {return face.face(p);}
  
struct line {
  public triple point;
  public triple dir;
}

line operator init() {return new line;}
  
line intersection(face a, face b) 
{
  line L;
  L.point=intersectionpoint(a.normal,a.point,b.normal,b.point);
  L.dir=unit(cross(a.normal,b.normal));
  return L;
}

struct half {
  pair[] left,right;
  
// Sort the points in the pair array z according to whether they lie on the
// left or right side of the line L in the direction dir passing through P.
// Points exactly on the L are considered to be on the right side.
// Also push any points of intersection of L with the path operator --(... z)
// onto each of the arrays left and right. 
  static half split(pair dir, pair P ... pair[] z) {
    half h=new half;
    pair lastz,invdir=1.0/dir;
    bool left,last;
    for(int i=0; i < z.length; ++i) {
      left=(invdir*z[i]).y > (invdir*P).y;
      if(i > 0 && last != left) {
	pair w=extension(P,P+dir,lastz,z[i]);
	h.left.push(w);
	h.right.push(w);
      }
      if(left) h.left.push(z[i]);
      else h.right.push(z[i]);
      last=left;
      lastz=z[i];
    }
    return h;  
  }
}
  
half operator init() {return new half;}

struct splitface {
  public face back,front;
}

splitface operator init() {return new splitface;}
  
// Return the pieces obtained by splitting face a by face cut.
splitface split(face a, face cut, projection P)
{
  splitface S;
  triple camera=P.camera;

  if(abs(a.normal-cut.normal) < epsilon ||
     abs(a.normal+cut.normal) < epsilon) {
    if(abs(dot(a.point-camera,a.normal)) > 
       abs(dot(cut.point-camera,cut.normal))) {
      S.back=a;
      S.front=null;
    } else {
      S.back=null;
      S.front=a;
    }
    return S;
  }
  
  line L=intersection(a,cut);
  pair point=a.t*project(L.point,P);
  pair dir=a.t*project(L.point+L.dir,P)-point;
  pair invdir=1.0/L.dir;
  triple apoint=L.point+cross(L.dir,a.normal);
  bool left=(invdir*(a.t*project(apoint,P))).y >= (invdir*point).y;
  real t=intersection(apoint,camera,cut.normal,cut.point);
  bool rightfront=left ^ (t <= 0 || t >= 1);
  
  face back,front;
  back=a;
  front=a.copy();
  pair max=max(a.fit);
  pair min=min(a.fit);
  half h=half.split(dir,point,max,(min.x,max.y),min,(max.x,min.y),max);
  if(h.right.length == 0) {
    if(rightfront) front=null;
    else back=null;
  } else if(h.left.length == 0) {
    if(rightfront) back=null;
    else front=null;
  }
  if(front != null)
    clip(front.fit,operator --(... rightfront ? h.right : h.left)--cycle);
  if(back != null)
    clip(back.fit,operator --(... rightfront ? h.left : h.right)--cycle);
  S.back=back;
  S.front=front;
  return S;
}

// A binary space partition
struct bsp
{
  bsp back;
  bsp front;
  face node;
  
  // Construct the bsp.
  static bsp split(face[] faces, projection P) {
    if(faces.length == 0) return null;
    bsp bsp=new bsp;
    bsp.node=faces.pop();
    face[] front,back;
    for(int i=0; i < faces.length; ++i) {
      splitface split=split(faces[i],bsp.node,P);
      if(split.front != null) front.push(split.front);
      if(split.back != null) back.push(split.back);
    }
    bsp.front=bsp.split(front,P);
    bsp.back=bsp.split(back,P);
    return bsp;
  }
  
  // Draw from back to front.
  void add(frame f) {
    if(back != null) back.add(f);
    add(f,node.fit,group=true);
    if(front != null) front.add(f);
  }
}

bsp operator init() {return new bsp;}
  
void add(picture pic=currentpicture, face[] faces,
	 projection P=currentprojection)
{
  int n=faces.length;
  face[] Faces=new face[n];
  for(int i=0; i < n; ++i)
    Faces[i]=faces[i].copy();
  
  pic.nodes.push(new void (frame f, transform t, transform T,
			   pair m, pair M) {
// Fit all of the pictures so we know their exact sizes.	   
   for(int i=0; i < n; ++i) {
     face F=Faces[i];
     F.t=t*T*F.pic.T;
     F.fit=F.pic.fit(t,T*F.pic.T,m,M);
   }
    
   bsp bsp;
   bsp=bsp.split(Faces,P);
   bsp.add(f);
  });
    
  for(int i=0; i < n; ++i) {
    picture F=Faces[i].pic;
    pic.userBox(F.userMin,F.userMax);
    pic.append(pic.bounds.point,pic.bounds.min,pic.bounds.max,F.T,F.bounds);
  }
}    
