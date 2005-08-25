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
// down the negative z axis) to sitting at the point "from" (looking at the
// origin). Since, in actuality, we are transforming the points instead of
// the camera, we calculate the inverse matrix.
transform3 lookAtOrigin(triple from)
{
  transform3 t=(from.x == 0 && from.y == 0) ? shift(-from) : 
    shift((0,0,-length(from)))*
    rotate(-90,Z)*
    rotate(-colatitude(from),Y)*
    rotate(-longitude(from),Z);
  return from.z >= 0 ? t : rotate(180,Y)*t;
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

projection orthographic(triple camera)
{
  projection P;
  P.init(camera,lookAtOrigin(camera));
  return P;
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
  
bool cyclic(explicit flatguide3 g) {return g.cyclic[g.cyclic.length-1];}
int size(explicit flatguide3 g) {
  return cyclic(g) ? g.nodes.length-1 : g.nodes.length;
}
int length(explicit flatguide3 g) {return g.nodes.length-1;}
triple point(explicit flatguide3 g, int k) {return g.nodes[k];}

void write(file file, flatguide3 g)
{
  if(size(g) == 0) {
    write("<nullguide3>");
    if(cyclic(g)) write(file,"..");
  } else for(int i=0; i < g.nodes.length; ++i) {
    if(i > 0) write(file);
    if(g.cyclic[i]) write(file,"cycle3");
    else write(file,g.nodes[i]);

    if(g.control[i].active) // Explicit control points trump other specifiers
      write(file,g.control[i]);
    else {
      write(file,g.out[i]);
      if(g.Tension[i].active) write(file,g.Tension[i]);
    }
    if(i < length(g)) write(file,"..");
    if(!g.control[i].active) write(file,g.in[i]);
  }
}
  
void write(file file=stdout, flatguide3 x, suffix s) {write(file,x); s(file);}
void write(flatguide3 g) {write(stdout,g,endl);}

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

void nullguide3(flatguide3) {};

guide3 operator init() {return nullguide3;}

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
  triple offset=t*(0,0,0);
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
    
  triple w;
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

path project(flatguide3 g, projection Q)
{
  project P=Q.project;
  int n=length(g);

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
    if(!g.control[i].active && !(g.out[i].Curl && g.in[i].Curl)) {
      Controls C;
      C.init(g.nodes[i],g.nodes[next],g.out[i].dir,g.in[i].dir,
	     g.Tension[i].out,g.Tension[i].in,g.Tension[i].atLeast);
      control c;
      c.init(C.c0,C.c1);
      g.control[i]=c;
    }
  }
  
  guide pg;
  
  // Construct the path.
  for(int i=0; i < size(g); ++i) {
    if(g.control[i].active)
      pg=pg..P(point(g,i))..
	controls P(g.control[i].post) and P(g.control[i].pre)..nullpath;
    else 
      pg=pg..P(point(g,i))--nullpath;
  }
  
  if(cyclic(g)) pg=g.control[n-1].active ? pg..cycle : pg--cycle;
  
  return pg;
}

pair project(triple v, projection P)
{
  project P=P.project;
  return P(v);
}

path[] project(flatguide3[] g, projection P)
{
  path[] p=new path[g.length];
  for(int i=0; i < g.length; ++i) 
    p[i]=project(g[i],P);
  return p;
}
  
public projection currentprojection=perspective((5,4,2));

pair operator cast(triple v) {
  return project(v,currentprojection);
}

path operator cast(triple v) {
  return project(v,currentprojection);
}

path operator cast(guide3 g) {
  return project(g,currentprojection);
}

path[] operator cast(guide3 g) {
  return new path[] {(path) g};
}

path[] operator cast(guide3[] g) {
  return project(g,currentprojection);
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

// The graph of a function along a path.
guide3 graph(triple F(path, real), path p, int n=10)
{
  guide3 g;
  for(int i=0; i < n*length(p); ++i)
    g=g--F(p,i/n);
  return cyclic(p) ? g--cycle3 : g--F(p,length(p));
}

guide3 graph(triple F(pair), path p, int n=10) 
{
  return graph(new triple(path p, real position) {
		 return F(point(p,position));
	       },p,n);
}

guide3 graph(real f(pair), path p, int n=10) 
{
  return graph(new triple(pair z) {return (z.x,z.y,f(z));},p,n);
}

picture surface(real f(pair z), pair back, pair front, int n=20, int subn=1, 
		pen surfacepen=lightgray, pen meshpen=currentpen,
		projection P=currentprojection)
{
  picture pic;

  void drawcell(pair a, pair b) {
    guide3 g=graph(f,box(a,b),subn);
    filldraw(pic,project(g,P),surfacepen,meshpen);
  }

  pair sample(int i, int j) {
    return (interp(back.x,front.x,i/n),
            interp(back.y,front.y,j/n));
  }

  for(int i=0; i < n; ++i)
    for(int j=0; j < n; ++j)
      drawcell(sample(i,j),sample(i+1,j+1));

  return pic;
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
