private import math;
import embedding;

string defaultembed3options="3Drender=Solid,3Dlights=White,toolbar=true";

real defaultshininess=0.25;

triple O=(0,0,0);
triple X=(1,0,0), Y=(0,1,0), Z=(0,0,1);

int ninterpolate=16;

pair operator ecast(real[] a)
{
  if(a[3] == 0) abort(tooclose);
  return (a[0],a[1])/a[3];
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
    {0.0, 0.0, 0.0, 1.0}
  }*shift(-u);
}

// Project u onto v.
triple project(triple u, triple v)
{
  v=unit(v);
  return dot(u,v)*v;
}

// Transformation corresponding to moving the camera from the origin
// (looking down the negative z axis) to the point 'eye' (looking at
// the origin), orienting the camera so that direction 'up' points upwards.
// Since, in actuality, we are transforming the points instead of
// the camera, we calculate the inverse matrix.
// Based on the gluLookAt implementation in the OpenGL manual.
transform3 look(triple eye, triple up=Z)
{
  triple f=unit(-eye);
  if(f == O)
    f=-Z; // The eye is already at the origin: look down.

  triple side=cross(f,up);
  if(side == O) {
    // The eye is pointing either directly up or down, so there is no
    // preferred "up" direction to rotate it.  Pick one arbitrarily.
    side=cross(f,Y);
    if(side == O) side=cross(f,Z);
  }
  triple s=unit(side);

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

projection currentprojection;

projection operator * (transform3 t, projection P)
{
  projection P=P.copy();
  if(!P.absolute) {
    P.camera=t*P.camera;
    P.target=t*P.target;
    P.project=P.projector(P.camera,P.up,P.target);
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
  return (pair) (P.project*(real[]) v);
}

// Uses the homogenous coordinate to perform perspective distortion.
// When combined with a projection to the XY plane, this effectively maps
// points in three space to a plane through target and
// perpendicular to the vector camera-target.
projection perspective(triple camera, triple up=Z, triple target=O)
{
  return projection(camera,target,up,
		    new transform3(triple camera, triple up, triple target) {
		      return shift(-target)*distort(camera-target)*
			look(camera-target,up);});
}

projection perspective(real x, real y, real z, triple up=Z, triple target=O)
{
  return perspective((x,y,z),up,target);
}

projection orthographic(triple camera, triple up=Z)
{
  return projection(camera,up,new transform3(triple camera, triple up, triple) {
      return look(camera,up);},infinity=true);
}

projection orthographic(real x, real y, real z, triple up=Z)
{
  return orthographic((x,y,z),up);
}

projection oblique(real angle=45)
{
  transform3 t=identity(4);
  real c2=Cos(angle)^2;
  real s2=1-c2;
  t[0][2]=-c2;
  t[1][2]=-s2;
  t[2][2]=0;
  return projection((c2,s2,1),up=Y,
		    new transform3(triple,triple,triple) {return t;},
		    infinity=true);
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
  return projection((1,c2,s2),
		    new transform3(triple,triple,triple) {return t;},
		    infinity=true);
}

projection obliqueY(real angle=45)
{
  transform3 t=identity(4);
  real c2=Cos(angle)^2;
  real s2=1-c2;
  t[0][1]=c2;
  t[1][1]=s2;
  t[1][2]=1;
  t[2][2]=0;
  return projection((c2,-1,s2),
		    new transform3(triple,triple,triple) {return t;},
		    infinity=true);
}

projection oblique=oblique();
projection obliqueX=obliqueX(), obliqueY=obliqueY(), obliqueZ=obliqueZ();

currentprojection=perspective(5,4,2);

// Map pair z onto a triple by inverting the projection P onto the 
// plane perpendicular to normal and passing through point.
triple invert(pair z, triple normal, triple point,
              projection P=currentprojection)
{
  transform3 t=P.project;
  real[][] A={{t[0][0]-z.x*t[3][0],t[0][1]-z.x*t[3][1],t[0][2]-z.x*t[3][2]},
              {t[1][0]-z.y*t[3][0],t[1][1]-z.y*t[3][1],t[1][2]-z.y*t[3][2]},
              {normal.x,normal.y,normal.z}};
  real[] b={z.x*t[3][3],z.y*t[3][3],dot(normal,point)};
  real[] x=solve(A,b);
  return (x[0],x[1],x[2]);
}

pair xypart(triple v)
{
  return (v.x,v.y);
}

struct control {
  triple post,pre;
  bool active=false;
  void init(triple post, triple pre) {
    this.post=post;
    this.pre=pre;
    active=true;
  }
}

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
  real out,in;
  bool atLeast;
  bool active;
  void init(real out=1, real in=1, bool atLeast=false, bool active=true) {
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
  Tension t=new Tension;
  t.init(false);
  return t;
}

Tension noTension;
  
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

void nullpath3(flatguide3) {};

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

// A version of acos that tolerates numerical imprecision
real acos1(real x)
{
  if(x < -1) x=-1;
  if(x > 1) x=1;
  return acos(x);
}
  
struct Controls {
  triple c0,c1;

  // 3D extension of John Hobby's control point formula
  // (cf. The MetaFont Book, page 131),
  // as described in John C. Bowman and A. Hammerlindl,
  // TUGBOAT: The Communications of th TeX Users Group 29:2 (2008).

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
  triple[] V;
  
  for(int i=1; i < n; ++i)
    V.push(cross(v[i]-v[i-1],v[i+1]-v[i])); 
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

struct node {
  triple pre,point,post;
  bool straight;
  node copy() {
    node n=new node;
    n.pre=pre;
    n.point=point;
    n.post=post;
    n.straight=straight;
    return n;
  }
}
  
void splitCubic(node[] sn, real t, node left_, node right_)
{
  node left=sn[0]=left_.copy(), mid=sn[1], right=sn[2]=right_.copy();
  triple x=interp(left.post,right.pre,t);
  left.post=interp(left.point,left.post,t);
  right.pre=interp(right.pre,right.point,t);
  mid.pre=interp(left.post,x,t);
  mid.post=interp(x,right.pre,t);
  mid.point=interp(mid.pre,mid.post,t);
}

node[] nodes(int n)
{
  return sequence(new node(int) {return new node;},n);
}

struct bbox3 {
  bool empty=true;
  triple min,max;
  
  void add(triple v) {
    if(empty) {
      min=max=v;
      empty=false;
    } else {
      real x=v.x; 
      real y=v.y;
      real z=v.z;
      
      real left=min.x;
      real bottom=min.y;
      real lower=min.z;
      
      real right=max.x;
      real top=max.y;
      real upper=max.z;
      
      if(x < left)
        left = x;  
      else if(x > right)
        right = x;  
      if(y < bottom)
        bottom = y;
      else if(y > top)
        top = y;
      if(z < lower)
        lower = z;
      else if(z > upper)
        upper = z;
      
      min=(left,bottom,lower);
      max=(right,top,upper);       
    }
  }

  void add(triple min, triple max) {
    add(min);
    add(max);
  }
  
  real diameter() {
    return length(max-min);
  }
  
  triple O() {return min;}
  triple X() {return (max.x,min.y,min.z);}
  triple XY() {return (max.x,max.y,min.z);}
  triple Y() {return (min.x,max.y,min.z);}
  triple YZ() {return (min.x,max.y,max.z);}
  triple Z() {return (min.x,min.y,max.z);}
  triple ZX() {return (max.x,min.y,max.z);}
  triple XYZ() {return max;}
}

bbox3 bbox3(triple min, triple max) 
{
  bbox3 b;
  b.add(min,max);
  return b;
}

private real Fuzz=10.0*realEpsilon;

triple XYplane(pair z) {return (z.x,z.y,0);}

struct path3 {
  node[] nodes;
  bool cycles;
  int n;
  real cached_length=-1;
  bbox3 box;
  
  void operator init(node[] nodes, bool cycles=false, real cached_length=-1) {
    for(int i=0; i < nodes.length; ++i)
      this.nodes[i]=nodes[i].copy();
    this.cycles=cycles;
    this.cached_length=cached_length;
    this.n=cycles ? nodes.length-1 : nodes.length;
  }
  
  void operator init(triple v) {
    node node;
    node.pre=node.point=node.post=v;
    node.straight=false;
    this.nodes.push(node);
    this.cycles=false;
    this.n=1;
  }
  
  void operator init(node n0, node n1) {
    node N0,N1;
    N0 = n0.copy();
    N1 = n1.copy();
    N0.pre = N0.point;
    N1.post = N1.point;
    this.nodes.push(N0);
    this.nodes.push(N1);
    this.cycles=false;
    this.n=2;
  }
  
  void operator init(path g, triple plane(pair)=XYplane) {
    this.cycles=cyclic(g);
    this.n=size(g);
    int N=this.cycles ? this.n+1 : this.n;
    node[] nodes=new node[N];
    for(int i=0; i < N; ++i) {
      node node;
      node.pre=plane(precontrol(g,i));
      node.point=plane(point(g,i));
      node.post=plane(postcontrol(g,i));
      node.straight=straight(g,i);
      nodes[i]=node;
    }
    this.nodes=nodes;  
  }

  int size() {return n;}
  int length() {return nodes.length-1;}
  bool empty() {return n == 0;}
  bool cyclic() {return cycles;}
  
  bool straight(int t) {
    if (cycles) return nodes[t % n].straight;
    return (t >= 0 && t < n) ? nodes[t].straight : false;
  }
  
  triple point(int t) {
    return nodes[adjustedIndex(t,n,cycles)].point;
  }

  triple precontrol(int t) {
    return nodes[adjustedIndex(t,n,cycles)].pre;
  }

  triple postcontrol(int t) {
    return nodes[adjustedIndex(t,n,cycles)].post;
  }

  triple point(real t) {
    checkEmpty(n);
    
    int i = Floor(t);
    t = fmod(t,1);
    if (t < 0) t += 1;

    int iplus;
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

    triple a = nodes[i].point,
      b = nodes[i].post,
      c = nodes[iplus].pre,
      d = nodes[iplus].point,
      ab   = interp(a,b,t),
      bc   = interp(b,c,t),
      cd   = interp(c,d,t),
      abc  = interp(ab,bc,t),
      bcd  = interp(bc,cd,t),
      abcd = interp(abc,bcd,t);

    return abcd;
  }
  
  triple precontrol(real t) {
    checkEmpty(n);
                     
    int i = Floor(t);
    t = fmod(t,1);
    if (t < 0) t += 1;

    int iplus;
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

    triple a = nodes[i].point,
      b = nodes[i].post,
      c = nodes[iplus].pre,
      ab   = interp(a,b,t),
      bc   = interp(b,c,t),
      abc  = interp(ab,bc,t);

    return (abc == a) ? nodes[i].pre : abc;
  }
        
 
  triple postcontrol(real t) {
    checkEmpty(n);
  
    int i = Floor(t);
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

    triple b = nodes[i].post,
      c = nodes[iplus].pre,
      d = nodes[iplus].point,
      bc   = interp(b,c,t),
      cd   = interp(c,d,t),
      bcd  = interp(bc,cd,t);

    return (bcd == d) ? nodes[iplus].post : bcd;
  }

  static real sqrtFuzz=sqrt(Fuzz);

  triple predir(int t) {
    if(!cycles && t <= 0) return (0,0,0);
    triple z0=point(t-1);
    triple z1=point(t);
    triple c1=precontrol(t);
    triple dir=z1-c1;
    real epsilon=Fuzz*abs(z0-z1);
    if(abs(dir) > epsilon) return unit(dir);
    triple c0=postcontrol(t-1);
    dir=2*c1-c0-z1;
    if(abs(dir) > epsilon) return unit(dir);
    return unit(z1-z0+3*(c0-c1));
  }

  triple postdir(int t) {
    if(!cycles && t >= n-1) return (0,0,0);
    triple z0=point(t);
    triple z1=point(t+1);
    triple c0=postcontrol(t);
    triple dir=c0-z0;
    real epsilon=Fuzz*abs(z0-z1);
    if(abs(dir) > epsilon) return unit(dir);
    triple c1=precontrol(t+1);
    dir=z0-2*c0+c1;
    if(abs(dir) > epsilon) return unit(dir);
    return unit(z1-z0+3*(c0-c1));
  }

  triple dir(int t) {
    return unit(predir(t)+postdir(t));
  }

  triple dir(int t, int sign) {
    if(sign == 0) return dir(t);
    else if(sign > 0) return postdir(t);
    else return predir(t);
  }

  path3 concat(path3 p1, path3 p2) {
    int n1 = p1.length(), n2 = p2.length();

    if (n1 == -1) return p2;
    if (n2 == -1) return p1;
    triple a=p1.point(n1);
    triple b=p2.point(0);

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
      nodes[i].straight = straight(j-1);
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
  
  path3 subpath(int a, int b) {
    if(empty()) return new path3;

    if (a > b) {
      path3 rp = reverse();
      int len=length();
      path3 result = rp.subpath(len-a, len-b);
      return result;
    }

    if (!cycles) {
      if (a < 0)
        a = 0;
      if (b > n-1)
        b = n-1;
    }

    int sn = b-a+1;
    node[] nodes=nodes(sn);
    for (int i = 0, j = a; j <= b; ++i, ++j) {
      nodes[i].pre = precontrol(j);
      nodes[i].point = point(j);
      nodes[i].post = postcontrol(j);
      nodes[i].straight = straight(j);
    }
    nodes[0].pre = nodes[0].point;
    nodes[sn-1].post = nodes[sn-1].point;

    return path3(nodes);
  }
  
  path3 subpath(real a, real b) {
    if(empty()) return new path3;
  
    if (a > b) {
      int len=length();
      return reverse().subpath(len-a, len-b);
    }

    node aL, aR, bL, bR;
    if (!cycles) {
      if (a < 0) {
        a = 0;
        if (b < 0)
          b = 0;
      }
      if (b > n-1) {
        b = n-1;
        if (a > n-1)
          a = n-1;
      }
      aL = nodes[floor(a)];
      aR = nodes[ceil(a)];
      bL = nodes[floor(b)];
      bR = nodes[ceil(b)];
    } else {
      if(fabs(a) > intMax || fabs(b) > intMax)
        abort("invalid path index");
      aL = nodes[floor(a) % n];
      aR = nodes[ceil(a) % n];
      bL = nodes[floor(b) % n];
      bR = nodes[ceil(b) % n];
    }

    if (a == b) return path3(point(a));
    
    node[] sn=nodes(3);
    path3 p = subpath(Ceil(a), Floor(b));
    if (a > floor(a)) {
      if (b < ceil(a)) {
        splitCubic(sn,a-floor(a),aL,aR);
        splitCubic(sn,(b-a)/(ceil(b)-a),sn[1],sn[2]);
        return path3(sn[0],sn[1]);
      }
      splitCubic(sn,a-floor(a),aL,aR);
      p=concat(path3(sn[1],sn[2]),p);
    }
    if (ceil(b) > b) {
      splitCubic(sn,b-floor(b),bL,bR);
      p=concat(p,path3(sn[0],sn[1]));
    }
    return p;
  }
  
  triple predir(real t) {
    if(!cycles) {
      if(t <= 0) return (0,0,0);
      if(t >= n-1) return predir(n-1);
    }
    int a=Floor(t);
    return (t-a < sqrtFuzz) ? predir(a) : subpath((real) a,t).predir(1);
  }

  triple postdir(real t) {
    if(!cycles) {
      if(t >= n-1) return (0,0,0);
      if(t <= 0) return postdir(0);
    }
    int b=Ceil(t);
    return (b-t < sqrtFuzz) ? postdir(b) : subpath(t,(real) b).postdir(0);
  }

  triple dir(real t) {
    return unit(predir(t)+postdir(t));
  }

  triple dir(real t, int sign) {
    if(sign == 0) return dir(t);
    else if(sign > 0) return postdir(t);
    else return predir(t);
  }

  bbox3 bounds() {
    if(!box.empty) return box;
    
    if (empty()) {
      // No bounds
      return new bbox3;
    }

    int len=length();
    for (int i = 0; i < len; ++i) {
      box.add(point(i));
      if(straight(i)) continue;
    
      triple z0=point(i);
      triple z0p=postcontrol(i);
      triple z1m=precontrol(i+1);
      triple z1=point(i+1);
      
      triple a=z1-z0+3.0*(z0p-z1m);
      triple b=2.0*(z0+z1m)-4.0*z0p;
      triple c=z0p-z0;
      
      // Check x coordinate
      real[] roots=quadraticroots(a.x,b.x,c.x);
      if(roots.length > 0) box.add(point(i+roots[0]));
      if(roots.length > 1) box.add(point(i+roots[1]));
    
      // Check y coordinate
      roots=quadraticroots(a.y,b.y,c.y);
      if(roots.length > 0) box.add(point(i+roots[0]));
      if(roots.length > 1) box.add(point(i+roots[1]));
    }
    box.add(point(length()));
    return box;
  }
  
  triple max() {
    checkEmpty(n);
    return bounds().max;
  }
  triple min() {
    checkEmpty(n);
    return bounds().min;
  }
}

bool cyclic(path3 p) {return p.cyclic();}
int size(path3 p) {return p.size();}
int length(path3 p) {return p.length();}

bool cyclic(guide3 g) {flatguide3 f; g(f); return f.cyclic();}
int size(guide3 g) {flatguide3 f; g(f); return f.size();}
int length(guide3 g) {flatguide3 f; g(f); return f.nodes.length-1;}

path3 operator * (transform3 t, path3 p) 
{
  int m=p.nodes.length;
  node[] nodes=nodes(m);
  for(int i=0; i < m; ++i) {
    nodes[i].pre=t*p.nodes[i].pre;
    nodes[i].point=t*p.nodes[i].point;
    nodes[i].post=t*p.nodes[i].post;
    nodes[i].straight=p.nodes[i].straight;
  }
  return path3(nodes,p.cycles);
}

path3[] path3(path[] g, triple plane(pair)=XYplane)
{
  return sequence(new path3(int i) {return path3(g[i],plane);},g.length);
}

path3[] operator * (transform3 t, path3[] p) 
{
  path3[] g=new path3[p.length];
  for(int i=0; i < p.length; ++i)
    g[i]=t*p[i];
  return g;
}

void write(file file, string s="", explicit path3 x, suffix suffix=none)
{
  write(file,s);
  if(size(x) == 0) write("<nullpath3>");
  else for(int i=0; i < x.nodes.length; ++i) {
      if(i == x.nodes.length-1 && x.cycles) write(file,"cycle");
      else write(file,x.nodes[i].point);
      if(i < length(x)) {
        if(x.nodes[i].straight) write(file,"--");
        else {
          write(file,".. controls ");
          write(file,x.nodes[i].post);
          write(file," and ");
          write(file,x.nodes[i+1].pre);
          write(file,"..",endl);
        }
      }
    }
  write(file,suffix);
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
  path3 p;

  // If duplicate points occur consecutively, add dummy controls (if absent).
  for(int i=0; i < n; ++i) {
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
    aim(g,i);
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
  if(g.nodes.length == 0) return new path3;
  bool cyclic=g.cyclic[g.cyclic.length-1];
  for(int i=0; i < g.nodes.length-1; ++i) {
    nodes[i].point=g.nodes[i];
    nodes[i].post=g.control[i].post;
    nodes[i+1].pre=g.control[i].pre;
    nodes[i].straight=!g.control[i].active;
  }
  nodes[g.nodes.length-1].point=g.nodes[g.nodes.length-1];
  if(cyclic) {
    nodes[0].pre=g.control[nodes.length-2].pre;
    nodes[g.nodes.length-1].post=g.control[nodes.length-1].post;
  } else {
    nodes[0].pre=nodes[0].point;
    nodes[g.nodes.length-1].post=nodes[g.nodes.length-1].point;
  }
  
  return path3(nodes,cyclic);
}

path nurb(path3 p, projection P=currentprojection,
          int ninterpolate=ninterpolate)
{
  triple f=P.camera;
  triple u=unit(P.camera-P.target);

  path nurb(triple v0, triple v1, triple v2, triple v3) {
    return nurb(project(v0,P),project(v1,P),project(v2,P),project(v3,P),
		dot(u,f-v0),dot(u,f-v1),dot(u,f-v2),dot(u,f-v3),ninterpolate);
  }

  path g;

  if(p.nodes[0].straight)
    g=project(p.nodes[0].point,P);

  for(int i=0; i < length(p); ++i) {
    if(p.nodes[i].straight)
      g=g--project(p.nodes[i+1].point,P);
    else
      g=g&nurb(p.nodes[i].point,p.nodes[i].post,p.nodes[i+1].pre,
               p.nodes[i+1].point);
  }

  int n=length(g);
  if(p.cycles) g=g&cycle;

  return g;
}

bool piecewisestraight(path3 p)
{
  int L=p.length();
  for(int i=0; i < L; ++i)
    if(!p.nodes[i].straight) return false;
  return true;
}

path project(explicit path3 p, projection P=currentprojection,
             int ninterpolate=ninterpolate)
{
  guide g;

  int last=p.nodes.length-1;
  if(last < 0) return g;
  
  if(P.infinity || ninterpolate == 1 || piecewisestraight(p)) {
    g=project(p.nodes[0].point,P);
    // Construct the path.
    for(int i=0; i < (p.cycles ? last-1 : last); ++i) {
      if(p.nodes[i].straight)
        g=g--project(p.nodes[i+1].point,P);
      else {
        g=g..controls project(p.nodes[i].post,P) and
	  project(p.nodes[i+1].pre,P)..project(p.nodes[i+1].point,P);
      }
    }
  } else return nurb(p,P,ninterpolate);
  
  if(p.cycles)
    g=p.nodes[last-1].straight ? g--cycle :
      g..controls project(p.nodes[last-1].post,P) and
      project(p.nodes[last].pre,P)..cycle;

  return g;
}

path project(flatguide3 g, projection P=currentprojection,
             int ninterpolate=ninterpolate)

{
  return project(solve(g),P,ninterpolate);
}

pair[] project(triple[] v, projection P=currentprojection)
{
  int n=v.length;
  pair[] z=new pair[n];
  for(int i=0; i < n; ++i)
    z[i]=project(v[i],P);
  return z;
}

path[] project(path3[] g, projection P=currentprojection,
               int ninterpolate=ninterpolate)
{
  path[] p=new path[g.length];
  for(int i=0; i < g.length; ++i) 
    p[i]=project(g[i],P,ninterpolate);
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
    if(p.nodes[i].straight) g=g--cycle;
    else g=g..controls p.nodes[i].post and p.nodes[i+1].pre..cycle;
  }
  
  return g;
}

bool cyclic(path3 p) {return p.cyclic();}

position operator cast(triple x) {return project(x);}

// Transforms that map XY plane to YX, YZ, ZY, ZX, and XZ planes.
restricted transform3 XY=identity4;
restricted transform3 YX=rotate(-90,O,Z);
restricted transform3 YZ=rotate(90,O,Z)*rotate(90,O,X);
restricted transform3 ZY=rotate(-90,O,X)*YZ;
restricted transform3 ZX=rotate(-90,O,Z)*rotate(-90,O,Y);
restricted transform3 XZ=rotate(-90,O,Y)*ZX;

// Transform for projecting onto plane through point O with normal cross(u,v).
transform transform(triple u, triple v, triple O=O,
                    projection P=currentprojection)
{
  transform3 t=P.project;
  real[] tO=t*(real[]) O;
  real[] x=new real[4];
  real[] y=new real[4];
  real[] t3=t[3];
  real tO3=tO[3];
  real factor=1.0/tO3^2;
  for(int i=0; i < 3; ++i) {
    x[i]=(tO3*t[0][i]-tO[0]*t3[i])*factor;
    y[i]=(tO3*t[1][i]-tO[1]*t3[i])*factor;
  }
  x[3]=1;
  y[3]=1;
  triple x=(triple) x;
  triple y=(triple) y;
  u=unit(u);
  v=unit(v);
  return (0,0,dot(u,x),dot(v,x),dot(u,y),dot(v,y));
}

// Project Label onto plane through point O with normal cross(u,v).
Label project(Label L, triple u, triple v, triple O=O,
              projection P=currentprojection) {
  Label L=L.copy();
  L.position=project(O,P);
  L.transform(transform(u,v,O,P)); 
  return L;
}

path3 operator cast(guide3 g) {return solve(g);}
path operator cast(path3 p) {return project(p);}
path operator cast(triple v) {return project(v);}
path operator cast(guide3 g) {return project(solve(g));}
path3 operator cast(triple v) {return path3(v);}

path[] operator cast(path3 p) {return new path[] {(path) p};}
path[] operator cast(guide3 g) {return new path[] {(path) g};}
path[] operator cast(path3[] g) {return project(g);}

guide3[] operator cast(triple[] v)
{
  guide3[] g=new guide3[v.length];
  for(int i=0; i < v.length; ++i)
    g[i]=v[i];
  return g;
}

path3[] operator cast(triple[] v)
{
  path3[] g=new path3[v.length];
  for(int i=0; i < v.length; ++i)
    g[i]=v[i];
  return g;
}

bool straight(path3 p, int t) {return p.straight(t);}
bool straight(explicit guide3 g, int t) {return ((path3) g).straight(t);}

triple point(path3 p, int t) {return p.point(t);}
triple point(explicit guide3 g, int t) {
  flatguide3 f;
  g(f);
  int n=f.size();
  return f.nodes[adjustedIndex(t,n,f.cyclic())];
}

triple point(path3 p, real t) {return p.point(t);}
triple point(explicit guide3 g, real t) {return ((path3) g).point(t);}

triple postcontrol(path3 p, int t) {return p.postcontrol(t);}
triple postcontrol(explicit guide3 g, int t) {
  return ((path3) g).postcontrol(t);
}
triple postcontrol(path3 p, real t) {return p.postcontrol(t);}
triple postcontrol(explicit guide3 g, real t) {
  return ((path3) g).postcontrol(t);
}

triple precontrol(path3 p, int t) {return p.precontrol(t);}
triple precontrol(explicit guide3 g, int t) {
  return ((path3) g).precontrol(t);
}
triple precontrol(path3 p, real t) {return p.precontrol(t);}
triple precontrol(explicit guide3 g, real t) {
  return ((path3) g).precontrol(t);
}

triple[] dirSpecifier(guide3 g, int t)
{
  flatguide3 f;
  g(f);
  bool cyclic=f.cyclic();
  int n=f.size();
  checkEmpty(n);
  if(cyclic) t=t % n;
  else if(t < 0 || t >= n-1) return new triple[] {O,O};
  return new triple[] {f.out[t].dir,f.in[t].dir};
}

triple[] controlSpecifier(guide3 g, int t) {
  flatguide3 f;
  g(f);
  bool cyclic=f.cyclic();
  int n=f.size();
  checkEmpty(n);
  if(cyclic) t=t % n;
  else if(t < 0 || t >= n-1) return new triple[];
  control c=f.control[t];
  if(c.active) return new triple[] {c.post,c.pre};
  else return new triple[];
}

tensionSpecifier tensionSpecifier(guide3 g, int t)
{
  flatguide3 f;
  g(f);
  bool cyclic=f.cyclic();
  int n=f.size();
  checkEmpty(n);
  if(cyclic) t=t % n;
  else if(t < 0 || t >= n-1) return operator tension(1,1,false);
  Tension T=f.Tension[t];
  return operator tension(T.out,T.in,T.atLeast);
}

real[] curlSpecifier(guide3 g)
{
  flatguide3 f;
  g(f);
  return new real[] {f.out[0].gamma,f.in[f.nodes.length-2].gamma};
}

triple dir(path3 p, int t, int sign=0) {return p.dir(t,sign);}
triple dir(explicit guide3 g, int t, int sign=0) {
  return ((path3) g).dir(t,sign);
}
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

path3 subpath(path3 p, int a, int b) {return p.subpath(a,b);}
path3 subpath(explicit guide3 g, int a, int b)
{
  return ((path3) g).subpath(a,b);
}

path3 subpath(path3 p, real a, real b) {return p.subpath(a,b);}
path3 subpath(explicit guide3 g, real a, real b) 
{
  return ((path3) g).subpath(a,b);
}

int maxdepth=round(1-log(realEpsilon)/log(2));
int mindepth=maxdepth-12;

real[] intersect(path3 p, path3 q, real fuzz, int depth)
{
  triple maxp=p.max();
  triple minp=p.min();
  triple maxq=q.max();
  triple minq=q.min();
  
  if(maxp.x+fuzz >= minq.x &&
     maxp.y+fuzz >= minq.y && 
     maxp.z+fuzz >= minq.z && 
     maxq.x+fuzz >= minp.x &&
     maxq.y+fuzz >= minp.y &&
     maxq.z+fuzz >= minp.z) {
    // Overlapping bounding boxes

    --depth;
    if(abs(maxp-minp)+abs(maxq-minq) <= fuzz || depth == 0) {
      return new real[] {0,0};
    }
    
    int lp=p.length();
    path3 p1,p2;
    real pscale,poffset;
    
    if(lp == 1) {
      node[] sn=nodes(3);
      splitCubic(sn,0.5,p.nodes[0],p.nodes[1]);
      p1=path3(new node[] {sn[0],sn[1]});
      p2=path3(new node[] {sn[1],sn[2]});
      pscale=poffset=0.5;
    } else {
      int tp=quotient(lp,2);
      p1=p.subpath(0,tp);
      p2=p.subpath(tp,lp);
      poffset=tp;
      pscale=1.0;
    }
      
    int lq=q.length();
    path3 q1,q2;
    real qscale,qoffset;
    
    if(lq == 1) {
      node[] sn=nodes(3);
      splitCubic(sn,0.5,q.nodes[0],q.nodes[1]);
      q1=path3(new node[] {sn[0],sn[1]});
      q2=path3(new node[] {sn[1],sn[2]});
      qscale=qoffset=0.5;
    } else {
      int tq=quotient(lq,2);
      q1=q.subpath(0,tq);
      q2=q.subpath(tq,lq);
      qoffset=tq;
      qscale=1.0;
    }
      
    real[] T;

    T=intersect(p1,q1,fuzz,depth);
    if(T.length > 0)
      return new real[] {pscale*T[0],qscale*T[1]};

    T=intersect(p1,q2,fuzz,depth);
    if(T.length > 0)
      return new real[] {pscale*T[0],qscale*T[1]+qoffset};

    T=intersect(p2,q1,fuzz,depth);
    if(T.length > 0)
      return new real[] {pscale*T[0]+poffset,qscale*T[1]};

    T=intersect(p2,q2,fuzz,depth);
    if(T.length > 0)
      return new real[] {pscale*T[0]+poffset,qscale*T[1]+qoffset};
  }

  return new real[];
}

private real computefuzz(path3 p, path3 q, real fuzz) {
  return max(fuzz,Fuzz*max(max(length(p.max()),length(p.min())),
                           max(length(q.max()),length(q.min()))));
}

real[] intersect(path3 p, path3 q, real fuzz=0)
{
  fuzz=computefuzz(p,q,fuzz);
  return intersect(p,q,fuzz,maxdepth);
}

real[] intersect(explicit guide3 p, explicit guide3 q, real fuzz=0)
{
  return intersect((path3) p,(path3) q,fuzz);
}

triple intersectionpoint(path3 p, path3 q, real fuzz=0)
{
  real[] t=intersect(p,q,fuzz);
  if(t.length == 0) abort("paths do not intersect");
  return point(p,t[0]);
}

triple intersectionpoint(explicit guide3 p, explicit guide3 q, real fuzz=0)
{
  return intersectionpoint((path3) p,(path3) q,fuzz);
}

// return an array containing all intersection times of p and q
real[][] intersections(path3 p, path3 q, real fuzz, int depth)
{
  triple maxp=max(p);
  triple minp=min(p);
  triple maxq=max(q);
  triple minq=min(q);

  if(maxp.x+fuzz >= minq.x &&
     maxp.y+fuzz >= minq.y && 
     maxp.z+fuzz >= minq.z && 
     maxq.x+fuzz >= minp.x &&
     maxq.y+fuzz >= minp.y &&
     maxq.z+fuzz >= minp.z) {
    // Overlapping bounding boxes

    --depth;
    if(abs(maxp-minp)+abs(maxq-minq) <= fuzz || depth == 0) {
      return new real[][] {{0,0}};
    }
    
    int lp=p.length();
    path3 p1,p2;
    real pscale,poffset;
    
    if(lp == 1) {
      node[] sn=nodes(3);
      splitCubic(sn,0.5,p.nodes[0],p.nodes[1]);
      p1=path3(new node[] {sn[0],sn[1]});
      p2=path3(new node[] {sn[1],sn[2]});
      pscale=poffset=0.5;
    } else {
      int tp=quotient(lp,2);
      p1=p.subpath(0,tp);
      p2=p.subpath(tp,lp);
      poffset=tp;
      pscale=1.0;
    }
      
    int lq=q.length();
    path3 q1,q2;
    real qscale,qoffset;
    
    if(lq == 1) {
      node[] sn=nodes(3);
      splitCubic(sn,0.5,q.nodes[0],q.nodes[1]);
      q1=path3(new node[] {sn[0],sn[1]});
      q2=path3(new node[] {sn[1],sn[2]});
      qscale=qoffset=0.5;
    } else {
      int tq=quotient(lq,2);
      q1=q.subpath(0,tq);
      q2=q.subpath(tq,lq);
      qoffset=tq;
      qscale=1.0;
    }
      
    real[][] S=new real[][];
    real[][] T;

    void add(real s, real t) {
      real fuzz=2*fuzz;
      for(int i=0; i < S.length; ++i) {
        real[] Si=S[i];
        if(abs(p.point(Si[0])-p.point(s)) <= fuzz &&
           abs(q.point(Si[1])-q.point(t)) <= fuzz) return;
      }
      S.push(new real[] {s,t});
    }
  
    void add(real pscale, real qscale, real poffset, real qoffset) {
      for(int j=0; j < T.length; ++j) {
        real[] Tj=T[j];
        add(pscale*Tj[0]+poffset,qscale*Tj[1]+qoffset);
      }
    }

    T=intersections(p1,q1,fuzz,depth);
    add(pscale,qscale,0,0);
    if(depth <= mindepth && T.length > 0)
      return S;

    T=intersections(p1,q2,fuzz,depth);
    add(pscale,qscale,0,qoffset);
    if(depth <= mindepth && T.length > 0)
      return S;

    T=intersections(p2,q1,fuzz,depth);
    add(pscale,qscale,poffset,0);
    if(depth <= mindepth && T.length > 0)
      return S;

    T=intersections(p2,q2,fuzz,depth);
    add(pscale,qscale,poffset,qoffset);
    return S;
  }
  return new real[][];
}

real[][] intersections(path3 p, path3 q, real fuzz=0)
{
  fuzz=computefuzz(p,q,fuzz);
  return intersections(p,q,fuzz,maxdepth);
}

// return an array containing all intersection points of p and q
triple[] intersectionpoints(path3 p, path3 q, real fuzz=0)
{
  real[][] t=intersections(p,q,fuzz);
  triple[] v=new triple[t.length];
  for(int i=0; i < t.length; ++i)
    v[i]=point(p,t[i][0]);
  return v;
}

triple[] intersectionpoints(explicit guide3 p, explicit guide3 q, real fuzz=0)
{
  return intersectionpoints((path3) p,(path3) q, fuzz);
}

path3 operator & (path3 p, path3 q) {return p.concat(p,q);}
path3 operator & (explicit guide3 p, explicit guide3 q)
{
  return ((path3) p).concat(p,q);
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
triple arcpoint(explicit guide3 p, real L)
{
  return arcpoint((path3) p,L);
}

// return the direction on path3 p at arclength L
triple arcdir(path3 p, real L)
{
  return dir(p,arctime(p,L));
}
triple arcdir(explicit guide3 p, real L)
{
  return arcdir((path3) p,L);
}

// return the time on path3 p at the relative fraction l of its arclength
real reltime(path3 p, real l)
{
  return arctime(p,l*arclength(p));
}
real reltime(explicit guide3 p, real l)
{
  return reltime((path3) p,l);
}

// return the point on path3 p at the relative fraction l of its arclength
triple relpoint(path3 p, real l)
{
  return point(p,reltime(p,l));
}
triple relpoint(explicit guide3 p, real l)
{
  return relpoint((path3) p,l);
}

// return the direction of path3 p at the relative fraction l of its arclength
triple reldir(path3 p, real l)
{
  return dir(p,reltime(p,l));
}
triple reldir(explicit guide3 p, real l)
{
  return reldir((path3) p,l);
}

// return the point on path3 p at half of its arclength
triple midpoint(path3 p)
{
  return relpoint(p,0.5);
}
triple midpoint(explicit guide3 p)
{
  return relpoint(p,0.5);
}

real relative(Label L, path3 g)
{
  return L.position.relative ? reltime(g,L.relative()) : L.relative();
}

// return a rotation that maps u to Z.
transform3 align(triple u) 
{
  triple xi=cross(u,Z);
  if(xi != O) return rotate(colatitude(u),xi);
  return u.z >= 0 ? identity(4) : diagonal(1,-1,-1,1);
}

// return the rotation that maps X,Y,Z to unit(u),unit(v),unit(cross(u,v))
transform3 transform3(triple u, triple v) 
{
  u=unit(u);
  v=unit(v);
  triple w=cross(u,v);

  return new real[][] {
    {u.x,v.x,w.x,0.0},
    {u.y,v.y,w.y,0.0},
    {u.z,v.z,w.z,0.0},
    {0.0,0.0,0.0,1.0}
  };
}

// return a transfrom that maps X,Y to the projection plane.
transform3 transform3(projection P)
{
  triple v=unit(P.camera-P.target);
  triple u=P.up-dot(P.up,v)*v;
  return transform3(cross(u,v),u);
}

transform rotate(explicit triple dir)
{
  return rotate(project(dir));
} 

triple[] triples(real[] x, real[] y, real[] z)
{
  if(x.length != y.length || x.length != z.length)
    abort("arrays have different lengths");
  return sequence(new triple(int i) {return (x[i],y[i],z[i]);},x.length);
}

path3[] operator ^^ (path3 p, path3  q) 
{
  return new path3[] {p,q};
}

path3[] operator ^^ (guide3 p, guide3 q) 
{
  return new path3[] {p,q};
}

path3[] operator ^^ (triple p, triple q) 
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
  for(int i=0; i < n; ++i) {
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

path3[] unitcube=box((0,0,0),(1,1,1));

path3 unitcircle3=X..Y..-X..-Y..cycle;

path3 circle(triple c, real r, triple normal=Z)
{
  path3 p=scale3(r)*unitcircle3;
  if(normal != Z) 
    p=rotate(longitude(normal,warn=false),Z)*rotate(colatitude(normal),Y)*p;
  return shift(c)*p;
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
  transform3 T=align(normal); 
  triple v1=T*dir(theta1,phi1);
  triple v2=T*dir(theta2,phi2);
  real[] t1=intersect(unitcircle3,O--2*(v1.x,v1.y,0));
  real[] t2=intersect(unitcircle3,O--2*(v2.x,v2.y,0));
  if(t1.length == 0 || t2.length == 0)
    abort("invalid normal vector");
  real t1=t1[0];
  real t2=t2[0];
  int n=length(unitcircle3);
  if(t1 >= t2 && direction) t1 -= n;
  if(t2 >= t1 && !direction) t2 -= n;
  return shift(c)*scale3(r)*inverse(T)*subpath(unitcircle3,t1,t2);
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
// The normal must be explicitly specified if c and the endpoints are colinear.
path3 arc(triple c, triple v1, triple v2, triple normal=O, bool direction=CCW)
{
  v1 -= c; v2 -= c;
  return arc(c,abs(v1),colatitude(v1),longitude(v1,warn=false),
             colatitude(v2),longitude(v2,warn=false),normal,direction);
}

private real epsilon=1000*realEpsilon;

// Return a representation of the plane through point O with normal cross(u,v).
path3 plane(triple u, triple v, triple O=O)
{
  return O--O+u--O+u+v--O+v--cycle;
}

// Return the unit normal vector to a planar path p.
triple normal(path3 p)
{
  triple normal;
  real abspoint,absnext;
  
  void check(triple n) {
    if(abs(n) > epsilon*max(abspoint,absnext)) {
      n=unit(n);
      if(normal != O && abs(normal-n) > epsilon && abs(normal+n) > epsilon)
        abort("path is not planar");
      normal=n;
    }
  }

  int L=length(p);
  triple nextpre=precontrol(p,0);
  triple nextpoint=point(p,0);
  absnext=abs(nextpoint);
  
  for(int i=0; i < L; ++i) {
    triple pre=nextpre;
    triple point=nextpoint;
    triple post=postcontrol(p,i);
    nextpre=precontrol(p,i+1);
    nextpoint=point(p,i+1);
    
    abspoint=abs(point);
    absnext=abs(nextpoint);
    
    check(cross(point-pre,post-point));
    check(cross(post-point,nextpoint-nextpre));
  }
  return normal;
}

// Routines for hidden surface removal (via binary space partition):
// Structure face is derived from picture.
struct face {
  picture pic;
  transform t;
  frame fit;
  triple normal,point;
  bbox3 box;
  void operator init(path3 p) {
    this.normal=normal(p);
    if(this.normal == O) abort("path is linear");
    this.point=point(p,0);
    this.box=bbox3(min(p),max(p));
  }
  face copy() {
    face f=new face;
    f.pic=pic.copy();
    f.t=t;
    f.normal=normal;
    f.point=point;
    f.box=box;
    add(f.fit,fit);
    return f;
  }
}

picture operator cast(face f) {return f.pic;}
face operator cast(path3 p) {return face(p);}
  
struct line {
  triple point;
  triple dir;
}

private line intersection(face a, face b) 
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
  // Points exactly on L are considered to be on the right side.
  // Also push any points of intersection of L with the path operator --(... z)
  // onto each of the arrays left and right. 
  void operator init(pair dir, pair P ... pair[] z) {
    pair lastz;
    pair invdir=dir != 0 ? 1/dir : 0;
    bool left,last;
    for(int i=0; i < z.length; ++i) {
      left=(invdir*z[i]).y > (invdir*P).y;
      if(i > 0 && last != left) {
        pair w=extension(P,P+dir,lastz,z[i]);
        this.left.push(w);
        this.right.push(w);
      }
      if(left) this.left.push(z[i]);
      else this.right.push(z[i]);
      last=left;
      lastz=z[i];
    }
  }
}
  
struct splitface {
  face back,front;
}

// Return the pieces obtained by splitting face a by face cut.
splitface split(face a, face cut, projection P)
{
  splitface S;

  void nointersection() {
    if(abs(dot(a.point-P.camera,a.normal)) >= 
       abs(dot(cut.point-P.camera,cut.normal))) {
      S.back=a;
      S.front=null;
    } else {
      S.back=null;
      S.front=a;
    }
  }

  if(P.infinity) {
    P=P.copy();
    static real factor=1/sqrt(realEpsilon);
    P.camera *= factor*max(abs(a.box.min),abs(a.box.max),
                           abs(cut.box.min),abs(cut.box.max));
  }

  if((abs(a.normal-cut.normal) < epsilon ||
      abs(a.normal+cut.normal) < epsilon)) {
    nointersection();
    return S;
  }

  line L=intersection(a,cut);

  if(dot(P.camera-L.point,P.camera-P.target) < 0) {
    nointersection();
    return S;
  }
    
  pair point=a.t*project(L.point,P);
  pair dir=a.t*project(L.point+L.dir,P)-point;
  pair invdir=dir != 0 ? 1/dir : 0;
  triple apoint=L.point+cross(L.dir,a.normal);
  bool left=(invdir*(a.t*project(apoint,P))).y >= (invdir*point).y;

  real t=intersect(apoint,P.camera,cut.normal,cut.point);
  bool rightfront=left ^ (t <= 0 || t >= 1);
  
  face back=a, front=a.copy();
  pair max=max(a.fit);
  pair min=min(a.fit);
  half h=half(dir,point,max,(min.x,max.y),min,(max.x,min.y),max);
  if(h.right.length == 0) {
    if(rightfront) front=null;
    else back=null;
  } else if(h.left.length == 0) {
    if(rightfront) back=null;
    else front=null;
  }
  if(front != null)
    clip(front.fit,operator --(... rightfront ? h.right : h.left)--cycle,
         zerowinding);
  if(back != null)
    clip(back.fit,operator --(... rightfront ? h.left : h.right)--cycle,
         zerowinding);
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
  void operator init(face[] faces, projection P) {
    if(faces.length != 0) {
      this.node=faces.pop();
      face[] front,back;
      for(int i=0; i < faces.length; ++i) {
        splitface split=split(faces[i],this.node,P);
        if(split.front != null) front.push(split.front);
        if(split.back != null) back.push(split.back);
      }
      this.front=bsp(front,P);
      this.back=bsp(back,P);
    }
  }
  
  // Draw from back to front.
  void add(frame f) {
    if(back != null) back.add(f);
    add(f,node.fit,group=true);
    if(labels(node.fit)) layer(f); // Draw over any existing TeX layers.
    if(front != null) front.add(f);
  }
}

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
                   face[] faces=new face[n];
                   for(int i=0; i < n; ++i) {
                     faces[i]=Faces[i].copy();
                     face F=faces[i];
                     F.t=t*T*F.pic.T;
                     F.fit=F.pic.fit(t,T*F.pic.T,m,M);
                   }
    
                   bsp bsp=bsp(faces,P);
                   if(bsp != null) bsp.add(f);
                 });
    
  for(int i=0; i < n; ++i) {
    picture F=Faces[i].pic;
    pic.userBox(F.userMin,F.userMax);
    pic.append(pic.bounds.point,pic.bounds.min,pic.bounds.max,F.T,F.bounds);
  }
}

triple size3(frame f)
{
  return max3(f)-min3(f);
}

// PRC support

private string[] file3;

string embedprc(string prefix=defaultfilename, frame f, string label="",
		string text=label,  string options="",
		real width=0, real height=0, real angle=30,
		pen background=white, projection P=currentprojection)
{
  if(!prc()) return "";

  if(width == 0) width=settings.paperwidth;
  if(height == 0) height=settings.paperheight;
  string prefix=outprefix();
  prefix += "-"+(string) file3.length;
  shipout3(prefix,f);
  prefix += ".prc";
  file3.push(prefix);

  string format(real x) {
    assert(abs(x) < 1e18,"Number too large: "+string(x));
    return format("%.18f",x,"C");
  }
  string format(triple v) {
    return format(v.x)+" "+format(v.y)+" "+format(v.z);
  }
  string format(pen p) {
    real[] c=colors(rgb(p));
    return format((c[0],c[1],c[2]));
  }

  triple v=(P.camera-P.target)/cm;
  triple u=unit(v);
  triple w=unit(Z-u.z*u);
  triple up=unit(P.up-dot(P.up,u)*u);
  real roll=degrees(acos1(dot(up,w)))*sgn(dot(cross(up,w),u));

  string options3="poster,text="+text+",label="+label+
    ",3Daac="+format(P.absolute ? P.angle : angle)+
    ",3Dc2c="+format(unit(v))+
    ",3Dcoo="+format(P.target/cm)+
    ",3Droll="+format(roll)+
    ",3Droo="+format(abs(v))+
    ",3Dbg="+format(background)+
    ","+defaultembed3options;
  if(options != "") options3 += ","+options;

  return embed(prefix,options3,width,height);
}

object embed(string prefix=defaultfilename, frame f, string label="",
	     string text=label, string options="",
	     real width=0, real height=0, real angle=30,
	     pen background=white, projection P=currentprojection)
{
  object F;

  if(prc())
    F.L=embedprc(prefix,f,label,text,options,width,height,angle,background,P);
  else
    F.f=f;
  return F;
}

object embed(string prefix=defaultfilename, picture pic, string label="",
	     string text=label, string options="",
	     real width=0, real height=0, real angle=0, 
	     pen background=white, projection P=currentprojection)
{
  object F;
  if(pic.empty3()) return F;
  real xsize=pic.xsize3, ysize=pic.ysize3, zsize=pic.zsize3;
  if(xsize == 0 && ysize == 0 && zsize == 0)
    xsize=ysize=zsize=max(pic.xsize,pic.ysize);
  transform3 t=pic.scaling(xsize,ysize,zsize,pic.keepAspect);
  picture pic2;
  size(pic2,pic);
  frame f=pic.fit3(t,pic2,P);

  if(!pic.bounds3.exact) {
    t=pic.scale3(f,pic.keepAspect)*t;
    pic.bounds3.exact=true;
    f=pic.fit3(t,pic2,P);
  }

  if(prc()) {
    projection P=P.copy();

    transform s=pic2.scaling(width == 0 ? pic.xsize : width,
			     height == 0 ? pic.ysize : height,pic.keepAspect);
    pair M=pic2.max(s);
    pair m=pic2.min(s);
    pair lambda=M-m;
    width=lambda.x;
    height=lambda.y;

    if(!P.absolute) {
      pair v=(s.xx,s.yy);
      pair x=project(X,P);
      pair y=project(Y,P);
      pair z=project(Z,P);
      real f(pair a, pair b) {
	return b == 0 ? (0.5*(a.x+a.y)) :
	  (b.x^2*a.x+b.y^2*a.y)/(b.x^2+b.y^2);
      }
      t=xscale3(f(v,x))*yscale3(f(v,y))*zscale3(f(v,z))*t;
      P=t*P;
      pair c=0.5*(M+m);
      triple origin=invert(c,unit(P.camera-P.target),P.target,P);
      if(P.infinity) {
	f=pic.fit3(shift(-origin)*t,pic2,P);
	P.camera=unit(P.camera)*max(abs(min3(f)),abs(max3(f)));
      } else {
	f=pic.fit3(t,pic2,P);
	P.target=origin;
      }
      if(angle == 0) {
	real r=min(M.x-c.x,M.y-c.y);
	// Choose the angle to be just large enough to view the entire image:
	angle=2.05*aTan(r/(abs(P.camera-P.target)));
      }
    }

    F.L=embedprc(prefix,f,label,text,options,width,height,angle,background,P);
  } else {
    //    P.adjust(m);
    //    P.adjust(M);
    F.f=pic2.fit(pic.xsize,pic.ysize,pic.keepAspect);
  }
  return F;
}

embed3=new object(string prefix, frame f) {return embed(prefix,f);};
embed3=new object(string prefix, picture pic) {return embed(prefix,pic);};

void add(picture dest=currentpicture, object src, pair position, pair align,
         bool group=true, filltype filltype=NoFill, bool put=Above)
{
  if(prc())
    label(dest,src,position,align);
  else
   plain.add(dest,src,position,align,group,filltype,put);
}

string cameralink(string label, string text="View Parameters")
{
  return link(label,text,"3Dgetview");
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

void drawprc(frame f, path3 g, pen p=currentpen)
{
  node[] nodes=g.nodes;

  bool straight=piecewisestraight(g);

  triple[] v;
  if(straight) {
    int n=nodes.length;
    v=new triple[n];
    for(int i=0; i < n; ++i)
      v[i]=nodes[i].point;
  } else {
    int n=nodes.length-1;
    v=new triple[3*n+1];
    int k=1;
    v[0]=nodes[0].point;
    v[1]=nodes[0].post;
    for(int i=1; i < n; ++i) {
      v[++k]=nodes[i].pre;
      v[++k]=nodes[i].point;
      v[++k]=nodes[i].post;
    }
    v[++k]=nodes[n].pre;
    v[++k]=nodes[n].point;
  }

  draw(f,v,p,straight,min(g),max(g));
}

void draw(frame f, path3 g, pen p=currentpen, projection P=currentprojection,
	  int ninterpolate=ninterpolate)
{
  if(prc()) drawprc(f,g,p);
  else draw(f,project(g,P,ninterpolate),p);
}

// Temporary
void draw(frame f, guide3 g, pen p=currentpen, projection P=currentprojection,
	  int ninterpolate=ninterpolate)
{
  draw(f,(path3) g,p,P,ninterpolate);
}

void draw(frame f, path3[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) draw(f,g[i],p);
}

include three_light;
include three_surface;

triple min3(pen p)
{
  return linewidth(p)*(-0.5,-0.5,-0.5);
}

triple max3(pen p)
{
  return linewidth(p)*(0.5,0.5,0.5);
}

void draw(picture pic=currentpicture, Label L="", path3 g, align align=NoAlign,
	  pen p=currentpen, int ninterpolate=ninterpolate)
{
  Label L=L.copy();
  L.align(align);
  if(L.s != "") {
    L.p(p);
    label(pic,L,g);
  }
  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      if(prc())
	drawprc(f,t*g,p);
      draw(pic,project(t*g,t*P,ninterpolate),p);
    },true);
  if(size(g) > 0) {
    pic.addPoint(min(g),min3(p));
    pic.addPoint(max(g),max3(p));
  }
}

void draw(picture pic=currentpicture, Label L="", explicit guide3 g,
	  pen p=currentpen)
{
  draw(pic,L,(path3) g,p);
}

void draw(picture pic=currentpicture, Label L="", path3[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) draw(pic,L,g[i],p);
}

void draw(transform t=identity(), frame f, surface s, int nu=nmesh, int nv=nu,
	  pen surfacepen=lightgray, pen meshpen=nullpen, pen ambientpen=black,
	  pen emissivepen=black, pen specularpen=mediumgray,
	  real opacity=opacity(surfacepen), real shininess=defaultshininess,
	  light light=currentlight, projection P=currentprojection)
{
  if(s.s.length == 0) return;

  // Draw a mesh in the absence of lighting (override with meshpen=invisible). 
  if(!light.on && meshpen == nullpen) meshpen=currentpen;

  if(prc()) {
    for(int i=0; i < s.s.length; ++i)
      drawprc(f,s.s[i],surfacepen,ambientpen,emissivepen,
	      specularpen,opacity,shininess,light);
  } else {
    if(surfacepen != nullpen) {
      // Sort patches by mean distance from camera
      triple camera=P.camera;
      if(P.infinity)
	camera *= max(abs(min(s)),abs(max(s)));

      real[][] depth;
    
      for(int i=0; i < s.s.length; ++i) {
	triple[][] P=s.s[i].P;
	  real d=abs(camera-0.25*(P[0][0]+P[0][3]+P[3][3]+P[3][0]));
	  depth.push(new real[] {d,i});
      }

      depth=sort(depth);

      // Draw from farthest to nearest
      while(depth.length > 0) {
	real[] a=depth.pop();
	int i=round(a[1]);
	tensorshade(t,f,s.s[i],surfacepen,light,P);
      }
    }
  }

  if(meshpen != nullpen && meshpen != invisible) {
    for(int k=0; k < s.s.length; ++k) {
      real step=nu == 0 ? 0 : 1/nu;
      for(int i=0; i <= nu; ++i)
	draw(f,s.s[k].uequals(i*step),meshpen,P);
      
      real step=nv == 0 ? 0 : 1/nv;
      for(int j=0; j <= nv; ++j)
	draw(f,s.s[k].vequals(j*step),meshpen,P);
    }
  }
}

void draw(picture pic=currentpicture, surface s, int nu=nmesh, int nv=nu,
	  pen surfacepen=lightgray, pen meshpen=nullpen, pen ambientpen=black,
	  pen emissivepen=black, pen specularpen=mediumgray,
	  real opacity=opacity(surfacepen), real shininess=defaultshininess,
	  light light=currentlight)
{
  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      surface S=t*s;
      projection P=t*P;
      if(prc()) {
      draw(f,S,nu,nv,surfacepen,invisible,ambientpen,emissivepen,specularpen,
	   opacity,shininess,light,P);
      } else
	pic.add(new void(frame f, transform T) {
	    draw(T,f,S,nu,nv,surfacepen,invisible,ambientpen,emissivepen,
		 specularpen,opacity,shininess,light,P);
	},true);
      pic.addPoint(min(S,P));
      pic.addPoint(max(S,P));
    },true);
  pic.addPoint(min(s));
  pic.addPoint(max(s));

  if(meshpen != nullpen && meshpen != invisible) {
    for(int k=0; k < s.s.length; ++k) {
      real step=nu == 0 ? 0 : 1/nu;
      for(int i=0; i <= nu; ++i)
	draw(pic,s.s[k].uequals(i*step),meshpen);
      
      real step=nv == 0 ? 0 : 1/nv;
      for(int j=0; j <= nv; ++j)
	draw(pic,s.s[k].vequals(j*step),meshpen);
    }
  }
}

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

