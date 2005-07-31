import math;

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
// See http://www.cprogramming.com/tutorial/3d/rotation.html
transform3 rotate(real angle, triple v)
{
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
transform3 rotate(real angle, triple v, triple u)
{
  return shift(u)*rotate(angle,v)*shift(-u);
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

// Uses the homogenous coordinate to perform perspective distortion.  When
// combined with a projection to the XY plane, this effectively maps
// points in three space to a plane at a distance d from the camera.
transform3 perspective(triple camera)
{
  transform3 t=identity(4);
  real d=length(camera);
  t[3][2]=-1/d;
  t[3][3]=0;
  return t*lookAtOrigin(camera);
}

transform3 orthographic(triple camera)
{
  return lookAtOrigin(camera);
}

transform3 oblique(real angle=45)
{
  transform3 t=identity(4);
  real x=Cos(angle)^2;
  t[0][2]=-x;
  t[1][2]=x-1;
  t[2][2]=0;
  return t;
}

transform3 oblique=oblique();

typedef pair projection(triple a);

pair projectXY(triple v)
{
  return (v.x,v.y);
}

projection operator cast(transform3 t)
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
  write(file,"..controls ");
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
  write(file,t.out);
  write(file," and ");
  write(file,t.in);
}
  
struct dir {
  public triple dir;
  public bool active=false;
  void init(triple v) {
    this.dir=v;
    active=true;
  }
}

dir operator init() {return new dir;}
dir nodir;

dir operator * (transform3 t, dir d) 
{
  dir D;
  D.dir=unit(t*d.dir-t*(0,0,0));
  D.active=d.active;
  return D;
}

// TODO: Relax the assumption that the cycle specifier is at the end.

struct flatguide3 {
  public triple[] nodes;
  public control[] control; // control points for segment starting at node
  public Tension[] Tension; // Tension parameters for segment starting at node
  public dir[] in,out;    // in and out directions for segment starting at node
  public bool[] straight; // true unless segment starting at node is a spline
  public bool cycles=false;

  void node(triple v) {
    nodes.push(v);
    control.push(nocontrol);
    Tension.push(noTension);
    in.push(nodir);
    out.push(nodir);
    straight.push(false);
 }

  void control(triple post, triple pre) {
    if(control.length > 0) {
      control c;
      c.init(post,pre);
      control[-1]=c;
    }
  }

  void Tension(real out, real in, bool atLeast) {
    if(Tension.length > 0) {
      Tension t;
      t.init(out,in,atLeast);
      Tension[-1]=t;
    }
  }

  void in(triple v) {
    if(in.length > 0) {
      dir d;
      d.init(v);
      in[-1]=d;
    }
  }

  void out(triple v) {
    if(out.length > 0) {
      dir d;
      d.init(v);
      out[-1]=d;
    }
  }

  void straight(bool b) {
    if(straight.length > 0) straight[-1]=b;
  }
}

flatguide3 operator init() {return new flatguide3;}
  
int size(explicit flatguide3 g) {return g.nodes.length;}
triple point(explicit flatguide3 g, int k) {return g.nodes[k];}
bool cyclic(explicit flatguide3 g) {return g.cycles;}
int length(explicit flatguide3 g) {
  return g.cycles ? g.nodes.length : g.nodes.length-1;
}

void write(file file, flatguide3 g)
{
  if(size(g) == 0) {
    write("<nullguide3>");
  if(cyclic(g) && g.straight.length > 0)
      write(file,g.straight[0] ? " --" : " ..");
  } else for(int i=0; i < size(g); ++i) {
    write(file,g.nodes[i],endl);
    if(g.out[i].active) {
      write(file,"{"); write(file,g.out[i].dir); write(file,"}");
    }
    if(g.control[i].active)
      write(file,g.control[i]);
    if(g.Tension[i].active)
      write(file,g.Tension[i]);
    if(i < length(g))
      write(file,g.straight[i] ? " --" : " ..");
    if(g.in[i].active) {
      write(file,"{"); write(file,g.in[i].dir); write(file,"}");
    }
  }
  if(cyclic(g))
    write(file,"cycle3");
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
  f.straight.push(false);
  f.cycles=true;
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

guide3 operator -- (... guide3[] g)
{
  return new void(flatguide3 f) {
    // Apply the subguides in order.
    for(int i=0; i < g.length; ++i) {
      g[i](f);
      f.straight(true);
    }
  };
}

guide3 operator .. (... guide3[] g)
{
  return new void(flatguide3 f) {
    for(int i=0; i < g.length; ++i) {
      g[i](f);
    }
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

guide3 operator spec(triple v, int p)
{
  return new void(flatguide3 f) {
    if(p == 0) f.out(v);
    else if(p == 1) f.in(v);
  };
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
      f.control[i]=t*f.control[i];
      f.Tension[i]=f.Tension[i];
      f.in[i]=t*f.in[i];
      f.out[i]=t*f.out[i];
      f.straight[i]=f.straight[i];
    }
    f.cycles=f.cycles;
  };
}

guide3[] operator * (transform3 t, guide3[] g) 
{
  guide3[] G=new guide3[g.length];
  for(int i=0; i < g.length; ++i)
    G[i]=t*g[i];
  return G;
}

struct Controls {
  triple c0,c1;

  void init(triple v0, triple v1, triple d0, triple d1, real tout, real tin,
	    bool atLeast) {
    triple v=v1-v0;
    triple u=unit(v);
    real L=length(v);
    real theta=acos(dot(unit(d0),u));
    real phi=acos(dot(unit(d1),u));
    if(dot(cross(d0,v),cross(v,d1)) < 0) phi=-phi;
    c0=v0+d0*L*relativedistance(theta,phi,tout,atLeast);
    c1=v1-d1*L*relativedistance(phi,theta,tin,atLeast);
  }
}

Controls operator init() {return new Controls;}
  
path project(flatguide3 g, projection P)
{
  guide pg;
  
  // Propagate directions across nodes.
  for(int i=0; i < length(g); ++i) {
    int next=(i+1 == size(g)) ? 0 : i+1;
    if(!g.in[i].active && g.out[next].active) {
      g.in[i]=g.out[next];
      g.in[i].active=true;
    }
    if(!g.out[next].active && g.in[i].active) {
      g.out[next]=g.in[i];
      g.out[next].active=true;
    }
  }
  
  // Compute missing control points in 3D when both directions are available.
  for(int i=0; i < length(g); ++i) {
    int next=(i+1 == size(g)) ? 0 : i+1;
    if(!g.control[i].active && g.out[i].active && g.in[i].active) {
      Controls C;
      C.init(point(g,i),point(g,next),g.out[i].dir,g.in[i].dir,
	     g.Tension[i].out,g.Tension[i].in,g.Tension[i].atLeast);
      control c;
      c.init(C.c0,C.c1);
      g.control[i]=c;
    }
  }
  
  // Construct the path.
  for(int i=0; i < size(g); ++i) {
    guide join(... guide[])=g.straight[i] ? operator -- : operator ..;
    if(g.control[i].active)
      pg=join(pg,P(point(g,i))..controls P(g.control[i].post) and 
	      P(g.control[i].pre)..nullpath);
    else if(g.out[i].active)
      pg=join(pg,P(point(g,i)){P(g.out[i].dir)}..nullpath);
    else if(g.in[i].active)
      pg=join(pg,P(point(g,i))..{P(g.in[i].dir)}nullpath);
    else pg=join(pg,P(point(g,i)));
  }
  return cyclic(g) ? (g.straight[-1] ? pg--cycle : pg..cycle) : pg;
}

pair project(triple v, projection P)
{
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

path operator cast(triple v) {
  return project(v,currentprojection);
}

path operator cast(guide3 g) {
  return project(g,currentprojection);
}

path[] operator cast(guide3[] g) {
  return project(g,currentprojection);
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
