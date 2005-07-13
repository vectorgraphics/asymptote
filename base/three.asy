import math;

triple X=(1,0,0), Y=(0,1,0), Z=(0,0,1);

real[] operator ecast(triple v) {
  return new real[] {v.x, v.y, v.z, 1};
}

triple operator ecast(real[] a) {
  if(a.length != 4) abort("vector length of "+(string) a.length+" != 4");
  return (a[0],a[1],a[2])/a[3];
}

typedef real[][] transform3;

triple operator * (transform3 t, triple v) {
  return (triple)(t*(real[]) v);
}

// A translation in 3d-space.
transform3 shift(triple v) {
  transform3 t=identity(4);
  t[0][3]=v.x;
  t[1][3]=v.y;
  t[2][3]=v.z;
  return t;
}

// A transformation representing rotation by an angle about an axis
// (in the right-handed direction).
// See http://www.cprogramming.com/tutorial/3d/rotation.html
transform3 rotate(real angle, triple axis) {
  real x=axis.x,y=axis.y,z=axis.z;
  real s=sin(angle),c=cos(angle),t=1-c;

  return new real[][] {
    {t*x^2+c,   t*x*y-s*z, t*x*z+s*y, 0},
    {t*x*y+s*z, t*y^2+c,   t*y*z-s*x, 0},
    {t*x*z-s*y, t*y*z+s*x, t*z^2+c,   0},
    {0,         0,         0,         1}};
}

// Transformation corresponding to moving the camera from the origin (looking
// down the negative z axis) to sitting at the point "from" (looking at the
// origin). Since, in actuality, we are transforming the points instead of
// the camera, we calculate the inverse matrix.
transform3 lookAtOrigin(triple from) {
  transform3 t=(from.x == 0 && from.y == 0) ? shift(-from) : 
    shift((0,0,-length(from)))*
    rotate(-pi/2,Z)*
    rotate(-colatitude(from),Y)*
    rotate(-azimuth(from),Z);
  return from.z >= 0 ? t : rotate(pi,Y)*t;
}

transform3 lookAt(triple from, triple to) {
  return lookAtOrigin(from-to)*shift(-to);
}

// Uses the homogenous coordinate to perform perspective distortion.  When
// combined with a projection to the XY plane, this effectively maps
// points in three space to a plane at a distance d from the camera.
transform3 perspective(triple camera) {
  transform3 t=identity(4);
  real d=length(camera);
  t[3][2]=-1/d;
  t[3][3]=0;
  return t*lookAtOrigin(camera);
}

transform3 orthographic(triple camera) {
  return lookAtOrigin(camera);
}

typedef pair projection(triple a);

pair projectXY(triple v) {
  return (v.x,v.y);
}

projection operator cast(transform3 t) {
  return new pair(triple v) {
    return projectXY(t*v);
  };
}

struct control {
  public triple pre,post;
  public bool active=false;
  void init(triple pre, triple post) {
    this.pre=pre;
    this.post=post;
    active=true;
  }
}

control operator init() {return new control;}

control nocontrol;
  
// Extend to cubic splines at some point.
struct path3 {
  public triple[] nodes;
  public control[] control;
  public bool cycles=false;
  
  triple pre(int i) {
    return control[i].pre;
  }
  
  triple post(int i) {
    return control[i].post;
  }
}

path3 operator init() {return new path3;}

int size(path3 g) {return g.nodes.length;}
triple point(path3 g, int k) {return g.nodes[k];}
bool cyclic(path3 g) {return g.cycles;}
int length(path3 g) {return g.cycles ? g.nodes.length : g.nodes.length-1;}
  
pair project(triple p, projection P)
{
  return P(p);
}

path project(path3 g, projection P)
{
  guide pg;
  for(int i=0; i < size(g); ++i) {
    if(g.control[i].active)
      pg=pg..P(point(g,i))..controls P(g.pre(i)) and P(g.post(i))..nullpath;
    else pg=pg--P(point(g,i));
  }
  if(cyclic(g))
    return (g.control[-1].active) ? pg..cycle : pg--cycle;
  return pg;
}

struct flatguide3 {
  public triple[] nodes;
  public control[] control;
  public bool cycles=false;

  void add(triple v) {
    nodes.push(v);
    control.push(nocontrol);
  }

  void control(triple pre, triple post) {
    control c;
    c.init(pre,post);
    control[-1]=c;
  }

  void add(path3 p) {
    if (nodes.length == 0) {
      nodes=p.nodes;
      control=p.control;
      cycles=p.cycles;
    } else {
      for(int i=0; i <= length(p); ++i)
        add(point(p,i));
    }
  }

  path3 solve() {
    path3 g;
    g.nodes=nodes;
    g.control=control;
    g.cycles=cycles;
    return g;
  }
}

flatguide3 operator init() {return new flatguide3;}
  
// A guide3 is most easily represented as something that modifies a flatguide.
typedef void guide3(flatguide3);

void nullpath3(flatguide3) {};

guide3 operator init() {return nullpath3;}

guide3 operator cast(triple v) {
  return new void(flatguide3 f) {
    f.add(v);
  };
}

guide3 operator cast(path3 p) {
  return new void(flatguide3 f) {
    f.add(p);
  };
}

guide3[] operator cast(triple[] v) {
  guide3[] g;
  for(int i=0; i < v.length; ++i)
    g[i]=v[i];
  return g;
}

guide3 operator cycle() {
  return new void(flatguide3 f) {
  f.cycles=true;
  };
}

guide3 operator controls(triple pre, triple post) {
  return new void(flatguide3 f) {
    f.control(pre,post);
  };
};
  
guide3 operator controls(triple v)
{
  return operator controls(v,v);
}

guide3 operator -- (... guide3[] g) {
  return new void(flatguide3 f) {
    // Apply the subguides in order.
    for(int i=0; i < g.length; ++i)
      g[i](f);
  };
}

guide3 operator .. (... guide3[] g) {
  return new void(flatguide3 f) {
    // Apply the subguides in order.
    for(int i=0; i < g.length; ++i)
      g[i](f);
  };
}

path3 operator cast(guide3 g) {
  flatguide3 f;
  g(f);
  return f.solve();
}

// The graph of a function along a path.
guide3 graph(real f(pair z), path p, int n=10) {
  triple F(pair z) {
    return (z.x,z.y,f(z));
  }

  guide3 g;
  for(int i=0; i < n*length(p); ++i) {
    pair z=point(p,i/n);
    g=g--F(z);
  }
  return cyclic(p) ? g--cycle : g--F(endpoint(p));
}

picture plot(real f(pair z), pair min, pair max,
             projection P, int n=20, int subn=1, 
	     pen surfacepen=lightgray, pen meshpen=currentpen)
{
  picture pic;

  void drawcell(pair a, pair b) {
    guide3 g=graph(f,box(a,b),subn);
    filldraw(pic,project(g,P),surfacepen,meshpen);
  }

  pair sample(int i, int j) {
    return (interp(min.x,max.x,i/n),
            interp(min.y,max.y,j/n));
  }

  for(int i=0; i < n; ++i)
    for(int j=0; j < n; ++j)
      drawcell(sample(i,j),sample(i+1,j+1));

  return pic;
}

/*{
  // A test.
  size(200,0);
  triple[] points={(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)};

  triple camera=(5,-5,2);
  projection P=perspective(camera);

  guide g;
  for(int i=0; i < points.length; ++i)
    g=g--P(points[i]);
  draw(g--cycle);
}*/

/*{
  // A test.
  size(200,0);
  guide3 g=(-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle;
 
  triple camera=(5,-5,2);
  projection P=perspective(camera);

  path pg=project(solve(g),P);
  draw(pg);
}*/

{
  size(200,0);
  real f(pair z) {
    return 0.5+exp(-abs(z)^2);
  }

  guide3 g=(-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle;
  guide3 eg=graph(f,(1,0)--(-1,0));
 
  triple camera=(5,4,2);
  projection P=perspective(camera);

  draw(project(g,P));

  real r=2;
  draw("$x$",project((0,0,0)--(r,0,0),P),1,red,Arrow);
  draw("$y$",project((0,0,0)--(0,r,0),P),1,red,Arrow);
  draw("$z$",project((0,0,0)--(0,0,r),P),1,red,Arrow);
  
  real a=4(sqrt(2)-1)/3;
  draw(project((1,0,0)..controls (1,a,0) and (a,1,0)..
       (0,1,0)..controls (-a,1,0) and (-1,a,0)..
       (-1,0,0)..controls (-1,-a,0) and (-a,-1,0)..
       (0,-1,0)..controls (a,-1,0) and (1,-a,0)..cycle,P),1,green);
  label("$O$",project((0,0,0),P),S);
  
  add(plot(f,(-1,-1),(1,1),P,n=10));
}
