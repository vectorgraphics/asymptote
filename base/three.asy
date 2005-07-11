import math;

triple X=(1,0,0), Y=(0,1,0), Z=(0,0,1);

real[] operator ecast(triple v) {
  return new real[] {v.x, v.y, v.z, 1};
}

triple operator ecast(real[] a) {
  //assert(a.length==4);

  real w=a[3];
  return w == 1 ? (a[0],a[1],a[2]) : (a[0]/w,a[1]/w,a[2]/w);
}

typedef real[][] transform3;

triple operator * (transform3 t, triple v) {
  return (triple)(t*(real[])v);
}

transform3 I=identity(4);

// A translation in 3d-space.
transform3 shift(triple v) {
  transform3 t=identity(4);
  real[] a=(real [])v;
  for (int i=0; i < 3; ++i)
    t[i][3]=a[i];
  return t;
}

// A transformation representing rotating by an angle about an axis.
// Copied from http://www.cprogramming.com/tutorial/3d/rotation.html
transform3 rotate(real angle, triple axis) {
  real x=axis.x,y=axis.y,z=axis.z;
  real s=sin(angle),c=cos(angle),t=1-c;

  return new real[][] {
    {t*x^2+c,   t*x*y-s*z, t*x*z+s*y, 0},
    {t*x*y+s*z, t*y^2+c,   t*y*z-s*x, 0},
    {t*x*z-s*y, t*y*z+s*x, t*z^2+c,   0},
    {0,         0,         0,         1}};
}

// Transformation corresponding to moving the camera from the origin looking
// down the negative axis to sitting at the point "from" looking to the origin.
// Since, in actuality, we are transforming the points instead of the camera,
// we calculate the inverse matrix.
transform3 lookAtOrigin(triple from) {
  if (from.x == 0 && from.y == 0)
    // Look up or down.
    return from.z >= 0 ? shift(-from) :
                    rotate(pi,Y) * shift(-from);
  else {
    real d=length(from);
    return shift((0,0,-d)) *
      rotate(-colatitude(from),X) *
           rotate(-azimuth(from),Z);
  }
}

transform3 lookAt(triple from, triple to) {
  return lookAtOrigin(from-to) * shift(-to);
}

// Uses the homogenous coordinate to perform perspective distortion.  When
// combined with the standard projection to 2d, this effectively maps points in
// three space to a plane at a distance d from the camera.
transform3 perspective(real d) {
  transform3 t=identity(4);
  t[3][2]=-1/d;
  t[3][3]=0;
  return t;
}

typedef pair projection(triple);

pair standardProjection(triple v) {
  return (v.x,v.y);
}

projection operator cast(transform3 t) {
  return new pair (triple v) {
    return standardProjection(t*v);
  };
}

// Extend to cubic splines at some point.
struct path3 {
  public triple[] nodes; 
  public bool cycles=false;
}

path3 operator init() {return new path3;}

int size(path3 g) { return g.nodes.length; }
triple point(path3 g,int k) { return g.nodes[k]; }
bool cyclic(path3 g) { return g.cycles; }
int length(path3 g) { return cyclic(g) ? size(g) : size(g)-1; }
  
path project(path3 g, projection P)
{
  guide pg;
  for (int i=0; i < size(g); ++i)
    pg=pg--P(point(g,i));
  return cyclic(g) ? pg--cycle : pg;
}

struct flatguide3 {
  public triple[] nodes;
  public bool cycles=false;

  void add(triple v) {
    nodes.push(v);
  }

  void add(path3 p) {
    if (nodes.length==0) {
      nodes=p.nodes;
      cycles=p.cycles;
    }
    else
      for (int i=0; i <= length(p); ++i)
        add(point(p, i));
  }

  path3 solve() {
    path3 g=new path3;
    g.nodes=nodes;
    g.cycles=cycles;
    return g;
  }
}

flatguide3 operator init () {return new flatguide3;}
  
// A guide is most easily represented as something that modifies a flatguide.
typedef void guide3(flatguide3);

void nullguide3(flatguide3) {};

guide3 operator init() {
  return nullguide3;
}

guide3 operator cast(triple v) {
  return new void(flatguide3 f) {
    f.add(v);
  };
}

guide3 operator cast (path3 p) {
  return new void(flatguide3 f) {
    f.add(p);
  };
}

guide3[] operator cast(triple[] v) {
  guide3[] g;
  for (int i=0; i < v.length; ++i)
    g[i]=v[i];
  return g;
}

void cycle3 (flatguide3 f) {
  f.cycles=true;
}

guide3 operator -- (... guide3[] g) {
  return new void(flatguide3 f) {
    // Apply the subguides in order.
    for (int i=0; i < g.length; ++i)
      g[i](f);
  };
}

path3 solve(guide3 g) {
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
  for (int i=0; i < n*length(p); ++i) {
    pair z=point(p,i/n);
    g=g--F(z);
  }
  return cyclic(p) ? g--cycle3 : g--F(endpoint(p));
}

picture plot(real f(pair z), pair min, pair max,
             projection P, int n=20, int subn=1)
{
  picture pic;

  void drawpath(path g) {
    fill(pic,g,grey);
    draw(pic,g);
  }

  void drawcell(pair a, pair b) {
    guide3 g=graph(f, box(a,b), subn);
    drawpath(project(solve(g), P));
  }

  pair sample(int i, int j) {
    return (interp(min.x,max.x,i/n),
            interp(min.y,max.y,j/n));
  }

  for (int i=0; i < n; ++i)
    for (int j=0; j < n; ++j)
      drawcell(sample(i,j),sample(i+1,j+1));

  return pic;
}

/*{
  // A test.
  size(200,0);
  triple[] points={(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)};

  triple camera=(5,-5,2);
  projection P=perspective(1) * lookAtOrigin(camera);

  guide g;
  for (int i=0; i < points.length; ++i)
    g=g--P(points[i]);
  draw(g--cycle);
}*/

/*{
  // A test.
  size(200,0);
  guide3 g=(-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0);
 
  triple camera=(5,-5,2);
  projection P=perspective(1) * lookAtOrigin(camera);

  path pg=project(solve(g),P);
  draw(pg);
}*/

{
  size(200,0);
  real f(pair z) {
    return exp(-abs(z)^2);
  }

  guide3 g=(-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle3;
  guide3 eg=graph(f,(1,0)--(-1,0));
 
  triple camera=(-5,4,2);
  projection P=perspective(1) * lookAtOrigin(camera);

  path pg=project(solve(g),P);
  draw(pg);

  add(plot(f,(-1,-1),(1,1),P,n=10));

  /*int n=20;
  real a=-1, b=1;
  for (int i=0; i <= n; ++i) {
    real y=a+(b-a)*i/n;
    draw(project(solve(graph(f, (1,y), (-1,y))), P), blue);
    draw(project(solve(graph(f, (y,1), (y,-1))), P), blue);
  }*/
}
