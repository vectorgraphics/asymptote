import math;

// A point in three-dimensional space.
// Because of the homogenized w-coordinate, it is actually in three-dimension
// projective space.
struct triple {
  // Remove the public modifier when permissions for static functions are
  // fixed.
  
  public real x,y,z;
  //real w=1;

  static triple init(real x, real y, real z) {
    triple v=new triple;
    v.x=x; v.y=y; v.z=z;
    return v;
  }
}

triple operator init() {return new triple;}

// Allows the notation triple(x,y,z).
triple triple(real,real,real)=triple.init;

triple operator + (triple u, triple v) {
  return triple(u.x+v.x,u.y+v.y,u.z+v.z);
}

triple operator - (triple u, triple v) {
  return triple(u.x-v.x,u.y-v.y,u.z-v.z);
}

triple operator - (triple v) {
  return triple(-v.x,-v.y,-v.z);
}

triple operator * (real t, triple v) {
  return triple(t*v.x,t*v.y,t*v.z);
}

triple X=triple(1,0,0), Y=triple(0,1,0), Z=triple(0,0,1);

/*triple homogenize(triple v) {
  return v.w==1 ? v : triple(v.x/v.w,v.y/v.w,v.z/v.w);
}*/

real[] operator cast(triple v) {
  return new real[] {v.x, v.y, v.z, 1};
}

triple operator cast(real[] a) {
  //assert(a.length==4);

  real w=a[3];
  return w==1 ? triple(a[0],a[1],a[2]) : triple(a[0]/w,a[1]/w,a[2]/w);
}

typedef real[][] transform3;

triple operator * (transform3 t, triple v) {
  return t*(real[])v;
}

transform3 I=identity(4);

// A translation in 3d-space.
transform3 shift(triple v) {
  transform3 t=identity(4);
  real[] a=v;
  for (int i=0; i<3; ++i)
    t[i][3]=a[i];
  return t;
}

// A transformation representing rotating by an angle about an axis.
// Copied from http://www.cprogramming.com/tutorial/3d/rotation.html
transform3 rotate(real angle, triple axis) {
  real x=axis.x,y=axis.y,z=axis.z;
  real s=sin(angle),c=cos(angle),t=1-c;

  return new real[][] {
    {t*x^2+c,   t*x*y-s*z, t*y*z+s*y, 0},
    {t*x*y+s*z, t*y^2+c,   t*y*z-s*x, 0},
    {t*y*z-s*y, t*y*z+s*x, t*z*z+c,   0},
    {0,         0,         0,         1}};
}

// Transformation corresponding to moving the camera from the origin looking
// down the negative axis to sitting at the point "from" looking to the origin.
// Since, in actuality, we are transforming the points instead of the camera,
// we calculate the inverse matrix.
transform3 lookAtOrigin(triple from) {
  real x=from.x,y=from.y,z=from.z;
  if (x==0 && y==0)
    // Look up or down.
    return z >= 0 ? shift(-from) :
                    rotate(pi,Y) * shift(-from);
  else {
    real d=sqrt(x^2+y^2+z^2);
    return shift(triple(0,0,-d)) *
           rotate(-acos(z/d),X) *
           rotate(-angle((x,y)),Z);
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
  
path project(path3 g, projection P)
{
  guide pg;
  for (int i=0; i<size(g); ++i)
    pg=pg--P(point(g,i));
  return cyclic(g) ? pg : pg--cycle;
}

struct flatguide3 {
  public triple[] nodes;
  public bool cycles=false;

  void add(triple v) {
    nodes.push(v);
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

guide3 operator cast(triple v) {
  return new void(flatguide3 f) {
    f.add(v);
  };
}

/*
guide3 operator ^^ (triple g, triple h) {
  return new void(flatguide3 f) {
    f.nodes.push(g);
    f.nodes.push(h);
  };
}

guide3 operator ^^ (guide3 g, triple h) {
  return new void(flatguide3 f) {
  };
}
*/

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

/*{
  // A test.
  size(200,0);
  triple[] points={triple(-1,-1,0),
                   triple(1,-1,0),
                   triple(1,1,0),
                   triple(-1,1,0)};

  triple camera=triple(5,-5,2);
  projection P=perspective(1) * lookAtOrigin(camera);

  guide g;
  for (int i=0; i<points.length; ++i)
    g=g--P(points[i]);
  draw(g--cycle);
}*/

{
  // A test.
  size(200,0);
  guide3 g=triple(-1,-1,0)--triple(1,-1,0)--triple(1,1,0)--triple(-1,1,0);
 
  triple camera=triple(5,-5,2);
  projection P=perspective(1) * lookAtOrigin(camera);

  path pg=project(solve(g),P);
  draw(pg);
}
