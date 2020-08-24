// Critical definitions for transform3 needed by projection and picture.

pair viewportmargin=settings.viewportmargin;

typedef real[][] transform3;
restricted transform3 identity4=identity(4);

// A uniform 3D scaling.
transform3 scale3(real s)
{
  transform3 t=identity(4);
  t[0][0]=t[1][1]=t[2][2]=s;
  return t;
}

// Simultaneous 3D scalings in the x, y, and z directions.
transform3 scale(real x, real y, real z)
{
  transform3 t=identity(4);
  t[0][0]=x;
  t[1][1]=y;
  t[2][2]=z;
  return t;
}

transform3 shiftless(transform3 t)
{
  transform3 T=copy(t);
  T[0][3]=T[1][3]=T[2][3]=0;
  return T;
}

real camerafactor=2;       // Factor used for camera adjustment.

struct transformation {
  transform3 modelview;  // For orientation and positioning
  transform3 projection; // For 3D to 2D projection
  bool infinity;
  void operator init(transform3 modelview) {
    this.modelview=modelview;
    this.projection=identity4;
    infinity=true;
  }
  void operator init(transform3 modelview, transform3 projection) {
    this.modelview=modelview;
    this.projection=projection;
    infinity=false;
  }
  transform3 compute() {
    return infinity ? modelview : projection*modelview;
  }
  transformation copy() {
    transformation T=new transformation;
    T.modelview=copy(modelview);
    T.projection=copy(projection);
    T.infinity=infinity;
    return T;
  }
}

struct projection {
  transform3 t;         // projection*modelview (cached)
  bool infinity;
  bool absolute=false;
  triple camera;        // Position of camera.
  triple up;            // A vector that should be projected to direction (0,1).
  triple target;        // Point where camera is looking at.
  triple normal;        // Normal vector from target to projection plane.
  pair viewportshift;   // Fractional viewport shift.
  real zoom=1;          // Zoom factor.
  real angle;           // Lens angle (for perspective projection).
  bool showtarget=true; // Expand bounding volume to include target?
  typedef transformation projector(triple camera, triple up, triple target);
  projector projector;
  bool autoadjust=true; // Adjust camera to lie outside bounding volume?
  bool center=false;    // Center target within bounding volume?
  int ninterpolate;     // Used for projecting nurbs to 2D Bezier curves.
  bool bboxonly=true;   // Typeset label bounding box only.
  
  transformation T;

  void calculate() {
    T=projector(camera,up,target);
    t=T.compute();
    infinity=T.infinity;
    ninterpolate=infinity ? 1 : 16;
  }

  triple vector() {
    return camera-target;
  }

  void operator init(triple camera, triple up=(0,0,1), triple target=(0,0,0),
                     triple normal=camera-target,
                     real zoom=1, real angle=0, pair viewportshift=0,
                     bool showtarget=true, bool autoadjust=true,
                     bool center=false, projector projector) {
    this.camera=camera;
    this.up=up;
    this.target=target;
    this.normal=normal;
    this.zoom=zoom;
    this.angle=angle;
    this.viewportshift=viewportshift;
    this.showtarget=showtarget;
    this.autoadjust=autoadjust;
    this.center=center;
    this.projector=projector;
    calculate();
  }

  projection copy() {
    projection P=new projection;
    P.t=t;
    P.infinity=infinity;
    P.absolute=absolute;
    P.camera=camera;
    P.up=up;
    P.target=target;
    P.normal=normal;
    P.zoom=zoom;
    P.angle=angle;
    P.viewportshift=viewportshift;
    P.showtarget=showtarget;
    P.autoadjust=autoadjust;
    P.center=center;
    P.projector=projector;
    P.ninterpolate=ninterpolate;
    P.bboxonly=bboxonly;
    P.T=T.copy();
    return P;
  }

  // Return the maximum distance of box(m,M) from target.
  real distance(triple m, triple M) {
    triple[] c={m,(m.x,m.y,M.z),(m.x,M.y,m.z),(m.x,M.y,M.z),
                (M.x,m.y,m.z),(M.x,m.y,M.z),(M.x,M.y,m.z),M};
    return max(abs(c-target));
  }
   
  
  // This is redefined here to make projection as self-contained as possible.
  static private real sqrtEpsilon = sqrt(realEpsilon);

  // Move the camera so that the box(m,M) rotated about target will always
  // lie in front of the clipping plane.
  bool adjust(triple m, triple M) {
    triple v=camera-target;
    real d=distance(m,M);
    static real lambda=camerafactor*(1-sqrtEpsilon);
    if(lambda*d >= abs(v)) {
      camera=target+camerafactor*d*unit(v);
      calculate();
      return true;
    }
    return false;
  }
}

projection currentprojection;

struct light {
  real[][] diffuse;
  real[][] specular;
  pen background=nullpen; // Background color of the 3D canvas.
  real specularfactor;
  triple[] position; // Only directional lights are currently implemented.

  transform3 T=identity(4); // Transform to apply to normal vectors.

  bool on() {return position.length > 0;}
  
  void operator init(pen[] diffuse,
                     pen[] specular=diffuse, pen background=nullpen,
                     real specularfactor=1,
                     triple[] position) {
    int n=diffuse.length;
    assert(specular.length == n && position.length == n);
    
    this.diffuse=new real[n][];
    this.specular=new real[n][];
    this.background=background;
    this.position=new triple[n];
    for(int i=0; i < position.length; ++i) {
      this.diffuse[i]=rgba(diffuse[i]);
      this.specular[i]=rgba(specular[i]);
      this.position[i]=unit(position[i]);
    }
    this.specularfactor=specularfactor;
  }

  void operator init(pen diffuse=white, pen specular=diffuse,
                     pen background=nullpen, real specularfactor=1 ...triple[] position) {
    int n=position.length;
    operator init(array(n,diffuse),array(n,specular),
                  background,specularfactor,position);
  }

  void operator init(pen diffuse=white, pen specular=diffuse,
                     pen background=nullpen, real x, real y, real z) {
    operator init(diffuse,specular,background,(x,y,z));
  }

  void operator init(explicit light light) {
    diffuse=copy(light.diffuse);
    specular=copy(light.specular);
    background=light.background;
    specularfactor=light.specularfactor;
    position=copy(light.position);
  }

  real[] background() {return rgba(background == nullpen ? white : background);}
}

light currentlight;

