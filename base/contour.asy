/*
  Contour routines written by Radoslav Marinov, John Bowman, and Chris Savage.
 
  [2009/10/15: C Savage] generate oriented contours
  [2009/10/19: C Savage] use boxes instead of triangles
*/

/*
  Contours lines/guides are oriented throughout.  By convention,
  for a single contour, higher values are to the left and/or lower
  values are to the right along the direction of the lines/guide.
*/

import graph_settings;

private real eps=sqrtEpsilon;

/*
  GRID CONTOURS
  
  Contours on a grid of points are determined as follows: 
  for each grid square, the function is approximated as the unique
  paraboloid passing through the function values at the four
  corners.  The intersection of a paraboloid with the f(x,y)=c
  plane is a line or hyperbola.
  
  Grid data structures: 
    
  boxcontour: 
  Describes a particular contour segment in a grid square.
    
  boxdata: 
  Describes contours in a grid square (holds boxcontours).
    
  segment: 
  Describes a contour line.  Usually a closed (interior) contour,
  a line that terminates on the border, or a border segment used
  to enclose a region.
    
  Segment: 
  Describes a contour line.
  
  Main grid routines: 
    
  setcontour: 
  Determines the contours in a grid square.
    
  contouredges: 
  Determines the contour segments over a grid of function values.
    
  connect: 
  Converts contours into guides
  
*/

private typedef int boxtype;
private boxtype exterior=-1;
private boxtype edge    = 0;
private boxtype interior=+1;

private typedef int contourshape;
private contourshape line     =1;
private contourshape hyperbola=2;

// Describe position by grid square and position in square
private struct gridpoint {
  int i,j;
  pair z;
  void operator init(int i, int j, pair z) {
    this.i=i;
    this.j=j;
    this.z=z;
  }
  void operator init(gridpoint gp) {
    this.i=gp.i;
    this.j=gp.j;
    this.z=gp.z;
  }
}

private bool same(gridpoint gp1, gridpoint gp2)
{
  return abs(gp2.z-gp1.z+(gp2.i-gp1.i,gp2.j-gp1.j)) < eps;
}


// Describe contour in unit square(scaling to be done later).
private struct boxcontour {
  bool active;
  contourshape type; // Shape of contour segment(line or hyperbola)
  pair a,b;          // Start/end point of contour segment.
                     // Higher values to left along a--b.
  real x0,y0,m;      // For hyperbola: (x-x0)*(y-y0)=m
  int signx,signy;   // Sign of x-x0&y-y0 for hyperbola piece;
                     // identifies which direction it opens
  int i,j;           // Indices of lower left corner in position or
                     // data array.
  int index;         // Contour index

  void operator init(contourshape type, pair a, pair b,
                     real x0, real y0, real m, int signx, int signy,
                     int i, int j, int index) {
    this.active=true;
    this.type=type;
    this.a=a;
    this.b=b;
    
    this.x0=x0;
    this.y0=y0;
    this.m=m;
    this.signx=signx;
    this.signy=signy;
    
    this.i=i;
    this.j=j;
    this.index=index;
  }
  // Generate list of points along the line/hyperbola segment
  // representing the contour in the box
  gridpoint[] points(int subsample=1, bool first=true, bool last=true) {
    gridpoint[] gp;
    if(first)
      gp.push(gridpoint(i,j,a));
    if(subsample > 0) {
      // Linear case
      if(type == line) {
        for(int k=1; k <= subsample; ++k) {
          pair z=interp(a,b,k/(subsample+1));
          gp.push(gridpoint(i,j,z));
        }
      } else if(type == hyperbola) {
        // Special hyperbolic case of m=0
        // The contours here are infinite lines at x=x0 and y=y0,
        // but handedness always connects a semi-infinite
        // horizontal segment with a semi-infinite vertical segment
        // connected at (x0,y0).
        // If (x0,y0) is outside the unit box, there is only one
        // line segment to include; otherwise, there are both
        // a horizontal and a vertical line segment to include.
        if(m == 0) {
          // Single line
          if(a.x == b.x || a.y == b.y) {
            for(int k=1; k <= subsample; ++k) {
              pair z=interp(a,b,k/(subsample+1));
              gp.push(gridpoint(i,j,z));
            }
            // Two lines(may get one extra point here)
          } else {
            int nsub=quotient(subsample,2);
            pair mid=(x0,y0);
            for(int k=1; k <= nsub; ++k) {
              pair z=interp(a,mid,k/(nsub+1));
              gp.push(gridpoint(i,j,z));
            }
            gp.push(gridpoint(i,j,mid));
            for(int k=1; k <= nsub; ++k) {
              pair z=interp(mid,b,k/(nsub+1));
              gp.push(gridpoint(i,j,z));
            }
          }
          // General hyperbolic case (m != 0).
          // Parametric equations(m > 0): 
          //   x(t)=x0 +/- sqrt(m)*exp(t)
          //   y(t)=y0 +/- sqrt(m)*exp(-t)
          // Parametric equations (m < 0): 
          //   x(t)=x0 +/- sqrt(-m)*exp(t)
          //   y(t)=y0 -/+ sqrt(-m)*exp(-t)
          // Points will be taken equally spaced in parameter t.
        } else {
          real sqrtm=sqrt(abs(m));
          real ta=log(signx*(a.x-x0)/sqrtm);
          real tb=log(signx*(b.x-x0)/sqrtm);
          real[] t=uniform(ta,tb,subsample+1);
          for(int k=1; k <= subsample; ++k) {
            pair z=(x0+signx*sqrtm*exp(t[k]),
                    y0+signy*sqrtm*exp(-t[k]));
            gp.push(gridpoint(i,j,z));
          }
        }
      }
    }
    if(last)
      gp.push(gridpoint(i,j,b));
    
    return gp;
  }
}

// Hold data for a single grid square
private struct boxdata {
  boxtype type;      // Does box contain a contour segment (edge of
                     // contour region) or is it entirely interior/
                     // exterior to contour region ? 
  real min,max;      // Smallest/largest corner value
  real max2;         // Second-largest corner value
  boxcontour[] data; // Stores actual contour segment data
  
  int count() {return data.length;}
  void operator init(real f00, real f10, real f01, real f11) {
    real[] X={f00,f10,f01,f11};
    min=min(X);
    max=max(X);
    X.delete(find(X == max));
    max2=max(X);
  }
  void settype(real c) {
    // Interior case(f >= c)
    if(min > c-eps) {
      type=interior;
      // Exterior case(f < c)
    } else if(max < c-eps) {
      type=exterior;
      // Special case: only one corner at f=c, f < c elsewhere
      //(no segment in this case)
    } else if((max < c+eps) && (max2 < c-eps)) {
      type=exterior;
      // Edge of contour passes through box
    } else {
      type=edge;
    }
  }
}


/*
  Determine contours within a unit square box.
  
  Here, we approximate the function on the unit square to be a quadric
  surface passing through the specified values at the four corners: 
  f(x,y)=(1-x)(1-y) f00+x(1-y) f10+(1-x)y f01+xy f11
  =a0+ax x+ay y+axy xy
  where f00, f10, f01 and f11 are the function values at the four
  corners of the unit square 0 < x < 1&0 < y < 1 and: 
  a0 =f00 
  ax =f10-f00
  ay =f01-f00
  axy=f00+f11-f10-f01
  This can also be expressed in paraboloid form as: 
  f(x,y)=alpha [(x+y-cp)^2-(x-y-cn)^2]+d
  where: 
  alpha=axy/4
  cp   =-(ax+ay)/a11
  cn   =-(ax-ay)/a11
  d    =(a0 axy-ax ay)/axy
  In the procedure below, we take f00 - > f00-c etc. for a contour
  level c and we search for f=0.
  
  For this surface, there are two possible contour shapes: 
  linear:     (y-y0)/(x-x0)=m
  hyperbolic: (x-x0)*(y-y0)=m
  The linear case has a single line.  The hyperbolic case may have
  zero, one or two segments within the box (there are two sides of
  a hyperbola, each of which may or may not pass through the unit
  square).  A hyperbola with m=0 is a special case that is handled
  separately below.
  
  If c0 is the desired contour level, we effectively find the
  contours at c0-epsilon for arbitrarily small epsilon.  Flat
  regions equal to c0 are considered to be interior to the
  contour curves.  Regions that lie at the contour level are
  considered to be interior to the contour curves.  As a result,
  contours are only constructed if they are immediately adjacent
  to some region interior to the square that falls below the
  contour value; in other words, if an edge falls on the contour
  value, but a point within the square arbitrarily close to the
  edge falls above the contour value, that edge (or applicable
  portion) is not included.  This requirement gives the following: 
  *) ensures contours on an edge are unique (do not appear in
  an adjacent square with the same orientation)
  *) no three line vertices (four line vertices are possible, but
  are not usually an issue)
  *) all segments can be joined into closed curves or curves that
  terminate on the boundary (no unclosed curves terminate in
  the interior region of the grid)
  
  Note the logic below skips cases that have been filtered out
  by the boxdata.settype() routine.
*/
private void setcontour(real f00, real f10, real f01, real f11, real epsf,
                        boxdata bd, int i, int j, int index) {
  // SPECIAL CASE: two diagonal corners at the contour level with
  // the other two below does not yield any contours within the
  // unit box, but may have been previously misidentified as an
  // edge containing region.
  if(((f00*f11 == 0) && (f10*f01 > 0)) || ((f01*f10 == 0) && (f00*f11 > 0))) {
    bd.type=exterior;
    return;
  }
  
  // NOTE: From this point on, we can assume at least one contour
  // segment exists in the square.  This allows several cases to
  // be ignored or simplified below, particularly edge cases.
  
  // Form used to approximate function on unit square
  real F(real x, real y) {
    return interp(interp(f00,f10,x),interp(f01,f11,x),y);
  }
  
  // Write contour as  a0+ax*x+ay*y +axy*x*y=0
  real a0 =f00;
  real ax =f10-f00;
  if(abs(ax) < epsf) ax=0;
  real ay =f01-f00;
  if(abs(ay) < epsf) ay=0;
  real axy=f00+f11-f01 -f10;
  if(abs(axy) < epsf) axy=0;
  
  // Linear contour(s)
  if(axy == 0) {
    pair a,b;
    // Horizontal
    if(ax == 0) {
      if(ay == 0) return; // Contour is at most an isolated point; ignore.
      real y0=-a0/ay;
      if(abs(y0-1) < eps) y0=1;
      if((f00 > 0) || (f01 < 0)) {
        a=(1,y0);
        b=(0,y0);
      } else {
        a=(0,y0);
        b=(1,y0);
      }
      // Vertical
    } else if(ay == 0) {
      real x0=-a0/ax;
      if(abs(x0-1) < eps) x0=1;
      if((f00 > 0) || (f10 < 0)) {
        a=(x0,0);
        b=(x0,1);
      } else {
        a=(x0,1);
        b=(x0,0);
      }
      // Angled line
    } else {
      real x0=-a0/ax;
      if(abs(x0-1) < eps) x0=1;
      real y0=-a0/ay;
      if(abs(y0-1) < eps) y0=1;
      int count=0;
      real[] farr={f00,f10,f11,f01};
      farr.cyclic=true;
      pair[] corners={(0,0),(1,0),(1,1),(0,1)};
      pair[] sidedir={(1,0),(0,1),(-1,0),(0,-1)};
      
      int count=0;
      for(int i=0; i < farr.length; ++i) {
        // Corner
        if(farr[i] == 0) {
          ++count;
          if(farr[i-1] > 0) {
            a=corners[i];
          } else {
            b=corners[i];
          }
          // Side
        } else if(farr[i]*farr[i+1] < 0) {
          ++count;
          if(farr[i] > 0) {
            a=corners[i]-(farr[i]/(farr[i+1]-farr[i]))*sidedir[i];
          } else {
            b=corners[i]-(farr[i]/(farr[i+1]-farr[i]))*sidedir[i];
          }
        }
      }
      // Check(if logic is correct above, this will not happen)
      if(count != 2) {
        abort("Unexpected error in setcontour routine: odd number of"
              +" crossings (linear case)");
      }
    }
    boxcontour bc=boxcontour(line,a,b,0,0,0,1,1,i,j,index);
    bd.data.push(bc);
    return;
  }
  
  // Hyperbolic contour(s)
  // Described in form: (x-x0)*(y-y0)=m
  real x0=-ay/axy;
  if(abs(x0-1) < eps) x0=1;
  real y0=-ax/axy;
  if(abs(y0-1) < eps) y0=1;
  real m =ay*ax-a0*axy;
  m=(abs(m) < eps) ? 0 : m/axy^2;
  
  // Special case here: straight segments (possibly crossing)
  if(m == 0) {
    pair a,b;
    int signx,signy;
    // Assuming at least one corner is below contour level here
    if(x0 == 0) {
      signx=+1;
      if(y0 == 0) {
        a=(1,0);
        b=(0,1);
        signy=+1;
      } else if(y0 == 1) {
        a=(0,0);
        b=(1,1);
        signy=-1;
      } else if(y0 < 0 || y0 > 1) {
        a=(0,0);
        b=(0,1);
        signy=y0 > 0 ? -1 : +1;
      } else {
        if(f10 > 0) {
          a=(1,y0);
          b=(0,1);
          signy=+1;
        } else {
          a=(0,0);
          b=(1,y0);
          signy=-1;
        }
      }
      boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,signx,signy,i,j,index);
      bd.data.push(bc);
      return;
    } else if(x0 == 1) {
      signx=-1;
      if(y0 == 0) {
        a=(1,1);
        b=(0,0);
        signy=+1;
      } else if(y0 == 1) {
        a=(0,1);
        b=(1,0);
        signy=-1;
      } else if(y0 < 0 || y0 > 1) {
        a=(1,1);
        b=(1,0);
        signy=y0 > 0 ? -1 : +1;
      } else {
        if(f01 > 0) {
          a=(0,y0);
          b=(1,0);
          signy=-1;
        } else {
          a=(1,1);
          b=(0,y0);
          signy=+1;
        }
      }
      boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,signx,signy,i,j,index);
      bd.data.push(bc);
      return;
    } else if(y0 == 0) {
      signy=+1;
      if(x0 < 0 || x0 > 1) {
        a=(1,0);
        b=(0,0);
        signx=x0 > 0 ? -1 : +1;
      } else {
        if(f11 > 0) {
          a=(x0,1);
          b=(0,0);
          signx=-1;
        } else {
          a=(1,0);
          b=(x0,1);
          signx=+1;
        }
      }
      boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,signx,signy,i,j,index);
      bd.data.push(bc);
      return;
    } else if(y0 == 1) {
      signy=-1;
      if(x0 < 0 || x0 > 1) {
        a=(0,1);
        b=(1,1);
        signx=x0 > 0 ? -1 : +1;
      } else {
        if(f00 > 0) {
          a=(x0,0);
          b=(1,1);
          signx=+1;
        } else {
          a=(0,1);
          b=(x0,0);
          signx=-1;
        }
      }
      boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,signx,signy,i,j,index);
      bd.data.push(bc);
      return;
    } else if(x0 < 0 || x0 > 1) {
      signx=x0 > 0 ? -1 : +1;
      if(f00 > 0) {
        a=(1,y0);
        b=(0,y0);
        signy=+1;
      } else {
        a=(0,y0);
        b=(1,y0);
        signy=-1;
      }
      boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,signx,signy,i,j,index);
      bd.data.push(bc);
      return;
    } else if(y0 < 0 || y0 > 1) {
      signy=y0 > 0 ? -1 : +1;
      if(f00 > 0) {
        a=(x0,0);
        b=(x0,1);
        signx=+1;
      } else {
        a=(x0,1);
        b=(x0,0);
        signx=-1;
      }
      boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,signx,signy,i,j,index);
      bd.data.push(bc);
      return;
    } else {
      if(f10 > 0) {
        a=(0,y0);
        b=(x0,0);
        boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,-1,-1,i,j,index);
        bd.data.push(bc);
        a=(1,y0);
        b=(x0,1);
        bc=boxcontour(hyperbola,a,b,x0,y0,m,+1,+1,i,j,index);
        bd.data.push(bc);
        return;
      } else {
        a=(x0,0);
        b=(1,y0);
        boxcontour bc=boxcontour(hyperbola,a,b,x0,y0,m,+1,-1,i,j,index);
        bd.data.push(bc);
        a=(x0,1);
        b=(0,y0);
        bc=boxcontour(hyperbola,a,b,x0,y0,m,-1,+1,i,j,index);
        bd.data.push(bc);
        return;
      }
    }
  }
  
  // General hyperbola case
  int signc=(F(x0,y0) > 0) ? +1 : -1;
  
  pair[] points;
  
  real xB=(y0 == 0) ? infinity : x0-m/y0;
  if(abs(xB) < eps) xB=0;
  if(xB >= 0 && xB <= 1-eps) points.push((xB,0));

  real xT=(y0 == 1) ? infinity : x0+m/(1-y0);
  if(abs(xT-1) < eps) xT=1;
  if(xT >= eps && xT <= 1) points.push((xT,1));

  real yL=(x0 == 0) ? infinity : y0-m/x0;
  if(abs(yL-1) < eps) yL=1;
  
  if(yL > eps && yL <= 1) points.push((0,yL));

  real yR=(x0 == 1) ? infinity : y0+m/(1-x0);
  if(abs(yR) < eps) yR=0;
  if(yR >= 0 && yR <= 1-eps) points.push((1,yR));

  // Check (if logic is correct above, this will not happen)
  if(!(points.length == 2 || points.length == 4)) {
    abort("Unexpected error in setcontour routine: odd number of"
          +" crossings (hyperbolic case)");
  }
  
  // Lower left side
  if((x0 > 0) && (y0 > 0) && (f00*signc < 0)) {
    pair[] pts0;
    for(int i=0; i < points.length; ++i) {
      if((points[i].x < x0) && (points[i].y < y0)) {
        pts0.push(points[i]);
      }
    }
    if(pts0.length == 2) {
      pair a0,b0;
      if((f00 > 0) ^(pts0[0].x < pts0[1].x)) {
        a0=pts0[0];
        b0=pts0[1];
      } else {
        a0=pts0[1];
        b0=pts0[0];
      }
      boxcontour bc=boxcontour(hyperbola,a0,b0,x0,y0,m,-1,-1,i,j,index);
      bd.data.push(bc);
    }
  }
  
  // Lower right side
  if((x0 < 1) && (y0 > 0) && (f10*signc < 0)) {
    pair[] pts0;
    for(int i=0; i < points.length; ++i) {
      if((points[i].x > x0) && (points[i].y < y0)) {
        pts0.push(points[i]);
      }
    }
    if(pts0.length == 2) {
      pair a0,b0;
      if((f10 > 0) ^(pts0[0].x < pts0[1].x)) {
        a0=pts0[0];
        b0=pts0[1];
      } else {
        a0=pts0[1];
        b0=pts0[0];
      }
      boxcontour bc=boxcontour(hyperbola,a0,b0,x0,y0,m,+1,-1,i,j,index);
      bd.data.push(bc);
    }
  }
  
  // Upper right side
  if((x0 < 1) && (y0 < 1) && (f11*signc < 0)) {
    pair[] pts0;
    for(int i=0; i < points.length; ++i) {
      if((points[i].x > x0) && (points[i].y > y0)) {
        pts0.push(points[i]);
      }
    }
    if(pts0.length == 2) {
      pair a0,b0;
      if((f11 > 0) ^(pts0[0].x > pts0[1].x)) {
        a0=pts0[0];
        b0=pts0[1];
      } else {
        a0=pts0[1];
        b0=pts0[0];
      }
      boxcontour bc=boxcontour(hyperbola,a0,b0,x0,y0,m,+1,+1,i,j,index);
      bd.data.push(bc);
    }
  }
  
  // Upper left side
  if((x0 > 0) && (y0 < 1) && (f01*signc < 0)) {
    pair[] pts0;
    for(int i=0; i < points.length; ++i) {
      if((points[i].x < x0) && (points[i].y > y0)) {
        pts0.push(points[i]);
      }
    }
    if(pts0.length == 2) {
      pair a0,b0;
      if((f01 > 0) ^(pts0[0].x > pts0[1].x)) {
        a0=pts0[0];
        b0=pts0[1];
      } else {
        a0=pts0[1];
        b0=pts0[0];
      }
      boxcontour bc=boxcontour(hyperbola,a0,b0,x0,y0,m,-1,+1,i,j,index);
      bd.data.push(bc);
    }
  }
  return;
}


// Checks if end of first contour segment matches the beginning of
// the second.
private bool connected(boxcontour bc1, boxcontour bc2) {
  return abs(bc2.a-bc1.b+(bc2.i-bc1.i,bc2.j-bc1.j)) < eps;
}

// Returns index of first active element in bca that with beginning
// that connects to the end of bc, or -1 if no such element.
private int connectedindex(boxcontour bc, boxcontour[] bca,
                           bool activeonly=true) {
  for(int i=0; i < bca.length; ++i) {
    if(!bca[i].active) continue;
    if(connected(bc,bca[i])) {
      return i;
    }
  }
  return -1;
}

// Returns index of first active element in bca with end that connects
// to the start of bc, or -1 if no such element.
private int connectedindex(boxcontour[] bca, boxcontour bc,
                           bool activeonly=true) {
  for(int i=0; i < bca.length; ++i) {
    if(!bca[i].active) continue;
    if(connected(bca[i],bc)) {
      return i;
    }
  }
  return -1;
}


// Processes indices for grid regions touching the
// end/start (forward=true/false) of the contour segment
private void searchindex(boxcontour bc, bool forward, void f(int i, int j)) {
  pair z=forward ? bc.b : bc.a;
  
  int i=bc.i;
  int j=bc.j;

  if(z == (0,0)) f(i-1,j-1);
  if(z.y == 0) f(i,j-1);
  if(z == (1,0)) f(i+1,j-1);
  if(z.x == 1) f(i+1,j);
  if(z == (1,1)) f(i+1,j+1);
  if(z.y == 1) f(i,j+1);
  if(z == (0,1)) f(i-1,j+1);
  if(z.x == 0) f(i-1,j);
}

// Contour segment
private struct segment {
  gridpoint[] data;
  void operator init() {
  }
  void operator init(boxcontour bc, int subsample=1) {
    bc.active=false;
    this.data.append(bc.points(subsample,first=true,last=true));
  }
  void operator init(int i, int j, pair z) {
    gridpoint gp=gridpoint(i,j,z);
    data.push(gp);
  }
  void operator init(gridpoint[] gp) {
    this.data.append(gp);
  }
  gridpoint start() {
    if(data.length == 0) {
      return gridpoint(-1,-1,(-infinity,-infinity));
    }
    gridpoint gp=data[0];
    return gridpoint(gp.i,gp.j,gp.z);
  }
  gridpoint end() {
    if(data.length == 0) {
      return gridpoint(-1,-1,(-infinity,-infinity));
    }
    gridpoint gp=data[data.length-1];
    return gridpoint(gp.i,gp.j,gp.z);
  }
  bool closed() {
    return same(this.start(),this.end());
  }
  void append(boxcontour bc, int subsample=1) {
    bc.active=false;
    data.append(bc.points(subsample,first=false,last=true));
  }
  void prepend(boxcontour bc, int subsample=1) {
    bc.active=false;
    data.insert(0 ... bc.points(subsample,first=true,last=false));
  }
  void append(int i, int j, pair z) {
    gridpoint gp=gridpoint(i,j,z);
    data.push(gp);
  }
  void prepend(int i, int j, pair z) {
    gridpoint gp=gridpoint(i,j,z);
    data.insert(0,gp);
  }
  segment copy() {
    segment seg=new segment;
    seg.data=new gridpoint[data.length];
    for(int i=0; i < data.length; ++i) {
      seg.data[i]=gridpoint(data[i].i,data[i].j,data[i].z);
    }
    return seg;
  }
  segment reversecopy() {
    segment seg=new segment;
    seg.data=new gridpoint[data.length];
    for(int i=0; i < data.length; ++i) {
      seg.data[data.length-i-1]=gridpoint(data[i].i,data[i].j,data[i].z);
    }
    return seg;
  }
}

// Container to hold edge and border segments that form one continuous line
private struct Segment {
  segment[] edges;
  segment[] borders;
  void operator init() {
  }
  void operator init(segment seg) {
    edges.push(seg);
  }
  void operator init(gridpoint[] gp) {
    segment seg=segment(gp);
    edges.push(seg);
  }
  gridpoint start() {
    if(edges.length == 0) {
      if(borders.length > 0) {
        return borders[0].start();
      }
      return gridpoint(-1,-1,(-infinity,-infinity));
    }
    return edges[0].start();
  }
  gridpoint end() {
    if(edges.length == 0 && borders.length == 0) {
      return gridpoint(-1,-1,(-infinity,-infinity));
    }
    if(edges.length > borders.length) {
      return edges[edges.length-1].end();
    } else {
      return borders[borders.length-1].end();
    }
  }
  bool closed() {
    return same(this.start(),this.end());
  }
  void addedge(segment seg) {
    edges.push(seg);
  }
  void addedge(gridpoint[] gp) {
    segment seg=segment(gp);
    edges.push(seg);
  }
  void addborder(segment seg) {
    borders.push(seg);
  }
  void addborder(gridpoint[] gp) {
    segment seg=segment(gp);
    borders.push(seg);
  }
  void append(Segment S) {
    edges.append(S.edges);
    borders.append(S.borders);
  }
}

private Segment[] Segment(segment[] s)
{
  return sequence(new Segment(int i) {return Segment(s[i]);},s.length);
}

private Segment[][] Segment(segment[][] s)
{
  Segment[][] S=new Segment[s.length][];
  for(int i=0; i < s.length; ++i)
    S[i]=Segment(s[i]);
  return S;
}

// Return contour points for a 2D data array.
// f:         two-dimensional array of corresponding f(z) data values
// c:         array of contour values
// subsample: number of points to use in each box in addition to endpoints
segment[][] contouredges(real[][] f, real[] c, int subsample=1)
{
  int nx=f.length-1;
  if(nx <= 0)
    abort("array f must have length >= 2");
  int ny=f[0].length-1;
  if(ny <= 0)
    abort("array f[0] must have length >= 2");

  c=sort(c);
  boxdata[][] bd=new boxdata[nx][ny];
  
  segment[][] result=new segment[c.length][];
  
  for(int i=0; i < nx; ++i) {
    boxdata[] bdi=bd[i];
    real[] fi=f[i];
    real[] fp=f[i+1];
    
    for(int j=0; j < ny; ++j) {
      boxdata bdij=bdi[j]=boxdata(fi[j],fp[j],fi[j+1],fp[j+1]);

      int checkcell(int cnt) {
        real C=c[cnt];
    
        real f00=fi[j];
        real f10=fp[j];
        real f01=fi[j+1];
        real f11=fp[j+1];

        real epsf=eps*max(abs(f00),abs(f10),abs(f01),abs(f11),abs(C));

        f00=f00-C;
        f10=f10-C;
        f01=f01-C;
        f11=f11-C;
  
        if(abs(f00) < epsf) f00=0;
        if(abs(f10) < epsf) f10=0;
        if(abs(f01) < epsf) f01=0;
        if(abs(f11) < epsf) f11=0;


        int countm=0;
        int countz=0;
        int countp=0;

        void check(real vertdat) {
          if(vertdat < -eps)++countm;
          else {
            if(vertdat <= eps)++countz; 
            else++countp;
          }
        }
        
        check(f00);
        check(f10);
        check(f01);
        check(f11);

        if(countm == 4) return 1;  // nothing to do 
        if(countp == 4) return -1; // nothing to do 
        if((countm == 3 || countp == 3) && countz == 1) return 0;

        // Calculate individual box contours
        bdij.settype(C);
        if(bdij.type == edge)
          setcontour(f00,f10,f01,f11,epsf,bdij,i,j,cnt);
        return 0;
      }
  
      void process(int l, int u) {
        if(l >= u) return;
        int i=quotient(l+u,2);
        int sign=checkcell(i);
        if(sign == -1) process(i+1,u);
        else if(sign == 1) process(l,i);
        else {
          process(l,i);
          process(i+1,u);
        }
      }
  
      process(0,c.length);
    }
  }
  
  // Find contours and follow them
  for(int i=0; i < nx; ++i) {
    boxdata[] bdi=bd[i];
    for(int j=0; j < ny; ++j) {
      boxdata bd0=bdi[j];
      if(bd0.count() == 0) continue;
      for(int k=0; k < bd0.count(); ++k) {
        boxcontour bc0=bd0.data[k];
          
        if(!bc0.active) continue;
          
        // Note: boxcontour set inactive when added to segment
        segment seg=segment(bc0,subsample);
          
        // Forward direction
        bool foundnext=true;
        while(foundnext) {
          foundnext=false;
          searchindex(bc0,true,new void(int i, int j) {
              if((i >= 0) && (i < nx) && (j >= 0) && (j < ny)) {
                boxcontour[] data=bd[i][j].data;
                int k0=connectedindex(bc0,data);
                if(k0 >= 0) {
                  bc0=data[k0];
                  seg.append(bc0,subsample);
                  foundnext=true;
                }
              }
            });
        }
          
        // Backward direction
        bc0=bd0.data[k];
        bool foundprev=true;
        while(foundprev) {
          foundprev=false;
          searchindex(bc0,false,new void(int i, int j) {
              if((i >= 0) && (i < nx) && (j >= 0) && (j < ny)) {
                boxcontour[] data=bd[i][j].data;
                int k0=connectedindex(data,bc0);
                if(k0 >= 0) {
                  bc0=data[k0];
                  seg.prepend(bc0,subsample);
                  foundprev=true;
                }
              }
            });
        }

        result[bc0.index].push(seg);
      }
    }
  }
  
  // Note: every segment here _should_ be cyclic or terminate on the
  // boundary
  return result;
}

// Connect contours into guides.
// Same initial/final points indicates a closed path.
// Borders are always joined using--.
private guide connect(Segment S, pair[][] z, interpolate join)
{
  pair loc(gridpoint gp) {
    pair offset=z[gp.i][gp.j];
    pair size=z[gp.i+1][gp.j+1]-z[gp.i][gp.j];
    return offset+(size.x*gp.z.x,size.y*gp.z.y);
  }
  pair[] loc(gridpoint[] gp) {
    pair[] result=new pair[gp.length];
    for(int i; i < gp.length; ++i) {
      result[i]=loc(gp[i]);
    }
    return result;
  }
  
  bool closed=S.closed();
  
  pair[][] edges=new pair[S.edges.length][];
  for(int i; i < S.edges.length; ++i) {
    edges[i]=loc(S.edges[i].data);
  }
  pair[][] borders=new pair[S.borders.length][];
  for(int i; i < S.borders.length; ++i) {
    borders[i]=loc(S.borders[i].data);
  }
  
  if(edges.length == 0 && borders.length == 1) {
    guide g=operator--(...borders[0]);
    if(closed) g=g--cycle;
    return g;
  }
  
  if(edges.length == 1 && borders.length == 0) {
    pair[] pts=edges[0];
    if(closed) pts.delete(pts.length-1);
    guide g=join(...pts);
    if(closed) g=join(g,cycle);
    return g;
  }
  
  guide[] ge=new guide[edges.length];
  for(int i=0; i < ge.length; ++i)
    ge[i]=join(...edges[i]);

  guide[] gb=new guide[borders.length];
  for(int i=0; i < gb.length; ++i)
    gb[i]=operator--(...borders[i]);
  
  guide g=ge[0];
  if(0 < gb.length) g=g&gb[0];
  for(int i=1; i < ge.length; ++i) {
    g=g&ge[i];
    if(i < gb.length) g=g&gb[i];
  }
  if(closed) g=g&cycle;
  return g;
}

// Connect contours into guides.
private guide[] connect(Segment[] S, pair[][] z, interpolate join)
{
  return sequence(new guide(int i) {return connect(S[i],z,join);},S.length);
}

// Connect contours into guides.
private guide[][] connect(Segment[][] S, pair[][] z, interpolate join)
{
  guide[][] result=new guide[S.length][];
  for(int i=0; i < S.length; ++i) {
    result[i]=connect(S[i],z,join);
  }
  return result;
}

// Return contour guides for a 2D data array.
// z:         two-dimensional array of nonoverlapping mesh points
// f:         two-dimensional array of corresponding f(z) data values
// c:         array of contour values
// join:      interpolation operator (e.g. operator--or operator ..)
// subsample: number of interior points to include in each grid square
//           (in addition to points on edge)
guide[][] contour(pair[][] z, real[][] f, real[] c,
                  interpolate join=operator--, int subsample=1)
{
  segment[][] seg=contouredges(f,c,subsample);
  Segment[][] Seg=Segment(seg);
  return connect(Seg,z,join);
}

// Return contour guides for a 2D data array on a uniform lattice
// f:         two-dimensional array of real data values
// a,b:       diagonally opposite vertices of rectangular domain
// c:         array of contour values
// join:      interpolation operator (e.g. operator--or operator ..)
// subsample: number of interior points to include in each grid square
//           (in addition to points on edge)
guide[][] contour(real[][] f, pair a, pair b, real[] c,
                  interpolate join=operator--, int subsample=1)
{
  int nx=f.length-1;
  if(nx == 0)
    abort("array f must have length >= 2");
  int ny=f[0].length-1;
  if(ny == 0)
    abort("array f[0] must have length >= 2");

  pair[][] z=new pair[nx+1][ny+1];
  for(int i=0; i <= nx; ++i) {
    pair[] zi=z[i];
    real xi=interp(a.x,b.x,i/nx);
    for(int j=0; j <= ny; ++j) {
      zi[j]=(xi,interp(a.y,b.y,j/ny));
    }
  }
  return contour(z,f,c,join,subsample);
}

// return contour guides for a real-valued function
// f:         real-valued function of two real variables
// a,b:       diagonally opposite vertices of rectangular domain
// c:         array of contour values
// nx,ny:     number of subdivisions in x and y directions(determines accuracy)
// join:      interpolation operator (e.g. operator--or operator ..)
// subsample: number of interior points to include in each grid square
//           (in addition to points on edge)
guide[][] contour(real f(real, real), pair a, pair b,
                  real[] c, int nx=ngraph, int ny=nx,
                  interpolate join=operator--, int subsample=1)
{
  // evaluate function at points and subsample
  real[][] dat=new real[nx+1][ny+1];
  
  for(int i=0; i <= nx; ++i) {
    real x=interp(a.x,b.x,i/nx);
    real[] dati=dat[i];
    for(int j=0; j <= ny; ++j) {
      dati[j]=f(x,interp(a.y,b.y,j/ny));
    }
  }

  return contour(dat,a,b,c,join,subsample);
}

guide[][] contour(real f(pair), pair a, pair b,
                  real[] c, int nx=ngraph, int ny=nx,
                  interpolate join=operator--, int subsample=1)
{
  return contour(new real(real x, real y) {return f((x,y));},
                 a,b,c,nx,ny,join,subsample);
}

void draw(picture pic=currentpicture, Label[] L=new Label[],
          guide[][] g, pen[] p)
{
  begingroup(pic);
  for(int cnt=0; cnt < g.length; ++cnt) {
    guide[] gcnt=g[cnt];
    pen pcnt=p[cnt];
    for(int i=0; i < gcnt.length; ++i)
      draw(pic,gcnt[i],pcnt);
    if(L.length > 0) {
      Label Lcnt=L[cnt];
      for(int i=0; i < gcnt.length; ++i) {
        if(Lcnt.s != "" && size(gcnt[i]) > 1)
          label(pic,Lcnt,gcnt[i],pcnt);
      }
    }
  }
  endgroup(pic);
}

void draw(picture pic=currentpicture, Label[] L=new Label[],
          guide[][] g, pen p=currentpen)
{
  draw(pic,L,g,sequence(new pen(int) {return p;},g.length));
}

// Draw the contour
void draw(picture pic=currentpicture, Label L,
          guide[] g, pen p=currentpen)
{
  draw(pic,g,p);
  for(int i=0; i < g.length; ++i) {
    if(L.s != "" && size(g[i]) > 1) {
      label(pic,L,g[i],p);
    }
  }
}

/* CONTOURS FOR IRREGULARLY SPACED POINTS  */
//                            
//               +---------+  
//               |\       /|
//               | \     / |
//               |  \   /  |
//               |   \ /   |
//               |    X    |  
//               |   / \   |
//               |  /   \  |
//               | /     \ |
//               |/       \|
//               +---------+       
//                            

// Is triangle p0--p1--p2--cycle counterclockwise ? 
private bool isCCW(pair p0, pair p1, pair p2) {return side(p0,p1,p2) < 0;}

private struct segment
{
  bool active;
  bool reversed;   // True if lower values are to the left along line a--b.
  pair a,b;        // Endpoints; a is always an edge point if one exists.
  int c;           // Contour value.
}

// Case 1: line passes through two vertices of a triangle
private segment case1(pair p0, pair p1, pair p2,
                      real v0, real v1, real v2)
{
  // Will cause a duplicate guide; luckily case1 is rare
  segment rtrn;
  rtrn.active=true;
  rtrn.a=p0;
  rtrn.b=p1;
  rtrn.reversed=(isCCW(p0,p1,p2) ^(v2 > 0));
  return rtrn;
}

// Cases 2 and 3: line passes through a vertex and a side of a triangle
//(the first vertex passed and the side between the other two)
private segment case2(pair p0, pair p1, pair p2,
                      real v0, real v1, real v2)
{
  segment rtrn;
  rtrn.active=true;
  pair val=interp(p1,p2,abs(v1/(v2-v1)));
  rtrn.a=val;
  rtrn.b=p0;
  rtrn.reversed=!(isCCW(p0,p1,p2) ^(v2 > 0));
  return rtrn;
}

private segment case3(pair p0, pair p1, pair p2,
                      real v0, real v1, real v2)
{
  segment rtrn;
  rtrn.active=true;
  pair val=interp(p1,p2,abs(v1/(v2-v1)));
  rtrn.a=p0;
  rtrn.b=val;
  rtrn.reversed=(isCCW(p0,p1,p2) ^(v2 > 0));
  return rtrn;
}

// Case 4: line passes through two sides of a triangle
//(through the sides formed by the first&second, and second&third vertices)
private segment case4(pair p0, pair p1, pair p2,
                      real v0, real v1, real v2)
{
  segment rtrn;
  rtrn.active=true;
  rtrn.a=interp(p1,p0,abs(v1/(v0-v1)));
  rtrn.b=interp(p1,p2,abs(v1/(v2-v1)));
  rtrn.reversed=(isCCW(p0,p1,p2) ^(v2 > 0));
  return rtrn;
}

// Check if a line passes through a triangle, and draw the required line.
private segment checktriangle(pair p0, pair p1, pair p2,
                              real v0, real v1, real v2)
{
  // default null return  
  static segment dflt;

  real eps=eps*max(abs(v0),abs(v1),abs(v2),1);
  
  if(v0 < -eps) {
    if(v1 < -eps) {
      if(v2 < -eps) return dflt; // nothing to do
      else if(v2 <= eps) return dflt; // nothing to do
      else return case4(p0,p2,p1,v0,v2,v1);
    } else if(v1 <= eps) {
      if(v2 < -eps) return dflt; // nothing to do
      else if(v2 <= eps) return case1(p1,p2,p0,v1,v2,v0);
      else return case3(p1,p0,p2,v1,v0,v2);
    } else {
      if(v2 < -eps) return case4(p0,p1,p2,v0,v1,v2);
      else if(v2 <= eps) 
        return case2(p2,p0,p1,v2,v0,v1);
      else return case4(p1,p0,p2,v1,v0,v2);
    } 
  } else if(v0 <= eps) {
    if(v1 < -eps) {
      if(v2 < -eps) return dflt; // nothing to do
      else if(v2 <= eps) return case1(p0,p2,p1,v0,v2,v1);
      else return case2(p0,p1,p2,v0,v1,v2);
    } else if(v1 <= eps) {
      if(v2 < -eps) return case1(p0,p1,p2,v0,v1,v2);
      else if(v2 <= eps) return dflt; // use finer partitioning.
      else return case1(p0,p1,p2,v0,v1,v2);
    } else {
      if(v2 < -eps) return case2(p0,p1,p2,v0,v1,v2);
      else if(v2 <= eps) return case1(p0,p2,p1,v0,v2,v1);
      else return dflt; // nothing to do
    } 
  } else {
    if(v1 < -eps) {
      if(v2 < -eps) return case4(p1,p0,p2,v1,v0,v2);
      else if(v2 <= eps)
        return case2(p2,p0,p1,v2,v0,v1);
      else return case4(p0,p1,p2,v0,v1,v2);
    } else if(v1 <= eps) {
      if(v2 < -eps) return case3(p1,p0,p2,v1,v0,v2);
      else if(v2 <= eps) return case1(p1,p2,p0,v1,v2,v0);
      else return dflt; // nothing to do
    } else {
      if(v2 < -eps) return case4(p0,p2,p1,v0,v2,v1);
      else if(v2 <= eps) return dflt; // nothing to do
      else return dflt; // nothing to do
    } 
  }      
}

// Collect connecting path segments.
private void collect(pair[][][] points, real[] c)
{
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] gdscnt=points[cnt];
    for(int i=0; i < gdscnt.length; ++i) {
      pair[] gig=gdscnt[i];
      int Li=gig.length;
      for(int j=i+1; j < gdscnt.length; ++j) {
        pair[] gjg=gdscnt[j];
        int Lj=gjg.length;
        if(abs(gig[0]-gjg[Lj-1]) < eps) {
          gig.delete(0);
          gdscnt[j].append(gig);
          gdscnt.delete(i);
          --i;
          break;
        } else if(abs(gig[Li-1]-gjg[0]) < eps) {
          gjg.delete(0);
          gig.append(gjg);
          gdscnt[j]=gig;
          gdscnt.delete(i);
          --i;
          break;
        }
      }
    }
  }
}

// Join path segments.
private guide[][] connect(pair[][][] points, real[] c, interpolate join)
{
  // set up return value
  guide[][] result=new guide[c.length][];
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] pointscnt=points[cnt];
    guide[] resultcnt=result[cnt]=new guide[pointscnt.length];
    for(int i=0; i < pointscnt.length; ++i) {
      pair[] pts=pointscnt[i];
      guide gd;
      if(pts.length > 0) {
        if(pts.length > 1 && abs(pts[0]-pts[pts.length-1]) < eps) {
          guide[] g=sequence(new guide(int i) {
              return pts[i];
            },pts.length-1);
          g.push(cycle);
          gd=join(...g);
        } else
          gd=join(...sequence(new guide(int i) {
                return pts[i];
              },pts.length));
      }
      resultcnt[i]=gd;
    }
  }
  return result;
}

guide[][] contour(pair[] z, real[] f, real[] c, interpolate join=operator--)
{
  if(z.length != f.length)
    abort("z and f arrays have different lengths");

  int[][] trn=triangulate(z);

  // array to store guides found so far
  pair[][][] points=new pair[c.length][][];
        
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] pointscnt=points[cnt];
    real C=c[cnt];
    for(int i=0; i < trn.length; ++i) {
      int[] trni=trn[i];
      int i0=trni[0], i1=trni[1], i2=trni[2];
      segment seg=checktriangle(z[i0],z[i1],z[i2],f[i0]-C,f[i1]-C,f[i2]-C);
      if(seg.active)
        pointscnt.push(seg.reversed ? new pair[] {seg.b,seg.a} : 
                       new pair[] {seg.a,seg.b});
    }
  }

  collect(points,c);

  return connect(points,c,join);
}

// Extend palette by the colors below and above at each end.
pen[] extend(pen[] palette, pen below, pen above) {
  pen[] p=copy(palette);
  p.insert(0,below);
  p.push(above);
  return p;
}

// Compute the interior palette for a sequence of cyclic contours
// corresponding to palette.
pen[][] interior(picture pic=currentpicture, guide[][] g, pen[] palette)
{
  if(palette.length != g.length+1)
    abort("Palette array must have length one more than guide array");
  pen[][] fillpalette=new pen[g.length][];
  for(int i=0; i < g.length; ++i) {
    guide[] gi=g[i];
    guide[] gp;
    if(i+1 < g.length) gp=g[i+1];
    guide[] gm;
    if(i > 0) gm=g[i-1];

    pen[] fillpalettei=new pen[gi.length];
    for(int j=0; j < gi.length; ++j) {
      path P=gi[j];
      if(cyclic(P)) {
        int index=i+1;
        bool nextinside;
        for(int k=0; k < gp.length; ++k) {
          path next=gp[k];
          if(cyclic(next)) {
            if(inside(P,point(next,0)))
              nextinside=true;
            else if(inside(next,point(P,0)))
              index=i;
          }
        }
        if(!nextinside) {
          // Check to see if previous contour is inside
          for(int k=0; k < gm.length; ++k) {
            path prev=gm[k];
            if(cyclic(prev)) {
              if(inside(P,point(prev,0)))
                index=i;
            }
          }
        } 
        fillpalettei[j]=palette[index];
      }
      fillpalette[i]=fillpalettei;
    }
  }
  return fillpalette;
}

// Fill the interior of cyclic contours with palette
void fill(picture pic=currentpicture, guide[][] g, pen[][] palette)
{
  for(int i=0; i < g.length; ++i) {
    guide[] gi=g[i];
    guide[] gp;
    if(i+1 < g.length) gp=g[i+1];
    guide[] gm;
    if(i > 0) gm=g[i-1];

    for(int j=0; j < gi.length; ++j) {
      path P=gi[j];
      path[] S=P;
      if(cyclic(P)) {
        for(int k=0; k < gp.length; ++k) {
          path next=gp[k];
          if(cyclic(next) && inside(P,point(next,0)))
            S=S^^next;
        }
        for(int k=0; k < gm.length; ++k) {
          path next=gm[k];
          if(cyclic(next) && inside(P,point(next,0)))
            S=S^^next;
        }
        fill(pic,S,palette[i][j]+evenodd);
      }
    }
  }
}
