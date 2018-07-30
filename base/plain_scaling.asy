real expansionfactor=sqrt(2);

// A coordinate in "flex space." A linear combination of user and true-size
// coordinates.
struct coord {
  real user,truesize;

  // Build a coord.
  static coord build(real user, real truesize) {
    coord c=new coord;
    c.user=user;
    c.truesize=truesize;
    return c;
  }

  // Deep copy of coordinate.  Users may add coords to the picture, but then
  // modify the struct. To prevent this from yielding unexpected results, deep
  // copying is used.
  coord copy() {
    return build(user, truesize);
  }
  
  void clip(real min, real max) {
    user=min(max(user,min),max);
    truesize=0;
  }
}

bool operator <= (coord a, coord b)
{
  return a.user <= b.user && a.truesize <= b.truesize;
}

bool operator >= (coord a, coord b)
{
  return a.user >= b.user && a.truesize >= b.truesize;
}

// Find the maximal elements of the input array, using the partial ordering
// given.
coord[] maxcoords(coord[] in, bool operator <= (coord,coord))
{
  // As operator <= is defined in the parameter list, it has a special
  // meaning in the body of the function.

  coord best;
  coord[] c;

  int n=in.length;
  
  if(n == 0)
    return c;

  int first=0;
  // Add the first coord without checking restrictions (as there are none).
  best=in[first];
  c.push(best);

  static int NONE=-1;

  int dominator(coord x)
  {
    // This assumes it has already been checked against the best.
    for(int i=1; i < c.length; ++i)
      if(x <= c[i])
        return i;
    return NONE;
  }

  void promote(int i)
  {
    // Swap with the top
    coord x=c[i];
    c[i]=best;
    best=c[0]=x;
  }

  void addmaximal(coord x)
  {
    coord[] newc;

    // Check if it beats any others.
    for(int i=0; i < c.length; ++i) {
      coord y=c[i];
      if(!(y <= x))
        newc.push(y);
    }
    newc.push(x);
    c=newc;
    best=c[0];
  }

  void add(coord x)
  {
    if(x <= best)
      return;
    else {
      int i=dominator(x);
      if(i == NONE)
        addmaximal(x);
      else
        promote(i);
    }
  }

  for(int i=1; i < n; ++i)
    add(in[i]);

  return c;
}

struct coords2 {
  coord[] x,y;
  void erase() {
    x.delete();
    y.delete();
  }
  // Only a shallow copy of the individual elements of x and y
  // is needed since, once entered, they are never modified.
  coords2 copy() {
    coords2 c=new coords2;
    c.x=copy(x);
    c.y=copy(y);
    return c;
  }
  void append(coords2 c) {
    x.append(c.x);
    y.append(c.y);
  }
  void push(pair user, pair truesize) {
    x.push(coord.build(user.x,truesize.x));
    y.push(coord.build(user.y,truesize.y));
  }
  void push(coord cx, coord cy) {
    x.push(cx);
    y.push(cy);
  }
  void push(transform t, coords2 c1, coords2 c2) {
    for(int i=0; i < c1.x.length; ++i) {
      coord cx=c1.x[i], cy=c2.y[i];
      pair tinf=shiftless(t)*(0,0);
      pair z=t*(cx.user,cy.user);
      pair w=(cx.truesize,cy.truesize);
      w=length(w)*unit(shiftless(t)*w);
      coord Cx,Cy;
      Cx.user=z.x;
      Cy.user=z.y;
      Cx.truesize=w.x;
      Cy.truesize=w.y;
      push(Cx,Cy);
    }
  }
  void xclip(real min, real max) {
    for(int i=0; i < x.length; ++i) 
      x[i].clip(min,max);
  }
  void yclip(real min, real max) {
    for(int i=0; i < y.length; ++i) 
      y[i].clip(min,max);
  }
}
  
// The scaling in one dimension:  x --> a*x + b
struct scaling {
  real a,b;
  static scaling build(real a, real b) {
    scaling s=new scaling;
    s.a=a; s.b=b;
    return s;
  }
  real scale(real x) {
    return a*x+b;
  }
  real scale(coord c) {
    return scale(c.user) + c.truesize;
  }
}

// Calculate the minimum point in scaling the coords.
real min(real m, scaling s, coord[] c) {
  for(int i=0; i < c.length; ++i)
    if(s.scale(c[i]) < m)
      m=s.scale(c[i]);
  return m;
}

// Calculate the maximum point in scaling the coords.
real max(real M, scaling s, coord[] c) {
  for(int i=0; i < c.length; ++i)
    if(s.scale(c[i]) > M)
      M=s.scale(c[i]);
  return M;
}

import simplex;

/*
 Calculate the sizing constants for the given array and maximum size.
 Solve the two-variable linear programming problem using the simplex method.
 This problem is specialized in that the second variable, "b", does not have
 a non-negativity condition, and the first variable, "a", is the quantity
 being maximized.
*/
real calculateScaling(string dir, coord[] m, coord[] M, real size,
                      bool warn=true) {
  real[][] A;
  real[] b;
  real[] c=new real[] {-1,0,0};

  void addMinCoord(coord c) {
    // (a*user + b) + truesize >= 0:
    A.push(new real[] {c.user,1,-1});
    b.push(-c.truesize);
  }     
  void addMaxCoord(coord c) {
    // (a*user + b) + truesize <= size:
    A.push(new real[] {-c.user,-1,1});
    b.push(c.truesize-size);
  }

  for (int i=0; i < m.length; ++i)
    addMinCoord(m[i]);
  for (int i=0; i < M.length; ++i)
    addMaxCoord(M[i]);

  int[] s=array(A.length,1);
  simplex S=simplex(c,A,s,b);

  if(S.case == S.OPTIMAL) {
    return S.x[0];
  } else if(S.case == S.UNBOUNDED) {
    if(warn) warning("unbounded",dir+" scaling in picture unbounded");
    return 0;
  } else {
    if(!warn) return 1;

    bool userzero(coord[] coords) {
      for(var coord : coords)
        if(coord.user != 0) return false;
      return true;
    }
    
    if((userzero(m) && userzero(M)) || size >= infinity) return 1;
    
    warning("cannotfit","cannot fit picture to "+dir+"size "+(string) size
            +"...enlarging...");
    
    return calculateScaling(dir,m,M,expansionfactor*size,warn);
  }
}

real calculateScaling(string dir, coord[] coords, real size, bool warn=true)
{
  coord[] m=maxcoords(coords,operator >=);
  coord[] M=maxcoords(coords,operator <=);

  return calculateScaling(dir, m, M, size, warn);
}
