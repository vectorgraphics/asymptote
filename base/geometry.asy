public real perpfactor=arrowfactor;

guide square(pair z1, pair z2)
{
  pair v=z2-z1;
  pair z3=z2+I*v;
  pair z4=z3-v;
  return z1--z2--z3--z4--cycle;
}

// Draw a perpendicular symbol at z aligned in the direction align
// relative to the path z--z+dir.
void perpendicular(picture pic=currentpicture, pair z, pair align,
		   pair dir=E, real size=0, pen p=currentpen) 
{
  if(size == 0) size=perpfactor*linewidth(p);
  picture apic;
  pair d1=size*align*unit(dir)*dir(-45);
  pair d2=I*d1;
  _draw(apic,d1--d1+d2--d2,p);
  add(pic,apic,z);
}
  
// Draw a perpendicular symbol at z aligned in the direction align
// relative to the path z--z+dir(g,0)
void perpendicular(picture pic=currentpicture, pair z, pair align, path g,
		   real size=0, pen p=currentpen) 
{
  perpendicular(pic,z,align,dir(g,0),size,p);
}

struct triangle {
  public pair A,B,C;

  static triangle SAS(real b, real alpha, real c, real angle=0, pair A=(0,0)) {
    triangle T=new triangle;
    T.A=A;
    T.B=A+c*dir(angle);
    T.C=A+b*dir(angle+alpha);
    return T;
  }

  static triangle vertices(pair A, pair B, pair C) {
    triangle T=new triangle;
    T.A=A;
    T.B=B;
    T.C=C;
    return T;
  }

  real a() {return length(C-B);}
  real b() {return length(A-C);}
  real c() {return length(B-A);}
  
  private real det(pair a, pair b) {return a.x*b.y-a.y*b.x;}
  real area() {return 0.5*abs(det(A,B)+det(B,C)+det(C,A));}
  
  real alpha() {return degrees(acos((b()^2+c()^2-a()^2)/(2b()*c())));}
  real beta()  {return degrees(acos((c()^2+a()^2-b()^2)/(2c()*a())));}
  real gamma() {return degrees(acos((a()^2+b()^2-c()^2)/(2a()*b())));}
  
  path Path() {return A--C--B--cycle;}
}

triangle operator init() {return new triangle;}
  
void draw(picture pic=currentpicture, triangle t, pen p=currentpen) 
{
  draw(pic,t.Path(),p);
}

triangle operator * (transform T, triangle t)
{
  return triangle.vertices(T*t.A,T*t.B,T*t.C);
}

// Return an interior arc BAC of triangle ABC, given a radius r > 0.
// If r < 0, return the corresponding exterior arc of radius |r|.
guide arc(explicit pair B, explicit pair A, explicit pair C,
	  real r=arrowfactor)
{
  return arc(A,r,degrees(B-A),degrees(C-A));
}

