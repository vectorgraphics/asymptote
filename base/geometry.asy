import math;

static public real perpfactor=arrowfactor;

// Draw a perpendicular symbol at z going from w to I*w.
void perpendicular(picture pic=currentpicture, pair z, pair w,
		   real size=0, pen p=currentpen) 
{
  if(size == 0) size=perpfactor*linewidth(p);
  picture apic;
  pair d1=size*w;
  pair d2=I*d1;
  _draw(apic,d1--d1+d2--d2,p);
  add(z,pic,apic);
}
  
// Draw a perpendicular symbol at z going from dir(g,0) to dir(g,0)+90
void perpendicular(picture pic=currentpicture, pair z, path g,
		   real size=0, pen p=currentpen) 
{
  if(size == 0) size=perpfactor*linewidth(p);
  perpendicular(pic,z,dir(g,0),size,p);
}

struct triangle {
  public pair A,B,C;

  void SAS(real b, real alpha, real c, real angle=0, pair A=(0,0))   {
    this.A=A;
    B=A+c*dir(angle);
    C=A+b*dir(angle+alpha);
  }

  real a() {return length(C-B);}
  real b() {return length(A-C);}
  real c() {return length(B-A);}
  
  private real det(pair a, pair b) {return a.x*b.y-a.y*b.x;}
  real area() {return 0.5*abs(det(A,B)+det(B,C)+det(C,A));}
  
  real alpha() {return degrees(acos((b()^2+c()^2-a()^2)/(2b()*c())));}
  real beta()  {return degrees(acos((c()^2+a()^2-b()^2)/(2c()*a())));}
  real gamma() {return degrees(acos((a()^2+b()^2-c()^2)/(2a()*b())));}
  
  void vertices(pair A, pair B, pair C) {
    this.A=A;
    this.B=B;
    this.C=C;
  }

  path Path() {return A--C--B--cycle;}
}

triangle operator init() {return new triangle;}
  
void draw(picture pic=currentpicture, triangle t, pen p=currentpen) 
{
  draw(pic,t.Path(),p);
}

triangle operator * (transform T, triangle t)
{
  triangle s;
  s.vertices(T*t.A,T*t.B,T*t.C);
  return s;
}

// Return an interior arc BAC of triangle ABC, given a radius r > 0.
// If r < 0, return the corresponding exterior arc of radius |r|.
guide arc(explicit pair B, explicit pair A, explicit pair C,
	  real r=arrowfactor)
{
  return arc(A,r,Angle(B-A),Angle(C-A));
}

