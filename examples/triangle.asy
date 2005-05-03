// Example of a transformable triangle structure

size(0,100);

struct triangle {
  public real a,c;   // side lengths
  public real beta;  // interior angle (degrees) at vertex B, opposite side b
  public real angle; // angle of B--A
  public pair B;     // vertex B

  void SAS(real a, real beta, real c, real angle=0, pair B=(0,0))   {
    this.a=a; this.beta=beta; this.c=c; this.angle=angle; this.B=B;

  }

  void vertices(pair A, pair B, pair C) {
    pair AB=A-B;
    c=length(AB);
    angle=Angle(AB);
    pair CB=C-B;
    a=length(CB);
    beta=Angle(CB)-angle;
    this.B=B;
  }

  // Vertices:
  pair A() {return B+c*dir(angle);}
  pair B() {return B;}
  pair C() {return B+a*dir(angle+beta);}

  path Path() {return A()--C()--B()--cycle;}
}

void draw(picture pic=currentpicture, triangle t, pen p=currentpen) 
{
  draw(pic,t.Path(),p);
}

triangle operator * (transform T, triangle t)
{
  triangle s=new triangle;
  s.vertices(T*t.A(),T*t.B(),T*t.C());
  return s;
}

triangle t=new triangle;
t.SAS(3,90,4);
  
dot((0,0));

draw(t);
draw(rotate(90)*t,red);
draw(shift((-4,0))*t,blue);
draw(reflect((0,0),(1,0))*t,green);
draw(slant(2)*t,magenta);
