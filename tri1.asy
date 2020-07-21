import three;
import math;

currentlight=nolight;

bool sameside(pair a, pair b, int s0, pair A, pair B, pair C)
{
  if(sgn(orient(a,b,A)) == s0) return true;
  if(sgn(orient(a,b,B)) == s0) return true;
  if(sgn(orient(a,b,C)) == s0) return true;
  return false;
}

// returns true iff 2D triangles abc and ABC intersect
bool intersect(pair a, pair b, pair c, pair A, pair B, pair C)
{
  int s0=sgn(orient(a,b,c)); // Optimize away
  int S0=sgn(orient(A,B,C)); // Optimize away
  return
    sameside(a,b,s0,A,B,C) &&
    sameside(b,c,s0,A,B,C) &&
    sameside(c,a,s0,A,B,C) &&
    sameside(A,B,S0,a,b,c) &&
    sameside(B,C,S0,a,b,c) &&
    sameside(C,A,S0,a,b,c);
}

triple[] vertex;

// Check if line p0--q0 intersects uniquely with line P0--Q0.
// If it does, push the intersection point onto the vertex array.
bool Intersect(triple p0, triple q0, triple P0, triple Q0,
               projection C=currentprojection)
{
  pair p=project(p0,C);
  pair q=project(q0,C);
  pair P=project(P0,C);
  pair Q=project(Q0,C);

  real a=q.x-p.x;
  real b=P.x-Q.x;
  real c=q.y-p.y;
  real d=P.y-Q.y;
  real e=P.x-p.x;
  real f=P.y-p.y;
  real det=a*d-b*c;
  if(det == 0) return false;
  real detinv=1/det;
  real t=(d*e-b*f)*detinv;
  real T=(a*f-e*c)*detinv;
  if(t < 0 || t > 1 || T < 0 || T > 1) return false;
  vertex.push(interp(p0,q0,t));
  return true;
}

// Find all projected intersections of a--b with triangle ABC.
int intersect(triple a, triple b, triple A, triple B, triple C,
              projection P=currentprojection)
{
  int count=0;
  int sum=0;
  if(Intersect(a,b,A,B,P)) {
    ++count;
    sum += 1;
    if(vertex.length == 3) return sum;
  }
  if(Intersect(a,b,B,C,P)) {
    ++count;
    sum += 2;
    if(count == 2 || vertex.length == 3) return sum;
  }
  if(Intersect(a,b,C,A,P))
    sum += 4;
  return sum;
}


real third=1.0/3.0;

bool sameside(triple A, triple B, triple C,
              projection P=currentprojection)
{
  dot(vertex,green);
  dot(third*sum(vertex),black);
  return sgn(orient(A,B,C,third*sum(vertex))) == sgn(orient(A,B,C,P.camera));
}

bool sameside(triple v, triple A, triple B, triple C,
              projection P=currentprojection)
{
  vertex.push(v);
  return sameside(A,B,C,P);
}

bool inside(pair a, pair b, pair c, pair z) {
  pair A=a-c;
  pair B=b-c;
  real[][] M={{A.x,B.x},{A.y,B.y}};
  real[] t=inverse(M)*new real[] {z.x-c.x,z.y-c.y};
  return t[0] > 0 && t[1] > 0 && t[0]+t[1] < 1;
}

// Return true if triangle abc can be rendered in front of triangle ABC,
// using projection P.
bool front(triple a, triple b, triple c, triple A, triple B, triple C,
           projection P=currentprojection) {
  int sum;
  vertex.delete();
// Find vertices of a triangle common to the projections of triangle abc
// and ABC.

  sum=intersect(a,b,A,B,C,P);
  write(vertex.length);
  if(vertex.length == 3) return sameside(A,B,C,P);

  sum += 8*intersect(b,c,A,B,C,P);
  write(vertex.length);
  if(vertex.length == 3) return sameside(A,B,C,P);

  sum += 64*intersect(c,a,A,B,C,P);
  write(vertex.length);
  if(vertex.length == 3) return sameside(A,B,C,P);

  if(vertex.length == 2) {
    path t=project(a,P)--project(b,P)--project(c,P)--cycle;
    path T=project(A,P)--project(B,P)--project(C,P)--cycle;

    write("sum=",sum);
    if(sum == 1*3 || sum == 8*3 || sum == 64*3)
      return !sameside(inside(t,project(B,P)) ? B : A,a,b,c,P);
    if(sum == 1*5 || sum == 8*5 || sum == 64*5)
      return !sameside(inside(t,project(A,P)) ? A : B,a,b,c,P);
    if(sum == 1*6 || sum == 8*6 || sum == 64*6)
      return !sameside(inside(t,project(C,P)) ? C : A,a,b,c,P);

    if(sum == 1*1+8*1 || sum == 1*2+8*2 || sum == 1*4+8*4)
      return sameside(inside(T,project(b,P)) ? b : a,A,B,C,P);
    if(sum == 64*1+1*1 || sum == 64*2+1*2 || sum == 64*4+1*4)
      return sameside(inside(T,project(a,P)) ? a : b,A,B,C,P);
    if(sum == 8*1+64*1 || sum == 8*2+64*2 || sum == 8*4+64*4)
      return sameside(inside(T,project(c,P)) ? c : a,A,B,C,P);
    
    if(sum == 64*4+1*2 || sum == 64*1+1*4 || sum == 64*2+1*1)
      return sameside(a,A,B,C,P);
    if(sum == 1*4+8*2 || sum == 1*1+8*4 || sum == 1*2+8*1)
      return sameside(b,A,B,C,P);
    if(sum == 8*4+64*2 || sum == 8*1+64*4 || sum == 8*2+64*1)
      return sameside(c,A,B,C,P);
  }
  return true; // Triangles do not intersect;
}

// returns true iff projection P of triangles abc and ABC intersect
bool intersect(triple a, triple b, triple c, triple A, triple B, triple C,
               projection P)
{
  return intersect(project(a,P),project(b,P),project(c,P),
                   project(A,P),project(B,P),project(C,P));
}

// returns true iff triangle abc is pierced by line segment AB
bool pierce(triple a, triple b, triple c, triple A, triple B)
{
  int sa=sgn(orient(A,b,c,B));
  int sb=sgn(orient(A,c,a,B));
  int sc=sgn(orient(A,a,b,B));
  return sa == sb && sb == sc; 
}

// returns true iff triangle abc is pierced by an edge of triangle ABC
bool intersect0(triple a, triple b, triple c, triple A, triple B, triple C,
                int sA, int sB, int sC)
{
  if(sA != sB) {
    if(pierce(a,b,c,A,B)) return true;
    if(sC != sA) {
      if(pierce(a,b,c,C,A)) return true;
    } else {
      if(pierce(a,b,c,B,C)) return true;
    }
  } else {
    if(pierce(a,b,c,B,C)) return true;
    if(pierce(a,b,c,C,A)) return true;
  }
  return false;
}  

// returns true iff triangle abc intersects triangle ABC
bool intersect(triple a, triple b, triple c, triple A, triple B, triple C)
{
  int sA=sgn(orient(a,b,c,A));
  int sB=sgn(orient(a,b,c,B));
  int sC=sgn(orient(a,b,c,C));
  if(sA == sB && sB == sC) return false;

  int sa=sgn(orient(A,B,C,a));
  int sb=sgn(orient(A,B,C,b));
  int sc=sgn(orient(A,B,C,c));
  if(sa == sb && sb == sc) return false;

  return intersect0(a,b,c,A,B,C,sA,sB,sC) || intersect0(A,B,C,a,b,c,sa,sb,sc);
}


size(10cm);

triple t0=-Z-0.5X+Y;
triple t0=-Z+X+2*Y;
triple t1=Y+2*Z+2Y;
triple t2=X+Z+2Y;

triple T0=O;
triple T1=4X;
triple T2=4Y;

triple T1=2X;
triple T2=2Y;


triple N=cross(T1-T0,T2-T0);
triple n=cross(t1-t0,t2-t0);

write(n,N);

triple a=point(t0--t1,intersect(t0,t1,N,T0));
triple b=point(t2--t0,intersect(t2,t0,N,T0));

dot("$a$",a);
dot("$b$",b);

triple e=currentprojection.camera;
write(e);

triple c0=(t0+a+b)/3;
triple c1=(t1+a+b)/3;
triple c2=(t2+t1+b)/3;

triple C=(T0+T1+T2)/3;

real s=0.1;

draw(T0--T0+s*N,Arrow3);

srand(seconds());

while(true) {
  currentprojection=orthographic(dir(180*unitrand(),360*unitrand()));     

  write(currentprojection.camera);
  erase();
  draw(surface(t0--a--b--cycle),red+opacity(0.5));
  draw(surface(T0--T1--T2--cycle),blue+opacity(0.5));
  if(intersect(a,b,t0,T0,T1,T2,currentprojection)) {
    write(front(a,b,t0,T0,T1,T2));
    shipout();
    exit();
  }
}
