import three;
import math;
import fontsize;

//size(80cm);
//settings.fitscreen=false;
defaultpen(fontsize(100pt)+linewidth(3));

currentlight=nolight;

// returns true if one of the points A, B, C have orientation s0 relative
// to a--b.
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

// Check if projections of the lines p0--q0 and P0--Q0 intersect uniquely.
// If they do, push the intersection point onto the vertex array.
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

// returns true iff v is on the same side of triangle ABC as P.camera
bool sameside(triple v, triple A, triple B, triple C,
              projection P=currentprojection)
{
  return sgn(orient(A,B,C,v)) == sgn(orient(A,B,C,P.camera));
}

triple centroid;

// returns true iff vertex centroid is on the same side of triangle
// ABC as P.camera
bool sameside(triple A, triple B, triple C,
              projection P=currentprojection)
{
  dot(vertex,green);
  centroid=third*sum(vertex);
  dot(centroid,black);
  return sameside(centroid,A,B,C,P);
}

// returns true iff the common triangle formed by v and the 2 elements of
// vertex are on the same side of triangle ABC as P.camera
bool Sameside(triple v, triple A, triple B, triple C,
              projection P=currentprojection)
{
  vertex.push(v);
  return sameside(A,B,C,P);
}

// returns true iff z is in 2D triangle abc
bool inside(pair a, pair b, pair c, pair z) {
  pair A=a-c;
  pair B=b-c;
  real[][] M={{A.x,B.x},{A.y,B.y}};
  real[] t=inverse(M)*new real[] {z.x-c.x,z.y-c.y};
  return t[0] > 0 && t[1] > 0 && t[0]+t[1] < 1;
}

int sum;

// Return true if triangle abc can be rendered in front of triangle ABC,
// using projection P.
bool front(triple a, triple b, triple c, triple A, triple B, triple C,
           projection P=currentprojection) {
  vertex.delete();
// Find vertices of a triangle common to the projections of triangle abc
// and ABC.

  sum=intersect(a,b,A,B,C,P);
  //  write(vertex.length);
  if(vertex.length == 3)
    return sameside(A,B,C,P);

  sum += 8*intersect(b,c,A,B,C,P);
  //  write(vertex.length);
  if(vertex.length == 3)
    return sameside(A,B,C,P);

  sum += 64*intersect(c,a,A,B,C,P);
  //  write(vertex.length);
  if(vertex.length == 3)
    return sameside(A,B,C,P);

  path T=project(A,P)--project(B,P)--project(C,P)--cycle;
  path t=project(a,P)--project(b,P)--project(c,P)--cycle;

  if(vertex.length == 2) {
    int o2=sum#64;
    int sum2=sum-64*o2;
    int o1=sum2#8;
    int o0=sum2-8*o1;

    int t1=AND(sum,7);
    int t2=AND(sum,7*8);
    int t3=AND(sum,7*64);

    if(t1 != sum && t2 != sum && t3 != sum) {
      // each side of t has at most 1 intersection
      if(t2 == 0)
        return Sameside(inside(T,project(a,P)) ? a : b,A,B,C,P);

      if(t3 == 0)
        return Sameside(inside(T,project(b,P)) ? b : c,A,B,C,P);

      if(t1 == 0)
        return Sameside(inside(T,project(c,P)) ? c : a,A,B,C,P);
    } else {
      // one side of t has exactly 2 intersections
      if(AND(sum,3*73) == sum)
        return !Sameside(inside(t,project(B,P)) ? B : C,a,b,c,P);

      if(AND(sum,5*73) == sum)
        return !Sameside(inside(t,project(A,P)) ? A : B,a,b,c,P);

      if(AND(sum,6*73) == sum)
        return !Sameside(inside(t,project(C,P)) ? C : A,a,b,c,P);
    }

    dot(vertex,brown);
    abort("Missing case: "+string(sum));
  }

  if(vertex.length == 0) {
    centroid=third*(a+b+c);
    if(inside(T,project(centroid,P))) {
      dot(centroid,black);
      return sameside(centroid,A,B,C,P);
    }
    centroid=third*(A+B+C);
    if(inside(t,project(centroid,P))) {
      dot(centroid,black);
      return !sameside(centroid,a,b,c,P);
    }
  }

  return true; // Triangle projections do not intersect.
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

triple t0=-Z+X+2*Y;
triple t1=Y+2*Z+2Y;
triple t2=X+Z+2Y;

//srand(seconds());

while(true) {
  //  currentprojection=orthographic(dir(180*unitrand(),360*unitrand()));
  currentprojection=orthographic(dir(180*unitrand(),360*unitrand()));
  //   write("Camera=",currentprojection.camera);


  /*
currentprojection=orthographic(
camera=(-9.30375447679876,0.959249381767673,13.6715710699745),
up=(0.0016510798297577,-0.000170232061679372,0.00113553418828932),
target=(-1.77635683940025e-15,0,0),
zoom=1);
  */

  //  currentprojection=absorthographic((0.389372307337626,-0.893583674375808,-0.223377311219387));

  //  currentprojection=absorthographic((1,1,1));

triple A,B,C;
triple a,b,c;


  triple A=(4*unitrand(),unitrand(),2*unitrand());
  triple B=(unitrand(),6*unitrand(),unitrand());
  triple C=(unitrand(),unitrand(),2*unitrand());

  triple a=(2*unitrand(),unitrand(),unitrand());
  triple b=(unitrand(),4*unitrand(),unitrand());
  triple c=(unitrand(),unitrand(),6*unitrand());

  //  write(A,B,C);
  //  write(a,b,c);

real f=600;
  A *= f;
  B *= f;
  C *= f;

  a *= f;
  b *= f;
  c *= f;

  if(!intersect(a,b,c,A,B,C) && intersect(a,b,c,A,B,C,currentprojection)) {
    erase();
    real opacity=1;//0.7;
    draw(surface(c--a--b--cycle),red+opacity(opacity));
    draw(surface(A--B--C--cycle),blue+opacity(opacity));

    front(a,b,c,A,B,C);

    currentprojection.camera += centroid-currentprojection.target;
    currentprojection.target=centroid;
    triple v=currentprojection.camera-centroid;
    real d=max(abs(dot(a-centroid,v)),abs(dot(b-centroid,v)),abs(dot(c-centroid,v)),
               abs(dot(A-centroid,v)),abs(dot(B-centroid,v)),abs(dot(C-centroid,v)));
    currentprojection.camera=centroid-d*unit(v);

    write(front(a,b,c,A,B,C));

    label("a",a,dir(c--a,b--a));
    label("b",b,dir(a--b,c--b));
    label("c",c,dir(a--c,b--c));
    label("A",A,dir(C--A,B--A));
    label("B",B,dir(A--B,C--B));
    label("C",C,dir(A--C,B--C));

    label("1",b--a);
    label("8",c--b);
    label("64",a--c);

    label("1",A--B);
    label("2",B--C);
    label("4",C--A);

    dot(centroid,yellow+10mm+opacity(0.25));
    draw(centroid--currentprojection.camera,magenta);
    dot(currentprojection.camera,purple);

    //       if(sum == 0) {
          //      write("Current camera:",currentprojection.camera);
          //      write("Current target:",currentprojection.target);
      shipout();
      //        }
    //    exit();

  }
}
