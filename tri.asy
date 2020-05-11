import three;
import math;

currentlight=nolight;

/*

currentprojection=orthographic(5,4,2);
currentprojection=orthographic(
camera=(9.39715488951148,7.28298997213219,-2.02067919410512),
up=(0.00167506907637499,0.00324746801807946,0.0194945681945002),
target=(0,0,0),
zoom=1);
*/


/*

currentprojection=orthographic(
camera=(6.93501776129247,9.43358189281555,-2.8886262151612),
up=(-0.000459939291657656,0.00319874068634107,0.00934211250583798),
target=(0,-8.88178419700125e-16,4.44089209850063e-16),
zoom=0.281240734950249);
*/


/*
currentprojection=orthographic(
camera=(-2.97757090340182,-11.0292755120778,3.86267888456667),
up=(-0.00163790175825422,-0.00270239617686973,-0.008978856560151),
target=(1.77635683940025e-15,0,8.88178419700125e-16),
zoom=0.231377448655857);
*/
/*
currentprojection=orthographic(
camera=(-11.9216035406615,-0.836122958672795,1.61473898736815),
up=(0.000563524767487652,0.0060750948777718,0.0073062461122266),
target=(1.77635683940025e-15,0,8.88178419700125e-16),
zoom=0.281240734950249);
*/
/*
currentprojection=orthographic(
camera=(-6.74227270292659,-8.82739026798862,4.69570424607574),
up=(0.0032355715066907,0.00215111278379648,0.00868959679385214),
target=(1.77635683940025e-15,0,8.88178419700125e-16),
zoom=0.281240734950249);
*/

currentprojection=orthographic(
camera=(-4.83137991441094,-8.506567974811,7.05170902399595),
up=(0.0039231528089854,0.00435516705806684,0.00794158661129773),
target=(2.66453525910038e-15,0,1.77635683940025e-15),
zoom=0.281240734950249);

currentprojection=orthographic(
camera=(-6.46309526458259,-9.12311485642375,4.51967971880273),
up=(0.00166637768812669,0.00328562291079432,0.00901502752598474),
target=(2.66453525910038e-15,0,2.66453525910038e-15),
zoom=0.505067952995518);
currentprojection=orthographic(
camera=(2.19215878353362,11.2648443737293,-3.7051346285006),
up=(-0.0095945796309076,0.00191901763523648,0.000157736578451513),
target=(1.77635683940025e-15,-1.77635683940025e-15,3.5527136788005e-15),
zoom=0.505067952995518);

currentprojection=orthographic(
camera=(9.35940303412626,4.48540694192949,-6.14110483286399),
up=(-0.00434221026946177,0.00689608251685773,-0.00158095383324197),
target=(2.22044604925031e-15,-1.77635683940025e-15,3.5527136788005e-15),
zoom=0.505067952995518);
currentprojection=orthographic(
camera=(11.3666219085399,3.18348315924905,-2.46889258061008),
up=(-0.00319025797315641,0.00723622684769123,-0.0053570804036643),
target=(1.77635683940025e-15,-1.77635683940025e-15,2.66453525910038e-15),
zoom=0.31006791028265);

/*
currentprojection=orthographic(
camera=(-4.81737745883932,-9.65676927692384,5.38241280519738),
up=(0.0010904000271421,0.00429082937595698,0.0086742538682318),
target=(1.77635683940025e-15,-1.77635683940025e-15,2.66453525910038e-15),
zoom=0.281240734950249);
*/
currentprojection=orthographic(
camera=(-11.1182640461483,-2.53321347658859,-3.92374211314538),
up=(-0.00313444067013077,0.00623439120658791,0.0048567371957692),
target=(1.77635683940025e-15,-1.77635683940025e-15,2.22044604925031e-15),
zoom=0.376889482873);
currentprojection=orthographic(
camera=(6.70252764387443,9.70237391784346,-2.52365151633353),
up=(-0.00694343386826759,0.00510769930674108,0.00119600375942015),
target=(1.77635683940025e-15,-2.66453525910038e-15,1.33226762955019e-15),
zoom=0.376889482873);
currentprojection=orthographic(
camera=(-7.09665507367602,-8.48386743085318,4.80520816296918),
up=(0.00359401205223605,-0.00656427517959168,-0.00628164533163974),
target=(1.77635683940025e-15,-5.32907051820075e-15,8.88178419700125e-16),
zoom=0.4581115219914);

currentprojection=orthographic(
camera=(-6.03202134573233,-9.57786294484806,4.16022643162396),
up=(0.00332937619285971,-0.00426714398083986,-0.00499666200849673),
target=(2.66453525910038e-15,-7.105427357601e-15,8.88178419700125e-16),
zoom=0.4581115219914);
currentprojection=orthographic(
camera=(6.23683985129294,1.57132247104473,-0.640743426881589),
up=(0.000877759925445611,0.000221144670685958,0.00908622301694504),
target=(8.88178419700125e-16,-4.44089209850063e-16,2.22044604925031e-16),
zoom=1);

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

// Check if line p--q intersects with line P--Q.
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
  dot(third*sum(vertex),red);
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


/*
write(intersect(t0,t1,t2,T0,T1,T2,currentprojection));
write(intersect(t0,t1,t2,T0,T1,T2));
write(intersect(T0,T1,T2,t0,t1,t2));
*/

//dot("$t_0$",t0);
//dot("$t_1$",t1);
//dot("$t_2$",t2);

/*
*/

/*
transform3 t=shift(3Z);
draw(t*surface(T0--T1--T2--cycle),red+opacity(0.5));
draw(t*surface(t0--t1--t2--cycle),blue+opacity(0.5));
*/

triple N=cross(T1-T0,T2-T0);
triple n=cross(t1-t0,t2-t0);

write(n,N);

triple a=point(t0--t1,intersect(t0,t1,N,T0));
triple b=point(t2--t0,intersect(t2,t0,N,T0));

dot("$a$",a);
dot("$b$",b);

//draw(surface(t0--t1--t2--cycle),blue+opacity(0.5));
//draw(surface(a--b--t1--cycle),blue+opacity(0.5));
//draw(surface(t1--t2--b--cycle),blue+opacity(0.5));

//draw(a--b--t1);

triple e=currentprojection.camera;
write(e);

triple c0=(t0+a+b)/3;
triple c1=(t1+a+b)/3;
triple c2=(t2+t1+b)/3;

triple C=(T0+T1+T2)/3;

real s=0.1;

//write(abs(T0+s*N-e));
//write(abs(T0-s*N-e));
draw(T0--T0+s*N,Arrow3);
//draw(T0--c0,Arrow3);

/*
real D=dot(e-T0,N);
real d=dot(e-t1,n);

write("red: ",D);
write("blue:",d);
*/
/*
write((D*dot(t0-T0,N)));

write();
write((D*dot(t0-T0,N)));
write((D*dot(a-T0,N)));
write((D*dot(b-T0,N)));
write();
*/

/*
write(intersect(e,c0,N,T0) > 1);
write(intersect(e,c1,N,T0) > 1);
write(intersect(e,c2,N,T0) > 1);

write();
write(-orient(T0,T1,T2,c0));
write(-orient(T0,T1,T2,c1));
write(-orient(T0,T1,T2,c2));
*/


/*
draw(surface(t0--a--b--cycle),red);//),blue+opacity(0.5));
draw(surface(T0--T1--T2--cycle),blue);//),blue+opacity(0.5));
write(front(a,b,t0,T0,T1,T2));
*/

while(true) {
  currentprojection=orthographic(dir(180*unitrand(),360*unitrand()));     
  //  currentprojection=orthographic((0.96492136982056,0.243104307203911,-0.0991314575829508));

  write(currentprojection.camera);
  erase();
  draw(surface(t0--a--b--cycle),red+opacity(0.8));
  draw(surface(T0--T1--T2--cycle),blue+opacity(0.8));
  if(intersect(a,b,t0,T0,T1,T2,currentprojection)) {
    write(front(a,b,t0,T0,T1,T2));
    shipout();
  }
}
