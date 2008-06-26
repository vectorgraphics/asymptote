// Roots.

import TestLib;
real x;
real[] r;

StartTest("quadratic roots");

r=quadraticroots(1,0,-8);
assert(r.length == 2);
r=sort(r);
x=2sqrt(2);
assert(close(r[0],-x));
assert(close(r[1],x));

r=quadraticroots(1,2,1);
assert(r.length == 2);
assert(close(r[0],-1));
assert(close(r[1],-1));

r=quadraticroots(1,0,8);
assert(r.length == 0);

r=quadraticroots(0,2,3);
assert(r.length == 1);
assert(close(r[0],-3/2));

EndTest();

StartTest("cubic roots");

r=cubicroots(1,0,0,-8);
assert(r.length == 1);
assert(close(r[0],2));

real[] r=cubicroots(1,3,3,1);
assert(r.length == 3);
assert(close(r[0],-1));
assert(close(r[1],-1));
assert(close(r[2],-1));

real[] r=cubicroots(1,-3,3,-1);
assert(r.length == 3);
assert(close(r[0],1));
assert(close(r[1],1));
assert(close(r[2],1));

r=cubicroots(1,0,0,0);
assert(r.length == 3);
assert(r[0] == 0);
assert(r[1] == 0);
assert(r[2] == 0);

r=cubicroots(1,0,-15,-4);
assert(r.length == 3);
r=sort(r);
assert(close(r[0],-2-sqrt(3)));
assert(close(r[1],-2+sqrt(3)));
assert(close(r[2],4));

r=cubicroots(1,0,-15,4);
assert(r.length == 3);
r=sort(r);
assert(close(r[0],-4));
assert(close(r[1],2-sqrt(3)));
assert(close(r[2],2+sqrt(3)));

r=cubicroots(1,0,-15,0);
assert(r.length == 3);
r=sort(r);
x=sqrt(15);
assert(close(r[0],-x));
assert(r[1] == 0);
assert(close(r[2],x));

r=cubicroots(1,1,1,0);
assert(r.length == 1);
assert(r[0] == 0);

r=cubicroots(1,0,20,-4);
assert(r.length == 1);
x=cbrt(54+6sqrt(6081));
assert(close(r[0],x/3-20/x));

EndTest();

StartTest("newton");

real f(real x) {return cos(x);}
real dfdx(real x) {return -sin(x);}

assert(close(newton(f,dfdx,1),pi/2));
assert(newton(f,dfdx,0) == realMax);
assert(newton(f,dfdx,0,2) == pi/2);

EndTest();
