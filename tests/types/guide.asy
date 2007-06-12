import TestLib;
StartTest("guide");

guide g;
assert(size(g) == 0);
assert(length(g) == -1);

guide g=(0,0){curl 2}..(1,1){W}..tension 5 and 6 ..{N}(4,2)..{curl 3}(0,1);

assert(size(g) == 4);
assert(length(g) == 3);

assert(close(point(g,-1),(0,0)));
assert(close(point(g,0),(0,0)));
assert(close(point(g,1),(1,1)));
assert(close(point(g,3),(0,1)));
assert(close(point(g,4),(0,1)));

tensionSpecifier t=tensionSpecifier(g,1);
assert(close(t.out,5));
assert(close(t.in,6));
assert(t.atLeast == false);

pair[] z=dirSpecifier(g,0);
assert(close(z[0],0));
assert(close(z[1],0));

pair[] z=dirSpecifier(g,1);
assert(close(z[0],W));
assert(close(z[1],N));

real[] x=curlSpecifier(g);
assert(close(x[0],2));
assert(close(x[1],3));

assert(!cyclic(g));

guide g=(0,0)..controls (3,5)..(1,1){W}..tension atleast 5 and 6 ..{N}(4,2)..
controls(1,2) and (2,4).. (0,1)..cycle;

assert(size(g) == 4);
assert(length(g) == 4);

tensionSpecifier t=tensionSpecifier(g,1);
assert(close(t.out,5));
assert(close(t.in,6));
assert(t.atLeast == true);

pair[] z=controlSpecifier(g,0);
assert(close(z[0],(3,5)));
assert(close(z[1],(3,5)));

pair[] z=controlSpecifier(g,2);
assert(close(z[0],(1,2)));
assert(close(z[1],(2,4)));

real[] x=curlSpecifier(g);
assert(close(x[0],1));
assert(close(x[1],1));

assert(cyclic(g));

guide g=(0,0)..controls (3,5)..(1,1){W}..tension atleast 5 and 6 ..{N}(4,2)..
cycle..controls(1,2) and (2,4)..(0,1)..cycle;

assert(size(g) == 5);
assert(length(g) == 5);

pair[] z=dirSpecifier(g,0);
assert(close(z[0],0));
assert(close(z[1],0));

pair[] z=dirSpecifier(g,1);
assert(close(z[0],0));
assert(close(z[1],0));

tensionSpecifier t=tensionSpecifier(g,1);
assert(close(t.out,1));
assert(close(t.in,1));
assert(t.atLeast == false);

pair[] z=controlSpecifier(g,0);
assert(close(z[0],(3,5)));
assert(close(z[1],(3,5)));

real[] x=curlSpecifier(g);
assert(close(x[0],1));
assert(close(x[1],1));

assert(cyclic(g));

EndTest();

