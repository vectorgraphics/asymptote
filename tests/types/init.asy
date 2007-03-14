import TestLib;
StartTest("init");
int operator init() { return 7; }
int x;
assert(x==7);

struct A {
  int x=3;
}
A a;
assert(a!=null);
assert(a.x==3);

A operator init() { return null; }
A aa;
assert(aa==null);
EndTest();
