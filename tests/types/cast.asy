import TestLib;
StartTest("cast");
struct A {
  public int x;
}
A operator cast(int x) {
  A a=new A;
  a.x=x;
  return a;
}
A a=7;
assert(a.x==7);
EndTest();
