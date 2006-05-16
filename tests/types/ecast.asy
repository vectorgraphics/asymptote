import TestLib;
StartTest("ecast");
struct A {
  public int x;
}
int operator ecast(A a) {
  return a.x;
}
A a=new A; a.x=5;
assert((int)a==5);
EndTest();
