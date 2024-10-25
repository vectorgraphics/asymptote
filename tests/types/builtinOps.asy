import TestLib;

StartTest('builtin record ops: external');
struct A {}
A a = new A;
assert(a == a);
assert(alias(a, a));
assert(a != null);
EndTest();

StartTest('builtin record ops: internal');
struct B {
  void runTest() {
    assert(this == this);
    assert(alias(this, this));
    assert(this != null);
  }
}
(new B).runTest();
EndTest();