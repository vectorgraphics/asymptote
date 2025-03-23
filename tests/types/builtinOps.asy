import TestLib;

{
  StartTest('builtin record ops: external');
  struct A {}
  A a = new A;
  assert(a == a);
  assert(alias(a, a));
  assert(a != null);
  EndTest();
}

{
  StartTest('builtin record ops: internal');
  struct A {
    void runTest() {
      assert(this == this);
      assert(alias(this, this));
      assert(this != null);
    }
  }
  (new A).runTest();
  EndTest();
}

{
  StartTest('builtin record ops: nested');
  struct A {
    struct B {}
  }
  A a;
  a.B b;
  from a unravel operator==;
  assert(b == b);
}