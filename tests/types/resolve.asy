import TestLib;
StartTest("resolve");
struct A {} struct B {} struct C {}

int f(B, real) { return 1; }
int f(C, int) { return 2; }
B operator cast(A) { return new B; }

assert(f(new A, 3) == 1);
C operator cast(A) { return new C; }
assert(f(new A, 3) == 2);

int givex(int x, int y) { return x; }
assert(givex(2002,3) == 2002);
assert(givex(2002,2002) == 2002);
assert(givex(-2005,2005) == -2005);
assert(givex(x=-77,205) == -77);
assert(givex(-77,y=205) == -77);
assert(givex(-77,x=205) == 205);
assert(givex(x=-77,y=205) == -77);
assert(givex(y=-77,x=205) == 205);

int g(real x, real y) { return 7; }
int g(int x, real y) { return 8; }

assert(g(4, 4) == 8);
assert(g(4, 4.4) == 8);
assert(g(4.4, 4) == 7);
assert(g(4.4, 4.4) == 7);

assert(g(x=4, y=4) == 8);
assert(g(x=4, y=4.4) == 8);
assert(g(x=4.4, y=4) == 7);
assert(g(x=4.4, y=4.4) == 7);

assert(g(x=4, 4) == 8);
assert(g(x=4, 4.4) == 8);
assert(g(x=4.4, 4) == 7);
assert(g(x=4.4, 4.4) == 7);

assert(g(4, y=4) == 8);
assert(g(4, y=4.4) == 8);
assert(g(4.4, y=4) == 7);
assert(g(4.4, y=4.4) == 7);

assert(g(y=4, x=4) == 8);
assert(g(y=4, x=4.4) == 7);
assert(g(y=4.4, x=4) == 8);
assert(g(y=4.4, x=4.4) == 7);

assert(g(4, x=4) == 8);
assert(g(4, x=4.4) == 7);
assert(g(4.4, x=4) == 8);
assert(g(4.4, x=4.4) == 7);

assert(g(y=4, 4) == 8);
assert(g(y=4, 4.4) == 7);
assert(g(y=4.4, 4) == 8);
assert(g(y=4.4, 4.4) == 7);

// Test exact matching over casting.
{
  void f(int x, real y=0.0, int z=0) {
    assert(x==1);
    assert(y==2.0);
    assert(z==0);
  }
  f(1,2);
}
{
  void f() {
    assert(false);
  }
  void f(int x, real y=0.0, int z=0) {
    assert(x==1);
    assert(y==2.0);
    assert(z==0);
  }
  f(1,2);
}
{
  void f() {
    assert(false);
  }
  void f(int x, int y) {
    assert(x==1);
    assert(y==2);
  }
  void f(int x, real y=0.0, int z=0) {
    assert(false);
  }
  f(1,2);
}
{
  struct A {}
  struct B {}
  struct C {}

  void f(B);
  void g(B);

  // Should resolve to void (B).
  assert(f == g);
  assert(g == f);
  assert(!(f != g));
  assert(!(g != f));
}
{
  struct A {}
  struct B {}
  struct C {}

  void f(A), f(B);
  void g(B), g(C);

  // Should resolve to void (B).
  assert(f == g);
  assert(g == f);
  assert(!(f != g));
  assert(!(g != f));
}
{
  struct A {}
  struct B {}
  struct C {}

  void f(B);
  void g(B), g(C);

  // Should resolve to void (B).
  assert(f == g);
  assert(g == f);
  assert(!(f != g));
  assert(!(g != f));
}
{
  struct A {}
  struct B {}
  struct C {}

  void f(B);
  void g(B), g(C);

  // Should resolve to void (B).
  assert(f == g);
  assert(g == f);
  assert(!(f != g));
  assert(!(g != f));
}
{
void foo() {}
assert((foo == null ? 5 : 8) == 8);
}

// TODO: Add packing vs. casting tests.

EndTest();
