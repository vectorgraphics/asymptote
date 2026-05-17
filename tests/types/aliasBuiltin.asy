import TestLib;

StartTest("alias builtin: records arrays functions");
{
  struct A {}

  A first = new A;
  A second = first;
  A third = new A;

  assert(alias(first, second));
  assert(!alias(first, third));
  assert(!alias(first, null));
  assert(!alias(null, first));

  int[] data = new int[] {1, 2, 3};
  int[] dataAlias = data;
  int[] copied = copy(data);

  assert(alias(data, dataAlias));
  assert(!alias(data, copied));

  int square(int x) { return x*x; }
  int cube(int x) { return x*x*x; }

  var squareAlias = square;
  var squareAgain = squareAlias;
  var cubeAlias = cube;

  assert(alias(squareAlias, squareAgain));
  assert(!alias(squareAlias, cubeAlias));
  assert(!alias(squareAlias, null));
}
EndTest();

StartTest("alias builtin: override");
{
  struct A {
    autounravel bool alias(A a, A b) {
      return true;
    }
  }
  A a = new A;
  A b = new A;
  assert(alias(a, b));
}
EndTest();

StartTest("alias builtin: scope");
{
  struct A {
    static struct B {
      static struct C {
        static struct D {}
        static D d1 = new D;
        static D d2 = new D;
      }
    }
  }
  assert(alias(A.B.C.d1, A.B.C.d1));
  assert(!alias(A.B.C.d1, A.B.C.d2));
}
EndTest();