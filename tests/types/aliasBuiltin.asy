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

StartTest("alias builtin: as function value");
{
  struct A {}
  A a = new A;
  A b = a;
  A c = new A;

  // Coerce the alias builtin to concrete function types for records, arrays,
  // and functions, mirroring `bool eq(R, R) = operator==;`.
  using aliasA = bool(A, A);
  aliasA aliasRecord = alias;
  assert(aliasRecord(a, b));
  assert(!aliasRecord(a, c));

  int[] data = new int[] {1, 2, 3};
  int[] dataAlias = data;
  int[] copied = copy(data);
  using aliasIntArray = bool(int[], int[]);
  aliasIntArray aliasArray = alias;
  assert(aliasArray(data, dataAlias));
  assert(!aliasArray(data, copied));

  using intFn = int(int);
  intFn square = new int(int x) { return x*x; };
  intFn cube = new int(int x) { return x*x*x; };
  intFn squareAlias = square;
  intFn squareAgain = squareAlias;
  using aliasIntFn = bool(intFn, intFn);
  aliasIntFn aliasFn = alias;
  assert(aliasFn(squareAlias, squareAgain));
  assert(!aliasFn(squareAlias, cube));
}
EndTest();