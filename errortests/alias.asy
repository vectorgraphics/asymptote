// Tests for alias call-time error constraints.
//
// The builtin 'alias' compares two nullable (reference-type) values for
// object identity.  It requires both arguments to be of the same nullable
// type.  No implicit casting is performed.  User declarations of alias
// are allowed without restriction and take priority over the builtin.
//
// NOTE: Each test case is wrapped in its own scope ({...}) so that error
// recovery in -debug mode (used by the error-test driver) does not leak
// state between cases.

{
  // Both arguments are null literals: not defined by the builtin alias.
  alias(null, null);  // error
}

{
  struct A {}
  struct B {}
  A a = new A;
  B b = new B;
  // Two non-null arguments of different nullable types.
  alias(a, b);  // error
}

{
  // Non-nullable (primitive) argument type.
  alias(1, 2);  // error
}

{
  // Two different types, even if one is castable to the other.
  struct A {}
  struct B {}
  A a = new A;
  B b = new B;
  A operator cast(B b) { return new A; }
  alias(a, b);  // error
  alias(b, a);  // error
}

{
  // Overloaded variable names for arguments.
  struct B {}
  B c;
  real c(real);
  real f(real);
  alias(c, f);  // no error: choose the c that matches the type of f
  alias(f, c);  // no error: choose the c that matches the type of f
  alias(c, c);  // error: ambiguous
}
{
  // Doubly-overloaded variable names for arguments.
  real f(real, real);
  real f(pair);
  real g(pair);
  real g(int);
  alias(f, g);  // no error: real(pair) is the only match for both f and g
  alias(g, f);  // no error: real(pair) is the only match for both g and f
  alias(f, null);  // error: ambiguous
  alias(null, f);  // error: ambiguous
}
