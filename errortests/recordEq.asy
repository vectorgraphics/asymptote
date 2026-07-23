// Tests for the default record `==` / `!=` operators.
//
// The built-in `==` / `!=` for structures compare object identity and require
// both operands to have the *same* record type.  Unlike an ordinary overloaded
// function, they perform no implicit casting to unify the operand types: if the
// operands are of different types, the comparison is an error even when a cast
// between them exists.  (A user-defined `operator ==` may still be declared for
// any signature and takes priority over the built-in.)
//
// NOTE: Each test case is wrapped in its own scope ({...}) so that error
// recovery in -debug mode (used by the error-test driver) does not leak state
// between cases.

{
  // Different record types: no matching built-in `==`.
  struct A {}
  struct B {}
  A a = new A;
  B b = new B;
  bool result = (a == b);  // error
}

{
  // Different record types, even though a cast B -> A is in scope: the built-in
  // `==` does not cast to unify operand types.
  struct A {}
  struct B {}
  A operator cast(B b) { return new A; }
  A a = new A;
  B b = new B;
  bool result = (a == b);  // error
  bool other = (a != b);   // error
}
