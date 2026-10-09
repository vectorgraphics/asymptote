
import TestLib;

// The built-in `==` for records requires both operands to have the *same*
// record type and performs no implicit casting (see errortests/recordEq.asy).
// A user-declared `operator==` may, however, be given any signature, including
// one that spans two different record types, and it takes priority over the
// built-in even when a cast between the types exists.
StartTest("crossTypeEq: user operator== spanning two record types");
{
  struct A {
    int x;
  }
  struct B {
    int y;
  }

  // A cast B -> A is in scope, but the built-in `==` must not use it: it only
  // compares same-type operands.  This also lets us tell, from the result, that
  // the comparison below is handled by the user operator (whose logic compares
  // field values) rather than by casting B to a fresh A and comparing identity.
  A operator cast(B b) {
    A r = new A;
    r.x = b.y;
    return r;
  }

  // Cross-type equality: takes priority over the built-in same-type operator.
  bool operator==(A a, B b) {
    return a.x == b.y;
  }

  A a = new A; a.x = 1;
  B b = new B; b.y = 1;
  B b2 = new B; b2.y = 2;

  // The user operator is used, not the built-in (which would be an error for
  // differing record types) and not the cast.
  assert(a == b);    // user operator: 1 == 1
  assert(!(a == b2));  // user operator: 1 != 2

  // Same-type comparison is unaffected: `operator==(A, B)` does not apply to
  // (A, A), so the built-in identity semantics still hold.
  A c = new A; c.x = 1;
  assert(!(a == c));  // different instances
}
EndTest();

