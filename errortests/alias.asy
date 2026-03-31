// Tests for alias declaration constraints.
//
// The builtin 'alias' compares two nullable (reference-type) values for
// object identity.  Any user declaration named 'alias' whose signature could
// intercept a call that would otherwise be dispatched to the builtin is
// rejected at declaration time.  The key criterion: two formals of the same
// reference (nullable) type, with all additional formals having defaults.

struct Obj {}

// ----- Rejected: function declarations -----

// Basic case: two nullable params of the same struct type.
bool alias(Obj, Obj);  // error

// Return type does not matter; the signature check fires regardless.
Obj alias(Obj, Obj);   // error

// Array types are reference (nullable) types.
bool alias(int[], int[]);  // error

// Function types are reference (nullable) types.
bool alias(void f(), void g());  // error

// Extra params are OK only when required; optional extras still conflict
// because alias(s1, s2) could be dispatched to the builtin.
bool alias(Obj, Obj, int extra = 0);  // error

// A rest parameter does not prevent rejection when the two named formals
// already match: alias(a, b) with two nullable args still resolves here.
bool alias(Obj, Obj ... int[] r);  // error

// Rest-only: alias(a, b) routes both args to the rest param without any cast.
bool alias(... Obj[] r);  // error

// One named nullable formal plus a rest of the same type: alias(a, b) still
// resolves — first arg matches named, second goes to rest as Obj.
bool alias(Obj a ... Obj[] r);  // error

// ----- Rejected: non-function-declaration forms -----

// A function parameter named 'alias' of a nullable-compatible type.
// Parameters are entered into the environment when the body is translated,
// so a function *with a body* triggers the error.
void takes_alias_param(bool alias(Obj, Obj)) {}  // error

// A variable declared via a typedef.  typedef is used here so the variable
// declaration is syntactically distinct from a function declaration.
{
  typedef bool ObjAlias(Obj, Obj);
  ObjAlias alias;  // error
}

// ----- No error: forward declaration without a body -----

// Parameters are not translated for a forward declaration, so the
// restriction does not fire here.
void takes_alias_param_forward(bool alias(Obj, Obj));  // no error

// ----- Allowed: signature cannot intercept the builtin -----
{
  struct A {}
  struct B {}

  // Primitive (non-reference) types: int, real, etc. are not nullable.
  bool alias(int, int);    // ok
  bool alias(real, real);  // ok
  bool alias(bool, bool);  // ok

  // Two different struct types: the two formals are not equivalent.
  bool alias(A, B);  // ok

  // Only one formal: cannot match a two-arg builtin call.
  bool alias(A);  // ok

  // Zero formals.
  bool alias();  // ok

  // Two same-type nullable formals plus a *required* extra formal:
  // a two-arg call cannot reach this overload at all.
  bool alias(A, A, int extra);  // ok

  // Rest-only but with a non-nullable (primitive) element type: int is not
  // a reference type, so alias(a, b) with nullable args would need a cast.
  bool alias(... int[] r);  // ok

  // One named nullable formal, rest of a *different* nullable type: the
  // second arg would need an implicit cast to reach the rest element type.
  bool alias(A a ... B[] r);  // ok

  // A non-function variable named alias.
  int alias;  // ok

  // A struct member variable (non-function) named alias.
  struct C {
    int alias;  // ok
  }
}
