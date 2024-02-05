import TestLib;

StartTest("singletype");

struct A { }
var a = new A;

// Desired behavior: We should not need to specify that operator== is accessed.
// Two possible justifications:
//
// 1. The function operator== has a required parameter of type Wrapper_T, so
//    overload resolution should prevent it from conflicting with any other
//    functions of the same name.
// 2. It is rather confusing to have to worry explicitly about importing a
//    basic operator like ==.
//
// I think that 1 is the stronger justification and would like if we could
// implement a general feature in which accessing a type also accesses
// all functions with at least one required parameter of that type.
from "template/imports/wrapper"(T=int) access Wrapper_T as Wrapper_int, wrap, operator ==;
// exit();
// unravel wrapper_int;
// typedef Wrapper_T wrapper_int;
//from wrapper_int unravel Wrapper_T as Wrapper_int;  // rename Wrapper_T to Wrapper_int
//from "template/imports/wrapper"(T=A) access Wrapper_T as Wrapper_A, wrap;

Wrapper_int w = wrap(5);
// Wrapper_T w = wrap(5);
assert(w.t == 5);
// Wrapper_A at = wrap(a);
// assert(at.t == a);

Wrapper_int w2 = wrap(5);
assert(w2 == w);

EndTest();
