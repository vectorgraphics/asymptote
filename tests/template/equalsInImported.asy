import TestLib;

StartTest("singletype");
write('\n');

struct A { }
var a = new A;
var b = new A;

// both of the following pass (comparing pointers, I assume)
assert(a == a);
assert(a != b);

// The next line gives error message:
//   return a.t == b.t;
//              ^
// template/imports/wrapper.asy: 15.14: no matching function 'operator ==(A, A)'
// from "template/imports/wrapper"(T=A) access Wrapper_T as Wrapper_A, wrap, operator==;
// ^
// template/equalsInImported.asy: 21.1: could not load module 'template/imports/wrapper'
from "template/imports/wrapper"(T=A) access Wrapper_T as Wrapper_A, wrap, operator==;
// Presumably, this indicates that somehow, operator==(A, A), which was defined implicitly,
// is not being passed to the imported module.

EndTest();
