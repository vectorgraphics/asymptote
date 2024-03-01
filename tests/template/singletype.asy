import TestLib;

StartTest("singletype");

struct A {}

// TODO: Should we import operator== and alias for free?
from "template/imports/wrapperWithEquals"(T=int) access
    Wrapper_T as Wrapper_int,
    wrap, operator ==, alias;
// TODO: Create a way to pass operator==(A, A) to the template, either
// implicitly or explicitly.
// from "template/imports/wrapperWithEquals"(T=A) access
//   Wrapper_T as Wrapper_A, wrap, operator ==, alias;  // error
from "template/imports/wrapper"(T=A)
    access Wrapper_T as Wrapper_A, wrap, alias;

// Basic functionality for ints:
Wrapper_int w1 = wrap(5);
Wrapper_int w2 = Wrapper_int(5);  // tests constructor
assert(w1.t == 5);
assert(w2.t == 5);
assert(w1 == w2);
assert(!alias(w1, w2));

// Basic functionality for A:
var a = new A;
Wrapper_A w3 = wrap(a);
Wrapper_A w4 = Wrapper_A(a);  // tests constructor
assert(w3.t == w4.t);
assert(!alias(w3, w4));

EndTest();
