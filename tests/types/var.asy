import TestLib;
StartTest("var");
var x = 4;
assert(x + x == 8);
assert(x^2 == 16);
assert((real)x == 4.0);

var y = x;
assert(y + y == 8);
assert(x + y == 8);
assert(y^2 == 16);
assert((real)y == 4.0);

var z = 2 * x;
assert(z + z == 16);
assert(x + z == 12);
assert(z^2 == 64);
assert((real)z == 8.0);

var t = sin(0);
assert(t == 0.0);
assert(2t == 0.0);

struct A {
  int x;
};

A a;
a.x = 3;

var b = a;
assert(a == b);
assert(a.x == 3);

A func(int x, int y) { return a; }

var c = func(2,3);
assert(a == b);
assert(a.x == 3);

int f(int x) { return x*x; }
var g = f;
assert(g == f);
for (int i = 0; i < 100; ++i)
    assert(g(i) == i*i);

/* var can be replaced with a normal type. */
{
    typedef int var;
    var x;
    assert(x == 0);

    var v = 14;
    assert(v == 14);
}

{
    struct var {
        int x;
    }

    var n = null;
    assert(n == null);

    var a;
    assert(a != null);
    assert(a.x == 0);
    a.x = 11;
    assert(a.x == 11);
}

// Test for single evaluation of the initializer.
{
    int x = 3;
    assert(x == 3);
    var y = ++x;
    assert(x == 4);
    assert(y == 4);
}

{
  int f = 4, f() = null;
  var x = (int)f;
  assert(x == 4);
}
EndTest();
