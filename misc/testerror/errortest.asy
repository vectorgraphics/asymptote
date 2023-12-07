/*****
 * errortest.asy
 * Andy Hammerlindl 2003/08/08
 *
 * This struct attempts to trigger every error message reportable by the
 * compiler to ensure that errors are being reported properly.
 *****/

// name.cc
{
  // line 33
  x.y = 5;
}
{
  // line 50
  int x = z;
}
{
  // line 67
  x = 5;
}
{
  // line 84 - unreachable
}
{
  // line 95
  x y1;
  x y2();
  x y3(int); 
  int y4(x);
  struct m {
    x y1;
    x y2();
    x y3(int); 
    int y4(x);
  }
}
{
  // line 130
  int x;
  x.y = 4;
}
{
  // line 156
  struct m {
    int x,y;
  }
  m.u.v = 5;
}
{
  // line 186
  struct m {
    int x,y;
  }
  int x = m.z;
}
{
  // line 217
  struct m {
    int x,y;
  }
  m.z = 5;
}
{
  // line 249 - unreachable
}
{
  // line 272 - not testable without typedef
}
{
  // line 283
  struct m {
    int x,y;
  }
  m.u v;
  struct mm {
   m.u v;
  }
}

// exp.h
{
  // line 109
  5 = 4;
}
{
  // line 136
  int f, f();
  f;
  struct m { int f, f(); }
  m.f;
}

// exp.cc
{
  // line 40
  int x = {};
  int y = {4, 5, 6};
}
{
  // line 110
  int f(), f(int);
  f.m = 5;
}
{
  // line 122
  int x;
  x.m = 5;
}
{
  // line 147
  struct point {
    int x,y;
  }
  point p;
  int x = p.z;
}
{
  // line 157
  struct point {
    int x, y;
  }
  point p;
  p.z;
}
{
  // line 163
  struct point {
    int f(), f(int);
  }
  point p;
  p.f;
}
{
  // line 204
  struct point {
    int x, y;
  }
  point p;
  p.z = 5;
}
{
  // line 229 - unreachable
}
{
  // lines 261 and 275 - wait for subscript to be fully implemented.
}
{
  // line 515
  void f(int), f(int, int);
  f();
  f("Hello?");
}
{
  // line 520 - not yet testable (need int->float casting or the like)
}
{
  // line 525
  void f(int);
  f();
}
{
  // line 649
  struct point {
    int x,y;
  }
  point p;
  p = +p;
}
{
  // line 1359
  int x, f();
  (true) ? x : f;
}
{
  // line 1359
  int a, a(), b, b();
  (true) ? a : b;
}
{
  // line 1581
  int x, f();
  x = f;
}
{
  // line 1586
  int a, a(), b, b();
  a = b;
}

// newexp.cc
{
  // line 34
  int f() = new int () {
    int x = 5;
  };
}
{
  // line 64
  int x = new int;
}
{
  // line 72
  struct a {
    struct b {
    }
  }

  new a.b;
}

// stm.cc
{
  // line 86
  5;
}
{
  // line 246
  break;
}
{
  // line 261
  continue;
}
{
  // line 282
  void f() {
    return 17;
  }
}
{
  // line 288
  int f() {
    return;
  }
}

// dec.cc
{
  // line 378
  int f() {
    int x = 5;
  }
  int g() {
    if (true)
      return 7;
  }
}

// env.h
{
  // line 99
  struct m {}
  struct m {}
}
{
  // line 109
  int f() {
    return 1;
  }
  int f() {
    return 2;
  }

  int x = 1;
  int x = 2;
 
  struct m {
    int f() {
      return 1;
    }
    int f() {
      return 2;
    }

    int x = 1;
    int x = 2;
  }
}

// env.cc
{
  // line 107 - currently unreachable as no built-ins are currently used
  // in records.
}
{
  // line 140
  // Assuming there is a built-in function void abort(string):
  void f(string);
  abort = f;
}
{
  // line 168 - currently unreachable as no built-in functions are
  // currently used in records.
}
{
  // line 222
  int x = "Hello";
}

// Test permissions.
{
  struct A {
    private int x=4;
    restricted int y=5;

    private void f() {}
    restricted void g() {}
  }

  A a;
  a.x;
  a.x = 4;
  a.x += 5;
  a.y = 6;
  a.y += 7;

  a.f;
  a.f();
  a.f = new void() {};
  a.g = new void() {};
}
{
  struct A {
    private int x=4;
    private void f() {}
  }

  A a;
  from a unravel x;
  from a unravel f;
}
{
  struct A {
    restricted int y=5;
    restricted void g() {}
  }

  A a;
  from a unravel y,g;
  y = 6;
  y += 7;

  g = new void() {};
}
{
  struct A {
    private typedef int T;
  }
  A.T x;
  A.T x=4;
  int y=2;
  A.T x=y;
  A.T x=17+3*y;
}
{
  struct A {
    private typedef int T;
  }
  from A unravel T;
}
{
  struct A {
    private typedef int T;
  }
  A a;
  a.T x;
  a.T x=4;
  int y=2;
  a.T x=y;
  a.T x=17+3*y;
}
{
  struct A {
    private typedef int T;
  }
  A a;
  from a unravel T;
}

// Read-only settings
// Ensure that the safe and globalwrite options can't be modified inside the
// code.
{
  access settings;
  settings.safe=false;
  settings.safe=true;
  settings.globalwrite=false;
  settings.globalwrite=true;
}
{
  from settings access safe, globalwrite;
  safe=false;
  safe=true;
  globalwrite=false;
  globalwrite=true;
}
{
  from settings access safe as x, globalwrite as y;
  x=false;
  x=true;
  y=false;
  y=true;
}
{
  import settings;
  settings.safe=false;
  settings.safe=true;
  settings.globalwrite=false;
  settings.globalwrite=true;
  safe=false;
  safe=true;
  globalwrite=false;
  globalwrite=true;
}

// Test cases where var is used outside of type inference.
{
  var x;
}
{
  var f() { return 4; }
}
{
  (var)3;
  var x = (var)3;
  int y = (var)3;
}
{
  var[] b = new var[] { 1, 2, 3};
  var[] b = new int[] { 1, 2, 3};
  var[] c = {1, 2, 3};
  new var[] { 4, 5, 6};
  int[] d = new var[] { 4, 5, 6};
  new var;
}
{
  int f(var x = 3) { return 0; }
}
{
  int f, f();
  var g = f;
}
{
  struct A { int f, f(); }
  var g = A.f;
  A a;
  var h = a.f;
}
{
  int x;
  for (var i : x)
    ;
}
{
  int x, x();
  for (var i : x)
    ;
}
{
  int x, x();
  int[] x = {2,3,4};
  for (var i : x)
    ;
}
{
  int[] temp={0};
  int[] v={0};

  temp[v]= v;
}

// Keyword and rest errors.
{
  int f(string s="" ... int[] a);
  f(1,2,3 ... new int[] {4,5,6}, "hi");
  f(1,2,3 ... new int[] {4,5,6} ... new int[] {7,8,9});
  f(... new int[] {4,5,6}, "hi");
  f(... new int[] {4,5,6} ... new int[] {7,8,9});
}
{
  int f(... int[] x, int y);
  int g(... int[] x, int y) { return 7; }
  int f(string s ... int[] x, int y);
  int g(string s ... int[] x, int y) { return 7; }
  int f(int keyword x, int y);
  int g(int keyword x, int y) { return 7; }
  int f(int keyword x, int y, string z);
  int g(int keyword x, int y, string z) { return 7; }
  int f(real t, int keyword x, int y);
  int g(real t, int keyword x, int y) { return 7; }
  int f(real t, int keyword x, int y, string z);
  int g(real t, int keyword x, int y, string z) { return 7; }
}
{
  int f(int notkeyword x);
  int g(int notkeyword x) { return 7; }
  int f(real w, int notkeyword x);
  int g(real w, int notkeyword x) { return 7; }
  int f(real w, int keyword y, int notkeyword x);
  int g(real w, int keyword y, int notkeyword x) { return 7; }
  int f(real w, int notkeyword y, int keyword x);
  int g(real w, int notkeyword y, int keyword x) { return 7; }
  int f(int notkeyword x, int y, string z);
  int g(int notkeyword x, int y, string z) { return 7; }
  int f(int notkeyword x, int y);
  int g(int notkeyword x, int y) { return 7; }
  int f(int notkeyword x, int y, string z);
  int g(int notkeyword x, int y, string z) { return 7; }
  int f(real t, int notkeyword x, int y);
  int g(real t, int notkeyword x, int y) { return 7; }
  int f(real t, int notkeyword x, int y, string z);
  int g(real t, int notkeyword x, int y, string z) { return 7; }
}
