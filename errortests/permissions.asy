// Permission and access control errors.
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
// Test more permissions.
{
  struct A {
    static int x;
    static int f;
    static void f();
    static struct R {}
  }
  struct T {
    private static from A unravel x;
    private static from A unravel f;
    private static from A unravel R;
  }
  T.x;  // incorrectly accessing private field
  (int)T.f;  // incorrectly accessing overloaded private field
  T.f();  // incorrectly accessing overloaded private field
  T.R r;  // correctly accessing private type
  struct U {
    private static unravel A;
  }
  U.x;  // incorrectly accessing private field
  (int)U.f;  // incorrectly accessing overloaded private field
  U.f();  // incorrectly accessing overloaded private field
  U.R r;  // correctly accessing private type
}
{
  struct A {
    static struct B {
      autounravel int x;
    }
  }
  struct T {
    private from A unravel B;  // x is autounraveled from B
  }
  T.x;  // incorrectly accessing private field
}
