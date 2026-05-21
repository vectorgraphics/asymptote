// Expression errors (exp.h, exp.cc)

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
