// Test cases where var is used outside of type inference.
{
  var x;
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
  int[] d = new var[] { 4, 5, 6};
  new var;
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
{
  var[] b = new var[] {1};
  b.push(2);
  b.insert(0, 3);
  var[] c = new var[] {4};
  b.append(c);
}
{
  var[] b = new var[] {1, 2.5, "s"};
  var x = b[0];
  int i = (int)x;
  real r = (real)x;
}

{
  void g() {}
  void h(var x) {}
  h(g());
  h(null);
  var[] a = {g()};
  var[] b = new var[] {null};
  write(g());
  write(1, g());
  var f() { return g(); }
  void k(var x=g()) {}
}
