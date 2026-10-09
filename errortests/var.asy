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
