import TestLib;
StartTest("keyword");
{
  int f(int keyword x) {
    return 2*x;
  }

  assert(f(x=17) == 34);
}

{
  int f(int keyword x = 10) {
    return 2*x;
  }

  assert(f() == 20);
}

{
  int f(int keyword x = 10, int keyword y = 20)
  {
    return 2x+y;
  }

  assert(f(x=1,y=2) == 4);
  assert(f(y=1,x=2) == 5);
  assert(f(x=1) == 22);
  assert(f(y=7) == 27);
  assert(f() == 40);
}

{
  int f(int keyword x, int keyword y = 20)
  {
    return x+y;
  }

  assert(f(x=1,y=2) == 3);
  assert(f(x=1) == 21);
}

{
  int f(int keyword x = 10, int keyword y)
  {
    return x+y;
  }

  assert(f(x=1,y=2) == 3);
  assert(f(y=2) == 12);
}

{
  int f(int keyword x, int keyword y)
  {
    return x+y;
  }

  assert(f(x=1,y=2) == 3);
}

{
  int f(int x, int keyword y)
  {
    return 2x+y;
  }

  assert(f(x=1,y=2) == 4);
  assert(f(1,y=2) == 4);
  assert(f(y=2,1) == 4);
  assert(f(y=2,x=1) == 4);
}

{
  int f(... int[] nums, int keyword r)
  {
    return r;
  }

  assert(f(r=3) == 3);
  assert(f(1,r=3) == 3);
  assert(f(1,2, r=3) == 3);
  assert(f(1,2,4,5,6, r=3) == 3);
  assert(f(r=3, 10, 20, 30) == 3);
  assert(f(4, 5, r=3, 10, 20, 30) == 3);
  assert(f(4, 5, r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(... new int[] {40,50,60}, r=3) == 3);
  assert(f(... new int[] {40,50,60}, r=3) == 3);
}

{
  int f(... int[] nums, int keyword r=77)
  {
    return r;
  }

  assert(f(r=3) == 3);
  assert(f(1,r=3) == 3);
  assert(f(1,2, r=3) == 3);
  assert(f(1,2,4,5,6, r=3) == 3);
  assert(f(r=3, 10, 20, 30) == 3);
  assert(f(4, 5, r=3, 10, 20, 30) == 3);
  assert(f(4, 5, r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(... new int[] {40,50,60}, r=3) == 3);
  assert(f(... new int[] {40,50,60}, r=3) == 3);

  assert(f() == 77);
  assert(f(1) == 77);
  assert(f(1,2) == 77);
  assert(f(1,2,4,5,6) == 77);
  assert(f(10, 20, 30) == 77);
  assert(f(4, 5, 10, 20, 30) == 77);
  assert(f(4, 5, 10, 20, 30 ... new int[] {40,50,60}) == 77);
  assert(f(10, 20, 30 ... new int[] {40,50,60}) == 77);
  assert(f(10, 20, 30 ... new int[] {40,50,60}) == 77);
  assert(f(... new int[] {40,50,60}) == 77);
  assert(f(... new int[] {40,50,60}) == 77);
}

{
  int f(int x ... int[] nums, int keyword r=77)
  {
    return r;
  }

  assert(f(345,r=3) == 3);
  assert(f(345,1,r=3) == 3);
  assert(f(345,1,2, r=3) == 3);
  assert(f(345,1,2,4,5,6, r=3) == 3);
  assert(f(345,r=3, 10, 20, 30) == 3);
  assert(f(345,4, 5, r=3, 10, 20, 30) == 3);
  assert(f(345,4, 5, r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(345,r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(345,r=3, 10, 20, 30 ... new int[] {40,50,60}) == 3);
  assert(f(345 ... new int[] {40,50,60}, r=3) == 3);
  assert(f(345 ... new int[] {40,50,60}, r=3) == 3);

  assert(f(345) == 77);
  assert(f(345,1) == 77);
  assert(f(345,1,2) == 77);
  assert(f(345,1,2,4,5,6) == 77);
  assert(f(345,10, 20, 30) == 77);
  assert(f(345,4, 5, 10, 20, 30) == 77);
  assert(f(345,4, 5, 10, 20, 30 ... new int[] {40,50,60}) == 77);
  assert(f(345,10, 20, 30 ... new int[] {40,50,60}) == 77);
  assert(f(345,10, 20, 30 ... new int[] {40,50,60}) == 77);
  assert(f(345 ... new int[] {40,50,60}) == 77);
  assert(f(345 ... new int[] {40,50,60}) == 77);
}

{
  int sqr(int x=7) { return x*x; }
  int f(int keyword x) = sqr;
  int g(int keyword x=666) = sqr;
  assert(f(x=5) == 25);
  assert(g(x=5) == 25);
  assert(g() == 49);
}
{
  int sqr(int n=7) { return n*n; }
  int f(int keyword x) = sqr;
  int g(int keyword x=666) = sqr;
  assert(f(x=5) == 25);
  assert(g(x=5) == 25);
  assert(g() == 49);
}
{
  int sqr(int keyword x=7) { return x*x; }
  int f(int x) = sqr;
  int g(int x=666) = sqr;
  assert(f(x=5) == 25);
  assert(g(x=5) == 25);
  assert(f(5) == 25);
  assert(g(5) == 25);
  assert(g() == 49);
}
{
  int sqr(int keyword n=7) { return n*n; }
  int f(int x) = sqr;
  int g(int x=666) = sqr;
  assert(f(x=5) == 25);
  assert(g(x=5) == 25);
  assert(f(5) == 25);
  assert(g(5) == 25);
  assert(g() == 49);
}
EndTest();
