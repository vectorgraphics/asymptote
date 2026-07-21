// Name resolution errors (name.cc)
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
