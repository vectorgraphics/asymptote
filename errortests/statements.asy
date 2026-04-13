// Statement and declaration errors (newexp.cc, stm.cc, dec.cc)

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
