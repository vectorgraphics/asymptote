// Environment errors (env.h, env.cc)

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
