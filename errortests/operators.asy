// Operator[] and iterator errors.
{
  // multiple signatures for operator[]
  struct A {
    int operator[](string);
    int operator[](int);
  }
}
{
  // operator[=] without operator[]
  struct A {
    void operator[=](int);
  }
}
{
  // non-void operator[=]
  struct A {
    int operator[](string);
    int operator[=](string, int);
  }
}
{
  // operator iter returns a non-iterable type
  struct A {
    int operator iter() { return 0; }
  }
  A a;
  for (var i : a)
    ;
}
{
  // Implicitly cast a function to an array
  using Function = int(int);
  int[] operator cast(Function f) {
    return sequence(f, 10);
  }
  int f(int i) { return i + 17; }
  for (var i : f)  // This would work if we used `int` rather than `var`.
    ;
}
{
  // Iterate over an ill-formed expression
  int f(int i) { return 7; }
  // cannot call 'int f(int i)' with parameter 'string'
  for (int i : f('asdf'))
    ;
  // cannot call 'int f(int i)' with parameter 'string'
  for (var i : f('asdf'))
    ;
}
