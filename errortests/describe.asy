{
  // describe takes exactly one unnamed argument.
  string s;
  s = describe();
  s = describe(1, 2);
  s = describe(x=1);
  s = describe(1 ... new int[] {2});
}
{
  // A void expression has no value, and an overloaded name with no value
  // does not say which function is meant.
  void nothing() { }
  int twice(int x) { return 2x; }
  real twice(real x) { return 2x; }
  string s;
  s = describe(nothing());
  s = describe(twice);
}
