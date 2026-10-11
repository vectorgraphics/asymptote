{
  // An overloaded name with no value: which function is meant is unknown.
  write(1, 2, sin);
  write(1, sin, 3);
}
{
  // A rest argument must be an array.
  write(1 ... 2);
}
{
  // Named arguments.
  write(1, x=2);
  write(1, s=2);
  write(1, file="a");
  write(1, suffix=3);
  write(1, s="a", s="b");
}
{
  // A void expression has no value to write.  The error is reported at the
  // offending argument.
  void nothing() { }
  write(1, nothing(), 2);
}
