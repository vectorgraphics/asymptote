{
  real notSuffix(real x) { return 7; }
  file f = output("write_errtest_tmp.txt");
  write(f, 1, notSuffix);
  close(f);
}
{
  // Function values and overloaded functions.
  write(1, 2, sin);
  write(1, sin, 3);
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
  // Rest arguments must be arrays of writeable values.
  struct NoWrite { }
  write(1 ... 2);
  write(1 ... new NoWrite[] {new NoWrite});
}
{
  // Each error is reported at the offending argument.
  struct NoWrite { }
  void suffix(file f) { }
  file g;
  write(1, g);
  write(1, suffix, 2);
  write(1, new NoWrite[] {new NoWrite});
}
