{
  real notSuffix(real x) { return 7; }
  file f = output("write_errtest_tmp.txt");
  write(f, 1, notSuffix);
  close(f);
}
