{
  struct PrivOnly {
    private void write(file f, void g(file)) { }
  }
  PrivOnly q;
  write(q);
}
