typedef import(Src, Dst);

private typedef Dst MapType(Src);

Dst[] map(MapType f, Src[] a) {
  return sequence(
      new Dst(int i) {return f(a[i]);},
      a.length);
}