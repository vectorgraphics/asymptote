typedef import(K, V);

struct Pair_K_V {
  restricted K k;
  restricted V v;
  void operator init(K k, V v) {
    this.k = k;
    this.v = v;
  }
  autounravel bool operator ==(Pair_K_V a, Pair_K_V b) {
    if (alias(a, null)) return alias(b, null);
    // a is not null.
    if (alias(b, null)) return false;
    // a and b are not null.
    // NOTE: This won't compile if K or V is an array type since == is
    // vectorized for arrays. We could locally define a cast operator from
    // bool[] to bool, but that would not behave as expected if comparing two
    // arrays of different lengths. (We would get an error instead of false.)
    return a.k == b.k && a.v == b.v;
  }
  autounravel bool operator !=(Pair_K_V a, Pair_K_V b) {
    return !(a == b);
  }
  int hash();  // To be overridden by the user.
}

Pair_K_V makePair(K k, V v) = Pair_K_V;