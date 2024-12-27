typedef import(K, V);

struct Pair_K_V {
  restricted K k;
  restricted V v;
  void operator init(K k, V v) {
    this.k = k;
    this.v = v;
  }
  autounravel Pair_K_V operator >> (K k, V v) {
    Pair_K_V pr = new Pair_K_V;
    pr.k = k;
    pr.v = v;
    return pr;
  }
  autounravel bool operator ==(Pair_K_V a, Pair_K_V b) {
    // NOTE: This won't compile if K or V is an array type since == is
    // vectorized for arrays. We could locally define a cast operator from
    // bool[] to bool, but that would not behave as expected if comparing two
    // arrays of different lengths. (We would get an error instead of false.)
    return a.k == b.k && a.v == b.v;
  }
  int hash();  // To be overridden by the user.
}

Pair_K_V makePair(K k, V v) = Pair_K_V;