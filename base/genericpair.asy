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
  autounravel bool alias(Pair_K_V, Pair_K_V) = alias;
  autounravel bool operator ==(Pair_K_V a, Pair_K_V b) {
    // NOTE: This won't compile if K or V is an array type since == is
    // vectorized for arrays.
    return a.k == b.k && a.v == b.v;
  }
}

Pair_K_V makePair(K k, V v) = Pair_K_V;