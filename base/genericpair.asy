typedef import(K, V);

struct Pair_K_V {
  restricted K k;
  restricted V v;
  void operator init(K k, V v) {
    this.k = k;
    this.v = v;
  }
}

Pair_K_V makePair(K k, V v) = Pair_K_V;

Pair_K_V operator >> (K k, V v) = makePair;