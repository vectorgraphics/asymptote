typedef import(K, V);

from collections.genericpair(K=K, V=V) access Pair_K_V;

from collections.iter(T=Pair_K_V) access
    Iter_T as Iter_Pair_K_V,
    Iterable_T as Iterable_Pair_K_V;
from collections.iter(T=K) access
    Iter_T as Iter_K,
    Iterable_T as Iterable_K;
from collections.iter(T=V) access
    Iter_T as Iter_V,
    Iterable_T as Iterable_V;

Iterable_Pair_K_V zip(Iterable_K a, Iterable_V b) {
  Iter_Pair_K_V iter() {
    Iter_K iterA = a.operator iter();
    Iter_V iterB = b.operator iter();
    Iter_Pair_K_V result;
    result.advance = new void() {
      iterA.advance();
      iterB.advance();
    };
    result.valid = new bool() {
      return iterA.valid() && iterB.valid();
    };
    result.get = new Pair_K_V() {
      return Pair_K_V(iterA.get(), iterB.get());
    };
    return result;
  }
  return Iterable_Pair_K_V(iter);
}