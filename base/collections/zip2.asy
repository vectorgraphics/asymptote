typedef import(K, V);

from collections.genericpair(K=K, V=V) access Pair_K_V, makePair;

from collections.iter(T=Pair_K_V) access
    Iter_T as Iter_Pair_K_V,
    Iterable_T as Iterable_Pair_K_V;
from collections.iter(T=K) access
    Iter_T as Iter_K,
    Iterable_T as Iterable_K;
from collections.iter(T=V) access
    Iter_T as Iter_V,
    Iterable_T as Iterable_V;

Iterable_Pair_K_V zip(Iterable_K a, Iterable_V b, Pair_K_V keyword default=null)
{
  Iter_Pair_K_V iter() {
    Iter_K iterA = a.operator iter();
    Iter_V iterB = b.operator iter();
    Iter_Pair_K_V result;
    if (alias(default, null)) {
      result.advance = new void() {
        iterA.advance();
        iterB.advance();
      };
      result.valid = new bool() {
        return iterA.valid() && iterB.valid();
      };
      result.get = new Pair_K_V() {
        return makePair(iterA.get(), iterB.get());
      };
    } else {
      result.advance = new void() {
        if (iterA.valid()) iterA.advance();
        if (iterB.valid()) iterB.advance();
      };
      result.valid = new bool() {
        return iterA.valid() || iterB.valid();
      };
      result.get = new Pair_K_V() {
        K k = iterA.valid() ? iterA.get() : default.k;
        V v = iterB.valid() ? iterB.get() : default.v;
        return makePair(k, v);
      };
    }
    return result;
  }
  return Iterable(iter);
}