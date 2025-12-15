typedef import(T);

from collections.iter(T=T) access Iter_T, Iterable_T;
from collections.genericpair(K=int, V=T) access
    Pair_K_V as Pair_int_T,
    makePair;
from collections.iter(T=Pair_int_T) access
    Iter_T as Iter_Pair_int_T,
    Iterable_T as Iterable_Pair_int_T,
    Iterable;

Iterable_Pair_int_T enumerate(Iterable_T iterable) {
  Iter_Pair_int_T iter() {
    int i = 0;
    Iter_T it = iterable.operator iter();
    Iter_Pair_int_T result;
    result.valid = it.valid;
    result.get = new Pair_int_T() {
      return makePair(i, it.get());
    };
    result.advance = new void() {
      ++i;
      it.advance();
    };
    return result;
  }
  return Iterable(iter);
}

Iterable_Pair_int_T enumerate(T[] array) {
  Iter_Pair_int_T iter() {
    int i = 0;
    Iter_Pair_int_T result;
    result.valid = new bool() {
      return i < array.length;
    };
    result.get = new Pair_int_T() {
      return makePair(i, array[i]);
    };
    result.advance = new void() {
      ++i;
    };
    return result;
  }
  return Iterable(iter);
}