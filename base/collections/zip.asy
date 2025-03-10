typedef import(T);

private using _ = T[];  // Add array ops.

T[][] zip(...T[][] arrays) {
  if (arrays.length == 0) {
    return new T[0][];
  }
  int minLength = arrays[0].length;
  for (int i = 1; i < arrays.length; ++i) {
    if (arrays[i].length < minLength) {
      minLength = arrays[i].length;
    }
  }
  T[][] truncated = new T[arrays.length][minLength];
  for (int i = 0; i < arrays.length; ++i) {
    truncated[i] = arrays[i][0:minLength];
  }
  return transpose(truncated);
}

T[][] zip(T keyword default ...T[][] arrays) {
  if (arrays.length == 0) {
    return new T[0][];
  }
  int maxLength = arrays[0].length;
  for (int i = 1; i < arrays.length; ++i) {
    if (arrays[i].length > maxLength) {
      maxLength = arrays[i].length;
    }
  }
  T[][] padded = new T[arrays.length][maxLength];
  for (int i = 0; i < arrays.length; ++i) {
    padded[i][:arrays[i].length] = arrays[i];
    padded[i][arrays[i].length:] = array(maxLength - arrays[i].length, default);
  }
  return transpose(padded);
}

from collections.iter(T=T) access
    Iter_T as Iter_T,
    Iterable_T as Iterable_T;
from collections.iter(T=T[]) access
    Iter_T as Iter_Array_T,
    Iterable_T as Iterable_Array_T;

Iterable_Array_T zip(...Iterable_T[] iterables) {
  Iter_Array_T iter() {
    Iter_T[] iters = new Iter_T[iterables.length];
    for (int i = 0; i < iterables.length; ++i) {
      iters[i] = iterables[i].operator iter();
    }
    Iter_Array_T result;
    result.advance = new void() {
      for (int i = 0; i < iters.length; ++i) {
        iters[i].advance();
      }
    };
    result.valid = new bool() {
      for (int i = 0; i < iters.length; ++i) {
        if (!iters[i].valid()) {
          return false;
        }
      }
      return true;
    };
    result.get = new T[]() {
      T[] result = new T[iters.length];
      for (int i = 0; i < iters.length; ++i) {
        result[i] = iters[i].get();
      }
      return result;
    };
    return result;
  }
  return Iterable_Array_T(iter);
}

Iterable_Array_T zip(T keyword default ...Iterable_T[] iterables) {
  Iter_Array_T iter() {
    Iter_T[] iters = new Iter_T[iterables.length];
    for (int i = 0; i < iterables.length; ++i) {
      iters[i] = iterables[i].operator iter();
    }
    Iter_Array_T result;
    result.advance = new void() {
      for (Iter_T iter : iters) {
        if (iter.valid()) iter.advance();
      }
    };
    result.valid = new bool() {
      for (Iter_T iter : iters) {
        if (iter.valid()) {
          return true;
        }
      }
      return false;
    };
    result.get = new T[]() {
      T[] result = new T[iters.length];
      for (int i = 0; i < iters.length; ++i) {
        if (iters[i].valid()) {
          result[i] = iters[i].get();
        } else {
          result[i] = default;
        }
      }
      return result;
    };
    return result;
  }
  return Iterable_Array_T(iter);
}
