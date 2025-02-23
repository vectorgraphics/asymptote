typedef import(T);

from collections.iter(T=T) access Iter_T, Iterable_T, Iterable;
from collections.repset(T=T) access RepSet_T;

struct SortedRepSet_T {
  RepSet_T repset;
  // Returns the least element > item, or nullT if there is no such
  // element.
  T after(T item);
  // Returns the greatest element < item, or nullT if there is no such
  // element.
  T before(T item);
  T firstGEQ(T item) {
    return repset.contains(item) ? repset.get(item) : after(item);
  }
  T firstLEQ(T item) {
    return repset.contains(item) ? repset.get(item) : before(item);
  }
  T min();               // Returns nullT if collection is empty.
  T popMin();            // Returns nullT if collection is empty.
  T max();               // Returns nullT if collection is empty.
  T popMax();            // Returns nullT if collection is empty.

  autounravel Iterable_T operator cast(SortedRepSet_T set) {
    return Iterable(set.repset.operator iter);
  }
  
  autounravel RepSet_T operator cast(SortedRepSet_T sorted_set) {
    return sorted_set.repset;
  }
  from repset unravel *;
}

// For testing purposes, we provide a naive implementation of SortedRepSet_T.
// This implementation is highly inefficient, but it is correct, and can be
// used to test other implementations of SortedRepSet_T.
struct Naive_T {
  struct _ { autounravel restricted SortedRepSet_T super; }
  from super unravel nullT, isNullT;
  private bool lt(T a, T b) = null;
  private T[] buffer = new T[0];

  private bool leq(T, T), gt(T, T), geq(T, T), equiv(T, T);
  
  leq = new bool(T a, T b) {
    return !lt(b, a);
  };

  gt = new bool(T a, T b) {
    return lt(b, a);
  };

  geq = new bool(T a, T b) {
    return leq(b, a);
  };
  
  equiv = new bool(T a, T b) {
    return leq(a, b) && leq(b, a);
  };

  void operator init(bool lessThan(T, T), T nullT) {
    this.lt = lessThan;
    super.operator init(nullT, equiv);
  }

  super.size = new int() {
    return buffer.length;
  };

  super.contains = new bool(T item) {
    for (T possibility : buffer) {
      if (equiv(possibility, item)) return true;
    }
    return false;
  };

  super.after = new T(T item) {
    for (T possibility : buffer) {
      if (gt(possibility, item)) return possibility;
    }
    assert(isNullT != null, 'No element after item to return');
    return nullT;
  };

  super.before = new T(T item) {
    for (int ii = buffer.length - 1; ii >= 0; --ii) {
      T possibility = buffer[ii];
      if (lt(possibility, item)) return possibility;
    }
    assert(isNullT != null, 'No element before item to return');
    return nullT;
  };

  super.min = new T() {
    if (buffer.length == 0) {
      assert(isNullT != null, 'No minimum element to return');
      return nullT;
    }
    return buffer[0];
  };

  super.popMin = new T() {
    if (buffer.length == 0) {
      assert(isNullT != null, 'No minimum element to return');
      return nullT;
    }
    T toreturn = buffer[0];
    buffer.delete(0);
    return toreturn;
  };

  super.max = new T() {
    if (buffer.length == 0) {
      assert(isNullT != null, 'No maximum element to return');
      return nullT;
    }
    return buffer[buffer.length - 1];
  };

  super.popMax = new T() {
    if (buffer.length == 0) {
      assert(isNullT != null, 'No maximum element to return');
      return nullT;
    }
    return buffer.pop();
  };

  super.add = new bool(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) return false;
      else if (gt(buffer[i], item)) {
        buffer.insert(i, item);
        return true;
      }
    }
    buffer.push(item);
    return true;
  };

  super.swap = new T(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) {
        T toreturn = buffer[i];
        buffer[i] = item;
        return toreturn;
      }
      else if (gt(buffer[i], item)) {
        buffer.insert(i, item);
        return nullT;
      }
    }
    buffer.push(item);
    return nullT;
  };

  super.get = new T(T item) {
    for (T possibility : buffer) {
      if (equiv(possibility, item)) return possibility;
    }
    return nullT;
  };

  super.delete = new T(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      T candidate = buffer[i];
      if (equiv(candidate, item)) {
        buffer.delete(i);
        return candidate;
      }
    }
    return nullT;
  };

  super.operator iter = new Iter_T() {
    return Iter_T(buffer);
  };

  autounravel SortedRepSet_T operator cast(Naive_T naive) {
    return naive.super;
  }

  // Compose cast operators, since implicit casting is not transitive.
  autounravel RepSet_T operator cast(Naive_T naive) {
    return (SortedRepSet_T)naive;
  }
  autounravel Iterable_T operator cast(Naive_T naive) {
    return Iterable(naive.super.operator iter);
  }

  from super unravel *;
}
