typedef import(T);

from collections.iter(T=T) access Iter_T, Iterable_T, Iterable;
from collections.set(T=T) access Set_T;

struct SortedSet_T {
  Set_T set;
  // Returns the least element > item, or nullT if there is no such
  // element.
  T after(T item);
  // Returns the greatest element < item, or nullT if there is no such
  // element.
  T before(T item);
  T firstGEQ(T item) {
    return set.contains(item) ? set.get(item) : after(item);
  }
  T firstLEQ(T item) {
    return set.contains(item) ? set.get(item) : before(item);
  }
  T min();               // Returns nullT if collection is empty.
  T popMin();            // Returns nullT if collection is empty.
  T max();               // Returns nullT if collection is empty.
  T popMax();            // Returns nullT if collection is empty.

  autounravel Iterable_T operator cast(SortedSet_T set) {
    return Iterable(set.set.operator iter);
  }
  
  autounravel Set_T operator cast(SortedSet_T sorted_set) {
    return sorted_set.set;
  }
  from set unravel *;
}

// For testing purposes, we provide a naive implementation of SortedSet_T.
// This implementation is highly inefficient, but it is correct, and can be
// used to test other implementations of SortedSet_T.
struct Naive_T {
  restricted SortedSet_T super;
  from super unravel nullT, isNullT;
  private bool lt(T a, T b) = null;
  private T[] buffer = new T[0];

  private bool leq(T a, T b) { return !lt(b, a); };
  private bool gt(T a, T b) { return lt(b, a); };
  private bool geq(T a, T b) { return leq(b, a); };
  private bool equiv(T a, T b) { return leq(a, b) && leq(b, a); };

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

  super.extract = new T(T item) {
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

  super.getRandom = new T() {
    if (buffer.length == 0) {
      assert(isNullT != null, 'No element to return');
      return nullT;
    }
    static int seed = 3567654160488757718;
    return buffer[(++seed).hash() % buffer.length];
  };

  autounravel SortedSet_T operator cast(Naive_T naive) {
    return naive.super;
  }

  // Compose cast operators, since implicit casting is not transitive.
  autounravel Set_T operator cast(Naive_T naive) {
    return (SortedSet_T)naive;
  }
  autounravel Iterable_T operator cast(Naive_T naive) {
    return Iterable(naive.super.operator iter);
  }


  from super unravel *;
}
