typedef import(T);

from pureset(T=T) access Set_T, operator cast, makeNaiveSet;

struct SortedSet_T {
  int size();
  bool empty() { return size() == 0; }
  bool contains(T item);
  // Returns the least element > item, or emptyresponse if there is no such
  // element.
  T after(T item);
  // Returns the greatest element < item, or emptyresponse if there is no such
  // element.
  T before(T item);
  T firstGEQ(T item) { return contains(item) ? item : after(item); }
  T firstLEQ(T item) { return contains(item) ? item : before(item); }
  T min();               // Returns emptyresponse if collection is empty.
  T popMin();            // Returns emptyresponse if collection is empty.
  T max();               // Returns emptyresponse if collection is empty.
  T popMax();            // Returns emptyresponse if collection is empty.
  bool insert(T item);   // Returns true iff the collection is modified.
  T replace(T item);     // Inserts item, and returns the item that was
                         // replaced, or emptyresponse if no item was replaced.
  T get(T item);         // Returns the item in the collection that is
                         // equivalent to item, or emptyresponse if there is no
                         // such item.
  bool delete(T item);   // Returns true iff the collection is modified.
  // Calls process on each item in the collection, in ascending order,
  // until process returns false.
  void forEach(bool process(T item));
}

T[] operator cast(SortedSet_T set) {
  T[] result;
  set.forEach(new bool(T item) {
                result.push(item);
                return true;
              });
  return result;
}

Set_T unSort(SortedSet_T sorted_set) {
  Set_T set = new Set_T;
  set.size = sorted_set.size;
  set.empty = sorted_set.empty;
  set.contains = sorted_set.contains;
  set.insert = sorted_set.insert;
  set.replace = sorted_set.replace;
  set.get = sorted_set.get;
  set.delete = sorted_set.delete;
  set.forEach = sorted_set.forEach;
  return set;
}

Set_T operator cast(SortedSet_T) = unSort;

// For testing purposes, we provide a naive implementation of SortedSet_T.
// This implementation is highly inefficient, but it is correct, and can be
// used to test other implementations of SortedSet_T.
struct NaiveSortedSet_T {
  private bool lt(T a, T b);
  private T[] buffer = new T[0];
  private T emptyresponse;

  private bool leq(T a, T b) {
    return !lt(b, a);
  }
  private bool gt(T a, T b) {
    return lt(b, a);
  }
  private bool geq(T a, T b) {
    return leq(b, a);
  }
  private bool equiv(T a, T b) {
    return leq(a, b) && leq(b, a);
  }

  void operator init(bool lessThan(T, T), T emptyresponse) {
    this.lt = lessThan;
    this.emptyresponse = emptyresponse;
  }

  int size() {
    return buffer.length;
  }

  bool contains(T item) {
    for (T possibility : buffer) {
      if (equiv(possibility, item)) return true;
    }
    return false;
  }

  T after(T item) {
    for (T possibility : buffer) {
      if (gt(possibility, item)) return possibility;
    }
    return emptyresponse;
  }

  T before(T item) {
    for (int ii = buffer.length - 1; ii >= 0; --ii) {
      T possibility = buffer[ii];
      if (lt(possibility, item)) return possibility;
    }
    return emptyresponse;
  }

  T min() {
    if (buffer.length == 0) return emptyresponse;
    return buffer[0];
  }

  T popMin() {
    if (buffer.length == 0) return emptyresponse;
    T toreturn = buffer[0];
    buffer.delete(0);
    return toreturn;
  }

  T max() {
    if (buffer.length == 0) return emptyresponse;
    return buffer[buffer.length - 1];
  }

  T popMax() {
    if (buffer.length == 0) return emptyresponse;
    return buffer.pop();
  }

  bool insert(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) return false;
      else if (gt(buffer[i], item)) {
        buffer.insert(i, item);
        return true;
      }
    }
    buffer.push(item);
    return true;
  }

  T replace(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) {
        T toreturn = buffer[i];
        buffer[i] = item;
        return toreturn;
      }
      else if (gt(buffer[i], item)) {
        buffer.insert(i, item);
        return emptyresponse;
      }
    }
    buffer.push(item);
    return emptyresponse;
  }

  T get(T item) {
    for (T possibility : buffer) {
      if (equiv(possibility, item)) return possibility;
    }
    return emptyresponse;
  }

  bool delete(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) {
        buffer.delete(i);
        return true;
      }
    }
    return false;
  }

  void forEach(bool process(T item)) {
    for (T item : buffer) {
      if (!process(item)) break;
    }
  }
}

SortedSet_T operator cast(NaiveSortedSet_T naive) {
  SortedSet_T toreturn;
  toreturn.size = naive.size;
  toreturn.contains = naive.contains;
  toreturn.after = naive.after;
  toreturn.before = naive.before;
  toreturn.min = naive.min;
  toreturn.popMin = naive.popMin;
  toreturn.max = naive.max;
  toreturn.popMax = naive.popMax;
  toreturn.insert = naive.insert;
  toreturn.replace = naive.replace;
  toreturn.get = naive.get;
  toreturn.delete = naive.delete;
  toreturn.forEach = naive.forEach;
  return toreturn;
}

// Compose cast operators, since implicit casting is not transitive.
T[] operator cast(NaiveSortedSet_T naive) {
  return (SortedSet_T)naive;
}

SortedSet_T makeNaiveSortedSet(bool lessThan(T, T), T emptyresponse) {
  return NaiveSortedSet_T(lessThan, emptyresponse);
}