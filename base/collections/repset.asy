typedef import(T);
from collections.iter(T=T) access Iter_T, Iterable_T;

// RepSet: set of representatives of equivalence classes. Contains at most one
// element from each equivalence class.


struct RepSet_T {
  restricted T nullT;
  restricted bool equiv(T, T) = operator ==;
  restricted bool isNullT(T) = null;
  restricted void operator init() {}
  restricted void operator init(T nullT,
      bool equiv(T a, T b) = operator ==,
      bool isNullT(T) = new bool(T t) { return equiv(t, nullT); }) {
    this.nullT = nullT;
    this.equiv = equiv;
    this.isNullT = isNullT;
  }

  // Creates a new, empty RepSet with the same implemention, nullT,
  // isNullT, and equiv as this one.
  RepSet_T newEmpty();

  int size();
  bool empty() {
    return size() == 0;
  }
  bool contains(T item);
  // Returns the equivalent item in the set, or nullT if the set
  // contains no equivalent item. Throws error if nullT was never set.
  T get(T item);
  // Returns an iterator over the items in the set.
  Iter_T operator iter();
  // If an equivalent item was already present, returns false. Otherwise, adds
  // the item and returns true. Noop if isNullT is defined and item is empty.
  bool add(T item);  
  // Inserts item, and returns the item that was replaced, or nullT if
  // no item was replaced. Throws error if there is no equivalent item and nullT
  // was never set. Noop if isNullT is defined and isNullT(item).
  T swap(T item);
  // Removes the equivalent item from the set, and returns it. Returns
  // nullT if there is no equivalent item. Throws error if
  // there is no equivalent item and nullT was never set.
  T delete(T item);
  // Used primarily in testing to get a random element. Most implementations
  // will not support this operation.
  T get_ith(int i) {
    assert(false, 'get_ith not implemented');
    return nullT;
  }

  autounravel Iterable_T operator cast(RepSet_T set) {
    return Iterable_T(set.operator iter);
  }

  void addAll(Iterable_T other) {
    for (T item : other) {
      add(item);
    }
  }
  void removeAll(Iterable_T other) {
    for (T item : other) {
      delete(item);
    }
  }

  autounravel bool operator <=(RepSet_T a, RepSet_T b) {
    for (var item : a) {
      if (!b.contains(item)) {
        return false;
      }
    }
    return true;
  }

  autounravel bool operator >=(RepSet_T a, RepSet_T b) {
    return b <= a;
  }

  autounravel bool operator ==(RepSet_T a, RepSet_T b) {
    return a <= b && a >= b;
  } 

  autounravel bool operator !=(RepSet_T a, RepSet_T b) {
    return !(a == b);
  }

  autounravel bool sameElementsInOrder(RepSet_T a, RepSet_T b) {
    bool equiv(T ai, T bi) {
      return a.equiv(ai, bi) && b.equiv(ai, bi);
    }
    var iterA = a.operator iter();
    var iterB = b.operator iter();
    while (iterA.valid() && iterB.valid()) {
      if (!equiv(iterA.get(), iterB.get())) {
        return false;
      }
      iterA.advance();
      iterB.advance();
    }
    return iterA.valid() == iterB.valid();
  }

  autounravel RepSet_T operator +(RepSet_T a, Iterable_T b) {
    RepSet_T result = a.newEmpty();
    for (T item : a) {
      result.add(item);
    }
    for (T item : b) {
      result.add(item);
    }
    return result;
  }

  autounravel RepSet_T operator -(RepSet_T a, RepSet_T b) {
    RepSet_T result = a.newEmpty();
    for (T item : a) {
      if (!b.contains(item)) {
        result.add(item);
      }
    }
    return result;
  }

}


// A reference implementation, inefficient but suitable for testing.
struct NaiveRepSet_T {
  RepSet_T super;
  unravel super;
  private T[] items;
  restricted void operator init() {
    typedef void F();
    ((F)super.operator init)();
  }
  restricted void operator init(T nullT,
      bool equiv(T a, T b) = operator ==,
      bool isNullT(T) = new bool(T t) { return equiv(t, nullT); }) {
    typedef void F(T, bool equiv(T, T), bool isNullT(T));
    ((F)super.operator init)(nullT, equiv, isNullT);
  }

  super.size = new int() {
    return items.length;
  };

  super.contains = new bool(T item) {
    for (T i : items) {
      if (equiv(i, item)) {
        return true;
      }
    }
    return false;
  };

  super.get = new T(T item) {
    for (T i : items) {
      if (equiv(i, item)) {
        return i;
      }
    }
    return nullT;
  };

  super.operator iter = new Iter_T() {
    return Iter_T(items);
  };

  super.add = new bool(T item) {
    if (isNullT != null && isNullT(item)) {
      return false;
    }
    if (contains(item)) {
      return false;
    }
    items.push(item);
    return true;
  };

  super.swap = new T(T item) {
    if (isNullT != null && isNullT(item)) {
      return nullT;
    }
    for (int i = 0; i < items.length; ++i) {
      if (equiv(items[i], item)) {
        T result = items[i];
        items[i] = item;
        return result;
      }
    }
    items.push(item);
    assert(isNullT != null, 'item not found');
    return nullT;
  };

  super.delete = new T(T item) {
    for (int i = 0; i < items.length; ++i) {
      if (equiv(items[i], item)) {
        T result = items[i];
        items.delete(i);
        return result;
      }
    }
    assert(isNullT != null, 'item not found');
    return nullT;
  };

  super.get_ith = new T(int i) {
    return items[i];
  };

  autounravel Iterable_T operator cast(NaiveRepSet_T set) {
    return Iterable_T(set.operator iter);
  }

  autounravel RepSet_T operator cast(NaiveRepSet_T set) {
    return set.super;
  }

  super.newEmpty = new RepSet_T() {
    return NaiveRepSet_T(nullT, equiv, isNullT);
  };

  autounravel T[] operator ecast(NaiveRepSet_T set) {
    T[] result;
    for (T item : set.items) {
      result.push(item);
    }
    return result;
  }
}
