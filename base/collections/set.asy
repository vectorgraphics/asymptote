typedef import(T);
from collections.iter(T=T) access Iter_T, Iterable_T;

// Set: set of representatives of equivalence classes. Contains at most one
// element from each equivalence class.


struct Set_T {
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

  // Creates a new, empty Set with the same implemention, nullT,
  // isNullT, and equiv as this one.
  Set_T newEmpty();

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
  // Returns a random, uniformly distributed element. The default
  // implementation is O(n) in the number of elements. Intended primarily for
  // testing purposes.
  T getRandom() {
    int size = this.size();
    static int seed = 3567654160488757718;
    int index = (++seed).hash() % size;
    for (T item : this) {
      if (index == 0) {
        return item;
      }
      --index;
    }
    assert(isNullT != null, 'Cannot get a random item from an empty set');
    return nullT;
  }

  autounravel Iterable_T operator cast(Set_T set) {
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

  autounravel bool operator <=(Set_T a, Set_T b) {
    for (var item : a) {
      if (!b.contains(item)) {
        return false;
      }
    }
    return true;
  }

  autounravel bool operator >=(Set_T a, Set_T b) {
    return b <= a;
  }

  autounravel bool operator ==(Set_T a, Set_T b) {
    return a <= b && a >= b;
  } 

  autounravel bool operator !=(Set_T a, Set_T b) {
    return !(a == b);
  }

  autounravel bool sameElementsInOrder(Set_T a, Set_T b) {
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

  autounravel Set_T operator +(Set_T a, Iterable_T b) {
    Set_T result = a.newEmpty();
    for (T item : a) {
      result.add(item);
    }
    for (T item : b) {
      result.add(item);
    }
    return result;
  }

  autounravel Set_T operator -(Set_T a, Set_T b) {
    Set_T result = a.newEmpty();
    for (T item : a) {
      if (!b.contains(item)) {
        result.add(item);
      }
    }
    return result;
  }

}


// A reference implementation, inefficient but suitable for testing.
struct NaiveSet_T {
  Set_T super;
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

  // This implementation is O(1).
  super.getRandom = new T() {
    if (items.length == 0) {
      assert(isNullT != null, 'Cannot get a random item from an empty set');
      return nullT;
    }
    static int seed = 3567654160488757718;
    int index = (++seed).hash() % items.length;
    return items[index];
  };

  autounravel Iterable_T operator cast(NaiveSet_T set) {
    return Iterable_T(set.operator iter);
  }

  autounravel Set_T operator cast(NaiveSet_T set) {
    return set.super;
  }

  super.newEmpty = new Set_T() {
    return NaiveSet_T(nullT, equiv, isNullT);
  };

  autounravel T[] operator ecast(NaiveSet_T set) {
    T[] result;
    for (T item : set.items) {
      result.push(item);
    }
    return result;
  }
}
