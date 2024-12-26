typedef import(T);
from 'collections/iter'(T=T) access Iter_T, Iterable_T;

// RepSet: set of representatives of equivalence classes. Contains at most one
// element from each equivalence class.


struct RepSet_T {
  restricted T emptyresponse;
  restricted bool equiv(T, T) = operator ==;
  restricted bool isEmpty(T) = null;
  restricted void operator init() {}
  restricted void operator init(T emptyresponse,
      bool equiv(T a, T b) = operator ==,
      bool isEmpty(T) = new bool(T t) { return equiv(t, emptyresponse); }) {
    this.emptyresponse = emptyresponse;
    this.equiv = equiv;
    this.isEmpty = isEmpty;
  }

  // Creates a new, empty RepSet with the same implemention, emptyresponse,
  // isEmpty, and equiv as this one.
  RepSet_T newEmpty();

  int size();
  bool empty() {
    return size() == 0;
  }
  bool contains(T item);
  // Returns the equivalent item in the set, or emptyresponse if the set
  // contains no equivalent item. Throws error if emptyresponse was never set.
  T get(T item);
  // We can make this an operator later, once we have made `for (T item : set)`
  // syntactic sugar for
  // `for (var iter = set.iter(); iter.valid(); iter.advance()) { T item = iter.get(); ... }`
  Iter_T iter();
  // If an equivalent item was already present, returns false. Otherwise, adds
  // the item and returns true. Noop if isEmpty is defined and item is empty.
  bool add(T item);  
  // Inserts item, and returns the item that was replaced, or emptyresponse if
  // no item was replaced. Throws error if emptyresponse was never set.
  // Noop if isEmpty is defined and item is empty.
  // QUESTION: Should we throw an error even if emptyresponse was not needed,
  // i.e., if there was already an equivalent item in the collection?
  T update(T item);
  // Removes the equivalent item from the set, and returns it. Returns
  // emptyresponse if there is no equivalent item. Throws error if
  // there is not equivalent item and emptyresponse was never set.
  T delete(T item);

  autounravel Iterable_T operator cast(RepSet_T set) {
    return Iterable_T(set.iter);
  }

  void addAll(Iterable_T other) {
    for (var iter = other.iter(); iter.valid(); iter.advance()) {
      add(iter.get());
    }
  }
  void removeAll(Iterable_T other) {
    for (var iter = other.iter(); iter.valid(); iter.advance()) {
      delete(iter.get());
    }
  }


  // Makes the notation `for (T item : (T[])set)` work for now, albeit inefficiently.
  autounravel T[] operator ecast(RepSet_T set) {
    return (T[])(Iterable_T)set;
  }

  autounravel bool operator <=(RepSet_T a, RepSet_T b) {
    for (var iter = a.iter(); iter.valid(); iter.advance()) {
      if (!b.contains(iter.get())) {
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
    var iterA = a.iter();
    var iterB = b.iter();
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
    for (var iter = a.iter(); iter.valid(); iter.advance()) {
      result.add(iter.get());
    }
    for (var iter = b.iter(); iter.valid(); iter.advance()) {
      result.add(iter.get());
    }
    return result;
  }

  autounravel RepSet_T operator -(RepSet_T a, RepSet_T b) {
    RepSet_T result = a.newEmpty();
    for (var iter = a.iter(); iter.valid(); iter.advance()) {
      if (!b.contains(iter.get())) {
        result.add(iter.get());
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
  restricted void operator init(T emptyresponse,
      bool equiv(T a, T b) = operator ==,
      bool isEmpty(T) = new bool(T t) { return equiv(t, emptyresponse); }) {
    typedef void F(T, bool equiv(T, T), bool isEmpty(T));
    ((F)super.operator init)(emptyresponse, equiv, isEmpty);
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
    return emptyresponse;
  };

  super.iter = new Iter_T() {
    return Iter_T(items);
  };

  super.add = new bool(T item) {
    if (isEmpty != null && isEmpty(item)) {
      return false;
    }
    if (contains(item)) {
      return false;
    }
    items.push(item);
    return true;
  };

  super.update = new T(T item) {
    if (isEmpty != null && isEmpty(item)) {
      return emptyresponse;
    }
    for (int i = 0; i < items.length; ++i) {
      if (equiv(items[i], item)) {
        T result = items[i];
        items[i] = item;
        return result;
      }
    }
    items.push(item);
    assert(isEmpty != null, 'No way to signal emptyresponse.');
    return emptyresponse;
  };

  super.delete = new T(T item) {
    for (int i = 0; i < items.length; ++i) {
      if (equiv(items[i], item)) {
        T result = items[i];
        items.delete(i);
        return result;
      }
    }
    assert(isEmpty != null, 'No way to signal emptyresponse.');
    return emptyresponse;
  };

  autounravel Iterable_T operator cast(NaiveRepSet_T set) {
    return Iterable_T(set.iter);
  }

  autounravel RepSet_T operator cast(NaiveRepSet_T set) {
    return set.super;
  }

  super.newEmpty = new RepSet_T() {
    return NaiveRepSet_T(emptyresponse, equiv, isEmpty);
  };

  autounravel T[] operator ecast(NaiveRepSet_T set) {
    T[] result;
    for (T item : set.items) {
      result.push(item);
    }
    return result;
  }
}
