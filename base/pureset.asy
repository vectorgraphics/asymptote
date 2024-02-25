typedef import(T);

struct Set_T {
  int size();
  bool empty() {
    return size() == 0;
  }
  bool contains(T item);
  bool insert(T item);
  T replace(T item);     // Inserts item, and returns the item that was
                         // replaced, or emptyresponse if no item was replaced.
  T get(T item);
  bool delete(T item);
  // Calls process on each item in the collection until process returns false.
  void forEach(bool process(T item));
}

struct NaiveSet_T {
  private T[] buffer = new T[0];
  private T emptyresponse;
  private bool equiv(T a, T b);

  void operator init(bool equiv(T a, T b), T emptyresponse) {
    this.equiv = equiv;
    this.emptyresponse = emptyresponse;
  }

  int size() {
    return buffer.length;
  }

  bool contains(T item) {
    for (T a : buffer) {
      if (equiv(a, item)) {
        return true;
      }
    }
    return false;
  }

  bool insert(T item) {
    if (contains(item)) {
      return false;
    }
    buffer.push(item);
    return true;
  }

  T replace(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) {
        T old = buffer[i];
        buffer[i] = item;
        return old;
      }
    }
    buffer.push(item);
    return emptyresponse;
  }

  T get(T item) {
    for (T a : buffer) {
      if (equiv(a, item)) {
        return a;
      }
    }
    return emptyresponse;
  }

  bool delete(T item) {
    for (int i = 0; i < buffer.length; ++i) {
      if (equiv(buffer[i], item)) {
        buffer[i] = buffer[buffer.length - 1];
        buffer.pop();
        return true;
      }
    }
    return false;
  }

  void forEach(bool process(T item)) {
    for (T a : buffer) {
      if (!process(a)) {
        return;
      }
    }
  }

}

Set_T operator cast(NaiveSet_T naiveSet) {
  Set_T set = new Set_T;
  set.size = naiveSet.size;
  set.contains = naiveSet.contains;
  set.insert = naiveSet.insert;
  set.replace = naiveSet.replace;
  set.get = naiveSet.get;
  set.delete = naiveSet.delete;
  set.forEach = naiveSet.forEach;
  return set;
}

T[] operator cast(Set_T set) {
  T[] buffer = new T[set.size()];
  int i = 0;
  set.forEach(new bool(T item) {
    buffer[i] = item;
    ++i;
    return true;
  });
  return buffer;
}

Set_T makeNaiveSet(bool equiv(T, T), T emptyresponse) {
  return NaiveSet_T(equiv, emptyresponse);
}