typedef import(T);

struct Iter_T {
  // Returns the current item. Error if the iterator is not valid.
  T get();
  // Advances the iterator to the next item. Error if the iterator is not valid.
  void advance();
  // Returns true if the iterator is valid. If the iterator is used without
  // modifying the datastructure, it will be valid as long as there is a next
  // item.
  bool valid();
}

Iter_T Iter_T(T[] items) {
  int index = 0;
  Iter_T retv;
  unravel retv;
  advance = new void() { ++index; };
  get = new T() { return items[index]; };
  valid = new bool() { return index < items.length; };
  return retv;
}

struct Iterable_T {
  // Returns an iterator over the collection.
  Iter_T operator iter();
  void operator init(Iter_T iter()) {
    this.operator iter = iter;
  }
  void operator init(T[] items) {
    this.operator iter = new Iter_T() {
      return Iter_T(items);
    };
  }
  autounravel T[] operator ecast(Iterable_T iterable) {
    T[] result;
    for (T item : iterable) {
      result.push(item);
    }
    return result;
  }
}

Iterable_T Iterable(Iter_T iter()) = Iterable_T;
Iterable_T Iterable(T[] items) = Iterable_T;
