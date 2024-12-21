typedef import(T);

struct Iter_T {
  // Returns the current item. Error if the iterator is not valid.
  T get();
  // Advances the iterator to the next item. Error if the iterator is not valid.
  void advance();
  // Returns true if the iterator is valid. If the iterator is used without
  // modifying the datastructure, it will be valid as long as there is a next
  // item.
  //
  // QUESTION: Do we want best-effort fail-fast iterators that set valid to false
  // if the datastructure is modified, or do we want to leave it the behavior
  // undefined in this case?
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
  Iter_T iter();
  void operator init(Iter_T iter()) {
    this.iter = iter;
  }
  autounravel T[] operator ecast(Iterable_T iterable) {
    T[] result;
    for (var iter = iterable.iter(); iter.valid(); iter.advance()) {
      result.push(iter.get());
    }
    return result;
  }
}
