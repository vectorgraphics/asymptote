typedef import(T);

from 'datastructures/repset'(T=T) access Iter_T, Iterable_T, RepSet_T;

private struct HashEntry {
  T item;
  int hash = -1;
  HashEntry newer = null;
  HashEntry older = null;
}

struct HashRepSet_T {
  RepSet_T super;
  unravel super;

  // These fields are mutable.
  private HashEntry[] buckets = array(16, (HashEntry)null);
  buckets.cyclic = true;
  private int size = 0;
  private int zombies = 0;
  private int numChanges = 0;  // Detect concurrent modification.
  HashEntry newest = null;
  HashEntry oldest = null;

  void operator init() {
    typedef void F();
    ((F)super.operator init)();
  }
  void operator init(T emptyresponse,
      bool equiv(T a, T b) = operator ==,
      bool isEmpty(T) = new bool(T t) { return equiv(t, emptyresponse); }) {
    typedef void F(T, bool equiv(T, T), bool isEmpty(T));
    ((F)super.operator init)(emptyresponse, equiv, isEmpty);
  }

  RepSet_T newEmpty() {
    return HashRepSet_T(emptyresponse, equiv, isEmpty).super;
  }

  size = new int() {
    return size;
  };

  contains = new bool(T item) {
    int bucket = hash(item);
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null || entry.hash < 0) {
        return false;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        return true;
      }
    }
    return false;
  };

  get = new T(T item) {
    write('get');
    int bucket = hash(item);
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null || entry.hash < 0) {
        return super.emptyresponse;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        return entry.item;
      }
    }
    return super.emptyresponse;
  };

  iter = new Iter_T() {
    Iter_T result = new Iter_T;
    HashEntry current = oldest;
    int expectedChanges = numChanges;
    result.valid = new bool() {
      assert(numChanges == expectedChanges, 'Concurrent modification');
      return current != null;
    };
    result.get = new T() {
      assert(numChanges == expectedChanges, 'Concurrent modification');
      assert(result.valid());
      return current.item;
    };
    result.advance = new void() {
      assert(numChanges == expectedChanges, 'Concurrent modification');
      assert(result.valid());
      current = current.newer;
    };
    return result;
  };

  private void addUnsafe(T item) {
    int bucket = hash(item);
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        entry = buckets[bucket + i] = new HashEntry;
      }
      if (entry.hash < 0) {
        entry.item = item;
        entry.hash = bucket;
        entry.older = newest;
        if (newest != null) {
          newest.newer = entry;
        }
        newest = entry;
        if (oldest == null) {
          oldest = entry;
        }
        return;
      }
    }
    assert(false, 'No space in hash table');
  }

  private void changeCapacity(int newCapacity = 2 * buckets.length) {
    ++numChanges;
    zombies = 0;
    buckets = array(newCapacity, (HashEntry)null);
    buckets.cyclic = true;
    for (HashEntry current = oldest; current != null; current = current.newer) {
      int bucket = current.hash;
      for (int i = 0; i < buckets.length; ++i) {
        if (buckets[bucket + i] == null) {
          buckets[bucket + i] = current;
          break;
        }
        assert(i < buckets.length - 1, 'No space in hash table; is the linked list circular?');
      }
    }
  }

  add = new bool(T item) {
    ++numChanges;
    if (isEmpty != null && isEmpty(item)) {
      return false;
    }
    if (2 * size >= buckets.length) {
      changeCapacity();
    }
    int bucket = hash(item);
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        entry = buckets[bucket + i] = new HashEntry;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        return false;
      }
      if (entry.hash < 0) {
        entry.item = item;
        entry.hash = bucket;
        entry.older = newest;
        if (newest != null) {
          newest.newer = entry;
        }
        newest = entry;
        if (oldest == null) {
          oldest = entry;
        }
        ++size;
        return true;
      }
    }
    assert(false, 'No space in hash table');
    return false;
  };

  update = new T(T item) {
    ++numChanges;
    if (isEmpty != null && isEmpty(item)) {
      return emptyresponse;
    }
    if (2 * size >= buckets.length) {
      changeCapacity();
    }
    int bucket = hash(item);
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        entry = buckets[bucket + i] = new HashEntry;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        T result = entry.item;
        entry.item = item;
        return result;
      }
      if (entry.hash < 0) {
        assert(isEmpty != null, 'Unable to report empty update.');
        entry.item = item;
        entry.hash = bucket;
        entry.older = newest;
        if (newest != null) {
          newest.newer = entry;
        }
        newest = entry;
        if (oldest == null) {
          oldest = entry;
        }
        ++size;
        return emptyresponse;
      }
    }
    assert(false, 'No space in hash table');
    return emptyresponse;
  };

  delete = new T(T item) {
    ++numChanges;
    int bucket = hash(item);
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        assert(isEmpty != null, 'Unable to report empty deletion.');
        return emptyresponse;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        T result = entry.item;
        entry.hash = -1;
        ++zombies;
        if (entry.older != null) {
          entry.older.newer = entry.newer;
        } else {
          oldest = entry.newer;
        }
        if (entry.newer != null) {
          entry.newer.older = entry.older;
        } else {
          newest = entry.older;
        }
        --size;
        if (2 * (size + zombies) > buckets.length) {
          changeCapacity(zombies > size ? buckets.length : 2 * buckets.length);
        }
        return result;
      }
    }
    assert(false, 'Overcrowded hash table; zombies: ' + string(zombies) +
           '; size: ' + string(size) +
           '; buckets.length: ' + string(buckets.length));
    assert(isEmpty != null, 'Unable to report empty deletion.');
    return emptyresponse;
  };

  autounravel T[] operator ecast(HashRepSet_T set) {
    // Make `operator ecast` static again:
    from RepSet_T unravel operator ecast;
    return (T[])set.super;
  }

  autounravel RepSet_T operator cast(HashRepSet_T set) {
    return set.super;
  }

  autounravel Iterable_T operator cast(HashRepSet_T set) {
    return Iterable_T(set.iter);
  }
}
    