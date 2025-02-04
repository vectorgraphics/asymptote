typedef import(T);

from collections.repset(T=T) access Iter_T, Iterable_T, RepSet_T;

private struct HashEntry {
  T item;
  int hash = -1;
  HashEntry newer = null;
  HashEntry older = null;
}

struct HashRepSet_T {
  struct _ { autounravel restricted RepSet_T super; }
  from super unravel nullT, equiv, isNullT;

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
  void operator init(T nullT,
      bool equiv(T a, T b) = operator ==,
      bool isNullT(T) = new bool(T t) { return equiv(t, nullT); }) {
    typedef void F(T, bool equiv(T, T), bool isNullT(T));
    ((F)super.operator init)(nullT, equiv, isNullT);
  }

  RepSet_T newEmpty() {
    return HashRepSet_T(nullT, equiv, isNullT).super;
  }

  super.size = new int() {
    return size;
  };

  super.contains = new bool(T item) {
    int bucket = item.hash();
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        return false;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        return true;
      }
    }
    return false;
  };

  super.get = new T(T item) {
    int bucket = item.hash();
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        return super.nullT;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        return entry.item;
      }
    }
    return super.nullT;
  };

  super.iter = new Iter_T() {
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

  private void changeCapacity() {
    ++numChanges;
    int newCapacity = (zombies > size ? buckets.length : 2 * buckets.length);
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

  super.add = new bool(T item) {
    ++numChanges;
    if (isNullT != null && isNullT(item)) {
      return false;
    }
    if (2 * (size + zombies) >= buckets.length) {
      changeCapacity();
    }
    int bucket = item.hash();
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        entry = buckets[bucket + i] = new HashEntry;
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
      } else if (entry.hash == bucket && equiv(entry.item, item)) {
        return false;
      }
    }
    assert(false, 'No space in hash table');
    return false;
  };

  super.update = new T(T item) {
    ++numChanges;
    if (isNullT != null && isNullT(item)) {
      return nullT;
    }
    if (2 * (size + zombies) >= buckets.length) {
      changeCapacity();
    }
    int bucket = item.hash();
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        entry = buckets[bucket + i] = new HashEntry;
        assert(isNullT != null,
               'Unable to report item addition of new item via update().');
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
        return nullT;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        T result = entry.item;
        entry.item = item;
        return result;
      }
    }
    assert(false, 'No space in hash table');
    return nullT;
  };

  super.delete = new T(T item) {
    ++numChanges;
    int bucket = item.hash();
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        assert(isNullT != null, 'Item is not present.');
        return nullT;
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
          changeCapacity();
        }
        return result;
      }
    }
    assert(false, 'Overcrowded hash table; zombies: ' + string(zombies) +
           '; size: ' + string(size) +
           '; buckets.length: ' + string(buckets.length));
    assert(isNullT != null, 'Item is not present.');
    return nullT;
  };

  autounravel T[] operator ecast(HashRepSet_T set) {
    return (T[])set.super;
  }

  autounravel RepSet_T operator cast(HashRepSet_T set) {
    return set.super;
  }

  autounravel Iterable_T operator cast(HashRepSet_T set) {
    return Iterable_T(set.super.iter);
  }
  unravel super;
}
    