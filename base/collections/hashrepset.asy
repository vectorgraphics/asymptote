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
  from super unravel emptyresponse, equiv, isEmpty;

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
        return super.emptyresponse;
      }
      if (entry.hash == bucket && equiv(entry.item, item)) {
        return entry.item;
      }
    }
    return super.emptyresponse;
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
    if (isEmpty != null && isEmpty(item)) {
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
    if (isEmpty != null && isEmpty(item)) {
      return emptyresponse;
    }
    if (2 * (size + zombies) >= buckets.length) {
      changeCapacity();
    }
    int bucket = item.hash();
    for (int i = 0; i < buckets.length; ++i) {
      HashEntry entry = buckets[bucket + i];
      if (entry == null) {
        entry = buckets[bucket + i] = new HashEntry;
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
      if (entry.hash == bucket && equiv(entry.item, item)) {
        T result = entry.item;
        entry.item = item;
        return result;
      }
    }
    assert(false, 'No space in hash table');
    return emptyresponse;
  };

  super.delete = new T(T item) {
    ++numChanges;
    int bucket = item.hash();
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
          changeCapacity();
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
    