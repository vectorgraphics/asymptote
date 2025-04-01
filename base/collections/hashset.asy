typedef import(T);

from collections.set(T=T) access Iter_T, Iterable_T, Set_T;

private struct HashEntry {
  T item;
  int hash = -1;
  HashEntry newer = null;
  HashEntry older = null;
}

struct HashSet_T {
  struct _ { autounravel restricted Set_T super; }
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
    using F = void();
    ((F)super.operator init)();
  }
  void operator init(T nullT,
      bool equiv(T a, T b) = operator ==,
      bool isNullT(T) = new bool(T t) { return equiv(t, nullT); }) {
    typedef void F(T, bool equiv(T, T), bool isNullT(T));
    ((F)super.operator init)(nullT, equiv, isNullT);
  }

  Set_T newEmpty() {
    return HashSet_T(nullT, equiv, isNullT).super;
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
    int end = bucket;
    int start = end - buckets.length;
    for (int i = start; i < end; ++i) {
      HashEntry entry = buckets[i];
      if (entry == null) {
        return super.nullT;
      }
      if (entry.hash == bucket) {
        var entryItem = entry.item;
        if (equiv(entryItem, item)) {
          return entryItem;
        }
      }
    }
    assert(isNullT != null, 'Item is not present.');
    return super.nullT;
  };

  super.operator iter = new Iter_T() {
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
        assert(i < buckets.length - 1, 'No space in hash table; '
                                       'is the linked list circular?');
      }
    }
  }

  // Returns an int as follows (note: "index" is modulo buckets.length):
  //   * If an equivalent item is in the set, returns its index.
  //   * Otherwise, if least one bucket is empty, returns the index of the empty
  //     bucket in which the item should be placed if added.
  //   * Otherwise, returns -1.
  private int find(T item, int hash) {
    for (int index = hash - buckets.length; index < hash; ++index) {
      HashEntry entry = buckets[index];
      if (entry == null) {
        return index;
      }
      if (entry.hash == hash && equiv(entry.item, item)) {
        return index;
      }
    }
    return -1;
  }

  super.add = new bool(T item) {
    if (isNullT != null && isNullT(item)) {
      return false;
    }
    int capacity = buckets.length;
    if (2 * (size + zombies) >= capacity) {
      changeCapacity();
      capacity = buckets.length;
    }
    int bucket = item.hash();
    int index = find(item, bucket);
    if (index == -1) {
      ++numChanges;
      changeCapacity();
      capacity = buckets.length;
      index = find(item, bucket);
      assert(index != -1, 'No space in hash table');
    }
    HashEntry entry = buckets[index];
    if (entry != null) {
      return false;
    }

    ++numChanges;
    if (2 * (size + zombies) >= capacity) {
      changeCapacity();
      capacity = buckets.length;
      index = find(item, bucket);
      assert(index != -1);
      assert(buckets[index] == null);
    }
    entry = buckets[index] = new HashEntry;
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
  };

  super.swap = new T(T item) {
    if (isNullT != null && isNullT(item)) {
      return nullT;
    }
    int bucket = item.hash();
    int index = find(item, bucket);
    if (index == -1) {
      changeCapacity();
      index = find(item, bucket);
      assert(index != -1, 'No space in hash table');
    }
    HashEntry entry = buckets[index];
    if (entry != null) {
      T result = entry.item;
      entry.item = item;
      return result;
    }
    assert(isNullT != null,
           'Adding item via swap() without defining nullT.');
    ++numChanges;
    if (2 * (size + zombies) >= buckets.length) {
      changeCapacity();
      index = find(item, bucket);
      assert(index != -1);
      assert(buckets[index] == null);
    }
    entry = buckets[index] = new HashEntry;
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
  };

  super.delete = new T(T item) {
    int bucket = item.hash();
    int index = find(item, bucket);
    HashEntry entry = buckets[index];
    if (index == -1) {
      assert(false, 'Overcrowded hash table; zombies: ' + string(zombies) +
             '; size: ' + string(size) +
             '; buckets.length: ' + string(buckets.length));
      return nullT;
    }
    if (entry == null) {
      assert(isNullT != null, 'Item is not present.');
      return nullT;
    }
    ++numChanges;
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
  };

  super.getRandom = new T() {
    if (size == 0) {
      assert(isNullT != null, 'Cannot get a random item from an empty set');
      return nullT;
    }
    static int seed = 3567654160488757718;
    if (size # 2 > buckets.length # size) {
      // Most buckets are empty, so it's faster to iterate over the linked list
      // of full buckets.
      int index = (++seed).hash() % size;
      for (T item : super) {
        if (index == 0) {
          return item;
        }
        --index;
      }
      assert(false, 'Unreachable code');
    }
    HashEntry entry = null;
    do {
      int index = (++seed).hash() % buckets.length;
      entry = buckets[index];
    } while (entry == null || entry.hash == -1);
    return entry.item;
  };


  autounravel Set_T operator cast(HashSet_T set) {
    return set.super;
  }

  autounravel Iterable_T operator cast(HashSet_T set) {
    return Iterable_T(set.super.operator iter);
  }
  unravel super;
}
    