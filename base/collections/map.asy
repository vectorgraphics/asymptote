typedef import(K, V);

from genericpair(K=K, V=V) access Pair_K_V;
from collections.iter(T=K) access Iter_T as Iter_K, Iterable_T as Iterable_K;
from collections.iter(T=Pair_K_V) access
    Iter_T as Iter_K_V,
    Iterable_T as Iterable_K_V;

struct Map_K_V {
  restricted V emptyresponse;
  restricted bool isEmpty(V) = null;
  void operator init() {}
  void operator init(V emptyresponse,
    bool isEmpty(V) = new bool(V v) { return v == emptyresponse; }
  ) {
    this.emptyresponse = emptyresponse;
    this.isEmpty = isEmpty;
    assert(isEmpty(emptyresponse), 'Emptyresponse must be empty');
  }
  // Remaining methods are not implemented here.
  int size();
  bool empty() { return size() == 0; }
  bool contains(K key);
  // If the key was not present already, returns emptyresponse, or throws error
  // if emptyresponse was never set.
  V operator [] (K key);
  // Adds the key-value pair, replacing both the key and value if the key was
  // already present.
  V operator [=] (K key, V value);
  // Removes the entry with the given key, if it exists.
  // QUESTION: Should we throw an error if the key was not present? (Current
  // implementation: yes, unless there is an emptyresponse to return.)
  void delete(K key);

  // TODO: Replace with operator iter.
  // This will be implemented later, once we have made `for (K key : map)`
  // syntactic sugar for
  // `for (var iter = map.iter(); iter.valid(); iter.advance()) { K key = iter.get(); ... }`
  Iter_K iter();

  autounravel Iterable_K operator cast(Map_K_V map) {
    return Iterable_K(map.iter);
  }

  // Makes the notation `for (K key: (K[])map)` work for now, albeit inefficiently.
  autounravel K[] operator ecast(Map_K_V map) {
    return (K[])(Iterable_K)map;
  }

  void addAll(Iterable_K_V other) {
    for (var iter = other.iter(); iter.valid(); iter.advance()) {
      Pair_K_V kv = iter.get();
      this[kv.k] = kv.v;
    }
  }
  void removeAll(Iterable_K other) {
    for (var iter = other.iter(); iter.valid(); iter.advance()) {
      delete(iter.get());
    }
  }
}

// Reference implementation for testing purposes.
struct NaiveMap_K_V {
  private K[] keys;
  private V[] values;
  private int size;
  private int numChanges = 0;
  restricted Map_K_V map;
  void operator init() {
    keys = new K[0];
    values = new V[0];
    size = 0;
    using F = void();
    ((F)map.operator init)();
  }
  void operator init(V emptyresponse, bool isEmpty(V) = null) {
    keys = new K[0];
    values = new V[0];
    size = 0;
    if (isEmpty == null) {
      map.operator init(emptyresponse);  // Let operator init supply its own default.
    } else {
      map.operator init(emptyresponse, isEmpty);
    }
  }
  map.size = new int() { return size; };
  map.contains = new bool(K key) {
    for (int i = 0; i < size; ++i) {
      if (keys[i] == key) {
        return true;
      }
    }
    return false;
  };
  map.operator[] = new V(K key) {
    for (int i = 0; i < size; ++i) {
      if (keys[i] == key) {
        return values[i];
      }
    }
    assert(map.isEmpty != null, 'Unable to report missing key');
    return map.emptyresponse;
  };
  map.operator[=] = new V(K key, V value) {
    ++numChanges;
    bool delete = false;
    if (map.isEmpty != null && map.isEmpty(value)) {
      delete = true;
    }
    for (int i = 0; i < size; ++i) {
      if (keys[i] == key) {
        if (delete) {
          keys.delete(i);
          values.delete(i);
          --size;
          return value;
        }
        keys[i] = key;
        values[i] = value;
        return value;
      }
    }
    if (!delete) {
      keys.push(key);
      values.push(value);
      ++size;
    }
    return value;
  };
  map.delete = new void(K key) {
    ++numChanges;
    for (int i = 0; i < size; ++i) {
      if (keys[i] == key) {
        keys.delete(i);
        values.delete(i);
        --size;
        return;
      }
    }
    assert(false, 'Nonexistent key cannot be deleted');
  };
  map.iter = new Iter_K() {
    int numChangesAtStart = numChanges;
    int i = 0;
    Iter_K result;
    result.valid = new bool() {
      assert(numChanges == numChangesAtStart, 'Map changed during iteration');
      return i < size;
    };
    result.advance = new void() { 
      assert(numChanges == numChangesAtStart, 'Map changed during iteration');
      ++i;
    };
    result.get = new K() {
      assert(numChanges == numChangesAtStart, 'Map changed during iteration');
      return keys[i];
    };
    return result;
  };
  autounravel Iterable_K operator cast(NaiveMap_K_V map) {
    return Iterable_K(map.map.iter);
  }
  autounravel K[] operator ecast(NaiveMap_K_V map) {
    return copy(map.keys);
  }
  autounravel Map_K_V operator cast(NaiveMap_K_V map) {
    return map.map;
  }
  from map unravel *;
}