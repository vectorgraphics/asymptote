typedef import(K, V);

from collections.genericpair(K=K, V=V) access Pair_K_V;
from collections.iter(T=K) access Iter_T as Iter_K, Iterable_T as Iterable_K;
from collections.iter(T=Pair_K_V) access
    Iter_T as Iter_K_V,
    Iterable_T as Iterable_K_V;

struct Map_K_V {
  restricted V nullValue;
  restricted bool isNullValue(V) = null;
  void operator init() {}
  void operator init(V nullValue,
    bool isNullValue(V) = new bool(V v) { return v == nullValue; }
  ) {
    this.nullValue = nullValue;
    this.isNullValue = isNullValue;
    assert(isNullValue(nullValue), 'nullValue must satisfy isNullValue');
  }
  // Remaining methods are not implemented here.
  int size();
  bool empty() { return size() == 0; }
  bool contains(K key);
  // If the key was not present already, returns nullValue, or throws error
  // if nullValue was never set.
  V operator [] (K key);
  // Adds the key-value pair, replacing both the key and value if the key was
  // already present.
  void operator [=] (K key, V value);
  // Removes the entry with the given key, if it exists.
  // QUESTION: Should we throw an error if the key was not present? (Current
  // implementation: yes, unless there is a nullValue to return.)
  void delete(K key);

  Iter_K operator iter();

  autounravel Iterable_K operator cast(Map_K_V map) {
    return Iterable_K(map.operator iter);
  }

  autounravel K[] operator ecast(Map_K_V map) {
    return (K[])(Iterable_K)map;
  }

  void addAll(Iterable_K_V other) {
    for (Pair_K_V kv : other) {
      this[kv.k] = kv.v;
    }
  }
  void removeAll(Iterable_K other) {
    for (K key : other) {
      delete(key);
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
  void operator init(V nullValue, bool isNullValue(V) = null) {
    keys = new K[0];
    values = new V[0];
    size = 0;
    if (isNullValue == null) {
      map.operator init(nullValue);  // Let operator init supply its own default.
    } else {
      map.operator init(nullValue, isNullValue);
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
    assert(map.isNullValue != null, 'Key not found in map');
    return map.nullValue;
  };
  map.operator[=] = new void(K key, V value) {
    bool delete = false;
    if (map.isNullValue != null && map.isNullValue(value)) {
      delete = true;
    }
    for (int i = 0; i < size; ++i) {
      if (keys[i] == key) {
        if (delete) {
          keys.delete(i);
          values.delete(i);
          ++numChanges;
          --size;
        } else {
          keys[i] = key;
          values[i] = value;
        }
        return;
      }
    }
    if (!delete) {
      keys.push(key);
      values.push(value);
      ++numChanges;
      ++size;
    }
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
  map.operator iter = new Iter_K() {
    int numChangesAtStart = numChanges;
    int i = 0;
    Iter_K result;
    result.valid = new bool() {
      assert(numChanges == numChangesAtStart,
             'Map keys changed during iteration');
      return i < size;
    };
    result.advance = new void() { 
      assert(numChanges == numChangesAtStart,
             'Map keys changed during iteration');
      ++i;
    };
    result.get = new K() {
      assert(numChanges == numChangesAtStart,
             'Map keys changed during iteration');
      return keys[i];
    };
    return result;
  };
  autounravel Iterable_K operator cast(NaiveMap_K_V map) {
    return Iterable_K(map.map.operator iter);
  }
  autounravel K[] operator ecast(NaiveMap_K_V map) {
    return copy(map.keys);
  }
  autounravel Map_K_V operator cast(NaiveMap_K_V map) {
    return map.map;
  }
  from map unravel *;
}