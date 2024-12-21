typedef import(K, V);

from genericpair(K=K, V=V) access Pair_K_V;

struct Map_K_V {
  int size();
  bool empty() {
    return size() == 0;
  }
  bool contains(K key);
  V get(K key);

  void forEach(bool process(K key, V value));
  // Returns the previous value associated with key, or emptyresponse if there
  // was no mapping for key.
  V put(Pair_K_V kv);
  V put(K key, V value) {
    return put(key >> value);
  }
  // Returns the previous value associated with key, or emptyresponse if there
  // was no mapping for key.
  V pop(K key);  

  autounravel Pair_K_V[] operator cast(Map_K_V map) {
    Pair_K_V[] result;
    map.forEach(new bool(K key, V value) {
      result.push(key >> value);
      return true;
    });
    return result;
  }

}

from pureset(T=Pair_K_V) access
    Set_T as Set_Pair_K_V,
    makeNaiveSet;

// Assumes that elements of `pairs` are considered equivalent if their keys
// are equivalent, and that pairs.emptyresponse is null.
Map_K_V makeMapHelper(Set_Pair_K_V pairs, V emptyresponse) {
  Map_K_V result = new Map_K_V;
  result.size = pairs.size;
  result.contains = new bool(K key) {
    return pairs.contains(key >> emptyresponse);
  };
  result.empty = new bool() {
    return pairs.empty();
  };
  result.get = new V(K key) {
    Pair_K_V kvpair = pairs.get(key >> emptyresponse);
    return alias(kvpair, null) ? emptyresponse : kvpair.v;
  };
  result.forEach = new void(bool process(K key, V value)) {
    pairs.forEach(new bool(Pair_K_V kvpair) {
      return process(kvpair.k, kvpair.v);
    });
  };
  result.put = new V(Pair_K_V kv) {
    Pair_K_V previous = pairs.update(kv);
    return alias(previous, null) ? emptyresponse : previous.v;
  };
  result.pop = new V(K key) {
    Pair_K_V previous = pairs.delete(key >> emptyresponse);
    return alias(previous, null) ? emptyresponse : previous.v;
  };
  return result;
}

Map_K_V makeNaiveMap(bool equiv(K a, K b), V emptyresponse) {
  bool keysEquiv(Pair_K_V a, Pair_K_V b) {
    return equiv(a.k, b.k);
  }
  Set_Pair_K_V pairs = makeNaiveSet(keysEquiv, null);
  return makeMapHelper(pairs, emptyresponse);
}