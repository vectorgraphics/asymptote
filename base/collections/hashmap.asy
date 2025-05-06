typedef import(K, V);

from collections.map(K=K, V=V) access Map_K_V, Iter_K, Iter_K_V, Iterable_K,
                                      Iterable_K_V;
from collections.genericpair(K=K, V=V) access Pair_K_V, makePair;
from collections.hashset(T=Pair_K_V) access
    HashSet_T as HashSet_K_V;

private Pair_K_V operator tuple(K k, V v) {
  Pair_K_V pair = makePair(k, v); 
  pair.hash = k.hash;
  return pair;
}

struct HashMap_K_V {
  struct _ { autounravel restricted Map_K_V map; }

  private HashSet_K_V pairs = HashSet_K_V(
    nullT=null,
    equiv = new bool(Pair_K_V a, Pair_K_V b) {
      // NOTE: This should never be called on a null pair.
      return a.k == b.k;
    },
    isNullT = new bool(Pair_K_V kv) { return alias(kv, null); }
  );

  void operator init() {
    using F = void();
    ((F)map.operator init)();
  }

  void operator init(V nullValue, bool isNullValue(V) = null) {
    using F = void(V, bool isNullValue(V)=null);  // The default value here is ignored.
    if (isNullValue == null) {
      ((F)map.operator init)(nullValue);  // Let operator init supply its own default.
    } else {
      ((F)map.operator init)(nullValue, isNullValue);
    }
  }

  map.size = pairs.size;

  map.contains = new bool(K key) {
    return pairs.contains((key, map.nullValue));
  };

  map.operator[] = new V(K key) {
    Pair_K_V pair = pairs.get((key, map.nullValue));
    if (!alias(pair, null)) {
      return pair.v;
    }
    assert(map.isNullValue != null, 'Key not found in map');
    return map.nullValue;
  };

  map.operator [=] = new void(K key, V value) {
    if (map.isNullValue != null && map.isNullValue(value)) {
      pairs.delete((key, value));
    } else {
      pairs.push((key, value));
    }
  };

  map.delete = new void(K key) {
    assert(pairs.delete((key, map.nullValue)),
           'Nonexistent key cannot be deleted');
  };

  map.operator iter = new Iter_K() {
    Iter_K_V it = pairs.operator iter();
    Iter_K result;
    result.valid = it.valid;
    result.advance = it.advance;
    result.get = new K() { return it.get().k; };
    return result;
  };

  map.pairs = new Iterable_K_V() { return pairs; };

  autounravel Iterable_K operator cast(HashMap_K_V map) {
    return Iterable_K(map.map.operator iter);
  }
  autounravel K[] operator ecast(HashMap_K_V map) {
    return (K[])(Iterable_K)map;
  }
  autounravel Map_K_V operator cast(HashMap_K_V map) {
    return map.map;
  }

  unravel map;
}