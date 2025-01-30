typedef import(K, V);

from collections.map(K=K, V=V) access Map_K_V, Iter_K, Iter_K_V, Iterable_K;
from genericpair(K=K, V=V) access Pair_K_V, makePair;
from collections.hashrepset(T=Pair_K_V) access
    HashRepSet_T as HashRepSet_K_V;

private Pair_K_V operator tuple(K k, V v) {
  Pair_K_V pair = makePair(k, v); 
  pair.hash = k.hash;
  return pair;
}

struct HashMap_K_V {
  struct _ { autounravel restricted Map_K_V map; }

  private HashRepSet_K_V pairs = HashRepSet_K_V(
    emptyresponse=null,
    equiv = new bool(Pair_K_V a, Pair_K_V b) {
      // NOTE: This should never be called on a null pair.
      return a.k == b.k;
    },
    isEmpty = new bool(Pair_K_V kv) { return alias(kv, null); }
  );

  void operator init() {
    using F = void();
    ((F)map.operator init)();
  }

  void operator init(V emptyresponse, bool isEmpty(V) = null) {
    using F = void(V, bool isEmpty(V)=null);  // The default value here is ignored.
    if (isEmpty == null) {
      ((F)map.operator init)(emptyresponse);  // Let operator init supply its own default.
    } else {
      ((F)map.operator init)(emptyresponse, isEmpty);
    }
  }

  map.size = pairs.size;

  map.contains = new bool(K key) {
    return pairs.contains((key, map.emptyresponse));
  };

  map.operator[] = new V(K key) {
    Pair_K_V pair = pairs.get((key, map.emptyresponse));
    if (!alias(pair, null)) {
      return pair.v;
    }
    assert(map.isEmpty != null, 'Unable to report missing key');
    return map.emptyresponse;
  };

  map.operator []= = new V(K key, V value) {
    if (map.isEmpty != null && map.isEmpty(value)) {
      pairs.delete((key, value));
    } else {
      pairs.update((key, value));
    }
    return value;
  };

  map.delete = new void(K key) {
    Pair_K_V removed = pairs.delete((key, map.emptyresponse));
    assert(!alias(removed, null), 'Nonexistent key cannot be deleted');
  };

  map.iter = new Iter_K() {
    Iter_K_V it = pairs.iter();
    Iter_K result;
    result.valid = it.valid;
    result.advance = it.advance;
    result.get = new K() { return it.get().k; };
    return result;
  };

  autounravel Iterable_K operator cast(HashMap_K_V map) {
    return Iterable_K(map.map.iter);
  }
  autounravel K[] operator ecast(HashMap_K_V map) {
    return (K[])(Iterable_K)map;
  }
  autounravel Map_K_V operator cast(HashMap_K_V map) {
    return map.map;
  }

  unravel map;
}