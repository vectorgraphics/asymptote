typedef import(K, V);

from genericpair(K=K, V=V) access Pair_K_V, operator >>, alias;
from hashset(T=Pair_K_V) access
    Set_T as Set_Pair_K_V,
    operator cast,
    makeHashSet;
from puremap(K=K, V=V) access Map_K_V, makeMapHelper;

Map_K_V makeHashMap(int hash(K, int bits), bool equiv(K, K), V emptyresponse) {
  Set_Pair_K_V hashSet = makeHashSet(
    new int(Pair_K_V kv, int bits) {
      return hash(kv.k, bits);
    },
    new bool(Pair_K_V a, Pair_K_V b) {
      return equiv(a.k, b.k);
    },
    emptyresponse=null
  );
  return makeMapHelper(hashSet, emptyresponse);
}