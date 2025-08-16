typedef import(T);

from collections.btreegeneral(T=T) access BTreeSet_T, Set_T, SortedSet_T;

private bool lt(T a, T b) = operator <;

BTreeSet_T BTreeSet_T() {
  return BTreeSet_T(lt);
}

BTreeSet_T BTreeSet_T(T nullT, bool isNullT(T) = null) {
  if (isNullT == null) {
    return BTreeSet_T(lt, nullT);
  }
  return BTreeSet_T(lt, nullT, isNullT);
}

BTreeSet_T BTreeSet_T(T nullT, bool isNullT(T) = null, int keyword maxPivots) {
  if (isNullT == null) {
    return BTreeSet_T(lt, nullT, maxPivots=maxPivots);
  }
  return BTreeSet_T(lt, nullT, isNullT, maxPivots=maxPivots);
}