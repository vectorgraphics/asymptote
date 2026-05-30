// Asy-side wrapper for the compound C++ port of
// base/collections/hashset.asy.
//
// The C++ core (`hashset_core`) provides HashSetCore_T and Cursor_T.
// This wrapper composes the user-visible HashSet_T on top of it,
// re-using base/collections/set.asy for the autounraveled operators
// (==, !=, +, -, &, ^, <=, >=) inherited via `unravel super`.
//
// The closure-installation block at the bottom of the struct mirrors
// the pattern used by base/collections/hashset.asy itself: struct-level
// statements run once per instance during initialization, after all
// operator init bodies.

typedef import(T);

from collections.set(T=T) access Iter_T, Iterable_T, Set_T;
from hashset_core(T=T)    access HashSetCore_T, Cursor_T;

struct HashSet_T {
  restricted Set_T super;
  from super unravel nullT, equiv, isNullT;

  // The C++ core. Held by reference (asy reference semantics).
  private HashSetCore_T core = HashSetCore_T();

  void operator init() {
    using F = void();
    ((F)super.operator init)();
    core.reset(
      new int(T item) { return item.hash(); },
      new bool(T a, T b) { return a == b; },
      null,
      16
    );
  }

  void operator init(T nullT,
      bool equiv(T a, T b) = operator ==,
      bool isNullT(T) = new bool(T t) { return equiv(t, nullT); }) {
    using F = void(T, bool equiv(T, T), bool isNullT(T));
    ((F)super.operator init)(nullT, equiv, isNullT);
    core.reset(
      new int(T item) { return item.hash(); },
      equiv,
      isNullT,
      16
    );
  }

  // ----------------------------------------------------------------
  // Per-instance closure installation. Runs once per HashSet_T
  // instance, after operator init. Each closure captures `core`,
  // `nullT`, `equiv`, `isNullT` from this instance.
  // ----------------------------------------------------------------

  super.newEmpty = new Set_T() {
    return HashSet_T(nullT, equiv, isNullT).super;
  };

  super.size     = new int() { return core.size(); };
  super.contains = new bool(T item) { return core.contains(item); };

  super.get = new T(T item) {
    var result = core.lookup(item);
    if (result.found) return (T)result.value;
    assert(isNullT != null, 'Item is not present.');
    return super.nullT;
  };

  super.operator iter = new Iter_T() {
    Cursor_T cur = core.beginCursor();
    Iter_T result = new Iter_T;
    result.valid = new bool() { return cur.valid(); };
    result.get = new T() { return (T)cur.get(); };
    result.advance = new void() { cur.advance(); };
    return result;
  };

  super.add = new bool(T item) { return core.add(item); };

  super.push = new T(T item) {
    var result = core.push(item);
    if (result.found) return (T)result.value;
    assert(isNullT != null,
           'Adding item via push() without defining nullT.');
    return nullT;
  };

  super.extract = new T(T item) {
    var result = core.extract(item);
    if (result.found) return (T)result.value;
    assert(isNullT != null, 'Item is not present.');
    return nullT;
  };

  super.delete = new bool(T item) { return core.deleteItem(item); };

  super.getRandom = new T() {
    var result = core.getRandom();
    if (result.found) return (T)result.value;
    assert(isNullT != null, 'Cannot get a random item from an empty set');
    return nullT;
  };

  autounravel Set_T operator cast(HashSet_T set) {
    return set.super;
  }

  autounravel Iterable_T operator cast(HashSet_T set) {
    return Iterable_T(set.super.operator iter);
  }

  unravel super;
}
