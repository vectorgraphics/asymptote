// Asy-side wrapper for the compound C++ port of the hash-set core.
//
// The compute-heavy parts (the bucket array, the doubly-linked
// oldest/newest list, find/changeCapacity/makeZombie, and the mutation
// counter) live in the sibling C++ plugin `collections.hashset_core`.
// This wrapper composes the user-visible HashSet_T on top of that core,
// re-using base/collections/set.asy for the autounraveled operators
// (==, !=, +, -, &, ^, <=, >=) inherited via `unravel super`.

typedef import(T);

from collections.set(T=T)           access Iter_T, Iterable_T, Set_T;
from collections.hashset_core(T=T)  access HashSetCore_T, Cursor_T;

struct HashSet_T {
  restricted Set_T super;
  from super unravel nullT, equiv, isNullT;

  // The C++ core. Held by reference (asy reference semantics).
  private HashSetCore_T core = HashSetCore_T();

  void operator init() {
    using F = void();
    ((F)super.operator init)();
    // Defaults already installed by the struct-level core.reset below.
  }

  void operator init(T nullT,
      bool equiv(T a, T b) = operator ==,
      bool isNullT(T) = new bool(T t) { return equiv(t, nullT); }) {
    using F = void(T, bool equiv(T, T), bool isNullT(T));
    ((F)super.operator init)(nullT, equiv, isNullT);
    // Override the struct-level defaults with the user-supplied
    // equiv / isNullT.
    core.reset(
      new int(T item) { return item.hash(); },
      equiv,
      isNullT,
      16
    );
  }

  // Install sensible defaults at struct level so that `HashSet_T s;`
  // (which skips all operator init bodies) yields a working empty
  // set rather than crashing on the first method call.
  core.reset(
    new int(T item) { return item.hash(); },
    new bool(T a, T b) { return a == b; },
    null,
    16
  );

  // ----------------------------------------------------------------
  // Per-instance closure installation. Runs once per HashSet_T
  // instance, after operator init.
  // ----------------------------------------------------------------

  super.newEmpty = new Set_T() {
    return HashSet_T(nullT, equiv, isNullT).super;
  };

  super.size     = core.size;
  super.contains = core.contains;

  super.get = new T(T item) {
    var result = core.lookup(item);
    if (result.found) return (T)result.value;
    assert(isNullT != null, 'Item is not present.');
    return super.nullT;
  };

  super.operator iter = new Iter_T() {
    Cursor_T cur = core.beginCursor();
    Iter_T result = new Iter_T;
    result.valid = cur.valid;
    result.get = cur.get;
    result.advance = cur.advance;
    return result;
  };

  super.add = core.add;

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

  super.delete = core.deleteItem;

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
