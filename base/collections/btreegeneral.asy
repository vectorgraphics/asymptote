// Asy-side wrapper for the compound C++ port of the B-tree set.
//
// The compute-heavy parts (node arena, locate / forceAdd / delete with
// split + rotate + merge, in-order cursor with concurrent-modification
// detection) live in the sibling C++ plugin `collections.btreegeneral_core`.
// This wrapper composes the user-visible BTreeSet_T on top of that core,
// re-using base/collections/sortedset.asy for the autounraveled operators
// (==, !=, +, -, &, ^, <=, >=) inherited via SortedSet_T's `from set
// unravel *`.

typedef import(T);

from collections.iter(T=T)               access Iter_T, Iterable_T, Iterable;
from collections.sortedset(T=T)          access Set_T, SortedSet_T;
from collections.btreegeneral_core(T=T)  access BTreeSetCore_T, Cursor_T;

struct BTreeSet_T {
  restricted SortedSet_T super;
  from super unravel nullT, isNullT;

  private bool lt(T, T) = null;
  private bool equiv(T a, T b) { return !(lt(a, b) || lt(b, a)); };

  // The C++ core. Held by reference (asy reference semantics).
  private BTreeSetCore_T core = BTreeSetCore_T();
  private int maxPivots = 128;

  // --- constructors -----------------------------------------------------

  void operator init(bool lessThan(T, T), T nullT,
                     bool isNullT(T) = new bool(T t) { return t == nullT; }) {
    this.lt = lessThan;
    super.operator init(nullT, equiv, isNullT);
    core.reset(lessThan, isNullT, maxPivots);
  }

  // Allows for adjusting the maximum number of pivots in a node.
  // Intended primarily for testing and benchmarking.
  void operator init(bool lessThan(T, T), T nullT,
                     bool isNullT(T) = new bool(T t) { return t == nullT; },
                     int keyword maxPivots) {
    this.maxPivots = maxPivots;
    this.operator init(lessThan, nullT, isNullT);
  }

  void operator init(bool lessThan(T, T)) {
    this.lt = lessThan;
    using Initializer = void();
    ((Initializer)super.operator init)();
    core.reset(lessThan, null, maxPivots);
  }

  // ----------------------------------------------------------------
  // Per-instance closure installation. Runs once per BTreeSet_T
  // instance, after operator init.
  // ----------------------------------------------------------------

  super.newEmpty = new Set_T() {
    if (isNullT == null) {
      return BTreeSet_T(lt).super;
    }
    return BTreeSet_T(lt, nullT, isNullT, maxPivots=maxPivots).super;
  };

  super.size      = core.size;
  super.contains  = core.contains;

  super.get = new T(T x) {
    var r = core.lookup(x);
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'Item is not present.');
    return nullT;
  };

  super.after = new T(T x) {
    var r = core.after(x);
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No element after item to return');
    return nullT;
  };

  super.before = new T(T x) {
    var r = core.before(x);
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No element before item to return');
    return nullT;
  };

  super.atOrAfter = new T(T x) {
    var r = core.atOrAfter(x);
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No element after item to return');
    return nullT;
  };

  super.atOrBefore = new T(T x) {
    var r = core.atOrBefore(x);
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No element before item to return');
    return nullT;
  };

  super.min = new T() {
    var r = core.minOpt();
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No minimum element to return');
    return nullT;
  };

  super.max = new T() {
    var r = core.maxOpt();
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No maximum element to return');
    return nullT;
  };

  super.popMin = new T() {
    var r = core.popMin();
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No minimum element to pop');
    return nullT;
  };

  super.popMax = new T() {
    var r = core.popMax();
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'No maximum element to pop');
    return nullT;
  };

  super.operator iter = new Iter_T() {
    Cursor_T cur = core.beginCursor();
    Iter_T result = new Iter_T;
    result.valid   = cur.valid;
    result.get     = new T() { return (T)cur.get(); };
    result.advance = cur.advance;
    return result;
  };

  super.add = core.add;

  super.push = new T(T item) {
    var r = core.push(item);
    if (r.found) return (T)r.value;
    assert(isNullT != null,
           'Adding item via push() without defining nullT.');
    return nullT;
  };

  super.extract = new T(T item) {
    var r = core.extract(item);
    if (r.found) return (T)r.value;
    assert(isNullT != null, 'Item not found');
    return nullT;
  };

  super.delete = core.deleteItem;

  // ----------------------------------------------------------------
  // Cast operators.
  // ----------------------------------------------------------------

  autounravel SortedSet_T operator cast(BTreeSet_T set) {
    return set.super;
  }
  autounravel Set_T operator cast(BTreeSet_T set) {
    return (SortedSet_T)set;  // Compose with the above cast.
  }
  autounravel Iterable_T operator cast(BTreeSet_T set) {
    return Iterable_T(set.super.operator iter);
  }

  from super unravel *;
}
