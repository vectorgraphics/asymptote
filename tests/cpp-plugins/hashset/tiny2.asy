from hashset_core(T=int) access HashSetCore_T;
HashSetCore_T c = HashSetCore_T();
c.reset(
  new int(int x) { return x.hash(); },
  new bool(int a, int b) { return a == b; },
  new bool(int x) { return false; },     // isNullTFn that always returns false
  16);

assert(c.capacity() == 16);
assert(!c.contains(3));
assert(c.add(3));
assert(c.contains(3));
assert(c.size() == 1);
