// Minimal smoke test of HashSetCore_T directly.
from hashset_core(T=int) access HashSetCore_T;

HashSetCore_T c = HashSetCore_T();
c.reset(
  new int(int x) { return x.hash(); },
  new bool(int a, int b) { return a == b; },
  null,
  16
);

assert(c.size() == 0);
assert(c.capacity() == 16);
assert(c.add(3));            // newly inserted
assert(c.add(1));            // newly inserted
assert(!c.add(3));           // duplicate
assert(c.size() == 2);
assert(c.contains(3));
assert(!c.contains(7));

var r = c.lookup(3);
assert(r.found);
assert((int)r.value == 3);
