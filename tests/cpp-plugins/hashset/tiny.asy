from hashset_core(T=int) access HashSetCore_T;

HashSetCore_T c = HashSetCore_T();
c.reset(
  new int(int x) { return x.hash(); },
  new bool(int a, int b) { return a == b; },
  null,
  16
);

assert(c.capacity() == 16);
assert(c.add(3));
