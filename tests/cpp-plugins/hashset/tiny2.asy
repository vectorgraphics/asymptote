from hashset_core(T=int) access HashSetCore_T;
HashSetCore_T c = HashSetCore_T();
c.reset(
  new int(int x) { return x.hash(); },
  new bool(int a, int b) { return a == b; },
  new bool(int x) { return false; },     // isNullTFn that always returns false
  16);

write('cap=', c.capacity());
write('cont=', c.contains(3));
write('add=', c.add_item(3));
write('cont=', c.contains(3));
write('size=', c.size());
