from hashset_core(T=int) access HashSetCore_T;

HashSetCore_T c = HashSetCore_T();
c.reset(
  new int(int x) { return x.hash(); },
  new bool(int a, int b) { return a == b; },
  null,
  16
);

write('cap=', c.capacity());

bool r1 = c.add_item(3);
write('r1=', r1);
