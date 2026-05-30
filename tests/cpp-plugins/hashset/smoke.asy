// Minimal smoke test of HashSetCore_T directly.
from hashset_core(T=int) access HashSetCore_T;

HashSetCore_T c = HashSetCore_T();
c.reset(
  new int(int x) { return x.hash(); },
  new bool(int a, int b) { return a == b; },
  null,
  16
);

write('size=', c.size());                   // 0
write('cap=',  c.capacity());                // 16
write('add 3:',  c.add_item(3));                  // true
write('add 1:',  c.add_item(1));                  // true
write('add 3:',  c.add_item(3));                  // false
write('size=', c.size());                    // 2
write('cont 3:', c.contains(3));             // true
write('cont 7:', c.contains(7));             // false

var r = c.lookup(3);
write('lookup 3 found:', r.found, 'value:', (int)r.value);  // true 3

write('done');
