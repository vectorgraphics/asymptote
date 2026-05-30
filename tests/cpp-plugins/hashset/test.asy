// Phase 4 fixture: compound C++ HashSet port.
// Exercises HashSet_T(T=int) and HashSet_T(T=string).

from hashset(T=int)    access HashSet_T as HashSet_int;
from hashset(T=string) access HashSet_T as HashSet_string;

write('--- int ---');
HashSet_int s = HashSet_int();
write(s.size());                 // 0
write(s.empty());                // true
write(s.add(3));                 // true
write(s.add(1));                 // true
write(s.add(4));                 // true
write(s.add(1));                 // false (duplicate)
write(s.add(5));                 // true
write(s.add(9));                 // true
write(s.add(2));                 // true
write(s.add(6));                 // true
write(s.size());                 // 8
write(s.contains(4));            // true
write(s.contains(7));            // false

// Iteration is in insertion order.
string itered;
for (int v : s) {
  itered += ',' + (string)v;
}
write(itered);                   // ,3,1,4,5,9,2,6

// Delete and verify.
write(s.delete(4));              // true
write(s.size());                 // 7
write(s.contains(4));            // false

// Iterate after delete to confirm 4 is gone.
itered = '';
for (int v : s) {
  itered += ',' + (string)v;
}
write(itered);                   // ,3,1,5,9,2,6

write('--- string ---');
HashSet_string ss = HashSet_string();
write(ss.add('hello'));          // true
write(ss.add('world'));          // true
write(ss.add('hello'));          // false
write(ss.size());                // 2
write(ss.contains('world'));     // true
write(ss.contains('foo'));       // false

// nullT-style construction (default value sentinel).
HashSet_int sn = HashSet_int(-1);
sn.add(10);
sn.add(20);
write(sn.get(10));               // 10
write(sn.get(99));               // -1   (nullT)
sn.delete(10);
write(sn.get(10));               // -1   (nullT)

write('done');
