// Phase 4 fixture: compound C++ HashSet port.
// Exercises HashSet_T(T=int) and HashSet_T(T=string).

from hashset(T=int)    access HashSet_T as HashSet_int;
from hashset(T=string) access HashSet_T as HashSet_string;

// --- int ---
HashSet_int s = HashSet_int();
assert(s.size() == 0);
assert(s.empty());
assert(s.add(3));
assert(s.add(1));
assert(s.add(4));
assert(!s.add(1));               // duplicate
assert(s.add(5));
assert(s.add(9));
assert(s.add(2));
assert(s.add(6));
assert(s.size() == 7);           // 7 distinct values inserted
assert(s.contains(4));
assert(!s.contains(7));

// Iteration is in insertion order.
string itered;
for (int v : s) {
  itered += ',' + (string)v;
}
assert(itered == ',3,1,4,5,9,2,6');

// Delete and verify.
assert(s.delete(4));
assert(s.size() == 6);
assert(!s.contains(4));

// Iterate after delete to confirm 4 is gone.
itered = '';
for (int v : s) {
  itered += ',' + (string)v;
}
assert(itered == ',3,1,5,9,2,6');

// --- string ---
HashSet_string ss = HashSet_string();
assert(ss.add('hello'));
assert(ss.add('world'));
assert(!ss.add('hello'));        // duplicate
assert(ss.size() == 2);
assert(ss.contains('world'));
assert(!ss.contains('foo'));

// nullT-style construction (default value sentinel).
HashSet_int sn = HashSet_int(-1);
sn.add(10);
sn.add(20);
assert(sn.get(10) == 10);
assert(sn.get(99) == -1);        // nullT
sn.delete(10);
assert(sn.get(10) == -1);        // nullT
