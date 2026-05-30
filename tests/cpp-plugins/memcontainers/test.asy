// Demonstration of ay::mem:: GC-friendly STL containers in a C++ plugin.
from memcontainers access
  VecBag, ListBag, DequeBag, StringMap, IntMap, reverse_via_mem_string;

// --- ay::mem::vector ---------------------------------------------------
VecBag v = VecBag();
for (int i = 1; i <= 1000; ++i) v.push(i);
assert(v.size() == 1000, 'VecBag size after 1000 pushes');
assert(v.sum() == 500500, 'VecBag sum 1..1000');
assert(v.pop() == 1000, 'VecBag pop returns last');
assert(v.size() == 999, 'VecBag size after pop');

// --- ay::mem::list -----------------------------------------------------
ListBag l = ListBag();
l.push_back(10);
l.push_back(20);
l.push_front(5);
assert(l.size() == 3, 'ListBag size');
assert(l.front() == 5, 'ListBag front');
assert(l.back() == 20, 'ListBag back');

// --- ay::mem::deque ----------------------------------------------------
DequeBag d = DequeBag();
for (int i = 0; i < 100; ++i) d.push_back(i);
int s = 0;
while (d.size() > 0) s += d.pop_front();
assert(s == 4950, 'DequeBag drain sum');

// --- ay::mem::map<ay::mem::string, long long> --------------------------
StringMap sm = StringMap();
sm.put('alpha', 1);
sm.put('beta', 2);
sm.put('gamma', 3);
sm.put('beta', 22);  // overwrite
assert(sm.size() == 3, 'StringMap size after overwrite');
assert(sm.get('alpha') == 1);
assert(sm.get('beta') == 22);
assert(sm.contains('gamma'));
assert(!sm.contains('delta'));

// --- ay::mem::unordered_map<long long, long long> ----------------------
IntMap im = IntMap();
for (int i = 0; i < 5000; ++i) im.put(i, i * i);
assert(im.size() == 5000, 'IntMap size');
assert(im.get(123) == 123 * 123, 'IntMap lookup');

// --- ay::mem::string ---------------------------------------------------
assert(reverse_via_mem_string('asymptote') == 'etotpmysa',
       'reverse_via_mem_string');

// Stress: force several GC cycles by allocating many transient
// containers, then verify the contents above are still intact.
for (int rep = 0; rep < 20; ++rep) {
  VecBag tmp = VecBag();
  for (int i = 0; i < 5000; ++i) tmp.push(i);
}
assert(v.size() == 999, 'VecBag survived GC pressure');
assert(im.get(4242) == 4242 * 4242, 'IntMap survived GC pressure');
assert(sm.get('beta') == 22, 'StringMap survived GC pressure');

write('memcontainers: PASSED');
