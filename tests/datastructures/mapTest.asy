import TestLib;

StartTest("PureMapTest");

from puremap(K=string, V=int) access
   Pair_K_V as Pair_string_int,
   operator >>,
   Map_K_V as Map_string_int,
   operator cast,
   makeNaiveMap;

bool stringsEqual(string, string) = operator ==;

Map_string_int testMap = makeNaiveMap(stringsEqual, intMin);

assert(testMap.empty());

assert(!testMap.contains('first'));
testMap.put('first' >> 1);
assert(testMap.size() == 1);
assert(testMap.contains('first'));
testMap.put('second' >> -7);
testMap.put('third' >> 34);
assert(testMap.size() == 3);
assert(testMap.get('undefined') == intMin);
assert(testMap.get('second') == -7);
assert(testMap.contains('first'));
assert(testMap.pop('first') == 1);
assert(!testMap.contains('first'));
assert(testMap.size() == 2);
assert(testMap.put('second' >> -8) == -7);
assert(testMap.size() == 2);
assert(testMap.get('second') == -8);

assert(testMap.size() == 2);
for (Pair_string_int kv : testMap) {
  testMap.pop(kv.k);
}
assert(testMap.size() == 0);

EndTest();