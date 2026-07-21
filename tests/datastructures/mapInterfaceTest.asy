import TestLib;

srand(2948571036482957103);

StartTest('Map: pairs() iteration');
{
  from collections.map(K=string, V=int) access
      Map_K_V as Map_string_int,
      NaiveMap_K_V as NaiveMap_string_int;
  from collections.genericpair(K=string, V=int) access
      Pair_K_V as Pair_string_int;

  Map_string_int map = NaiveMap_string_int(nullValue=0);
  map['a'] = 1;
  map['b'] = 2;
  map['c'] = 3;

  int sum = 0;
  int count = 0;
  for (var kv : map.pairs()) {
    assert(map[kv.k] == kv.v);
    sum += kv.v;
    ++count;
  }
  assert(count == 3);
  assert(sum == 6);
}
EndTest();

StartTest('Map: keys() method');
{
  from collections.map(K=int, V=string) access
      Map_K_V as Map_int_string,
      NaiveMap_K_V as NaiveMap_int_string;

  Map_int_string map = NaiveMap_int_string(nullValue='');
  map[10] = 'ten';
  map[20] = 'twenty';
  map[30] = 'thirty';

  int[] keys = map.keys();
  assert(keys.length == 3);
  // All keys should be in the result.
  bool found10 = false, found20 = false, found30 = false;
  for (int k : keys) {
    if (k == 10) found10 = true;
    if (k == 20) found20 = true;
    if (k == 30) found30 = true;
  }
  assert(found10 && found20 && found30);
}
EndTest();

StartTest('Map: empty() and size()');
{
  from collections.map(K=string, V=int) access
      Map_K_V as Map_string_int,
      NaiveMap_K_V as NaiveMap_string_int;

  Map_string_int map = NaiveMap_string_int(nullValue=0);
  assert(map.empty());
  assert(map.size() == 0);

  map['x'] = 1;
  assert(!map.empty());
  assert(map.size() == 1);

  map['x'] = 0;  // soft delete via nullValue
  assert(map.empty());
  assert(map.size() == 0);
}
EndTest();

StartTest('Map: nullValue soft delete');
{
  from collections.hashmap(K=string, V=int) access
      HashMap_K_V as HashMap_string_int;

  HashMap_string_int map = HashMap_string_int(nullValue=0);

  map['a'] = 10;
  map['b'] = 20;
  assert(map.size() == 2);
  assert(map.contains('a'));

  // Setting to nullValue should delete the entry.
  map['a'] = 0;
  assert(!map.contains('a'));
  assert(map.size() == 1);

  // Reading a non-existent key returns nullValue.
  assert(map['nonexistent'] == 0);
}
EndTest();

StartTest('Map: explicit delete');
{
  from collections.hashmap(K=string, V=int) access
      HashMap_K_V as HashMap_string_int;

  HashMap_string_int map = HashMap_string_int(nullValue=0);
  map['a'] = 1;
  map['b'] = 2;
  map['c'] = 3;

  map.delete('b');
  assert(!map.contains('b'));
  assert(map.size() == 2);
  assert(map.contains('a'));
  assert(map.contains('c'));
}
EndTest();

StartTest('Map: add from pairs');
{
  from collections.map(K=string, V=int) access
      Map_K_V as Map_string_int,
      NaiveMap_K_V as NaiveMap_string_int;

  Map_string_int src = NaiveMap_string_int(nullValue=0);
  src['a'] = 1;
  src['b'] = 2;

  Map_string_int dst = NaiveMap_string_int(nullValue=0);
  dst['c'] = 3;

  dst.add(src.pairs());

  assert(dst.size() == 3);
  assert(dst['a'] == 1);
  assert(dst['b'] == 2);
  assert(dst['c'] == 3);
}
EndTest();