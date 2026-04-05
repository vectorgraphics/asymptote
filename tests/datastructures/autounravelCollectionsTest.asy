import TestLib;

// Test that autounravel works correctly when accessing collection types
// through template imports. This is a central feature of the PR:
// autounravel declarations in datastructures should be properly propagated.

StartTest('autounravel: hashmap operators via import');
{
  from collections.hashmap(K=string, V=int) access
      HashMap_K_V as HashMap_string_int;

  // The autounravel cast operators should make HashMap usable as a Map
  // and as an Iterable without explicit casting.
  HashMap_string_int map = HashMap_string_int(nullValue=0);
  map['alpha'] = 1;
  map['beta'] = 2;
  map['gamma'] = 3;

  // Test that autounraveled Iterable_K cast works (for-each loop).
  int count = 0;
  for (string key : map) {
    assert(map[key] > 0);
    ++count;
  }
  assert(count == 3);

  // Test that autounraveled Map_K_V cast works.
  from collections.map(K=string, V=int) access Map_K_V as Map_string_int;
  Map_string_int asMap = map;
  assert(asMap.size() == 3);
  assert(asMap['alpha'] == 1);
}
EndTest();

StartTest('autounravel: hashset operators via import');
{
  from collections.hashset(T=string) access
    HashSet_T as HashSet_string;

  HashSet_string set = HashSet_string();
  set.add('x');
  set.add('y');
  set.add('z');
  set.add('x');  // duplicate

  assert(set.size() == 3);

  // for-each should work due to autounraveled Iterable_T cast.
  int count = 0;
  for (string s : set) {
    assert(set.contains(s));
    ++count;
  }
  assert(count == 3);
}
EndTest();

StartTest('autounravel: queue cast and iteration');
{
  from collections.queue(T=int) access
      Queue_T as Queue_int;

  // makeQueue should be available via autounravel from Queue_T.
  Queue_int q = makeQueue(new int[] {10, 20, 30});

  // Autounraveled Iterable cast should allow for-each.
  int[] items;
  for (int item : q) {
    items.push(item);
  }
  assert(items.length == 3);
  assert(items[0] == 10);
  assert(items[1] == 20);
  assert(items[2] == 30);
}
EndTest();

StartTest('autounravel: wraparray operators');
{
  from collections.wraparray(T=int) access
      Array_T as Array_int,
      wrap;

  Array_int a = wrap(new int[] {1, 2, 3});
  Array_int b = wrap(new int[] {1, 2, 3});
  Array_int c = wrap(new int[] {1, 2, 4});

  // autounraveled operator == and operator !=
  assert(a == b);
  assert(a != c);

  // autounraveled cast from T[] to Array_T
  int[] raw = {5, 6, 7};
  Array_int fromCast = raw;
  assert(fromCast.length() == 3);
  assert(fromCast[0] == 5);

  // autounraveled cast from Array_T to T[]
  int[] backToRaw = a;
  assert(all(backToRaw == new int[] {1, 2, 3}));
}
EndTest();

StartTest('autounravel: btreemap operators');
{
  from collections.btreemap(K=int, V=string) access
      BTreeMap_K_V as BTreeMap_int_string;

  BTreeMap_int_string btmap = BTreeMap_int_string(nullValue='');
  btmap[1] = 'one';
  btmap[2] = 'two';
  btmap[3] = 'three';

  // autounraveled Map cast
  from collections.map(K=int, V=string) access Map_K_V as Map_int_string;
  Map_int_string asMap = btmap;
  assert(asMap.size() == 3);
  assert(asMap[2] == 'two');

  // autounraveled Iterable_K cast for for-each
  int count = 0;
  int prev = -1;
  for (int key : btmap) {
    assert(key > prev);  // BTreeMap iterates in sorted order
    prev = key;
    ++count;
  }
  assert(count == 3);
}
EndTest();

StartTest('autounravel: nested access through typedef');
{
  // Test that autounravel works when a struct type is accessed via typedef
  // through multiple levels.
  struct Container {
    static struct Inner {
      autounravel int value = 42;
      autounravel int doubled() { return 2 * value; }
    }
  }
  using Inner = Container.Inner;
  // autounraveled fields should be accessible.
  assert(value == 42);
  assert(doubled() == 84);
  value = 10;
  assert(Inner.value == 10);
  assert(doubled() == 20);
}
EndTest();

StartTest('autounravel: from unravel propagation');
{
  struct Container {
    static struct Data {
      autounravel int value = 42;
    }
  }
  unravel Container;
  // After unravel, Data and its autounraveled members are accessible.
  assert(Data.value == 42);
  assert(value == 42);
  value = 100;
  assert(Data.value == 100);
  Data.value = 200;
  assert(value == 200);
}
EndTest();

StartTest('autounravel: from-unravel with functions');
{
  struct Outer {
    static struct MySet {
      autounravel int tally = 0;
      autounravel void increment() { ++tally; }
      autounravel void reset() { tally = 0; }
    }
  }
  from Outer unravel MySet;
  assert(tally == 0);
  increment();
  increment();
  assert(tally == 2);
  assert(MySet.tally == 2);
  reset();
  assert(tally == 0);
  assert(MySet.tally == 0);
}
EndTest();
