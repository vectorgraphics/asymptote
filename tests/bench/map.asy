//from collections.repset(T=int) access RepSet_T as RepSet_int;
//from collections.hashrepset(T=int) access HashRepSet_T as HashRepSet_int;
//from collections.btree(T=int) access BTreeRepSet_T as BTreeRepSet_int;
from collections.map(K=int, V=int) access Map_K_V as Map_int_int;
from collections.hashmap(K=int, V=int) access HashMap_K_V as HashMap_int_int;
from collections.btreemap(K=int, V=int) access BTreeMap_K_V as BTreeMap_int_int;

from collections.zip2(K=Map_int_int, V=string) access zip, operator cast;


// Run a sequence of operations on a Map_K_V and measure elapsed time.
cputime benchmark(Map_int_int map, int operations, int size=operations#2) {
  assert(size % 43 != 0, 'size must not be divisible by 43');
  assert(size % 31667 != 0, 'size must not be divisible by 31667');
  cputime();

  // Insert operations
  int i = 0;
  do {
    i = (i + 43) % size;
    map[i] = i;
  } while(i != 0);

  // Retrieve operations
  int t = 0;
  int ceiling = size + size;
  for(int i=0; i<operations; ++i) {
    t = (t + 31667) % ceiling;
    map[t];
  }

  // Iterate over map
  for (int i : map) {
    i = i;
  }

  // Delete half the keys
  for(int i=0; i<size; i += 2) {
    map.delete(i);
  }

  return cputime();
}

string[] names;
Map_int_int[] sets;
sets.push(HashMap_int_int(nullValue=-1));
names.push('HashMap');
sets.push(BTreeMap_int_int(nullValue=-1));
names.push('BTreeMap');
//sets.push(BTreeMap_int_int(operator <, nullValue=-1, maxPivots=8));
//names.push('BTreeMap (8 pivots max)');

// Run a benchmark
for (int operations = 1000; operations <= 1e6; operations *= 10) {
  for (var labeledMap : zip(sets, names)) {
    cputime time = benchmark(labeledMap.k, operations);
    write(labeledMap.v + ': user time for ' + (string)operations + ' operations: ', suffix=flush);
    write(format('%#.2fs', time.change.user));
  }
}
