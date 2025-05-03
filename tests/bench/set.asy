from collections.set(T=int) access Set_T as Set_int;
from collections.hashset(T=int) access HashSet_T as HashSet_int;
from collections.btree(T=int) access BTreeSet_T as BTreeSet_int;
from collections.zip2(K=Set_int, V=string) access zip, operator cast;

// Run a sequence of operations on a Map_K_V and measure elapsed time.
cputime benchmark(Set_int set, int operations, int size=operations#2) {
  assert(size % 43 != 0, 'size must not be divisible by 43');
  assert(size % 31667 != 0, 'size must not be divisible by 31667');
  cputime();

  // Insert operations
  int i = 0;
  do {
    i = (i + 43) % size;
    set.add(i);
  } while(i != 0);

  // Retrieve operations
  int t = 0;
  int ceiling = size + size;
  for(int i=0; i<operations; ++i) {
    t = (t + 31667) % ceiling;
    set.get(t);
  }

  // Iterate over set
  for (int i : set) {
    i = i;
  }

  // Delete half the keys
  for(int i=0; i<size; i += 2) {
    set.delete(i);
  }

  return cputime();
}

string[] names;
Set_int[] sets;
sets.push(HashSet_int(nullT=-1));
names.push('HashSet');
sets.push(BTreeSet_int(nullT=-1));
names.push('BTreeSet');
sets.push(BTreeSet_int(nullT=-1, maxPivots=8));
names.push('BTreeSet (8 pivots max)');

// Run a benchmark
int operations = 5000000;
for (var labeledMap : zip(sets, names)) {
  cputime time = benchmark(labeledMap.k, operations);
  write(labeledMap.v + ': user time for ' + (string)operations + ' operations: ', suffix=flush);
  write(format('%#.2fs', time.change.user));
}