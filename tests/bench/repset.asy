from collections.repset(T=int) access RepSet_T as RepSet_int;
from collections.hashrepset(T=int) access HashRepSet_T as HashRepSet_int;
from collections.btree(T=int) access BTreeRepSet_T as BTreeRepSet_int;
from collections.zip2(K=RepSet_int, V=string) access zip, operator cast;

// Run a sequence of operations on a Map_K_V and measure elapsed time.
cputime benchmark(RepSet_int set, int operations, int size=operations#2) {
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
RepSet_int[] sets;
sets.push(HashRepSet_int(nullT=-1));
names.push('HashRepSet');
sets.push(BTreeRepSet_int(operator <, nullT=-1));
names.push('BTreeRepSet');
sets.push(BTreeRepSet_int(operator <, nullT=-1, maxPivots=8));
names.push('BTreeRepSet (8 pivots max)');

// Run a benchmark
int operations = 5000000;
for (var labeledMap : zip(sets, names)) {
  cputime time = benchmark(labeledMap.k, operations);
  write(labeledMap.v + ': user time for ' + (string)operations + ' operations: ', suffix=flush);
  write(format('%#.2fs', time.change.user));
}