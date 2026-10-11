import TestLib;

StartTest('Writing collections');

from collections.hashset(T=int) access HashSet_T as HashSet_int,
                                       Set_T as Set_int;
from collections.set(T=string) access NaiveSet_T as NaiveSet_string;
from collections.btree(T=int) access BTreeSet_T as BTreeSet_int,
                                     SortedSet_T as SortedSet_int;
from collections.sortedset(T=int) access Naive_T as NaiveSorted_int;
from collections.splaytree(T=int) access SplayTree_T as SplayTree_int;
from collections.hashmap(K=string, V=int) access
    HashMap_K_V as HashMap_string_int,
    Map_K_V as Map_string_int;
from collections.btreemap(K=int, V=string) access
    BTreeMap_K_V as BTreeMap_int_string;
from collections.map(K=int, V=int) access NaiveMap_K_V as NaiveMap_int_int;
from collections.queue(T=real) access Queue_T as Queue_real,
                                      makeNaiveQueue,
                                      ArrayQueue_T as ArrayQueue_real,
                                      LinkedQueue_T as LinkedQueue_real;
from collections.wraparray(T=int) access Array_T as Array_int, wrap;
from collections.wrapper(T=pair) access Wrapped_T as Wrapped_pair, wrap;
from collections.genericpair(K=int, V=string) access
    Pair_K_V as Pair_int_string,
    makePair;

string tmpfile = 'collections_write_test_tmp.txt';
file out;

// Reads back the lines written to out since beginWrite().
string[] written;
void beginWrite() { out = output(tmpfile); }
void endWrite() {
  close(out);
  file in = input(tmpfile);
  written = in.line();
  close(in);
  delete(tmpfile);
}

// Sets.
{
  HashSet_int hashSet;
  hashSet.add(5);
  NaiveSet_string naiveSet;
  naiveSet.add('a');
  naiveSet.add('b');
  BTreeSet_int btree = BTreeSet_int();
  btree.add(3); btree.add(1); btree.add(2);
  NaiveSorted_int naiveSorted = NaiveSorted_int(operator <, 0);
  naiveSorted.add(2); naiveSorted.add(1);
  SplayTree_int splay = SplayTree_int(operator <, 0);
  splay.add(2); splay.add(1);
  HashSet_int empty;
  Set_int unimplemented;

  beginWrite();
  write(out, hashSet, endl);
  write(out, (Set_int) hashSet, endl);
  write(out, naiveSet, endl);
  write(out, btree, endl);
  write(out, (SortedSet_int) btree, endl);
  write(out, (Set_int) btree, endl);
  write(out, naiveSorted, endl);
  write(out, splay, endl);
  write(out, empty, endl);
  write(out, unimplemented, endl);
  write(out, 's=', btree, 7, endl);
  endWrite();
  assert(all(written == new string[] {
    '{5}', '{5}', '{"a", "b"}', '{1, 2, 3}', '{1, 2, 3}', '{1, 2, 3}', '{1, 2}',
    '{1, 2}', '{}', '<no iterator>', 's={1, 2, 3}' + '\t' + '7'}));
}

// Maps.
{
  HashMap_string_int hashMap;
  hashMap['x'] = 1;
  BTreeMap_int_string btreeMap;
  btreeMap[2] = 'two';
  btreeMap[1] = 'one';
  NaiveMap_int_int naiveMap;
  naiveMap[1] = 10;
  naiveMap[2] = 20;

  beginWrite();
  write(out, hashMap, endl);
  write(out, (Map_string_int) hashMap, endl);
  write(out, btreeMap, endl);
  write(out, naiveMap, endl);
  endWrite();
  assert(all(written == new string[] {
    '{("x", 1)}', '{("x", 1)}', '{(1, "one"), (2, "two")}',
    '{(1, 10), (2, 20)}'}));
}

// Queues, wrappers and pairs.
{
  Queue_real queue = makeNaiveQueue(new real[] {1.5, 2.5});
  ArrayQueue_real arrayQueue = ArrayQueue_real(new real[] {1, 2});
  LinkedQueue_real linkedQueue;
  linkedQueue.push(7);

  beginWrite();
  write(out, queue, endl);
  write(out, arrayQueue, endl);
  write(out, linkedQueue, endl);
  write(out, wrap(new int[] {1, 2, 3}), endl);
  write(out, wrap((1, 2)), endl);
  write(out, makePair(1, 'one'), endl);
  endWrite();
  assert(all(written == new string[] {
    '{1.5, 2.5}', '{1, 2}', '{7}', '{1, 2, 3}', '(1,2)', '(1, "one")'}));
}

// Collections that are fields of another struct.
{
  struct Holder {
    HashSet_int set;
    HashMap_string_int map;
    Array_int array = wrap(new int[] {4, 5});
  }
  Holder holder;
  holder.set.add(9);
  holder.map['k'] = 7;

  beginWrite();
  write(out, holder, endl);
  endWrite();
  assert(all(written == new string[] {'(set={9}, map={("k", 7)}, array={4, 5})'}));
}

EndTest();

StartTest('Writing collections of arrays and structs');
{
  from collections.hashset(T=Array_int) access HashSet_T as HashSet_Array_int;
  struct Point { int x; int y; }
  from collections.queue(T=Point) access makeNaiveQueue;
  from collections.queue(T=int[]) access makeNaiveQueue;

  HashSet_Array_int arrays;
  arrays.add(wrap(new int[] {1, 2}, hashElement=new int(int x) { return x; }));
  Point p;
  p.x = 1; p.y = 2;

  beginWrite();
  write(out, arrays, endl);
  write(out, makeNaiveQueue(new Point[] {p, null}), endl);
  write(out, makeNaiveQueue(new int[][] {{1, 2}, {3}}), endl);
  endWrite();
  assert(all(written == new string[] {
    '{{1, 2}}', '{(x=1, y=2), null}', '{{1, 2}, {3}}'}));
}
EndTest();
