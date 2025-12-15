import TestLib;
StartTest('wraparray');

from collections.wraparray(T=int) access Array_T as Array_int, wrap;

// Test initialization and basic operations
int[] data = {1, 2, 3, 4, 5};
Array_int array = wrap(data);

assert(array.length() == 5, 'Length test failed');
assert(array[0] == 1, 'Indexing test failed');
assert(array[4] == 5, 'Indexing test failed');

array[2] = 10;
assert(array[2] == 10, 'Assignment test failed');
// {1, 2, 10, 4, 5}

// Test append and push
array.push(6);
assert(array.length() == 6, 'Push test failed');
assert(array[5] == 6, 'Push test failed');
// {1, 2, 10, 4, 5, 6}

int[] moreData = {7, 8, 9};
array.append(moreData);
assert(array.length() == 9, 'Append test failed');
assert(array[8] == 9, 'Append test failed');
// {1, 2, 10, 4, 5, 6, 7, 8, 9}

// Test pop
int popped = array.pop();
assert(popped == 9, 'Pop test failed');
assert(array.length() == 8, 'Pop test failed');
// {1, 2, 10, 4, 5, 6, 7, 8}

// Test insert and delete
array.insert(2, 20, 30);
assert(array.length() == 10, 'Insert test failed');
assert(array[2] == 20, 'Insert test failed');
assert(array[3] == 30, 'Insert test failed');
// {1, 2, 20, 30, 10, 4, 5, 6, 7, 8}

array.delete(2, 3);
assert(array.length() == 8, 'Delete test failed');
assert(array[2] == 10, 'Delete test failed');
// {1, 2, 10, 4, 5, 6, 7, 8}

// Test equality and inequality
assert(array == array, 'Equality test failed');
assert(array != wrap(new int[]), 'Different lengths inequality test failed'); 
int[] data2 = {1, 2, 10, 4, 5, 6, 7, 8};
Array_int array2 = wrap(data2);
assert(array == array2, 'Equality test failed');
array2[2] = 11;
assert(array != array2, 'Inequality test failed');


// Test equality and inequality with null arrays
Array_int nullArray = null;
assert(nullArray == null, 'Null equality test failed');
assert(null == nullArray, 'Null equality test failed');
assert(nullArray != array, 'Null inequality test failed');
assert(array != nullArray, 'Null inequality test failed');

Array_int wrappedNullArray = wrap(null);
assert(wrappedNullArray != nullArray, 'Wrapped null inequality test failed');
assert(nullArray != wrappedNullArray, 'Wrapped null inequality test failed');
assert(wrappedNullArray != array, 'Wrapped null inequality test failed');
assert(array != wrappedNullArray, 'Wrapped null inequality test failed');
assert(wrappedNullArray == wrap(null), 'Wrapped null equality test failed');

// Test cyclic property
array.cyclic(true);
assert(array.cyclic() == true, 'Cyclic test failed');
assert(array[-1] == array[array.length() - 1], 'Cyclic test failed');

array.cyclic(false);
assert(array.cyclic() == false, 'Cyclic test failed');

// Test keys
int[] keys = array.keys();
assert(keys.length == array.length(), 'Keys test failed');
for (int i = 0; i < keys.length; ++i) {
  assert(keys[i] == i, 'Keys test failed');
}

// Test initialized (and also casting to and from Array_T)
assert(array.initialized(0), 'Initialized test failed');
assert(!array.initialized(100), 'Initialized test failed');
array2 = copy(array);
assert(array2.length() == array.length(), 'Cast test failed');
assert(array2 == array, 'Cast test failed');
assert(!alias(array2.data, array.data), 'Cast test failed');
array2.append(new int[100]);
assert(!array2.initialized(100), 'Initialized test failed');

// Test hash
int hashElement(int x) { return x; }
Array_int arrayWithHash = wrap(data, hashElement);
assert(arrayWithHash.hash() == hash(data), 'Hash test failed');

// Test use as hashmap key
from collections.hashmap(K=Array_int, V=int) access
    HashMap_K_V as HashMap_Array_int_int;
var map = HashMap_Array_int_int();
map[arrayWithHash] = 10191;
assert(map[arrayWithHash] == 10191, 'Hashmap test failed');
Array_int arrayWithHash2 = wrap(new int[], hashElement);
for (int i : arrayWithHash) {
  arrayWithHash2.push(i);
}
assert(map[arrayWithHash2] == 10191, 'Hashmap test failed');

EndTest();