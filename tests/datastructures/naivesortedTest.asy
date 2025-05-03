// Check that NaiveSortedSet behaves identically to HashSet except for
// iteration order. This should increase confidence when using NaiveSortedSet
// to test other implementations of SortedSet_T.
import TestLib;

srand(4282308941601638229);

StartTest("NaiveSortedSet");

// from wrapper(T=int) access
//     Wrapper_T as wrapped_int,
//     wrap;
struct wrapped_int {
  restricted int t;
  void operator init(int t) {
    this.t = t;
  }
  autounravel bool operator ==(wrapped_int a, wrapped_int b) {
    if (alias(a, null)) return alias(b, null);
    if (alias(b, null)) return false;
    return a.t == b.t;
  }
  autounravel bool operator !=(wrapped_int a, wrapped_int b) {
    return !(a == b);
  }
  autounravel bool operator <(wrapped_int a, wrapped_int b) {
    return a.t < b.t;
  }
  int hash() { return t.hash(); }
}

wrapped_int wrap(int t) = wrapped_int;  // `wrap` is alias for constructor

from collections.set(T=wrapped_int) access
    Set_T as Set_wrapped_int;

from collections.hashset(T=wrapped_int) access
    HashSet_T as HashSet_wrapped_int;

// TODO: Change to sortedset
from collections.sortedset(T=wrapped_int) access
    Naive_T as NaiveSortedSet_wrapped_int;

from collections.btreegeneral(T=wrapped_int) access
    BTreeSet_T as BTreeSet_wrapped_int;

struct ActionEnum {
  static restricted int num = 0;
  static private int make() {
    return ++num - 1;
  }
  autounravel restricted int CONTAINS = make();
  autounravel restricted int GET = make();
  autounravel restricted int ADD = make();
  autounravel restricted int SWAP = make();
  autounravel restricted int DELETE = make();
  autounravel restricted int DELETE_CONTAINS = make();
}

from collections.zip(T=int) access zip;
from mapArray(Src=wrapped_int, Dst=int) access map;
int get(wrapped_int a) {
  return a.t;
}

int[] operator cast(wrapped_int[] a) {
  for (wrapped_int x : a) {
    assert(!alias(x, null), 'Null element in array');
  }
  return map(get, a);
}

string differences(wrapped_int[] aArray, wrapped_int[] bArray) {
  if (aArray.length != bArray.length) {
    return 'Different sizes: ' + string(aArray.length) + ' vs ' +
           string(bArray.length);
  }
  int[] aIntArray = sort(map(get, aArray));
  int[] bIntArray = sort(map(get, bArray));
  string arrayValues = '[\n';
  bool different = false;
  for (int i = 0; i < aIntArray.length; ++i) {
    arrayValues += '  [' + format('%5d', aIntArray[i]) + ',' 
                   + format('%5d', bIntArray[i]) + ']';
    if (!alias(aArray[i], bArray[i])) {
      arrayValues += '  <---';
      different = true;
    }
    arrayValues += '\n';
  }
  arrayValues += ']';
  // write(arrayValues + '\n');
  if (different) {
    return arrayValues;
  }
  return '';
}

string differences(Set_wrapped_int a, Set_wrapped_int b) {
  if (a.size() != b.size()) {
    return 'Different sizes: ' + string(a.size()) + ' vs ' + string(b.size());
  }
  wrapped_int[] aArray, bArray;
  for (wrapped_int x : a) {
    aArray.push(x);
  }
  for (wrapped_int x : b) {
    bArray.push(x);
  }
  aArray = sort(aArray, operator<);
  bArray = sort(bArray, operator<);
  return differences(aArray, bArray);
}

string string(int[] a) {
  string result = '[';
  for (int i = 0; i < a.length; ++i) {
    if (i > 0) {
      result += ', ';
    }
    result += string(a[i]);
  }
  result += ']';
  return result;
}

string string(wrapped_int[] a) {
  string result = '[';
  for (int i = 0; i < a.length; ++i) {
    if (i > 0) {
      result += ', ';
    }
    if (alias(a[i], null)) {
      result += 'null';
    } else {
      result += string(a[i].t);
    }
  }
  result += ']';
  return result;
}

string string(bool[] a) {
  string result = '[';
  for (int i = 0; i < a.length; ++i) {
    if (i > 0) {
      result += ', ';
    }
    result += a[i] ? 'true' : 'false';
  }
  result += ']';
  return result;
}

typedef void Action(int ...Set_wrapped_int[]);

Action[] actions = new Action[ActionEnum.num];
actions[ADD] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toInsert = wrap(rand() % maxItem);
  // write('Inserting ' + string(toInsert.t) + '\n');
  bool[] results = new bool[];
  for (Set_wrapped_int s : sets) {
    results.push(s.add(toInsert));
  }
  if (results.length > 0) {
    bool expected = results[0];
    for (bool r : results) {
      assert(r == expected, 'Different results: ' + string(results));
    }
  }
};
actions[SWAP] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toReplace = wrap(rand() % maxItem);
  // write('Replacing ' + string(toReplace.t) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (Set_wrapped_int s : sets) {
    results.push(s.swap(toReplace));
  }
  if (results.length > 0) {
    wrapped_int expected = results[0];
    for (wrapped_int r : results) {
      if (!alias(r, expected)) {
        write(flush);
        assert(false, 'Different results: ' + string(results));
      }
    }
  }
};
actions[DELETE] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toDelete = wrap(rand() % maxItem);
  // write('Deleting ' + string(toDelete.t) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (Set_wrapped_int s : sets) {
    results.push(s.extract(toDelete));
  }
  if (results.length > 0) {
    wrapped_int expected = results[0];
    for (wrapped_int r : results) {
      if (!alias(r, expected)) {
        assert(false, 'Different results: ' + string(results));
      }
    }
  }
};
actions[CONTAINS] = new void(int maxItem ...Set_wrapped_int[] sets)
{
  int toCheck = rand() % maxItem;
  // write('Checking ' + string(toCheck) + '\n');
  bool[] results = new bool[];
  for (Set_wrapped_int s : sets) {
    results.push(s.contains(wrap(toCheck)));
  }
  if (results.length > 0) {
    bool expected = results[0];
    for (bool r : results) {
      assert(r == expected, 'Different results: ' + string(results));
    }
  }
};
actions[GET] = new void(int maxItem ...Set_wrapped_int[] sets)
{
  int toCheck = rand() % maxItem;
  // write('Getting ' + string(toCheck) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (Set_wrapped_int s : sets) {
    results.push(s.get(wrap(toCheck)));
  }
  if (results.length > 0) {
    wrapped_int expected = results[0];
    for (wrapped_int r : results) {
      if (!alias(r, expected)) {
        assert(false, 'Different results: ' + string(results));
      }
    }
  }
};
actions[DELETE_CONTAINS] = new void(int ...Set_wrapped_int[] sets) {
  if (sets.length == 0) {
    return;
  }
  int initialSize = sets[0].size();
  if (initialSize == 0) {
    return;
  }
  wrapped_int toDelete = sets[0].getRandom();
  // write('Deleting ' + string(toDelete.t));
  int i = 0;
  for (Set_wrapped_int s : sets) {
    assert(s.contains(toDelete), 'Contains failed ' + string(i));
    wrapped_int deleted = s.extract(toDelete);
    assert(!alias(deleted, null), 'Delete returned null');
    typedef bool F(wrapped_int, wrapped_int);
    assert(((F)operator ==) != ((F)alias));
    assert(deleted == toDelete, 'Delete returned ' + string(deleted.t) +
                      ' instead of ' + string(toDelete.t));
    assert(!s.contains(toDelete), 'Contains failed');
    assert(s.size() == initialSize - 1, 'Size failed');
    ++i;
  }
};
real[] increasingProbs = new real[ActionEnum.num];
increasingProbs[ADD] = 0.7;
increasingProbs[SWAP] = 0.1;
increasingProbs[DELETE] = 0.05;
increasingProbs[CONTAINS] = 0.05;
increasingProbs[GET] = 0.05;
increasingProbs[DELETE_CONTAINS] = 0.05;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = new real[ActionEnum.num];
decreasingProbs[ADD] = 0.1;
decreasingProbs[SWAP] = 0.1;
decreasingProbs[DELETE] = 0.4;
decreasingProbs[CONTAINS] = 0.05;
decreasingProbs[GET] = 0.05;
decreasingProbs[DELETE_CONTAINS] = 0.3;
assert(sum(decreasingProbs) == 1, 'Probabilities do not sum to 1');

Set_wrapped_int naiveSet = NaiveSortedSet_wrapped_int(operator <, null);
HashSet_wrapped_int hashSet = HashSet_wrapped_int(null);
Set_wrapped_int btreeSet = BTreeSet_wrapped_int(operator <, null);

int chooseAction(real[] probs) {
  real r = unitrand();
  real sum = 0;
  for (int i = 0; i < probs.length; ++i) {
    sum += probs[i];
    if (r < sum) {
      return i;
    }
  }
  return probs.length - 1;
} 

from collections.zip(T=wrapped_int) access zip;

int maxSize = 0;
for (int i = 0; i < 2000; ++i) {
  real[] probs = i < 800 ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  actions[choice](100, naiveSet, hashSet, btreeSet);
  bool differenceFound = false;
  var ia = naiveSet.operator iter();
  var ib = btreeSet.operator iter();
  while (ia.valid() && ib.valid()) {
    var a = ia.get(), b = ib.get();
    if (!alias(a, b)) {
      differenceFound = true;
      break;
    }
    if (!alias(a, hashSet.get(a))) {
      differenceFound = true;
      break;
    }
    ia.advance();
    ib.advance();
  }
  if (ia.valid() || ib.valid()) {
    differenceFound = true;
  }
  for (wrapped_int a : hashSet) {
    if (!btreeSet.contains(a)) {
      differenceFound = true;
      break;
    }
  }
  if (differenceFound) {
    write('Difference Found:\n');
    write('Naive vs Hash: ' + differences(naiveSet, hashSet) + '\n');
    write('Naive vs BTree: ' + differences(naiveSet, btreeSet) + '\n');
    write('Hash vs BTree: ' + differences(hashSet, btreeSet) + '\n');
    assert(false);
  }

  maxSize = max(maxSize, naiveSet.size());
}

EndTest();
