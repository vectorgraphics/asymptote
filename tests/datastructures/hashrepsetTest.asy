import TestLib;

StartTest("HashRepSet");

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
    write('difference: ' + string(a.t - b.t));
    return a.t == b.t;
  }
  autounravel bool operator !=(wrapped_int a, wrapped_int b) {
    return !(a == b);
  }
  autounravel bool operator <(wrapped_int a, wrapped_int b) {
    return a.t < b.t;
  }
  autounravel int hash(wrapped_int a) {
    if (alias(a, null)) return 0;
    return hash(a.t, 62);
  }
}
{
  typedef bool F(wrapped_int, wrapped_int);
  assert(((F)operator ==) != ((F)alias));
}

wrapped_int wrap(int t) = wrapped_int;  // `wrap` is alias for constructor

// from pureset(T=wrapped_int) access
//     Set_T as Set_wrapped_int,
//     makeNaiveSet;
from 'datastructures/repset'(T=wrapped_int) access
    RepSet_T as Set_wrapped_int,
    NaiveRepSet_T as NaiveSet_wrapped_int;

from 'datastructures/hashrepset'(T=wrapped_int) access
    HashRepSet_T as HashSet_wrapped_int;

struct ActionEnum {
  static restricted int num = 0;
  static private int make() {
    return ++num - 1;
  }
  autounravel restricted int CONTAINS = make();
  autounravel restricted int GET = make();
  autounravel restricted int ADD = make();
  autounravel restricted int UPDATE = make();
  autounravel restricted int DELETE = make();
  autounravel restricted int DELETE_CONTAINS = make();
}

from zip(T=int) access zip;
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
    return 'Different sizes: ' + string(aArray.length) + ' vs ' + string(bArray.length);
  }
  int[] aIntArray = map(get, aArray);
  int[] bIntArray = map(get, bArray);
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
  wrapped_int[] aArray = sort((wrapped_int[])a, operator<);
  wrapped_int[] bArray = sort((wrapped_int[])b, operator<);
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
  write('Inserting ' + string(toInsert.t) + '\n');
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
actions[UPDATE] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toReplace = wrap(rand() % maxItem);
  write('Replacing ' + string(toReplace.t) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (Set_wrapped_int s : sets) {
    results.push(s.update(toReplace));
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
actions[DELETE] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toDelete = wrap(rand() % maxItem);
  write('Deleting ' + string(toDelete.t) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (Set_wrapped_int s : sets) {
    results.push(s.delete(toDelete));
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
  write('Checking ' + string(toCheck) + '\n');
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
  write('Getting ' + string(toCheck) + '\n');
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
  int indexToDelete = rand() % initialSize;
  write('Iterating to ' + string(indexToDelete));
  var it = sets[0].iter();
  for (int i = 0; i < indexToDelete; ++i) {
    it.advance();
  }
  wrapped_int toDelete = wrap(it.get().t);
  write('Deleting ' + string(toDelete.t));
  int i = 0;
  for (Set_wrapped_int s : sets) {
    assert(s.contains(toDelete), 'Contains failed ' + string(i));
    wrapped_int deleted = s.delete(toDelete);
    assert(!alias(deleted, null), 'Delete returned null');
    write('AAA');
    typedef bool F(wrapped_int, wrapped_int);
    assert(((F)operator ==) != ((F)alias));
    write(deleted == toDelete);
    write('BBB');
    assert(deleted == toDelete, 'Delete returned ' + string(deleted.t) + ' instead of ' + string(toDelete.t));
    assert(!s.contains(toDelete), 'Contains failed');
    assert(s.size() == initialSize - 1, 'Size failed');
    ++i;
  }
};
real[] increasingProbs = new real[ActionEnum.num];
increasingProbs[ADD] = 0.7;
increasingProbs[UPDATE] = 0.1;
increasingProbs[DELETE] = 0.05;
increasingProbs[CONTAINS] = 0.05;
increasingProbs[GET] = 0.05;
increasingProbs[DELETE_CONTAINS] = 0.05;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = new real[ActionEnum.num];
decreasingProbs[ADD] = 0.1;
decreasingProbs[UPDATE] = 0.1;
decreasingProbs[DELETE] = 0.4;
decreasingProbs[CONTAINS] = 0.05;
decreasingProbs[GET] = 0.05;
decreasingProbs[DELETE_CONTAINS] = 0.3;
assert(sum(decreasingProbs) == 1, 'Probabilities do not sum to 1');

Set_wrapped_int naiveSet = NaiveSet_wrapped_int(null);
Set_wrapped_int hashSet = HashSet_wrapped_int(null);

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

int maxSize = 0;
for (int i = 0; i < 2000; ++i) {
  real[] probs = i < 800 ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  actions[choice](100, naiveSet, hashSet);
  bool differenceFound = false;
  assert(naiveSet.iter != null, 'Naive set has no iter');
  assert(hashSet.iter != null, 'Hash set has no iter');
  for (var ita = naiveSet.iter(), itb = hashSet.iter(); ita.valid() && itb.valid(); ita.advance(), itb.advance()) {
    if (!alias(ita.get(), itb.get())) {
      differenceFound = true;
      break;
    }
  }
  if (differenceFound) {
    assert(false, 'Naive vs hash: \n' + differences((wrapped_int[])naiveSet, (wrapped_int[])hashSet));
  }

  maxSize = max(maxSize, naiveSet.size());
}
// write('Max size: ' + string(maxSize) + '\n');

// int maxSize = 0;
// for (int i = 0; i < 2000; ++i) {
//   real[] probs = i < 800 ? increasingProbs : decreasingProbs;
//   int choice = chooseAction(probs);
//   actions[choice](1000, naiveSet, hashSet);
//   string diffs = differences(naiveSet, hashSet);
//   assert(diffs == '', 'Naive vs hash: \n' + diffs);
//   maxSize = max(maxSize, naiveSet.size());
// }
// write('Max size: ' + string(maxSize) + '\n');

EndTest();