import TestLib;

StartTest("NaiveSortedSet");

from wrapper(T=int) access
    Wrapper_T as wrapped_int,
    wrap,
    alias;

bool operator < (wrapped_int a, wrapped_int b) {
  return a.t < b.t;
}

bool operator == (wrapped_int a, wrapped_int b) {
  return a.t == b.t;
}

// ISSUE: We have to import these from sortedset. If we import directly from
// pureset instead, identical types are not recognized as such when resolving
// function calls and implicit casts.
// from pureset(T=wrapped_int) access
//     Set_T as Set_wrapped_int,
//     makeNaiveSet,
//     operator cast;

from sortedset(T=wrapped_int) access
    Set_T as Set_wrapped_int,
    makeNaiveSet,
    SortedSet_T as SortedSet_wrapped_int,
    makeNaiveSortedSet,
    operator cast,
    unSort;

struct ActionEnum {
  static restricted int numActions = 0;
  static private int next() {
    return ++numActions - 1;
  }
  static restricted int INSERT = next();
  static restricted int REPLACE = next();
  static restricted int DELETE = next();
  static restricted int CONTAINS = next();
  static restricted int DELETE_CONTAINS = next();
}

from zip(T=int) access zip;
//from sort(T=wrapped_int) access mergeSort as sort;
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

string differences(Set_wrapped_int a, Set_wrapped_int b) {
  if (a.size() != b.size()) {
    return 'Different sizes: ' + string(a.size()) + ' vs ' + string(b.size());
  }
  wrapped_int[] aArray = sort(a, operator<);
  int[] aIntArray = map(get, aArray);
  wrapped_int[] bArray = sort(b, operator<);
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

Action[] actions = new Action[ActionEnum.numActions];
actions[ActionEnum.INSERT] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toInsert = wrap(rand() % maxItem);
  // write('Inserting ' + string(toInsert.t) + '\n');
  for (Set_wrapped_int s : sets) {
    s.insert(toInsert);
  }
};
actions[ActionEnum.REPLACE] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toReplace = wrap(rand() % maxItem);
  // write('Replacing ' + string(toReplace.t) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (Set_wrapped_int s : sets) {
    results.push(s.replace(toReplace));
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
actions[ActionEnum.DELETE] = new void(int maxItem ...Set_wrapped_int[] sets) {
  wrapped_int toDelete = wrap(rand() % maxItem);
  // write('Deleting ' + string(toDelete.t) + '\n');
  bool[] results = new bool[];
  for (Set_wrapped_int s : sets) {
    results.push(s.delete(toDelete));
  }
  if (results.length > 0) {
    bool expected = results[0];
    for (bool r : results) {
      assert(r == expected, 'Different results: ' + string(results));
    }
  }
};
actions[ActionEnum.CONTAINS] = new void(int maxItem ...Set_wrapped_int[] sets) {
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
actions[ActionEnum.DELETE_CONTAINS] = new void(int ...Set_wrapped_int[] sets) {
  if (sets.length == 0) {
    return;
  }
  int initialSize = sets[0].size();
  if (initialSize == 0) {
    return;
  }
  int indexToDelete = rand() % initialSize;
  int i = 0;
  wrapped_int toDelete = null;
  bool process(wrapped_int a) {
    if (i == indexToDelete) {
      toDelete = wrap(a.t);
      return false;
    }
    ++i;
    return true;
  }
  sets[0].forEach(process);
  assert(i < initialSize, 'Index out of range');
  // write('Deleting ' + string(toDelete.t) + '\n');
  int i = 0;
  for (Set_wrapped_int s : sets) {
    assert(s.contains(toDelete), 'Contains failed ' + string(i));
    assert(s.delete(toDelete), 'Delete failed');
    assert(!s.contains(toDelete), 'Contains failed');
    assert(s.size() == initialSize - 1, 'Size failed');
    ++i;
  }
};
real[] increasingProbs = new real[ActionEnum.numActions];
increasingProbs[ActionEnum.INSERT] = 0.7;
increasingProbs[ActionEnum.REPLACE] = 0.1;
increasingProbs[ActionEnum.DELETE] = 0.05;
increasingProbs[ActionEnum.CONTAINS] = 0.1;
increasingProbs[ActionEnum.DELETE_CONTAINS] = 0.05;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = new real[ActionEnum.numActions];
decreasingProbs[ActionEnum.INSERT] = 0.1;
decreasingProbs[ActionEnum.REPLACE] = 0.1;
decreasingProbs[ActionEnum.DELETE] = 0.4;
decreasingProbs[ActionEnum.CONTAINS] = 0.1;
decreasingProbs[ActionEnum.DELETE_CONTAINS] = 0.3;
assert(sum(decreasingProbs) == 1, 'Probabilities do not sum to 1');

Set_wrapped_int pure_set = makeNaiveSet(operator ==, (wrapped_int)null);
SortedSet_wrapped_int sorted_set =
    makeNaiveSortedSet(operator <, (wrapped_int)null);

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

bool isStrictlySorted(wrapped_int[] arr) {
  for (int i = 1; i < arr.length; ++i) {
    if (!(arr[i - 1] < arr[i])) {
      return false;
    }
  }
  return true;
}

int maxSize = 0;
for (int i = 0; i < 2000; ++i) {
  real[] probs = i < 800 ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  actions[choice](100, pure_set, sorted_set);
  string diffs = differences(pure_set, sorted_set);
  assert(diffs == '', 'Pure vs sorted: \n' + diffs);
  assert(isStrictlySorted(sorted_set), 'Not sorted');
  maxSize = max(maxSize, pure_set.size());
}
// write('Max size: ' + string(maxSize) + '\n');

// int maxSize = 0;
// for (int i = 0; i < 2000; ++i) {
//   real[] probs = i < 800 ? increasingProbs : decreasingProbs;
//   int choice = chooseAction(probs);
//   actions[choice](1000, pure_set, unSort(sorted_set));
//   string diffs = differences(pure_set, sorted_set);
//   assert(diffs == '', 'Pure vs sorted: \n' + diffs);
//   maxSize = max(maxSize, pure_set.size());
// }
// write('Max size: ' + string(maxSize) + '\n');

EndTest();