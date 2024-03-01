
import TestLib;

StartTest("SplayTree_as_Set");

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

from splaytree(T=wrapped_int) access
    makeNaiveSortedSet,
    SortedSet_T as SortedSet_wrapped_int,
    SplayTree_T as SplayTree_wrapped_int,
    operator cast;

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

string differences(SortedSet_wrapped_int a, SortedSet_wrapped_int b) {
  if (a.size() != b.size()) {
    return 'Different sizes: ' + string(a.size()) + ' vs ' + string(b.size());
  }
  wrapped_int[] aArray = a;
  int[] aIntArray = aArray;
  wrapped_int[] bArray = b;
  int[] bIntArray = bArray;
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

typedef void Action(int ...SortedSet_wrapped_int[]);

Action[] actions = new Action[ActionEnum.numActions];
actions[ActionEnum.INSERT] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toInsert = wrap(rand() % maxItem);
      // write('Inserting ' + string(toInsert.t) + '\n');
      for (SortedSet_wrapped_int s : sets) {
        s.insert(toInsert);
      }
    };
actions[ActionEnum.REPLACE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toReplace = wrap(rand() % maxItem);
      // write('Replacing ' + string(toReplace.t) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
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
actions[ActionEnum.DELETE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toDelete = wrap(rand() % maxItem);
      // write('Deleting ' + string(toDelete.t) + '\n');
      bool[] results = new bool[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.delete(toDelete));
      }
      if (results.length > 0) {
        bool expected = results[0];
        for (bool r : results) {
          assert(r == expected, 'Different results: ' + string(results));
        }
      }
    };
actions[ActionEnum.CONTAINS] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      // write('Checking ' + string(toCheck) + '\n');
      bool[] results = new bool[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.contains(wrap(toCheck)));
      }
      if (results.length > 0) {
        bool expected = results[0];
        for (bool r : results) {
          assert(r == expected, 'Different results: ' + string(results));
        }
      }
    };
actions[ActionEnum.DELETE_CONTAINS] =
    new void(int ...SortedSet_wrapped_int[] sets) {
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
      for (SortedSet_wrapped_int s : sets) {
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

SortedSet_wrapped_int sorted_set =
    makeNaiveSortedSet(operator <, (wrapped_int)null);
SplayTree_wrapped_int splayset =
    SplayTree_wrapped_int(operator <, (wrapped_int)null);

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
  actions[choice](100, sorted_set, splayset);
  string diffs = differences(sorted_set, splayset);
  assert(diffs == '', 'Naive vs splayset: \n' + diffs);
  assert(isStrictlySorted(splayset), 'Not sorted');
  maxSize = max(maxSize, splayset.size());
}
EndTest();

StartTest("SplayTree_as_SortedSet");

struct ActionEnum {
  static restricted int numActions = 0;
  static private int next() {
    return ++numActions - 1;
  }
  static restricted int CONTAINS = next();
  static restricted int AFTER = next();
  static restricted int BEFORE = next();
  static restricted int FIRST_GEQ = next();
  static restricted int FIRST_LEQ = next();
  static restricted int MIN = next();
  static restricted int POP_MIN = next();
  static restricted int MAX = next();
  static restricted int POP_MAX = next();
  static restricted int INSERT = next();
  static restricted int REPLACE = next();
  static restricted int GET = next();
  static restricted int DELETE = next();
  static restricted int DELETE_CONTAINS = next();
}

Action[] actions = new Action[ActionEnum.numActions];
actions[ActionEnum.CONTAINS] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      // write('Checking ' + string(toCheck) + '\n');
      bool[] results = new bool[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.contains(wrap(toCheck)));
      }
      if (results.length > 0) {
        bool expected = results[0];
        for (bool r : results) {
          assert(r == expected, 'Different results: ' + string(results));
        }
      }
    };
actions[ActionEnum.AFTER] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      // write('After ' + string(toCheck) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.after(wrap(toCheck)));
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
actions[ActionEnum.BEFORE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      // write('Before ' + string(toCheck) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.before(wrap(toCheck)));
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
actions[ActionEnum.FIRST_GEQ] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      // write('First greater or equal ' + string(toCheck) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.firstGEQ(wrap(toCheck)));
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
actions[ActionEnum.FIRST_LEQ] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      // write('First less or equal ' + string(toCheck) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.firstLEQ(wrap(toCheck)));
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
actions[ActionEnum.MIN] = new void(int ...SortedSet_wrapped_int[] sets) {
  // write('Min\n');
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.min());
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
actions[ActionEnum.POP_MIN] = new void(int ...SortedSet_wrapped_int[] sets) {
  // write('Pop min\n');
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.popMin());
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
actions[ActionEnum.MAX] = new void(int ...SortedSet_wrapped_int[] sets) {
  // write('Max\n');
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.max());
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
actions[ActionEnum.POP_MAX] = new void(int ...SortedSet_wrapped_int[] sets) {
  // write('Pop max\n');
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.popMax());
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
actions[ActionEnum.INSERT] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toInsert = wrap(rand() % maxItem);
      // write('Inserting ' + string(toInsert.t) + '\n');
      for (SortedSet_wrapped_int s : sets) {
        s.insert(toInsert);
      }
    };
actions[ActionEnum.REPLACE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toReplace = wrap(rand() % maxItem);
      // write('Replacing ' + string(toReplace.t) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
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
actions[ActionEnum.GET] = new void(int maxItem ...SortedSet_wrapped_int[] sets)
{
  wrapped_int toGet = wrap(rand() % maxItem);
  // write('Getting ' + string(toGet) + '\n');
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.get(toGet));
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
actions[ActionEnum.DELETE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toDelete = wrap(rand() % maxItem);
      // write('Deleting ' + string(toDelete.t) + '\n');
      bool[] results = new bool[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.delete(toDelete));
      }
      if (results.length > 0) {
        bool expected = results[0];
        for (bool r : results) {
          assert(r == expected, 'Different results: ' + string(results));
        }
      }
    };
actions[ActionEnum.DELETE_CONTAINS] =
    new void(int ...SortedSet_wrapped_int[] sets) {
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
      for (SortedSet_wrapped_int s : sets) {
        assert(s.delete(toDelete), 'Delete failed');
        assert(!s.contains(toDelete), 'Contains failed');
        assert(s.size() == initialSize - 1, 'Size failed');
        ++i;
      }
    };

real[] increasingProbs = array(n=ActionEnum.numActions, value=0.0);
// Actions that don't modify the set (except for rebalancing):
increasingProbs[ActionEnum.CONTAINS] = 1 / 2^5;
increasingProbs[ActionEnum.AFTER] = 1 / 2^5;
increasingProbs[ActionEnum.BEFORE] = 1 / 2^5;
increasingProbs[ActionEnum.FIRST_GEQ] = 1 / 2^5;
increasingProbs[ActionEnum.FIRST_LEQ] = 1 / 2^5;
increasingProbs[ActionEnum.MIN] = 1 / 2^5;
increasingProbs[ActionEnum.MAX] = 1 / 2^5;
increasingProbs[ActionEnum.GET] = 1 / 2^5;
// 1/4 probability of this sort of action:
assert(sum(increasingProbs) == 8 / 2^5);
// Actions that might add an element:
increasingProbs[ActionEnum.INSERT] = 1 / 4;
increasingProbs[ActionEnum.REPLACE] = 1 / 4;
assert(sum(increasingProbs) == 3/4);
// Actions that might remove an element:
increasingProbs[ActionEnum.POP_MIN] = 1 / 16;
increasingProbs[ActionEnum.POP_MAX] = 1 / 16;
increasingProbs[ActionEnum.DELETE] = 1 / 16;
increasingProbs[ActionEnum.DELETE_CONTAINS] = 1 / 16;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = copy(increasingProbs);
// Actions that might add an element:
decreasingProbs[ActionEnum.INSERT] = 1 / 8;
decreasingProbs[ActionEnum.REPLACE] = 1 / 8;
// Actions that might remove an element:
decreasingProbs[ActionEnum.POP_MIN] = 1 / 8;
decreasingProbs[ActionEnum.POP_MAX] = 1 / 8;
decreasingProbs[ActionEnum.DELETE] = 1 / 8;
decreasingProbs[ActionEnum.DELETE_CONTAINS] = 1 / 8;
assert(sum(decreasingProbs) == 1, 'Probabilities do not sum to 1');

SortedSet_wrapped_int sorted_set =
    makeNaiveSortedSet(operator <, (wrapped_int)null);
SplayTree_wrapped_int splayset =
    SplayTree_wrapped_int(operator <, (wrapped_int)null);


int maxSize = 0;
for (int i = 0; i < 2000; ++i) {
  real[] probs = i < 800 ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  actions[choice](100, sorted_set, splayset);
  string diffs = differences(sorted_set, splayset);
  assert(diffs == '', 'Naive vs splayset: \n' + diffs);
  assert(isStrictlySorted(splayset), 'Not sorted');
  maxSize = max(maxSize, splayset.size());
}

EndTest();