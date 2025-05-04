import TestLib;

srand(4282308941601638229);

StartTest("BTree_as_SortedSet");

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

from collections.iter(T=wrapped_int) access
    Iterable_T as Iterable_wrapped_int;
from collections.set(T=wrapped_int) access
    Set_T as Set_wrapped_int;
from collections.hashset(T=wrapped_int) access
    HashSet_T as HashSet_wrapped_int;
from collections.sortedset(T=wrapped_int) access
    SortedSet_T as SortedSet_wrapped_int,
    Naive_T as NaiveSortedSet_wrapped_int;
from collections.btree(T=wrapped_int) access
    BTreeSet_T as BTreeSet_wrapped_int;

struct ActionEnum {
  static restricted int n = 0;
  static private int next() {
    return ++n - 1;
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
  static restricted int ADD = next();
  static restricted int SWAP = next();
  static restricted int GET = next();
  static restricted int DELETE = next();
  static restricted int DELETE_CONTAINS = next();
  // no dedicated action for: size(), empty(), operator iter(), getRandom(),
  // addAll, removeAll, operator <=, operator >=, operator ==, operator !=,
  // sameElementsInOrder, operator+, operator-.
}

from mapArray(Src=wrapped_int, Dst=int) access map;

int[] operator cast(wrapped_int[] a) {
  for (wrapped_int x : a) {
    assert(!alias(x, null), 'Null element in array');
  }
  static int get(wrapped_int w) { return w.t; }
  return map(get, a);
}

from collections.zip(T=wrapped_int) access zip;

string differences(SortedSet_wrapped_int a, SortedSet_wrapped_int b) {
  if (a.size() != b.size()) {
    return 'Different sizes: ' + string(a.size()) + ' vs ' + string(b.size());
  }
  string arrayValues = '[\n';
  bool diff = false;
  for (wrapped_int[] ab : zip(default=null, a, b)) {
    wrapped_int aItem = ab[0]; assert(!alias(aItem, null));
    wrapped_int bItem = ab[1]; assert(!alias(bItem, null));
    arrayValues += '  [' + format('%5d', aItem.t) + ',' + format('%5d', bItem.t)
                         + ']';
    if (!alias(aItem, bItem)) {
      arrayValues += '  <---';
      diff = true;
    }
    arrayValues += '\n';
  }
  arrayValues += ']';
  if (diff) {
    return arrayValues;
  }
  return '';
}

bool different(SortedSet_wrapped_int a, SortedSet_wrapped_int b) {
  if (a.size() != b.size()) {
    return true;
  }
  for (wrapped_int[] ab : zip(default=null, a, b)) {
    wrapped_int aItem = ab[0];
    wrapped_int bItem = ab[1];
    if (!alias(aItem, bItem)) {
      return true;
    }
  }
  return false;
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


using Action = void(int ...SortedSet_wrapped_int[]);
Action[] actions = new Action[ActionEnum.n];
actions[ActionEnum.CONTAINS] =
    new void(int  maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      bool[] results;
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.contains(wrap(toCheck)));
      }
      if (results.length > 1) {
        bool expected = results[0];
        for (bool r : results) {
          if (r != expected) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
    };
actions[ActionEnum.AFTER] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      wrapped_int[] results;
      for (SortedSet_wrapped_int s : sets) {
        wrapped_int w = s.after(wrap(toCheck));
        results.push(w);
        if (!alias(w, null)) {
          assert(w.t > toCheck);
        }
      }
      if (results.length > 1) {
        wrapped_int expected = results[0];
        for (wrapped_int r : results) {
          if (!alias(r, expected)) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
    };
actions[ActionEnum.BEFORE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      wrapped_int[] results;
      for (SortedSet_wrapped_int s : sets) {
        wrapped_int w = s.before(wrap(toCheck));
        results.push(w);
        if (!alias(w, null)) {
          assert(w.t < toCheck);
        }
      }
      if (results.length > 1) {
        wrapped_int expected = results[0];
        for (wrapped_int r : results) {
          if (!alias(r, expected)) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
    };
actions[ActionEnum.FIRST_GEQ] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      wrapped_int[] results;
      for (SortedSet_wrapped_int s : sets) {
        wrapped_int w = s.firstGEQ(wrap(toCheck));
        results.push(w);
        if (!alias(w, null)) {
          assert(w.t >= toCheck);
        }
      }
      if (results.length > 1) {
        wrapped_int expected = results[0];
        for (wrapped_int r : results) {
          if (!alias(r, expected)) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
    };
actions[ActionEnum.FIRST_LEQ] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      int toCheck = rand() % maxItem;
      wrapped_int[] results;
      for (SortedSet_wrapped_int s : sets) {
        wrapped_int w = s.firstLEQ(wrap(toCheck));
        results.push(w);
        if (!alias(w, null)) {
          assert(w.t <= toCheck);
        }
      }
      if (results.length > 1) {
        wrapped_int expected = results[0];
        for (wrapped_int r : results) {
          if (!alias(r, expected)) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
    };
actions[ActionEnum.MIN] = new void(int ...SortedSet_wrapped_int[] sets) {
  wrapped_int[] results;
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.min());
  }
  if (results.length > 1) {
    wrapped_int expected = results[0];
    for (wrapped_int r : results) {
      if (!alias(r, expected)) {
        assert(false, 'Different results: ' + string(results));
        break;
      }
    }
  }
};
actions[ActionEnum.POP_MIN] = new void(int ...SortedSet_wrapped_int[] sets) {
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.popMin());
  }
  if (results.length > 0) {
    wrapped_int expected = results[0];
    for (wrapped_int r : results) {
      if (!alias(r, expected)) {
        assert(false, 'Different results: ' + string(results));
        break;
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
        break;
      }
    }
  }
};
actions[ActionEnum.POP_MAX] = new void(int ...SortedSet_wrapped_int[] sets) {
  wrapped_int[] results = new wrapped_int[];
  for (SortedSet_wrapped_int s : sets) {
    results.push(s.popMax());
  }
  if (results.length > 0) {
    wrapped_int expected = results[0];
    for (wrapped_int r : results) {
      if (!alias(r, expected)) {
        assert(false, 'Different results: ' + string(results));
        break;
      }
    }
  }
};
actions[ActionEnum.ADD] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toInsert = wrap(rand() % maxItem);
      // write('Inserting ' + string(toInsert.t) + '\n');
      bool[] results;
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.add(toInsert));
      }
      if (results.length > 1) {
        bool expected = results[0];
        for (bool r : results) {
          if (r != expected) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
    };
actions[ActionEnum.SWAP] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toReplace = wrap(rand() % maxItem);
      // write('Replacing ' + string(toReplace.t) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.swap(toReplace));
      }
      if (results.length > 0) {
        wrapped_int expected = results[0];
        for (wrapped_int r : results) {
          if (!alias(r, expected)) {
            assert(false, 'Different results: ' + string(results));
            break;
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
        break;
      }
    }
  }
};
actions[ActionEnum.DELETE] =
    new void(int maxItem ...SortedSet_wrapped_int[] sets) {
      wrapped_int toDelete = wrap(rand() % maxItem);
      // write('Deleting ' + string(toDelete.t) + '\n');
      wrapped_int[] results = new wrapped_int[];
      for (SortedSet_wrapped_int s : sets) {
        results.push(s.extract(toDelete));
      }
      if (results.length > 1) {
        wrapped_int expected = results[0];
        for (wrapped_int r : results) {
          if (!alias(r, expected)) {
            assert(false, 'Different results: ' + string(results));
            break;
          }
        }
      }
      if (results.length > 0) {
        wrapped_int r = results[0];
        if (!alias(r, null)) {
          for (SortedSet_wrapped_int s : sets) {
            assert(!s.contains(r));
          }
        }
      }
    };
actions[ActionEnum.DELETE_CONTAINS] =
    new void(int ...SortedSet_wrapped_int[] sets) {
      if (sets.length == 0 || sets[0].empty()) {
        return;
      }
      int initialsize = sets[0].size();
      wrapped_int toDelete = wrap(sets[0].getRandom().t);
      for (SortedSet_wrapped_int s : sets) {
        assert(s.extract(toDelete) == toDelete, 'Delete failed');
        assert(!s.contains(toDelete), 'Contains failed');
        assert(s.size() == initialsize - 1, 'Size failed');
      }
    };
real[] increasingProbs = array(n=ActionEnum.n, value=0.0);
// Actions that don't modify the set:
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
increasingProbs[ActionEnum.ADD] = 1 / 4;
increasingProbs[ActionEnum.SWAP] = 1 / 4;
assert(sum(increasingProbs) == 3/4);
// Actions that might remove an element:
increasingProbs[ActionEnum.POP_MIN] = 1 / 16;
increasingProbs[ActionEnum.POP_MAX] = 1 / 16;
increasingProbs[ActionEnum.DELETE] = 1 / 16;
increasingProbs[ActionEnum.DELETE_CONTAINS] = 1 / 16;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = copy(increasingProbs);
// Actions that might add an element:
decreasingProbs[ActionEnum.ADD] = 1 / 8;
decreasingProbs[ActionEnum.SWAP] = 1 / 8;
// Actions that might remove an element:
decreasingProbs[ActionEnum.POP_MIN] = 1 / 8;
decreasingProbs[ActionEnum.POP_MAX] = 1 / 8;
decreasingProbs[ActionEnum.DELETE] = 1 / 8;
decreasingProbs[ActionEnum.DELETE_CONTAINS] = 1 / 8;
assert(sum(decreasingProbs) == 1, 'Probabilities do not sum to 1');

SortedSet_wrapped_int naive = NaiveSortedSet_wrapped_int(operator <, null);
SortedSet_wrapped_int btree1 = BTreeSet_wrapped_int(null, maxPivots=4);
SortedSet_wrapped_int btree2 = BTreeSet_wrapped_int(null, maxPivots=128);

bool isStrictlySorted(SortedSet_wrapped_int s) {
  wrapped_int last = null;
  for (wrapped_int w : s) {
    if (!alias(last, null) && last.t >= w.t) {
      return false;
    }
    last = w;
  }
  return true;
}

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

int[] counts = array(n=ActionEnum.n, value=0);
string[] names = {
  'CONTAINS', 'AFTER', 'BEFORE', 'FIRST_GEQ', 'FIRST_LEQ', 'MIN', 'POP_MIN',
  'MAX', 'POP_MAX', 'ADD', 'SWAP', 'GET', 'DELETE', 'DELETE_CONTAINS'
};

int maxSize = 0;
int numActions = 2000;
for (int i : range(numActions)) {
  real[] probs = i < numActions * 2 # 5 ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  ++counts[choice];
  actions[choice](100, naive, btree1, btree2);
  for (var btree : new SortedSet_wrapped_int[] {btree1, btree2}) {
    if (different(naive, btree)) {
      write('Different sets after action ' + names[choice]);
      assert(false, 'Naive vs btree: \n' + differences(naive, btree));
    }
    assert(isStrictlySorted(btree), 'Not sorted');
    maxSize = max(maxSize, btree.size());
  }
}

if (false) {
  write('Max size: ' + string(maxSize));
  write('Action counts: {');
  for (int i : range(ActionEnum.n)) {
    write('  ' + names[i] + ': ' + string(counts[i]) + ',');
  }
}

EndTest();

StartTest('BTree_binary_ops');

Set_wrapped_int a = BTreeSet_wrapped_int();
a.add(wrap(1));
a.add(wrap(2));
a.add(wrap(3));
Set_wrapped_int b = HashSet_wrapped_int();
b.add(wrap(2));
b.add(wrap(3));
b.add(wrap(4));

{
  Set_wrapped_int c = a + b;
  assert(c.size() == 4, 'Union failed: wrong size ' + string(c.size()));
  assert(c.contains(wrap(1)), 'Union failed: missing 1');
  assert(c.contains(wrap(2)), 'Union failed: missing 2');
  assert(c.contains(wrap(3)), 'Union failed: missing 3');
  assert(c.contains(wrap(4)), 'Union failed: missing 4');
}
{
  Set_wrapped_int c = a - b;
  assert(c.size() == 1, 'Difference failed: wrong size ' + string(c.size()));
  assert(c.contains(wrap(1)), 'Difference failed: missing 1');
}
{
  Set_wrapped_int c = a & b;
  assert(c.size() == 2, 'Intersection failed: wrong size ' + string(c.size()));
  assert(c.contains(wrap(2)), 'Intersection failed: missing 2');
  assert(c.contains(wrap(3)), 'Intersection failed: missing 3');
}
{
  Set_wrapped_int c = a ^ b;
  assert(c.size() == 2, 'Symmetric difference failed: wrong size ' +
         string(c.size()));
  assert(c.contains(wrap(1)), 'Symmetric difference failed: missing 1');
  assert(c.contains(wrap(4)), 'Symmetric difference failed: missing 4');
}
EndTest();
