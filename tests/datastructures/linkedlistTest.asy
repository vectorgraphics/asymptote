import TestLib;

StartTest("LinkedList");

access "datastructures/linkedlist"(T=int) as list_int;

struct ListActionEnum {
  static restricted int numActions = 0;
  static private int next() {
    return ++numActions - 1;
  }
  static restricted int ADD = next();
  static restricted int INSERT = next();
  static restricted int ITERATE = next();
  restricted int choice;
  void operator init(int choice) {
    assert(choice < numActions, 'Invalid ListActionEnum choice: ' + string(choice));
    assert(choice >= 0, 'Invalid ListActionEnum choice: ' + string(choice));
    this.choice = choice;
  }
  static ListActionEnum add = ListActionEnum(ADD);
  static ListActionEnum insert = ListActionEnum(INSERT);
  static ListActionEnum iterate = ListActionEnum(ITERATE);
}
bool operator == (ListActionEnum a, ListActionEnum b) {
  return a.choice == b.choice;
}
bool operator == (ListActionEnum a, int b) {
  return a.choice == b;
}
string operator ecast(ListActionEnum a) {
  if (a == ListActionEnum.add) {
    return 'ADD';
  } else
  if (a == ListActionEnum.insert) {
    return 'INSERT';
  } else if (a == ListActionEnum.iterate) {
    return 'ITERATE';
  }
  assert(false);
  return '';
}

// Actions that can be taken using an iterator.
struct IterActionEnum {
  static restricted int numActions = 0;
  static private int next() {
    return ++numActions - 1;
  }
  static restricted int TRY_NEXT = next();
  static restricted int TRY_DELETE = next();
  static restricted int END_EARLY = next();
  restricted int choice;
  void operator init(int choice) {
    assert(choice < numActions, 'Invalid IterActionEnum choice: ' + string(choice));
    assert(choice >= 0, 'Invalid IterActionEnum choice: ' + string(choice));
    this.choice = choice;
  }
  static IterActionEnum tryNext = IterActionEnum(TRY_NEXT);
  static IterActionEnum tryDelete = IterActionEnum(TRY_DELETE);
  static IterActionEnum endEarly = IterActionEnum(END_EARLY);
}
bool operator == (IterActionEnum a, IterActionEnum b) {
  return a.choice == b.choice;
}
bool operator == (IterActionEnum a, int b) {
  return a.choice == b;
}
string operator ecast(IterActionEnum a) {
  if (a == IterActionEnum.tryNext) {
    return 'TRY_NEXT';
  } else if (a == IterActionEnum.tryDelete) {
    return 'TRY_DELETE';
  } else if (a == IterActionEnum.endEarly) {
    return 'END_EARLY';
  }
  assert(false);
  return '';
}

from zip(T=int) access zip;

void writeArrays(int unused);
void writeArrays(...int[][] arrays) {
  write(zip(...arrays));
}

string differences(list_int.L a, list_int.L b) {
  if (a.size() != b.size()) {
    return 'Different sizes: ' + string(a.size()) + ' vs ' + string(b.size());
  }
  if (!all(list_int.toArray(a) == list_int.toArray(b))) {
    writeArrays(list_int.toArray(a), list_int.toArray(b));
    return 'Different contents';
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

typedef void ListAction(...list_int.L[]);
list_int.Iter[] iters;
void endIters() {
  iters.delete();
}
bool inIterMode() {
  return iters.length > 0;
}
bool canDelete;
int numDeletions = 0;

ListAction[] listActions = new ListAction[ListActionEnum.numActions];
listActions[ListActionEnum.ADD] = new void(...list_int.L[] lists) {
  int toAdd = rand() % 100;
  for (list_int.L list : lists) {
    list.add(toAdd);
  }
};
listActions[ListActionEnum.INSERT] = new void(...list_int.L[] lists) {
  int toAdd = rand() % 100 - 100;
  for (list_int.L list : lists) {
    list.insertAtBeginning(toAdd);
  }
};
listActions[ListActionEnum.ITERATE] = new void(...list_int.L[] lists) {
  for (list_int.L list : lists) {
    iters.push(list.iterator());
  }
  canDelete = false;
};

typedef void IterAction();
IterAction[] iterActions = new IterAction[IterActionEnum.numActions];
iterActions[IterActionEnum.TRY_NEXT] = new void() {
  if (iters.length == 0) {
    return;
  }
  int[] nexts;
  if (!iters[0].hasNext()) {
    // write('no next');
    for (list_int.Iter iter : iters) {
      if (iter.hasNext()) {
        writeArrays(0);
        write('hasNext should be false');
        assert(false);
      }
    }
    endIters();
    return;
  }
  for (list_int.Iter iter : iters) {
    // write('examining next');
    if (!iter.hasNext()) {
      writeArrays(0);
      write('hasNext should be true');
      assert(false);
    }
    nexts.push(iter.next());
  }
  canDelete = true;
  int val = nexts[0];
  // write('next: ' + string(val));
  if (!all(nexts == val)) {
    writeArrays(0);
    write('Nexts should all be ' + string(val));
    write('Nexts: ' + string(nexts));
    assert(false);
  }
};
iterActions[IterActionEnum.TRY_DELETE] = new void() {
  if (!canDelete) {
    return;
  }
  for (list_int.Iter iter : iters) {
    iter.delete();
  }
  // write('deleted item');
  ++numDeletions;
  canDelete = false;
};
iterActions[IterActionEnum.END_EARLY] = endIters;

real clamp(real x, real keyword min, real keyword max) {
  if (x < min) {
    return min;
  }
  if (x > max) {
    return max;
  }
  return x;
}

ListActionEnum nextListAction(real desiredLength, int length) {
  real lengthenProb = clamp(0.1 + 0.1 * (desiredLength - length), min=0, max=1);
  if (unitrand() < lengthenProb) {
    if (rand() % 2 == 0) {
      return ListActionEnum.add;
    } else {
      return ListActionEnum.insert;
    }
  }
  return ListActionEnum.iterate;
}
IterActionEnum nextIterAction(real desiredLength, int length) {
  real deleteProb = clamp(0.1 + 0.1 * (length - desiredLength), min=0, max=1);
  if (unitrand() < deleteProb)
    return IterActionEnum.tryDelete;
  real endProb = 1 / (2 * length);
  if (unitrand() < endProb)
    return IterActionEnum.endEarly;
  return IterActionEnum.tryNext;
}

list_int.L naive = list_int.makeNaive();
list_int.L linked = list_int.make();

writeArrays = new void(int unused) {
  writeArrays(list_int.toArray(naive), list_int.toArray(linked));
};

for (int i = 0; i < 2000; ++i) {
  // write('Step ' + string(i));
  // writeArrays(0);
  int desiredLength = (i < 800 ? 100 : 1);
  if (inIterMode()) {
    IterActionEnum action = nextIterAction(desiredLength, iters.length);
    // write('next action: ' + (string)action);
    iterActions[action.choice]();
  } else {
    ListActionEnum action = nextListAction(desiredLength, linked.size());
    // write('next action: ' + (string)action);
    listActions[action.choice](naive, linked);
  }
  string diffs = differences(naive, linked);
  assert(diffs == '', diffs);
}

EndTest();