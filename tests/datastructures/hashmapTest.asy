import TestLib;

StartTest('collections.hashmap');

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
  int hash() {
    return hash(t);
  }
}

wrapped_int wrap(int t) = wrapped_int;  // `wrap` is alias for constructor
wrapped_int operator cast(int t) = wrap;


from collections.map(K=wrapped_int, V=real) access
    Map_K_V as Map_int_real,
    NaiveMap_K_V as NaiveMap_int_real;

from collections.hashmap(K=wrapped_int, V=real) access
    HashMap_K_V as HashMap_int_real;

// from collections.smallintmap(T=real) access
//     SmallIntMap_V as MapSmallint_real;

struct ActionEnum {
  static restricted int num = 0;
  static private int next() {
    int result = num;
    ++num;
    return result;
  }
  static restricted int SIZE = next();
  static restricted int EMPTY = next();
  static restricted int CONTAINS = next();
  static restricted int FOR_EACH_CONTAINS = next();
  static restricted int PUT = next();
  static restricted int SOFT_DELETE = next();
  static restricted int FIND_DELETE = next();

  static string toString(int action) {
    if (action == SIZE) return 'SIZE';
    if (action == EMPTY) return 'EMPTY';
    if (action == CONTAINS) return 'CONTAINS';
    if (action == FOR_EACH_CONTAINS) return 'FOR_EACH_CONTAINS';
    if (action == PUT) return 'PUT';
    if (action == SOFT_DELETE) return 'SOFT_DELETE';
    if (action == FIND_DELETE) return 'FIND_DELETE';
    return 'UNKNOWN';
  }
}

using Action=void(int maxItem...Map_int_real[]);

Action[] actions = new Action[ActionEnum.num];
actions[ActionEnum.SIZE] = new void(int maxItem ...Map_int_real[] maps) {
  int referenceIndex = rand() % maps.length;
  int referenceSize = maps[referenceIndex].size();
  for (Map_int_real map : maps) {
    assert(map.size() == referenceSize);
  }
  //write('size: ' + (string)referenceSize);
};
actions[ActionEnum.EMPTY] = new void(int maxItem ...Map_int_real[] maps) {
  bool referenceEmpty = maps[rand() % maps.length].empty();
  for (Map_int_real map : maps) {
    assert(map.empty() == referenceEmpty);
  }
};
actions[ActionEnum.CONTAINS] = new void(int maxItem ...Map_int_real[] maps) {
  int key = rand() % maxItem;
  bool referenceContains = maps[rand() % maps.length].contains(key);
  for (Map_int_real map : maps) {
    assert(map.contains(key) == referenceContains);
  }
};
actions[ActionEnum.FOR_EACH_CONTAINS] = new void(
    int maxItem
    ...Map_int_real[] maps
) {
  for (Map_int_real map : maps) {
    for (var it=map.iter(); it.valid(); it.advance()) {
      for (Map_int_real map_ : maps) {
        wrapped_int key = it.get();
        assert(map_.contains(key));
        if (isnan(map[key]))
          assert(isnan(map_[key]));
        else
          assert(map_[key] == map[key]);
      }
    }
  }
};
actions[ActionEnum.PUT] = new void(int maxItem ...Map_int_real[] maps) {
  wrapped_int key = rand() % maxItem;
  real value = rand();
  for (Map_int_real map : maps) {
    map[key] = value;
  }
};
actions[ActionEnum.SOFT_DELETE] = new void(int maxItem ...Map_int_real[] maps) {
  wrapped_int key = rand() % maxItem;
  for (Map_int_real map : maps) {
    map[key] = nan;
  }
};
actions[ActionEnum.FIND_DELETE] = new void(int maxItem ...Map_int_real[] maps) {
  int whichmap = rand() % maps.length;
  Map_int_real referenceMap = maps[whichmap];
  int size = referenceMap.size();
  if (size == 0)
    return;
  int index = rand() % size;
  wrapped_int key = ((wrapped_int[])referenceMap)[index];
  for (Map_int_real map : maps) {
    assert(map.contains(key));
    map.delete(key);
  }
};

real[] increasingProbs = new real[ActionEnum.num];
increasingProbs[ActionEnum.SIZE] = 0.1;
increasingProbs[ActionEnum.EMPTY] = 0.1;
increasingProbs[ActionEnum.CONTAINS] = 0.1;
increasingProbs[ActionEnum.FOR_EACH_CONTAINS] = 0.05;
increasingProbs[ActionEnum.PUT] = 0.4;
increasingProbs[ActionEnum.SOFT_DELETE] = 0.15;
increasingProbs[ActionEnum.FIND_DELETE] = 0.1;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = new real[ActionEnum.num];
decreasingProbs[ActionEnum.SIZE] = 0.1;
decreasingProbs[ActionEnum.EMPTY] = 0.1;
decreasingProbs[ActionEnum.CONTAINS] = 0.1;
decreasingProbs[ActionEnum.FOR_EACH_CONTAINS] = 0.05;
decreasingProbs[ActionEnum.PUT] = 0.1;
decreasingProbs[ActionEnum.SOFT_DELETE] = 0.35;
decreasingProbs[ActionEnum.FIND_DELETE] = 0.2;
assert(sum(decreasingProbs) == 1, 'Probabilities do not sum to 1');

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

bool intsEqual(int, int) = operator ==;
Map_int_real naiveMap = NaiveMap_int_real(nan, isnan);
Map_int_real hashMap = HashMap_int_real(nan, isnan);
// Map_int_real smallintMap = makeMapSmallint(nan, isnan);

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
    return 'Different sizes: ' + string(aArray.length) + ' vs '
            + string(bArray.length);
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

int n = 2000;
int startDecreasing = n * 2 # 5;  // two-fifths of the way through
int maxKey = 100;
for (int i = 0; i < n; ++i) {
  real[] probs = i < startDecreasing ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  actions[choice](maxKey, naiveMap, hashMap);
  //write(naiveMap.size());
  if (naiveMap.size() != hashMap.size()) {
    write('Naive size: ' + (string)naiveMap.size() + ' Hash size: '
          + (string)hashMap.size());
    assert(false, 'Sizes do not match');
  }

  bool keyDifferenceFound = false;
  bool valueDifferenceFound = false;
  assert(naiveMap.iter != null, 'Naive set has no iter');
  assert(hashMap.iter != null, 'Hash set has no iter');
  for (var ita = naiveMap.iter(), itb = hashMap.iter();
       ita.valid() && itb.valid();
       ita.advance(), itb.advance())
  {
    wrapped_int a = ita.get();
    wrapped_int b = itb.get();
    if (!alias(a, b)) {
      keyDifferenceFound = true;
      break;
    }
    if (naiveMap[a] != hashMap[b]) {
      valueDifferenceFound = true;
      break;
    }
  }
  if (keyDifferenceFound) {
    assert(false, 'Naive vs hash: \n'
                  + differences((wrapped_int[])naiveMap, (wrapped_int[])hashMap)
          );
  }
  if (valueDifferenceFound) {
    write('value difference found');
    for (var ita = naiveMap.iter(), itb = hashMap.iter();
         ita.valid() && itb.valid();
         ita.advance(), itb.advance())
    {
      wrapped_int a = ita.get();
      wrapped_int b = itb.get();
      if (naiveMap[a] != hashMap[b]) {
        write('key: ' + (string)a.t + ' value: ' + (string)naiveMap[a] + ' '
              + (string)hashMap[b]);
      }
    }
    assert(false);
  }
}


EndTest();