import TestLib;

StartTest('HashMap');

from genericpair(K=int, V=real) access
    Pair_K_V as Pair_int_real,
    operator >>;

from puremap(K=int, V=real) access
    Map_K_V as Map_int_real,
    operator cast,
    makeNaiveMap;

from hashmap(K=int, V=real) access
    makeHashMap;

from map_smallint(T=real) access
    makeMapSmallint;

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
  static restricted int POP = next();
  static restricted int GET_POP = next();

  static string toString(int action) {
    if (action == SIZE) return 'SIZE';
    if (action == EMPTY) return 'EMPTY';
    if (action == CONTAINS) return 'CONTAINS';
    if (action == FOR_EACH_CONTAINS) return 'FOR_EACH_CONTAINS';
    if (action == PUT) return 'PUT';
    if (action == POP) return 'POP';
    if (action == GET_POP) return 'GET_POP';
    return 'UNKNOWN';
  }
}

struct PutEnum {
  static restricted int num = 0;
  static private int next() {
    return ++num - 1;
  }
  static restricted int PAIR = next();
  static restricted int COMMA_SEPARATED = next();

  static int random() {
    return rand() % num;
  }
}

typedef void Action(int maxItem...Map_int_real[]);

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
    map.forEach(new bool(int key, real value) {
      for (Map_int_real map_ : maps) {
        assert(map_.contains(key));
        if (isnan(value))
          assert(isnan(map_.get(key)));
        else
          assert(map_.get(key) == value);
      }
      return true;
    });
  }
};
actions[ActionEnum.PUT] = new void(int maxItem ...Map_int_real[] maps) {
  int key = rand() % maxItem;
  real value = rand();
  real[] returned;
  for (Map_int_real map : maps) {
    if (PutEnum.random() == PutEnum.PAIR)
      returned.push(map.put(key >> value));
    else
      returned.push(map.put(key, value));
  }
  real reference = returned[0];
  for (real value : returned) {
    if (isnan(reference))
      assert(isnan(value));
    else
      assert(value == reference);
  }
};
actions[ActionEnum.POP] = new void(int maxItem ...Map_int_real[] maps) {
  int key = rand() % maxItem;
  real[] returned;
  for (Map_int_real map : maps) {
    returned.push(map.pop(key));
  }
  real reference = returned[0];
  for (real value : returned) {
    if (isnan(reference))
      assert(isnan(value));
    else
      assert(value == reference);
  }
};
actions[ActionEnum.GET_POP] = new void(int maxItem ...Map_int_real[] maps) {
  Map_int_real referenceMap = maps[rand() % maps.length];
  int size = referenceMap.size();
  if (size == 0)
    return;
  int index = rand() % size;
  Pair_int_real referencePair = ((Pair_int_real[])referenceMap)[index];
  unravel referencePair;
  for (Map_int_real map : maps) {
    real value = map.pop(k);
    if (isnan(v))
      assert(isnan(value));
    else
      assert(value == v);
  }
};

real[] increasingProbs = new real[ActionEnum.num];
increasingProbs[ActionEnum.SIZE] = 0.1;
increasingProbs[ActionEnum.EMPTY] = 0.1;
increasingProbs[ActionEnum.CONTAINS] = 0.1;
increasingProbs[ActionEnum.FOR_EACH_CONTAINS] = 0.05;
increasingProbs[ActionEnum.PUT] = 0.4;
increasingProbs[ActionEnum.POP] = 0.15;
increasingProbs[ActionEnum.GET_POP] = 0.1;
assert(sum(increasingProbs) == 1, 'Probabilities do not sum to 1');

real[] decreasingProbs = new real[ActionEnum.num];
decreasingProbs[ActionEnum.SIZE] = 0.1;
decreasingProbs[ActionEnum.EMPTY] = 0.1;
decreasingProbs[ActionEnum.CONTAINS] = 0.1;
decreasingProbs[ActionEnum.FOR_EACH_CONTAINS] = 0.05;
decreasingProbs[ActionEnum.PUT] = 0.1;
decreasingProbs[ActionEnum.POP] = 0.35;
decreasingProbs[ActionEnum.GET_POP] = 0.2;
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
Map_int_real pureMap = makeNaiveMap(intsEqual, nan);
Map_int_real hashMap = makeHashMap(hash, intsEqual, nan);
Map_int_real smallintMap = makeMapSmallint(nan, isnan);

for (int i = 0; i < 2000; ++i) {
  real[] probs = i < 800 ? increasingProbs : decreasingProbs;
  int choice = chooseAction(probs);
  actions[choice](100, pureMap, hashMap, smallintMap);
}


EndTest();