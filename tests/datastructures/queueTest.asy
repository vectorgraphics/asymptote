import TestLib;

StartTest("Queue");

from queue(T=int) access
    Queue_T as Queue_int,
    makeNaiveQueue,
    makeArrayQueue,
    makeLinkedQueue,
    makeQueue,
    operator cast;


struct ActionEnum {
  static restricted int numActions = 0;
  static private int next() {
    return ++numActions - 1;
  }
  static restricted int PUSH = next();
  static restricted int POP = next();
}

from zip(T=int) access zip;

string differences(Queue_int a, Queue_int b) {
  if (a.size() != b.size()) {
    return 'Different sizes: ' + string(a.size()) + ' vs ' + string(b.size());
  }
  if (a.size() != 0) {
    if (a.peek() != b.peek()) {
      return 'Different peek: ' + string(a.peek()) + ' vs ' + string(b.peek());
    }
  }
  if (!all(a.toArray() == b.toArray())) {
    write(zip(a.toArray(), b.toArray()));
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

typedef void Action(...Queue_int[]);

Action[] actions = new Action[ActionEnum.numActions];
actions[ActionEnum.PUSH] = new void(...Queue_int[] qs) {
  int toPush = rand();
  for (Queue_int q : qs) {
    q.push(toPush);
  }
};
actions[ActionEnum.POP] = new void(...Queue_int[] qs) {
  int[] results = new int[];
  for (Queue_int q : qs) {
    if (q.size() > 0) {
      results.push(q.pop());
    }
  }
  if (results.length > 0) {
    int expected = results[0];
    for (int r : results) {
      assert(r == expected, 'Different results: ' + string(results));
    }
  }
};

real[] increasingProbs = new real[ActionEnum.numActions];
increasingProbs[ActionEnum.PUSH] = 0.7;
increasingProbs[ActionEnum.POP] = 0.3;

real[] decreasingProbs = new real[ActionEnum.numActions];
decreasingProbs[ActionEnum.PUSH] = 0.3;
decreasingProbs[ActionEnum.POP] = 0.7;

Queue_int naive = makeNaiveQueue(new int[]);
Queue_int array = makeArrayQueue(new int[]);
Queue_int linked = makeLinkedQueue(new int[]);

for (int i = 0; i < 2000; ++i) {
  // if (i % 100 == 0) {
  //   write('Step ' + string(i));
  //   write('Naive: ' + string(naive.toArray()));
  //   write('Array: ' + string(array.toArray()));
  //   write('Linked: ' + string(linked.toArray()));
  // }
  real[] probs = i < 800 ? increasingProbs : decreasingProbs;
  int choice = (unitrand() < probs[ActionEnum.PUSH]
                ? ActionEnum.PUSH
                : ActionEnum.POP);
  actions[choice](naive, array, linked);
  string diffs = differences(naive, array);
  assert(diffs == '', 'Naive vs array: \n' + diffs);
  diffs = differences(naive, linked);
  assert(diffs == '', 'Naive vs linked: \n' + diffs);
}

EndTest();