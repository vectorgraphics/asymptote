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

// Shouldn't this be builtin?
int[][] transpose(int[][] a) {
  int n = a.length;
  int m = a[0].length;
  int[][] b = new int[m][n];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      b[j][i] = a[i][j];
    }
  }
  return b;
}

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
    write(transpose(new int[][]{a.toArray(), b.toArray()}));
    return 'Different contents';
  }
  return '';
}

typedef void Action(Queue_int);

var actions = new Action[ActionEnum.numActions];