import TestLib;

StartTest('zip_arrays');
{
  from collections.zip(T=int) access zip;
  int[] a = {1, 2, 3};
  int[] b = {4, 5, 6};
  int[] c = {7, 8, 9, 10};
  {
    int i = 0;
    for (int[] xy : zip(a, b)) {
      assert(xy[0] == a[i]);
      assert(xy[1] == b[i]);
      ++i;
    }
    assert(i == 3);
  }
  {
    int i = 0;
    for (int[] xyz : zip(a, b, c)) {
      assert(xyz[0] == a[i]);
      assert(xyz[1] == b[i]);
      assert(xyz[2] == c[i]);
      ++i;
    }
    assert(i == 3);
  }
  {
    int i = 0;
    for (int[] xyz : zip(default=427, a, b, c)) {
      assert(xyz[0] == (i < 3 ? a[i] : 427));
      assert(xyz[1] == (i < 3 ? b[i] : 427));
      assert(xyz[2] == c[i]);
      ++i;
    }
    assert(i == 4);
  }
}
EndTest();

StartTest('zip_iterables');
{
  from collections.iter(T=int) access range;
  from collections.zip(T=int) access zip;
  int[] a = {1, 2, 3};
  int[] b = {4, 5, 6};
  int[] c = {7, 8, 9, 10};
  var A = range(a);
  var B = range(b);
  var C = range(c);
  {
    int i = 0;
    for (int[] xy : zip(A, B)) {
      assert(xy[0] == a[i]);
      assert(xy[1] == b[i]);
      ++i;
    }
    assert(i == 3);
  }
  {
    int i = 0;
    for (int[] xyz : zip(A, B, C)) {
      assert(xyz[0] == a[i]);
      assert(xyz[1] == b[i]);
      assert(xyz[2] == c[i]);
      ++i;
    }
    assert(i == 3);
  }
  {
    int i = 0;
    for (int[] xyz : zip(default=427, A, B, C)) {
      assert(xyz[0] == (i < 3 ? a[i] : 427));
      assert(xyz[1] == (i < 3 ? b[i] : 427));
      assert(xyz[2] == c[i]);
      ++i;
    }
    assert(i == 4);
  }
}
EndTest();

StartTest('zip_heterogeneous');
{
  from collections.zip2(K=int, V=string) access zip, makePair;
  from collections.iter(T=int) access range;
  from collections.iter(T=string) access range;
  int[] a = {1, 2, 3};
  string[] b = {'one', 'two', 'three', 'four'};
  int[] c = {1, 2, 3, 4, 5};
  var A = range(a);
  var B = range(b);
  var C = range(c);
  {
    int i = 0;
    for (var kv : zip(A, B)) {
      assert(kv.k == a[i]);
      assert(kv.v == b[i]);
      ++i;
    }
    assert(i == 3);
  }
  {
    int i = 0;
    for (var kv : zip(default=makePair(-1, ''), A, B)) {
      assert(kv.k == (i < 3 ? a[i] : -1));
      assert(kv.v == b[i]);
      ++i;
    }
    assert(i == 4);
  }
  {
    int i = 0;
    for (var kv : zip(C, B)) {
      assert(kv.k == c[i]);
      assert(kv.v == b[i]);
      ++i;
    }
    assert(i == 4);
  }
  {
    int i = 0;
    for (var kv : zip(default=makePair(-1, ''), C, B)) {
      assert(kv.k == c[i]);
      assert(kv.v == (i < 4 ? b[i] : ''));
      ++i;
    }
    assert(i == 5);
  }
}
EndTest();