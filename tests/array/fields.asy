import TestLib;

StartTest("fields");

{
  int[] z = {1, 2, 3};
  assert(z.length == 3);
  int[] keys = z.keys;
  assert(keys.length == 3);
  for (int i; i<3; ++i)
    assert(keys[i] == i);
  for (int j = 0; j < 10; ++j) {
    assert(z.cyclic == false);
    z.cyclic=true;
    assert(z.cyclic == true);
    z.cyclic=false;
  }
}

{
  int[] z = {2, 3, 5};
  for (int k = -100; k <= 100; ++k)
    assert(z.initialized(k) == (k >= 0 && k < 3));
}

{
  int[] z;
  for (int i=0; i<10; ++i) {
    for (int k = 0; k <= 100; ++k) {
      assert(z.length == k);
      z.push(k*k+3k+1);
      assert(z.length == k+1);
    }
    for (int k = 100; k >= 0; --k) {
      assert(z.length == k+1);
      assert(z.pop() == k*k+3k+1);
      assert(z.length == k);
    }
  }
  z.cyclic=true;
  for (int i=0; i<10; ++i) {
    for (int k = 0; k <= 100; ++k) {
      assert(z.length == k);
      z.push(k*k+3k+1);
      assert(z.length == k+1);
    }
    for (int k = 100; k >= 0; --k) {
      assert(z.length == k+1);
      z.delete(quotient(k,2));
      assert(z.length == k);
    }
  }
}

{
  int[] base={4,5,9,5,0,2,3};
  int[] z;
  for (int i=0; i<9; ++i) {
    assert(z.length == i*base.length);
    for (int j : z.keys)
      assert(z[j] == base[j%base.length]);
    z.append(base);
  }
}

{
  int[] z = {1,2,3,4,6,7,8,9};
  assert(z.length == 8);
  z.insert(4, 5);
  assert(z.length == 9);
  z.insert(0, 0);
  assert(z.length == 10);
  for (int i=0; i<10; ++i)
    assert(z[i] == i);
  z.insert(7, 100, 101, 102, 103);
  assert(z.length == 14);

  // TODO: Test inserting/deleting lengths more seriously.
}

{
  // Test extended for.
  int[] a = {1,4,6,2,7,4,8,9,1,3,-1};
  int i = 0;
  for (int x : a) {
    assert(x == a[i]);
    ++i;
  }
  assert(i == a.length);
}

{
  // Test extended for.
  int[] a = {1,4,6,2,7,4,8,9,1,3,-1};
  int i = 0;
  for (var x : a) {
    assert(x == a[i]);
    ++i;
  }
  assert(i == a.length);
}

EndTest();
