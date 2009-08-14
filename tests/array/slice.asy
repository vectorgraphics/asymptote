import TestLib;

StartTest("slice");

int[] x={0,1,2,3,4,5,6,7,8,9};

// Non-cyclic cases.
assert(all(x[:] == x));
assert(!alias(x[:],x));

assert(all(x[0:4] == new int[] {0,1,2,3} ));
assert(all(x[2:4] == new int[] {2,3} ));

assert(all(x[5:] == new int[] {5,6,7,8,9} ));
assert(all(x[:5] == new int[] {0,1,2,3,4} ));

assert(all(x[3:3] == new int[] {} ));
assert(all(x[3:4] == new int[] {3} ));
assert(all(x[98:99] == new int[] {} ));

assert(x[:].cyclic == false);
assert(x[2:].cyclic == false);
assert(x[:7].cyclic == false);
assert(x[3:3].cyclic == false);
assert(x[2:9].cyclic == false);

// Cyclic cases
x.cyclic=true;

assert(all(x[:] == new int[] {0,1,2,3,4,5,6,7,8,9} ));
assert(all(x[0:4] == new int[] {0,1,2,3} ));
assert(all(x[2:4] == new int[] {2,3} ));

assert(all(x[5:] == new int[] {5,6,7,8,9} ));
assert(all(x[-5:] == new int[] {5,6,7,8,9,0,1,2,3,4,5,6,7,8,9} ));
assert(all(x[:5] == new int[] {0,1,2,3,4} ));

assert(all(x[3:3] == new int[] {} ));
assert(all(x[3:4] == new int[] {3} ));

assert(all(x[-1:1] == new int[] {9,0} ));
assert(all(x[9:11] == new int[] {9,0} ));
assert(all(x[9:21] == new int[] {9,0,1,2,3,4,5,6,7,8,9,0} ));
assert(all(x[-15:15] == new int[] {5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9,0,1,2,3,4}));
assert(all(x[6728:6729] == new int[] {8} ));
assert(all(x[-6729:-6728] == new int[] {1} ));

assert(x[:].cyclic == false);
assert(x[2:].cyclic == false);
assert(x[:7].cyclic == false);
assert(x[3:3].cyclic == false);
assert(x[2:9].cyclic == false);
assert(x[5:100].cyclic == false);

pair[] z={(1,2), (3,4), (5,6)};
assert(all(z[1:1] == new pair[] {}));
assert(all(z[:1] == new pair[] {(1,2)}));
assert(all(z[1:] == new pair[] {(3,4), (5,6)}));
assert(all(z[:] == z));
assert(all(z[1:2] == new pair[] {(3,4)}));

// Writing tests.
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  int[] z={56,67,78};

  y[:] = z;
  assert(all(y == z));
  assert(!alias(y,z));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  int[] z={56,67,78};
  z.cyclic=true;

  y[:] = z;
  assert(all(y == z));
  assert(!alias(y,z));
  assert(y.cyclic == false);
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};

  y[2:3] = y[5:6] = new int[] {77};
  assert(all(y == new int[] {0,1,77,3,4,77,6,7,8,9}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};

  y[:3] = y[7:] = new int[] {};
  assert(all(y == new int[] {3,4,5,6}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};

  y[3:5] = new int[] {13,14,15,16,17};
  assert(all(y == new int[] {0,1,2,13,14,15,16,17,5,6,7,8,9}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;
  int[] z={56,67,78};

  y[:] = z;
  assert(all(y == z));
  assert(!alias(y,z));
  assert(y.cyclic == true);
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;
  int[] z={56,67,78};
  z.cyclic=true;

  y[:] = z;
  assert(all(y == z));
  assert(!alias(y,z));
  assert(y.cyclic == true);
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[2:3] = y[5:6] = new int[] {77};
  assert(all(y == new int[] {0,1,77,3,4,77,6,7,8,9}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[:3] = y[7:] = new int[] {};
  assert(all(y == new int[] {3,4,5,6}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[8:] = new int[] {18,19,20,21,22};
  assert(all(y == new int[] {0,1,2,3,4,5,6,7,18,19,20,21,22}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[-2:0] = new int[] {18,19,20,21,22};
  assert(all(y == new int[] {0,1,2,3,4,5,6,7,18,19,20,21,22}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[18:20] = new int[] {18,19,20,21,22};
  assert(all(y == new int[] {0,1,2,3,4,5,6,7,18,19,20,21,22}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[3:5] = new int[] {13,14,15,16,17};
  assert(all(y == new int[] {0,1,2,13,14,15,16,17,5,6,7,8,9}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[13:15] = new int[] {13,14,15,16,17};
  assert(all(y == new int[] {0,1,2,13,14,15,16,17,5,6,7,8,9}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[3-10:5-10] = new int[] {13,14,15,16,17};
  assert(all(y == new int[] {0,1,2,13,14,15,16,17,5,6,7,8,9}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[8:12] = new int[] {18,19,20,21};
  assert(all(y == new int[] {20,21,2,3,4,5,6,7,18,19}));
}
{
  int[] y={0,1,2,3,4,5,6,7,8,9};
  y.cyclic=true;

  y[-2:2] = new int[] {18,19,20,21};
  assert(all(y == new int[] {20,21,2,3,4,5,6,7,18,19}));
}

// Side Effect Test
{
  int state=0;
  int[] x={0,1,2,3,4,5,6,7,8,9};

  int[] a() {
    assert(state==0);
    ++state;

    return x;
  }

  int l() {
    assert(state==1);
    ++state;

    return 2;
  }

  int r() {
    assert(state==2);
    ++state;

    return 6;
  }

  int[] b() {
    assert(state==3);
    ++state;

    return new int[] {77,77};
  }

  assert(state==0);
  a()[l():r()]=b();
  assert(state==4);
  assert(all(x == new int[] {0,1,77,77,6,7,8,9}));
}


EndTest();
