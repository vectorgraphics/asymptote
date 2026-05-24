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

StartTest("multislice");

{
  int[][][][][][][] x = new int[2][2][2][2][2][2][2];
  for (int i0=0; i0<2; ++i0)
    for (int i1=0; i1<2; ++i1)
      for (int i2=0; i2<2; ++i2)
        for (int i3=0; i3<2; ++i3)
          for (int i4=0; i4<2; ++i4)
            for (int i5=0; i5<2; ++i5)
              for (int i6=0; i6<2; ++i6)
                x[i0][i1][i2][i3][i4][i5][i6] = i0+i1+i2+i3+i4+i5+i6;

  int[][][][][][][] y = x[1:][1:][1:][1:][1:][1:][1:];
  assert(y.length == 1);
  assert(y[0].length == 1);
  assert(y[0][0].length == 1);
  assert(y[0][0][0].length == 1);
  assert(y[0][0][0][0].length == 1);
  assert(y[0][0][0][0][0].length == 1);
  assert(y[0][0][0][0][0][0].length == 1);
  assert(y[0][0][0][0][0][0][0] == 7);
  y[0][0][0][0][0][0][0] = 77;
  assert(x[1][1][1][1][1][1][1] == 7);  // Check that y is a copy, not a view.
}

{
  int[][] x = {{1,2,3},{4,5,6}};
  int[] y = x[:][1];
  assert(all(y == new int[] {2,5}));
  y[0] = 77;
  assert(x[0][1] == 2);  // Check that y is a copy, not a view.
}

// Comprehensive tests for intermixed slice + subscript indexers.
// Each indexer in a [..]-chain that begins with a slice consumes one axis of
// the source array.  Slices preserve that axis; subscripts collapse it.
// Slicing yields copy semantics throughout, unless the chain beginning with
// the slice has length 1.

{
  // Basic 2D: slice then subscript over each column.
  int[][] x = {{1,2,3},{4,5,6}};

  assert(all(x[:][0] == new int[] {1,4}));
  assert(all(x[:][1] == new int[] {2,5}));
  assert(all(x[:][2] == new int[] {3,6}));

  assert(all(x[0:1][1] == new int[] {2}));
  assert(all(x[1:][0] == new int[] {4}));
  assert(all(x[:1][2] == new int[] {3}));

  assert(x[0:0][0].length == 0);
  assert(x[1:1][2].length == 0);
}

{
  // Two-slice 2D yields a deep-copied 2D array.
  int[][] x = {{1,2,3},{4,5,6}};
  int[][] y = x[:][:];
  assert(y.length == 2);
  assert(y[0].length == 3);
  assert(y[1].length == 3);
  assert(all(y[0] == new int[] {1,2,3}));
  assert(all(y[1] == new int[] {4,5,6}));
  assert(!alias(y, x));
  assert(!alias(y[0], x[0]));
  assert(!alias(y[1], x[1]));
  y[0][0] = 77;
  assert(x[0][0] == 1);
}

{
  // Partial slices on both axes.
  int[][] x = {{1,2,3},{4,5,6}};
  int[][] y = x[1:][1:];
  assert(y.length == 1);
  assert(y[0].length == 2);
  assert(all(y[0] == new int[] {5,6}));
  y[0][0] = 77;
  assert(x[1][1] == 5);
}

{
  // Subscript-then-slice: parses as subscriptExp then a single-slice sliceExp.
  int[][] x = {{1,2,3},{4,5,6}};
  int[] y = x[1][1:];
  assert(all(y == new int[] {5,6}));
  y[0] = 77;
  assert(x[1][1] == 5);
}

{
  // 3D: a[i][j][k] = 100*i + 10*j + k, shape (3, 2, 4).
  int[][][] a = new int[3][2][4];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 4; ++k)
        a[i][j][k] = 100*i + 10*j + k;

  // [:][0][:] slices outer + inner, subscripts middle.
  int[][] r1 = a[:][0][:];
  assert(r1.length == 3);
  for (int i = 0; i < 3; ++i) {
    assert(r1[i].length == 4);
    for (int k = 0; k < 4; ++k)
      assert(r1[i][k] == 100*i + k);
  }
  // Copy semantics.
  r1[0][0] = 999;
  assert(a[0][0][0] == 0);

  // [:][:][0] slices outer two axes, subscripts innermost.
  int[][] r2 = a[:][:][0];
  assert(r2.length == 3);
  for (int i = 0; i < 3; ++i) {
    assert(r2[i].length == 2);
    for (int j = 0; j < 2; ++j)
      assert(r2[i][j] == 100*i + 10*j);
  }

  // a[1][:][0]: subscriptExp on axis 0 then sliceExp [:][0] on the rest.
  int[] r3 = a[1][:][0];
  assert(all(r3 == new int[] {100, 110}));
  r3[0] = 999;
  assert(a[1][0][0] == 100);

  // a[1:3][1][:].
  int[][] r4 = a[1:3][1][:];
  assert(r4.length == 2);
  for (int i = 0; i < 2; ++i) {
    assert(r4[i].length == 4);
    for (int k = 0; k < 4; ++k)
      assert(r4[i][k] == 100*(i+1) + 10 + k);
  }

  // a[:][:][1:3]: all slices, 3D result.
  int[][][] r5 = a[:][:][1:3];
  assert(r5.length == 3);
  for (int i = 0; i < 3; ++i) {
    assert(r5[i].length == 2);
    for (int j = 0; j < 2; ++j) {
      assert(r5[i][j].length == 2);
      for (int k = 0; k < 2; ++k)
        assert(r5[i][j][k] == 100*i + 10*j + (k+1));
    }
  }

  // a[:][1:2][0]: alternates slice/sub/slice/sub style at axes 0,1,2.
  // Wait, only 3 axes -- slice axis 0, slice axis 1 (range 1:2), subscript axis 2.
  int[][] r6 = a[:][1:2][0];
  assert(r6.length == 3);
  for (int i = 0; i < 3; ++i) {
    assert(r6[i].length == 1);
    assert(r6[i][0] == 100*i + 10);
  }
}

{
  // 4D mix: pattern slice, subscript, slice, subscript.
  int[][][][] a = new int[2][3][4][5];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        for (int l = 0; l < 5; ++l)
          a[i][j][k][l] = 1000*i + 100*j + 10*k + l;

  // [:][1][:][2]: slice axis 0, subscript axis 1, slice axis 2, subscript axis 3.
  // Result shape: (2, 4) over (i, k), value = 1000*i + 100*1 + 10*k + 2.
  int[][] r = a[:][1][:][2];
  assert(r.length == 2);
  for (int i = 0; i < 2; ++i) {
    assert(r[i].length == 4);
    for (int k = 0; k < 4; ++k)
      assert(r[i][k] == 1000*i + 100 + 10*k + 2);
  }
  r[0][0] = -1;
  assert(a[0][1][0][2] == 102);
}

{
  // Cyclic outer array: outer slice cycles, then subscript per element.
  int[][] x = {{1,2,3},{4,5,6}};
  x.cyclic = true;
  // x[0:4] -> x[0], x[1], x[0], x[1].
  int[] y = x[0:4][1];
  assert(all(y == new int[] {2,5,2,5}));
  // x[-1:1] -> x[1], x[0].
  int[] z = x[-1:1][0];
  assert(all(z == new int[] {4,1}));
  // x[9:12] also cycles.
  int[] w = x[9:12][2];
  assert(all(w == new int[] {6,3,6}));
}

{
  // Cyclic inner arrays: subscript uses cyclic indexing.
  int[][] x = {{1,2,3},{4,5,6}};
  x[0].cyclic = true;
  x[1].cyclic = true;
  // 4 mod 3 = 1.
  int[] y = x[:][4];
  assert(all(y == new int[] {2,5}));
  // -1 mod 3 = 2.
  int[] z = x[:][-1];
  assert(all(z == new int[] {3,6}));
}

{
  // Side-effect ordering: array first, then each indexer's expressions left
  // to right.  For a slice, left is evaluated before right.
  int state = 0;
  int[][] data = {{1,2,3},{4,5,6}};

  int[][] arr() {
    assert(state == 0);
    ++state;
    return data;
  }

  int l() {
    assert(state == 1);
    ++state;
    return 0;
  }

  int r() {
    assert(state == 2);
    ++state;
    return 2;
  }

  int i() {
    assert(state == 3);
    ++state;
    return 1;
  }

  int[] y = arr()[l():r()][i()];
  assert(state == 4);
  assert(all(y == new int[] {2, 5}));
}

{
  // Parenthesizing a single slice restores old semantics for the trailing
  // subscript.  (x[:]) is a copy of x; (x[:])[1] is the second row of that
  // copy.
  int[][] x = {{1,2,3},{4,5,6}};
  int[] y = (x[:])[1];
  assert(all(y == new int[] {4,5,6}));
}

{
  // The original 2D test the user wrote, kept for regression.
  int[][] x = {{1,2,3},{4,5,6}};
  int[] y = x[:][1];
  assert(all(y == new int[] {2,5}));
  y[0] = 77;
  assert(x[0][1] == 2);
}

EndTest();
