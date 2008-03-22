import TestLib;

StartTest("array");

{
  int[] x=array(10, 7);
  assert(x.length == 10);
  for (int i=0; i<x.length; ++i)
    assert(x[i] == 7);
}
{
  int[][] y=array(10, array(10, 7));
  assert(y.length == 10);
  for (int i=0; i<y.length; ++i) {
    assert(y[i].length == 10);
    for (int j=0; j<y[i].length; ++j)
      assert(y[i][j] == 7);
  }
}
{
  int[][] y=array(10, array(10, 7));
  y[4][5] = 9;
  assert(y.length == 10);
  for (int i=0; i<y.length; ++i) {
    assert(y[i].length == 10);
    for (int j=0; j<y[i].length; ++j)
      if (i==4 && j==5)
        assert(y[i][j] == 9);
      else
        assert(y[i][j] == 7);
  }
}
{
  int[][] y=array(10, array(10, 7), depth=0);
  y[4][5] = 9;
  assert(y.length == 10);
  for (int i=0; i<y.length; ++i) {
    assert(y[i].length == 10);
    for (int j=0; j<y[i].length; ++j)
      if (j==5)
        assert(y[i][j] == 9);
      else
        assert(y[i][j] == 7);
  }
}



EndTest();
