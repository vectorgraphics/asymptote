// Slice errors (exp.h, exp.cc)

{
  // Not an error: omitted endpoints are allowed for a single slice.
  int[] a = new int[] {0,1,2};
  a[:];
}
{
  // expression to slice must be an array.
  int x = 0;
  x[:];
}
{
  // left bound of a single slice must have type int.
  int[] a = new int[] {0,1,2};
  a["bad":];
}
{
  // right bound of a single slice must have type int.
  int[] a = new int[] {0,1,2};
  a[:"bad"];
}
{
  // slice assignment requires an array value on the right-hand side.
  int[] a = new int[] {0,1,2};
  a[:] = 1;
}
{
  // slice assignment preserves the array element type.
  int[] a = new int[] {0,1,2};
  a[:] = new string[] {"bad"};
}
{
  // Not an error: a multi-slice read is allowed when the array is deep enough.
  int[][] a = new int[2][2];
  a[:][:];
}
{
  // too many slice dimensions for the array.
  int[] a = new int[] {0,1,2};
  a[:][:];
}
{
  // a later multi-slice bound must also have type int.
  int[][] a = new int[2][2];
  a[:][:"bad"];
}
{
  // too many slice dimensions are rejected even after valid earlier slices.
  int[][] a = new int[2][2];
  a[:][:][:];
}
{
  // cannot assign through a multi-dimensional slice.
  int[][] a = new int[2][2];
  a[:][:] = new int[0][0];
}
{
  // A scalar RHS is rejected by normal assignment typing before slice-write checks.
  int[][] a = new int[2][2];
  a[:][:] = 0;
}
{
  // intermixed indexers: too many slice/subscript dimensions for the array.
  int[] a = new int[] {0,1,2};
  a[:][0];
}
{
  // intermixed indexers: subscript after slice must have type int.
  int[][] a = new int[2][2];
  a[:]["bad"];
}
{
  // intermixed indexers: too many dimensions after slice+subscript chain.
  int[][] a = new int[2][2];
  a[:][0][:];
}
{
  // cannot assign through a multi-dimensional slice that mixes subscripts.
  int[][][] a = new int[2][2][2];
  a[:][0][:] = new int[0][0];
}