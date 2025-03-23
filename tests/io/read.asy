import TestLib;

StartTest("read1");
{
  file fin=input("io/data1.txt").read(1);
  real[] a=fin;
  assert(a.length == 10);
  int c=-1;
  for(int i=0; i < a.length; ++i)
    assert(a[i] == ++c);
  seek(fin,0);
  real[] b=fin.line();
  assert(b[0] == a.length);
  real[] b=fin.line();
  c=-1;
  for(int i=0; i < b.length; ++i)
    assert(b[i] == ++c);
  seek(fin,0);
  real nx=fin;
  real[] b=fin.dimension(5);
  c=-1;
  for(int i=0; i < 5; ++i)
    assert(b[i] == ++c);
}
EndTest();

StartTest("read2");
{
  file fin=input("io/data2.txt").read(2);
  real[][] a=fin;
  assert(a.length == 2);
  assert(a[0].length == 3);
  int c=-1;
  for(int i=0; i < a.length; ++i)
    for(int j=0; j < a[0].length; ++j)
      assert(a[i][j] == ++c);
  seek(fin,0);
   real nx=fin;
  real ny=fin;
  c=-1;
 real[][] b=fin.dimension(a.length,a[0].length);
  for(int i=0; i < a.length; ++i)
    for(int j=0; j < a[0].length; ++j)
      assert(b[i][j] == ++c);
}
EndTest();

StartTest("read3");
{
  file fin=input("io/data3.txt").read(3);
  real[][][] a=fin;
  assert(a.length == 2);
  assert(a[0].length == 3);
  assert(a[0][0].length == 4);
  int c=-1;
  for(int i=0; i < a.length; ++i)
    for(int j=0; j < a[0].length; ++j)
      for(int k=0; k < a[0][0].length; ++k)
        assert(a[i][j][k] == ++c);
  seek(fin,0);
  real nx=fin;
  real ny=fin;
  real nz=fin;
  real[][][] b=fin.dimension(a.length,a[0].length,a[0][0].length);
  c=-1;
  for(int i=0; i < a.length; ++i)
    for(int j=0; j < a[0].length; ++j)
      for(int k=0; k < a[0][0].length; ++k)
        assert(b[i][j][k] == ++c);
}
EndTest();
