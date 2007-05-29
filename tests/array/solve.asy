import TestLib;

StartTest("solve");
real[][] a=new real[][] {{1,1,1},{1,2,2},{0,0,1}};
real[][] b=new real[][] {{3,9},{5,11},{1,3}};
real[][] x=new real[][] {{1,7},{1,-1,},{1,3}};
real[][] c=solve(a,b);
for(int i=0; i < c.length; ++i)
  for(int j=0; j < c[i].length; ++j)
    assert(close(c[i][j],x[i][j]));

real[][] a={{1,-2,3,0},{4,-5,6,2},{-7,-8,10,5},{1,50,1,-2}};
real[] b={7,19,33,3};
real[] x=solve(a,b);
real[] c=a*x;
for(int i=0; i < c.length; ++i)
  assert(close(c[i],b[i]));
EndTest();

StartTest("inverse");
real[][] a=new real[][] {{1,1,1},{1,2,2},{0,0,1}};
real[][] ainverse=new real[][] {{2,-1,0},{-1,1,-1},{0,0,1}};
real[][] d=inverse(a);
real[][] l=d*a;
real[][] r=a*d;
real[][] I=identity(a.length);
for(int i=0; i < d.length; ++i) {
  for(int j=0; j < d[i].length; ++j) {
    assert(close(d[i][j],ainverse[i][j]));
    assert(I[i][j] == (i == j ? 1 : 0));
    assert(close(l[i][j],I[i][j]));
    assert(close(r[i][j],I[i][j]));
  }
}
EndTest();
