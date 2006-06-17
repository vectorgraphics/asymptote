import TestLib;

StartTest("solve");
real[][] a=new real[][] {{1,1,1},{1,2,2},{0,0,1}};
real[][] b=new real[][] {{3,9},{5,11},{1,3}};
real[][] x=new real[][] {{1,7},{1,-1,},{1,3}};

real[][] c=solve(a,b);
for(int i=0; i < 3; ++i)
  for(int j=0; j < 2; ++j)
    assert(close(c[i][j],x[i][j]));
EndTest();

StartTest("inverse");
real[][] ai=new real[][] {{2,-1,0},{-1,1,-1},{0,0,1}};
real[][] d=inverse(a);
for(int i=0; i < 3; ++i)
  for(int j=0; j < 3; ++j)
    assert(close(d[i][j],ai[i][j]));
EndTest();
