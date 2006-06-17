import TestLib;

StartTest("solve");
real[][] a=new real[3][3];
a[0][0]=1; a[0][1]=1; a[0][2]=1; 
a[1][0]=1; a[1][1]=2; a[1][2]=2;
a[2][0]=0; a[2][1]=0; a[2][2]=1;

real[][] b=new real[3][2];
b[0][0]=3; b[1][0]=5; b[2][0]=1;
b[0][1]=9; b[1][1]=11; b[2][1]=3;

real[][] x=new real[3][2];
x[0][0]=1; x[1][0]=1; x[2][0]=1;
x[0][1]=7; x[1][1]=-1; x[2][1]=3;

real[][] c=solve(a,b);
for(int i=0; i < 3; ++i)
  for(int j=0; j < 2; ++j)
    assert(close(c[i][j],x[i][j]));


real[][] ai=new real[3][3];
ai[0][0]=2; ai[0][1]=-1; ai[0][2]=0; 
ai[1][0]=-1; ai[1][1]=1; ai[1][2]=-1;
ai[2][0]=0; ai[2][1]=0; ai[2][2]=1;
EndTest();

StartTest("inverse");
real[][] d=inverse(a);
for(int i=0; i < 3; ++i)
  for(int j=0; j < 3; ++j)
    assert(close(d[i][j],ai[i][j]));
EndTest();
