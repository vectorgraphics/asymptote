import TestLib;
import math;

StartTest("transpose");
int n=3;
real[][] a=new real[n][n]; 
real[][] b=new real[n][n]; 
for(int i=0; i < n; ++i) {
  for(int j=0; j < n; ++j) {
    a[i][j]=b[j][i]=rand();
  }
}

bool operator == (real[][] a, real[][] b)
{
  int n=a.length;
  for(int i=0; i < n; ++i)
    if(!all(a[i] == b[i])) return false;
  return true;
}

bool operator == (real[][][] a, real[][][] b)
{
  int n=a.length;
  for(int i=0; i < n; ++i) {
    real[][] ai=a[i];
    real[][] bi=b[i];
    int m=ai.length;
    for(int j=0; j < m; ++j) {
      if(!all(ai[j] == bi[j])) return false;
    }
  }
  return true;
}

assert(a == transpose(b));

int n=3;
real[][][] a=new real[n][n][n]; 
real[][][] b=new real[n][n][n]; 
real[][][] c=new real[n][n][n]; 
real[][][] d=new real[n][n][n]; 
for(int i=0; i < n; ++i) {
  for(int j=0; j < n; ++j) {
    for(int k=0; k < n; ++k) {
      a[i][j][k]=b[j][i][k]=c[i][k][j]=d[k][j][i]=rand();
    }
  }
}

assert(a == transpose(b,new int[] {1,0,2}));
assert(a == transpose(c,new int[] {0,2,1}));
assert(a == transpose(d,new int[] {2,1,0}));

EndTest();
