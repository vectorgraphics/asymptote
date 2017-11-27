import math;

int OPTIMAL=0;
int UNBOUNDED=1;
int INFEASIBLE=2;

struct solution {

  int type;
  real[] x;
  real cost;
}

int m,n;

// Row reduce based on pivot E[I][J]
void rowreduce(real[][] E, int N, int I, int J)
{
  real[] EI=E[I];
  real v=EI[J];
  for(int j=0; j < J; ++j) EI[j] /= v;
  EI[J]=1;
  for(int j=J+1; j <= N; ++j) EI[j] /= v;

  for(int i=0; i < I; ++i) {
    real EiJ=E[i][J];
    for(int j=0; j <= N; ++j)
      E[i][j] -= EI[j]*EiJ;
  }
  for(int i=I+1; i <= m; ++i) {
    real EiJ=E[i][J];
    for(int j=0; j <= N; ++j)
      E[i][j] -= EI[j]*EiJ;
  }
}

int[] Bindices;
int J;

solution Solution;

void iterate(real[][] E, int N)
{
  while(true) {
    // Find first negative entry in bottom row
    for(J=0; J < N; ++J)
      if(E[m][J] < 0) break;

    if(J == N)
      return;

    int I=-1;
    real M=inf;
    for(int i=0; i < m; ++i) {
      real e=E[i][J];
      if(e > 0) {
        real v=E[i][N]/e;
        if(v < M) {M=v; I=i;}
      }
    }
    if(I == -1) {
      Solution.type=UNBOUNDED; // Can only happen in Phase 2.
      return;
    }

    Bindices[I]=J;

    // Generate new tableau
    rowreduce(E,N,I,J);

    //    write();
    //    write(E);
  }
}


// Try to find a solution x to Ax=b that minimizes the cost c^T x.
// A is an m x n matrix
solution simplex(real[] c, real[][] A, real[] b)
{
  
  // Phase 1    
  //  write(A);
  assert(rectangular(A));
  assert(all(b >= 0));
  
  m=A.length;
  n=A[0].length;
  
  real[][] E=new real[m+1][n+m+1];

  for(int j=0; j < n; ++j) {
    real sum=0;
    for(int i=0; i < m; ++i) { 
      real Aij=A[i][j];
      E[i][j]=Aij;
      sum += Aij;
    }
    E[m][j]=-sum;
  }

  for(int j=0; j < m; ++j) {
    E[m][n+j]=0;
    for(int i=0; i < m; ++i) { 
      E[i][n+j]=i == j ? 1 : 0;
    }
  }

  for(int i=0; i < m; ++i) { 
    E[i][n+m]=b[i];
  }
  E[m][n+m]=-sum(b);

  //  write(E);
  
  Bindices=sequence(n,n+m-1);
  iterate(E,n+m);

  if(E[m][J] != 0) {
    Solution.type=INFEASIBLE;
    return Solution;
  }

  //  write("Done with Phase 1");
  //  write("Bindices:",Bindices);

  real[][] D=new real[m+1][n+1];

  real[] cb=new real[m];

  int ip=0; // reduced i
  for(int i=0; i < m; ++i) {
    int k=Bindices[i];
    if(k >= n) continue;
    Bindices[ip]=k; 
    cb[ip]=c[k];
    for(int j=0; j < n; ++j)
      D[ip][j]=E[i][j];
    D[ip][n]=E[i][n+m];
    ++ip;
  }

  for(int j=0; j < n; ++j)
    D[ip][j]=E[m][j];
  D[ip][n]=E[m][n+m];

  m=ip;
  //  write("Reduced Bindices:",Bindices[0:m]);

  for(int j=0; j < n; ++j) {
    real sum=0;
    for(int k=0; k < m; ++k)
      sum += cb[k]*D[k][j];
    D[m][j]=c[j]-sum;
  }
  
  real sum=0;
  for(int k=0; k < m; ++k)
    sum += cb[k]*D[k][n];
  D[m][n]=-sum;

  //  write();
  //  write(D);

  iterate(D,n);

  if(Solution.type == UNBOUNDED)
    return Solution;

  for(int j=0; j < n; ++j)
    Solution.x[j]=0;

  for(int k=0; k < m; ++k) {
    int i=Bindices[k];
    Solution.x[i]=D[k][n];
  }

  Solution.cost=-D[m][n];

  Solution.type=OPTIMAL;
  return Solution;
}

// Try to find a solution x to sgn(Ax-b)=sgn(s) that minimizes the cost c^T x.
solution simplex(real[] c, real[][] A, int[] s, real[] b)
{
  int m=A.length;
  int n=A[0].length;

  int count=0;
  for(int i=0; i < m; ++i)
    if(s[i] != 0) ++count;

  real[][] a=new real[m][n+count];

  for(int i=0; i < m; ++i) {
    for(int j=0; j < n; ++j) {
      a[i][j]=A[i][j];
    }
  }
  
  int k=0;

  for(int i=0; i < m; ++i) {
    for(int j=0; j < k; ++j)
      a[i][n+j]=0;
    if(k < count)
      a[i][n+k]=-s[i];
    for(int j=k+1; j < count; ++j)
      a[i][n+j]=0;
    if(s[i] != 0) ++k;
  }

  solution S=simplex(concat(c,array(count,0.0)),a,b);
  S.x.delete(n,n+count-1);
  return S;
}

/*
solution S=simplex(new real[] {4,1,1},
                   new real[][] {{2,1,2},{3,3,1}},
                   new real[] {4,3});
*/


/*
solution S=simplex(new real[] {2,6,1,1},
                   new real[][] {{1,2,0,1},{1,2,1,1},{1,3,-1,2},{1,1,1,0}},
                   new real[] {6,7,7,5});
*/


/*
solution S=simplex(new real[] {-10,-12,-12,0,0,0},
                   new real[][] {{1,2,2,1,0,0},
                                 {2,1,2,0,1,0},
                                 {2,2,1,0,0,1}},
                   new real[] {20,20,20});
write();
write("x:",S.x);
write("Cost=",S.cost);
*/

/*
solution S=simplex(new real[] {-10,-12,-12},
                   new real[][] {{1,2,2},
                                 {2,1,2},
                                 {2,2,1}},
                   new int[] {0,0,-1},
                   new real[] {20,20,20});
write();
write("x:",S.x);
write("Cost=",S.cost);
*/

 /*

solution S=simplex(new real[] {1,1,1,0},
                   new real[][] {{1,2,3,0},
                                 {-1,2,6,0},
                                 {0,4,9,0},
                                 {0,0,3,1}},
                   new real[] {3,2,5,1});

write();
write("x:",S.x);
write("Cost=",S.cost);
 */

