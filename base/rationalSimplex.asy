import math;
import rational;

struct simplex {
  static int OPTIMAL=0;
  static int UNBOUNDED=1;
  static int INFEASIBLE=2;

  int case;
  rational[] x;
  rational cost;

  int m,n;
  int[] Bindices;
  int J;

  // Row reduce based on pivot E[I][J]
  void rowreduce(rational[][] E, int N, int I, int J) {
    rational[] EI=E[I];
    rational v=EI[J];
    for(int j=0; j < J; ++j) EI[j] /= v;
    EI[J]=1;
    for(int j=J+1; j <= N; ++j) EI[j] /= v;

    for(int i=0; i < I; ++i) {
      rational[] Ei=E[i];
      rational EiJ=Ei[J];
      for(int j=0; j < J; ++j)
        Ei[j] -= EI[j]*EiJ;
      Ei[J]=0;
      for(int j=J+1; j <= N; ++j)
        Ei[j] -= EI[j]*EiJ;
    }
    for(int i=I+1; i <= m; ++i) {
      rational[] Ei=E[i];
      rational EiJ=Ei[J];
      for(int j=0; j < J; ++j)
        Ei[j] -= EI[j]*EiJ;
      Ei[J]=0;
      for(int j=J+1; j <= N; ++j)
        Ei[j] -= EI[j]*EiJ;
    }
  }

  int iterate(rational[][] E, int N) {
    while(true) {
      // Find first negative entry in bottom (reduced cost) row
      for(J=0; J < N; ++J)
        if(E[m][J] < 0) break;

      if(J == N)
        return 0;

      int I=-1;
      rational M;
      for(int i=0; i < m; ++i) {
        rational e=E[i][J];
        if(e > 0) {
          M=E[i][N]/e;
          I=i;
          break;
        }
      }
      for(int i=I+1; i < m; ++i) {
        rational e=E[i][J];
        if(e > 0) {
          rational v=E[i][N]/e;
          if(v < M) {M=v; I=i;}
        }
      }
      if(I == -1)
        return UNBOUNDED; // Can only happen in Phase 2.

      Bindices[I]=J;

      // Generate new tableau
      rowreduce(E,N,I,J);
    }
    return 0;
  }

  void operator init() {}
    
  // Try to find a solution x to Ax=b that minimizes the cost c^T x.
  // A is an m x n matrix
  void operator init(rational[] c, rational[][] A, rational[] b) {
  
    // Phase 1    
    //  write(A);
    assert(rectangular(A));
    //  assert(all(b >= 0));
  
    m=A.length;
    n=A[0].length;
  
    rational[][] E=new rational[m+1][n+m+1];

    for(int j=0; j < n; ++j) {
      rational sum=0;
      for(int i=0; i < m; ++i) { 
        rational Aij=A[i][j];
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
  
    Bindices=sequence(n,n+m-1);
    iterate(E,n+m);
  
    if(abs(E[m][J]) > 0) {
    case=INFEASIBLE;
    return;
    }

    write("Done with Phase 1");
    //  write("Bindices:",Bindices);

    rational[][] D=new rational[m+1][n+1];
    rational[] cb=new rational[m];

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
      rational sum=0;
      for(int k=0; k < m; ++k)
        sum += cb[k]*D[k][j];
      D[m][j]=c[j]-sum;
    }
  
    rational sum=0;
    for(int k=0; k < m; ++k)
      sum += cb[k]*D[k][n];
    D[m][n]=-sum;

    //  write();
    //  write(D);

    if(iterate(D,n) == UNBOUNDED) {
    case=UNBOUNDED;
    return;
    }

    for(int j=0; j < n; ++j)
      x[j]=0;

    for(int k=0; k < m; ++k) {
      int i=Bindices[k];
      x[i]=D[k][n];
    }

    cost=-D[m][n];
    case=OPTIMAL;
  }

  // Try to find a solution x to sgn(Ax-b)=sgn(s) that minimizes the cost
  // c^T x.
  void operator init(rational[] c, rational[][] A, int[] s, rational[] b) {
    int m=A.length;
    int n=A[0].length;

    int count=0;
    for(int i=0; i < m; ++i)
      if(s[i] != 0) ++count;

    rational[][] a=new rational[m][n+count];

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

    operator init(concat(c,array(count,rational(0))),a,b);
    if(case == OPTIMAL)
      x.delete(n,n+count-1);
  }
}

/*
simplex S=simplex(new rational[] {4,1,1},
                  new rational[][] {{2,1,2},{3,3,1}},
                  new rational[] {4,3});

simplex S=simplex(new rational[] {2,6,1,1},
                  new rational[][] {{1,2,0,1},{1,2,1,1},{1,3,-1,2},{1,1,1,0}},
                  new rational[] {6,7,7,5});

simplex S=simplex(new rational[] {-10,-12,-12,0,0,0},
                  new rational[][] {{1,2,2,1,0,0},
                                    {2,1,2,0,1,0},
                                    {2,2,1,0,0,1}},
                  new rational[] {20,20,20});

simplex S=simplex(new rational[] {-10,-12,-12},
                  new rational[][] {{1,2,2},
                                    {2,1,2},
                                    {2,2,1}},
                  new int[] {0,0,-1},
                  new rational[] {20,20,20});
*/

simplex S=simplex(new rational[] {1,1,1,0},
                  new rational[][] {{1,2,3,0},
                                    {-1,2,6,0},
                                    {0,4,9,0},
                                    {0,0,3,1}},
                  new rational[] {3,2,5,1});

write();
write("case:",S.case);
write("x:",S.x);
write("Cost=",S.cost);

