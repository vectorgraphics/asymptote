// Rational simplex solver written by John C. Bowman and Pouria Ramazi, 2018.
import rational;

struct simplex {
  static int OPTIMAL=0;
  static int UNBOUNDED=1;
  static int INFEASIBLE=2;

  int case;
  rational[] x;
  rational cost;

  int m,n;
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

  int iterate(rational[][] E, int N, int[] Bindices) {
    while(true) {
      // Find first negative entry in bottom (reduced cost) row
      rational[] Em=E[m];
      for(J=0; J < N; ++J)
        if(Em[J] < 0) break;

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
          if(v <= M) {M=v; I=i;}
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

  // Try to find a solution x to Ax=b that minimizes the cost c^T x,
  // where A is an m x n matrix, x is a vector of length n, b is a
  // vector of length m, and c is a vector of length n.
  void operator init(rational[] c, rational[][] A, rational[] b,
                     bool phase1=true) {
    // Phase 1    
    m=A.length;
    n=A[0].length;

    int N=phase1 ? n+m : n;
    rational[][] E=new rational[m+1][N+1];
    rational[] Em=E[m];

    for(int j=0; j < n; ++j)
      Em[j]=0;

    for(int i=0; i < m; ++i) {
      rational[] Ai=A[i];
      rational[] Ei=E[i];
      if(b[i] >= 0) {
        for(int j=0; j < n; ++j) {
          rational Aij=Ai[j];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      } else {
        for(int j=0; j < n; ++j) {
          rational Aij=-Ai[j];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      }
    }

    if(phase1) {
      for(int i=0; i < m; ++i) { 
        rational[] Ei=E[i];
        for(int j=0; j < i; ++j)
          Ei[n+j]=0;
        Ei[n+i]=1;
        for(int j=i+1; j < m; ++j)
          Ei[n+j]=0;
      }
    }

    rational sum=0;
    for(int i=0; i < m; ++i) {
      rational B=abs(b[i]);
      E[i][N]=B;
      sum -= B;
    }
    Em[N]=sum;

    if(phase1)
      for(int j=0; j < m; ++j)
        Em[n+j]=0;
   
    int[] Bindices=sequence(new int(int x){return x;},m)+n;

    if(phase1) {
      iterate(E,N,Bindices);
  
      if(Em[J] != 0) {
      case=INFEASIBLE;
      return;
      }
    }
    
    rational[][] D=phase1 ? new rational[m+1][n+1] : E;
    rational[] Dm=D[m];
    rational[] cb=phase1 ? new rational[m] : c[n-m:n];
    if(phase1) {
      int ip=0; // reduced i
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k >= n) continue;
        Bindices[ip]=k; 
        cb[ip]=c[k];
        rational[] Dip=D[ip];
        rational[] Ei=E[i];
        for(int j=0; j < n; ++j)
          Dip[j]=Ei[j];
        Dip[n]=Ei[N];
        ++ip;
      }

      rational[] Dip=D[ip];
      rational[] Em=E[m];
      for(int j=0; j < n; ++j)
        Dip[j]=Em[j];
      Dip[n]=Em[N];

      m=ip;

      for(int j=0; j < n; ++j) {
        rational sum=0;
        for(int k=0; k < m; ++k)
          sum += cb[k]*D[k][j];
        Dm[j]=c[j]-sum;
      }

      // Done with Phase 1
    }
   
    rational sum=0;
    for(int k=0; k < m; ++k)
      sum += cb[k]*D[k][n];
    Dm[n]=-sum;

    if(iterate(D,n,Bindices) == UNBOUNDED) {
    case=UNBOUNDED;
    return;
    }

    for(int j=0; j < n; ++j)
      x[j]=0;

    for(int k=0; k < m; ++k)
      x[Bindices[k]]=D[k][n];

    cost=-Dm[n];
    case=OPTIMAL;
  }

  // Try to find a solution x to sgn(Ax-b)=sgn(s) that minimizes the cost
  // c^T x, where A is an m x n matrix, x is a vector of length n, b is a
  // vector of length m, and c is a vector of length n.
  void operator init(rational[] c, rational[][] A, int[] s, rational[] b) {
    int m=A.length;
    int n=A[0].length;

    int count=0;
    for(int i=0; i < m; ++i)
      if(s[i] != 0) ++count;

    rational[][] a=new rational[m][n+count];

    for(int i=0; i < m; ++i) {
      rational[] ai=a[i];
      rational[] Ai=A[i];
      for(int j=0; j < n; ++j) {
        ai[j]=Ai[j];
      }
    }
  
    int k=0;

    for(int i=0; i < m; ++i) {
      rational[] ai=a[i];
      for(int j=0; j < k; ++j)
        ai[n+j]=0;
      if(k < count)
        ai[n+k]=-s[i];
      for(int j=k+1; j < count; ++j)
        ai[n+j]=0;
      if(s[i] != 0) ++k;
    }

    bool phase1=!all(s == -1);
    operator init(concat(c,array(count,rational(0))),a,b,phase1);

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
*/
