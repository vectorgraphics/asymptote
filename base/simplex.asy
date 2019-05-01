// Real simplex solver written by John C. Bowman and Pouria Ramazi, 2018.

struct simplex {
  static int OPTIMAL=0;
  static int UNBOUNDED=1;
  static int INFEASIBLE=2;

  int case;
  real[] x;
  real cost;

  int m,n;
  int J;
  real epsilonA;

  // Row reduce based on pivot E[I][J]
  void rowreduce(real[][] E, int N, int I, int J) {
    real[] EI=E[I];
    real v=EI[J];
    for(int j=0; j < J; ++j) EI[j] /= v;
    EI[J]=1.0;
    for(int j=J+1; j <= N; ++j) EI[j] /= v;

    for(int i=0; i < I; ++i) {
      real[] Ei=E[i];
      real EiJ=Ei[J];
      for(int j=0; j < J; ++j)
        Ei[j] -= EI[j]*EiJ;
      Ei[J]=0.0;
      for(int j=J+1; j <= N; ++j)
        Ei[j] -= EI[j]*EiJ;
    }
    for(int i=I+1; i <= m; ++i) {
      real[] Ei=E[i];
      real EiJ=Ei[J];
      for(int j=0; j < J; ++j)
        Ei[j] -= EI[j]*EiJ;
      Ei[J]=0.0;
      for(int j=J+1; j <= N; ++j)
        Ei[j] -= EI[j]*EiJ;
    }
  }

  int iterate(real[][] E, int N, int[] Bindices) {
    while(true) {
      // Find first negative entry in bottom (reduced cost) row
      real[] Em=E[m];
      for(J=0; J < N; ++J)
        if(Em[J] < 0) break;

      if(J == N)
        break;

      int I=-1;
      real M;
      for(int i=0; i < m; ++i) {
        real e=E[i][J];
        if(e > epsilonA) {
          M=E[i][N]/e;
          I=i;
          break;
        }
      }
      for(int i=I+1; i < m; ++i) {
        real e=E[i][J];
        if(e > epsilonA) {
          real v=E[i][N]/e;
          if(v < M) {M=v; I=i;} // Bland's rule: choose smallest argmin
        }
      }
      if(I == -1)
        return UNBOUNDED; // Can only happen in Phase 2.

      // Generate new tableau
      Bindices[I]=J;
      rowreduce(E,N,I,J);
    }
    return OPTIMAL;
  }

  int iterateDual(real[][] E, int N, int[] Bindices) {
    while(true) {
      // Find first negative entry in right (basic variable) column
      real[] Em=E[m];
      int I;
      for(I=0; I < m; ++I) {
        if(E[I][N] < 0) break;
      }

      if(I == m)
        break;

      int J=-1;
      real M;
      for(int j=0; j < N; ++j) {
        real e=E[I][j];
        if(e < epsilonA) {
          M=-E[m][j]/e;
          J=j;
          break;
        }
      }
      for(int j=J+1; j < N; ++j) {
        real e=E[I][j];
        if(e < epsilonA) {
          real v=-E[m][j]/e;
          if(v < M) {M=v; J=j;} // Bland's rule: choose smallest argmin
        }
      }
      if(J == -1)
        return INFEASIBLE; // Can only happen in Phase 2.

      // Generate new tableau
      Bindices[I]=J;
      rowreduce(E,N,I,J);
    }
    return OPTIMAL;
  }

  // Try to find a solution x to Ax=b that minimizes the cost c^T x,
  // where A is an m x n matrix, x is a vector of n non-negative numbers,
  // b is a vector of length m, and c is a vector of length n.
  // Can set phase1=false if the last m columns of A form the identity matrix.
  void operator init(real[] c, real[][] A, real[] b, bool phase1=true,
                     bool dual=false) {
    if(dual) phase1=false;
    static real epsilon=sqrt(realEpsilon);
    epsilonA=epsilon*norm(A);

    // Phase 1
    m=A.length;
    if(m == 0) {case=INFEASIBLE; return;}
    n=A[0].length;
    if(n == 0) {case=INFEASIBLE; return;}

    int N=phase1 ? n+m : n;
    real[][] E=new real[m+1][N+1];
    real[] Em=E[m];

    for(int j=0; j < n; ++j)
      Em[j]=0;

    for(int i=0; i < m; ++i) {
      real[] Ai=A[i];
      real[] Ei=E[i];
      if(b[i] >= 0 || dual) {
        for(int j=0; j < n; ++j) {
          real Aij=Ai[j];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      } else {
        for(int j=0; j < n; ++j) {
          real Aij=-Ai[j];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      }
    }

    if(phase1) {
      for(int i=0; i < m; ++i) { 
        real[] Ei=E[i];
        for(int j=0; j < i; ++j)
          Ei[n+j]=0.0;
        Ei[n+i]=1.0;
        for(int j=i+1; j < m; ++j)
          Ei[n+j]=0.0;
      }
    }

    real sum=0;
    for(int i=0; i < m; ++i) {
      real B=dual ? b[i] : abs(b[i]);
      E[i][N]=B;
      sum -= B;
    }
    Em[N]=sum;

    if(phase1)
      for(int j=0; j < m; ++j)
        Em[n+j]=0.0;
   
    int[] Bindices;

    if(phase1) {
      Bindices=sequence(new int(int x){return x;},m)+n;
      iterate(E,N,Bindices);
  
      if(abs(Em[J]) > epsilonA) {
      case=INFEASIBLE;
      return;
      }
    } else Bindices=sequence(new int(int x){return x;},m)+n-m;
    
    real[] cB=phase1 ? new real[m] : c[n-m:n];
    real[][] D=phase1 ? new real[m+1][n+1] : E;
    if(phase1) {
      // Drive artificial variables out of basis.
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k >= n) {
          real[] Ei=E[i];
          int j;
          for(j=0; j < n; ++j)
            if(Ei[j] != 0) break;
          if(j == n) continue;
          Bindices[i]=j;
          rowreduce(E,n,i,j);
        }
      }
      int ip=0; // reduced i
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k >= n) continue;
        Bindices[ip]=k; 
        cB[ip]=c[k];
        real[] Dip=D[ip];
        real[] Ei=E[i];
        for(int j=0; j < n; ++j)
          Dip[j]=Ei[j];
        Dip[n]=Ei[N];
        ++ip;
      }

      real[] Dip=D[ip];
      real[] Em=E[m];
      for(int j=0; j < n; ++j)
        Dip[j]=Em[j];
      Dip[n]=Em[N];

      if(m > ip) {
        Bindices.delete(ip,m-1);
        D.delete(ip,m-1);
        m=ip;
      }
    }

    real[] Dm=D[m];
    for(int j=0; j < n; ++j) {
      real sum=0;
      for(int k=0; k < m; ++k)
        sum += cB[k]*D[k][j];
      Dm[j]=c[j]-sum;
    }

    real sum=0;
    for(int k=0; k < m; ++k)
      sum += cB[k]*D[k][n];
    Dm[n]=-sum;

    case=(dual ? iterateDual : iterate)(D,n,Bindices);
    if(case != OPTIMAL)
      return;

    for(int j=0; j < n; ++j)
      x[j]=0;

    for(int k=0; k < m; ++k)
      x[Bindices[k]]=D[k][n];

    cost=-Dm[n];
  }

  // Try to find a solution x to sgn(Ax-b)=sgn(s) that minimizes the cost
  // c^T x, where A is an m x n matrix, x is a vector of n non-negative
  // numbers, b is a vector of length m, and c is a vector of length n.
  void operator init(real[] c, real[][] A, int[] s, real[] b) {
    int m=A.length;
    if(m == 0) {case=INFEASIBLE; return;}
    int n=A[0].length;
    if(n == 0) {case=INFEASIBLE; return;}

    int count=0;
    for(int i=0; i < m; ++i)
      if(s[i] != 0) ++count;

    real[][] a=new real[m][n+count];

    for(int i=0; i < m; ++i) {
      real[] ai=a[i];
      real[] Ai=A[i];
      for(int j=0; j < n; ++j) {
        ai[j]=Ai[j];
      }
    }
  
    int k=0;

    bool phase1=false;
    bool dual=count == m && all(c >= 0);

    for(int i=0; i < m; ++i) {
      real[] ai=a[i];
      for(int j=0; j < k; ++j)
        ai[n+j]=0;
      if(k < count)
        ai[n+k]=-s[i];
      for(int j=k+1; j < count; ++j)
        ai[n+j]=0;
      int si=s[i];
      if(si == 0) phase1=true;
      else {
        ++k;
        real bi=b[i];
        if(bi == 0) {
          if(si == 1) {
            s[i]=-1;
            for(int j=0; j < n+count; ++j)
              ai[j]=-ai[j];
          }
        } else if(si*bi > 0) {
          if(dual && si == 1) {
            b[i]=-bi;
            s[i]=-1;
            for(int j=0; j < n+count; ++j)
              ai[j]=-ai[j];
          } else
            phase1=true;
        }
      }
    }

    operator init(concat(c,array(count,0.0)),a,b,phase1,dual);

    if(case == OPTIMAL && count > 0)
      x.delete(n,n+count-1);
  }
}
