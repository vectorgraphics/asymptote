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
  real EpsilonA;

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
      for(J=1; J <= N; ++J)
        if(Em[J] < 0) break;

      if(J > N)
        break;

      int I=-1;
      real t;
      for(int i=0; i < m; ++i) {
        real u=E[i][J];
        if(u > EpsilonA) {
          t=E[i][0]/u;
          I=i;
          break;
        }
      }
      for(int i=I+1; i < m; ++i) {
        real u=E[i][J];
        if(u > EpsilonA) {
          real r=E[i][0]/u;
          if(r <= t && (r < t || Bindices[i] < Bindices[I])) {
            t=r; I=i;
          } // Bland's rule: exiting variable has smallest minimizing index
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
      // Find first negative entry in zeroth (basic variable) column
      real[] Em=E[m];
      int I;
      for(I=0; I < m; ++I) {
        if(E[I][0] < 0) break;
      }

      if(I == m)
        break;

      int J=0;
      real t;
      for(int j=1; j <= N; ++j) {
        real u=E[I][j];
        if(u < -EpsilonA) {
          t=-E[m][j]/u;
          J=j;
          break;
        }
      }
      for(int j=J+1; j <= N; ++j) {
        real u=E[I][j];
        if(u < -EpsilonA) {
          real r=-E[m][j]/u;
          if(r <= t && (r < t || j < J)) {
            t=r; J=j;
          } // Bland's rule: exiting variable has smallest minimizing index
        }
      }
      if(J == 0)
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
    real normA=norm(A);
    real epsilonA=100.0*realEpsilon*normA;
    EpsilonA=epsilon*normA;

    // Phase 1
    m=A.length;
    if(m == 0) {case=INFEASIBLE; return;}
    n=A[0].length;
    if(n == 0) {case=INFEASIBLE; return;}

    real[][] E=new real[m+1][n+1];
    real[] Em=E[m];

    for(int j=1; j <= n; ++j)
      Em[j]=0;

    for(int i=0; i < m; ++i) {
      real[] Ai=A[i];
      real[] Ei=E[i];
      if(b[i] >= 0 || dual) {
        for(int j=1; j <= n; ++j) {
          real Aij=Ai[j-1];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      } else {
        for(int j=1; j <= n; ++j) {
          real Aij=-Ai[j-1];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      }
    }

    void basicValues() {
      real sum=0;
      for(int i=0; i < m; ++i) {
        real B=dual ? b[i] : abs(b[i]);
        E[i][0]=B;
        sum -= B;
      }
      Em[0]=sum;
    }

    int[] Bindices;

    if(phase1) {
      Bindices=new int[m];
      int p=0;

      // Check for redundant basis vectors.
      bool checkBasis(int j) {
        for(int i=0; i < m; ++i) {
          real[] Ei=E[i];
          if(i != p ? abs(Ei[j]) >= epsilonA : Ei[j] <= epsilonA) return false;
        }
        return true;
      }

      int checkTableau() {
        for(int j=1; j <= n; ++j)
          if(checkBasis(j)) return j;
        return 0;
      }

      int k=0;
      while(p < m) {
        int j=checkTableau();
        if(j > 0)
          Bindices[p]=j;
        else { // Add an artificial variable
          Bindices[p]=n+1+k;
          for(int i=0; i < p; ++i)
            E[i].push(0.0);
          E[p].push(1.0);
          for(int i=p+1; i < m; ++i)
            E[i].push(0.0);
          E[m].push(0.0);
          ++k;
        }
        ++p;
      }

      basicValues();
      iterate(E,n+k,Bindices);
  
      if(abs(Em[0]) > EpsilonA) {
      case=INFEASIBLE;
      return;
      }
    } else {
       Bindices=sequence(new int(int x){return x;},m)+n-m+1;
       basicValues();
    }

    real[] cB=phase1 ? new real[m] : c[n-m:n];
    real[][] D=phase1 ? new real[m+1][n+1] : E;
    if(phase1) {
      // Drive artificial variables out of basis.
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k > n) {
          real[] Ei=E[i];
          int j;
          for(j=1; j <= n; ++j)
            if(abs(Ei[j]) > EpsilonA) break;
          if(j > n) continue;
          Bindices[i]=j;
          rowreduce(E,n,i,j);
        }
      }
      int ip=0; // reduced i
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k > n) continue;
        Bindices[ip]=k; 
        cB[ip]=c[k-1];
        real[] Dip=D[ip];
        real[] Ei=E[i];
        for(int j=1; j <= n; ++j)
          Dip[j]=Ei[j];
        Dip[0]=Ei[0];
        ++ip;
      }

      real[] Dip=D[ip];
      real[] Em=E[m];
      for(int j=1; j <= n; ++j)
        Dip[j]=Em[j];
      Dip[0]=Em[0];

      if(m > ip) {
        Bindices.delete(ip,m-1);
        D.delete(ip,m-1);
        m=ip;
      }
    }

    real[] Dm=D[m];
    for(int j=1; j <= n; ++j) {
      real sum=0;
      for(int k=0; k < m; ++k)
        sum += cB[k]*D[k][j];
      Dm[j]=c[j-1]-sum;
    }

    real sum=0;
    for(int k=0; k < m; ++k)
      sum += cB[k]*D[k][0];
    Dm[0]=-sum;

    case=(dual ? iterateDual : iterate)(D,n,Bindices);
    if(case != OPTIMAL)
      return;

    for(int j=0; j < n; ++j)
      x[j]=0;

    for(int k=0; k < m; ++k)
      x[Bindices[k]-1]=D[k][0];
    cost=-Dm[0];
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
