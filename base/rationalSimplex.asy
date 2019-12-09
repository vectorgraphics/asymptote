// Rational simplex solver written by John C. Bowman and Pouria Ramazi, 2018.
import rational;

void simplexStandard(rational[] c, rational[][] A, int[] s=new int[],
                     rational[] b) {}
void simplexTableau(rational[][] E, int[] Bindices, int I=-1, int J=-1) {}
void simplexPhase1(rational[] c, rational[][] A, rational[] b,
                   int[] Bindices) {}
void simplexPhase2() {}

void simplexWrite(rational[][] E, int[] Bindices, int, int)
{
  int m=E.length-1;
  int n=E[0].length-1;

  write(E[m][0],tab);
  for(int j=1; j <= n; ++j)
    write(E[m][j],tab);
  write();

  for(int i=0; i < m; ++i) {
    write(E[i][0],tab);
    for(int j=1; j <= n; ++j) {
      write(E[i][j],tab);
    }
    write();
  }
  write();
};

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
      for(J=1; J <= N; ++J)
        if(Em[J] < 0) break;

      if(J > N)
        break;

      int I=-1;
      rational t;
      for(int i=0; i < m; ++i) {
        rational u=E[i][J];
        if(u > 0) {
          t=E[i][0]/u;
          I=i;
          break;
        }
      }
      for(int i=I+1; i < m; ++i) {
        rational u=E[i][J];
        if(u > 0) {
          rational r=E[i][0]/u;
          if(r <= t && (r < t || Bindices[i] < Bindices[I])) {
            t=r; I=i;
          } // Bland's rule: exiting variable has smallest minimizing index
        }
      }
      if(I == -1)
        return UNBOUNDED; // Can only happen in Phase 2.

      simplexTableau(E,Bindices,I,J);

      // Generate new tableau
      Bindices[I]=J;
      rowreduce(E,N,I,J);
    }
    return OPTIMAL;
  }

  int iterateDual(rational[][] E, int N, int[] Bindices) {
    while(true) {
      // Find first negative entry in zeroth (basic variable) column
      rational[] Em=E[m];
      int I;
      for(I=0; I < m; ++I) {
        if(E[I][0] < 0) break;
      }

      if(I == m)
        break;

      int J=0;
      rational t;
      for(int j=1; j <= N; ++j) {
        rational u=E[I][j];
        if(u < 0) {
          t=-E[m][j]/u;
          J=j;
          break;
        }
      }
      for(int j=J+1; j <= N; ++j) {
        rational u=E[I][j];
        if(u < 0) {
          rational r=-E[m][j]/u;
          if(r <= t && (r < t || j < J)) {
            t=r; J=j;
          } // Bland's rule: exiting variable has smallest minimizing index
        }
      }
      if(J == 0)
        return INFEASIBLE; // Can only happen in Phase 2.

      simplexTableau(E,Bindices,I,J);

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
  void operator init(rational[] c, rational[][] A, rational[] b,
                     bool phase1=true, bool dual=false) {
    if(dual) phase1=false;
    // Phase 1
    m=A.length;
    if(m == 0) {case=INFEASIBLE; return;}
    n=A[0].length;
    if(n == 0) {case=INFEASIBLE; return;}

    rational[][] E=new rational[m+1][n+1];
    rational[] Em=E[m];

    for(int j=1; j <= n; ++j)
      Em[j]=0;

    for(int i=0; i < m; ++i) {
      rational[] Ai=A[i];
      rational[] Ei=E[i];
      if(b[i] >= 0 || dual) {
        for(int j=1; j <= n; ++j) {
          rational Aij=Ai[j-1];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      } else {
        for(int j=1; j <= n; ++j) {
          rational Aij=-Ai[j-1];
          Ei[j]=Aij;
          Em[j] -= Aij;
        }
      }
    }

    void basicValues() {
      rational sum=0;
      for(int i=0; i < m; ++i) {
        rational B=dual ? b[i] : abs(b[i]);
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
          rational[] Ei=E[i];
          if(i != p ? Ei[j] != 0 : Ei[j] <= 0) return false;
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
            E[i].push(0);
          E[p].push(1);
          for(int i=p+1; i < m; ++i)
            E[i].push(0);
          E[m].push(0);
          ++k;
        }
        ++p;
      }

      basicValues();

      simplexPhase1(c,A,b,Bindices);

      iterate(E,n+k,Bindices);
  
      if(Em[0] != 0) {
        simplexTableau(E,Bindices);
      case=INFEASIBLE;
      return;
      }
    } else {
       Bindices=sequence(new int(int x){return x;},m)+n-m+1;
       basicValues();
    }

    rational[] cB=phase1 ? new rational[m] : c[n-m:n];
    rational[][] D=phase1 ? new rational[m+1][n+1] : E;
    if(phase1) {
      bool output=true;
      // Drive artificial variables out of basis.
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k > n) {
          rational[] Ei=E[i];
          int j;
          for(j=1; j <= n; ++j)
            if(Ei[j] != 0) break;
          if(j > n) continue;
          output=false;
          simplexTableau(E,Bindices,i,j);
          Bindices[i]=j;
          rowreduce(E,n,i,j);
        }
      }
      if(output) simplexTableau(E,Bindices);
      int ip=0; // reduced i
      for(int i=0; i < m; ++i) {
        int k=Bindices[i];
        if(k > n) continue;
        Bindices[ip]=k; 
        cB[ip]=c[k-1];
        rational[] Dip=D[ip];
        rational[] Ei=E[i];
        for(int j=1; j <= n; ++j)
          Dip[j]=Ei[j];
        Dip[0]=Ei[0];
        ++ip;
      }

      rational[] Dip=D[ip];
      rational[] Em=E[m];
      for(int j=1; j <= n; ++j)
        Dip[j]=Em[j];
      Dip[0]=Em[0];

      if(m > ip) {
        Bindices.delete(ip,m-1);
        D.delete(ip,m-1);
        m=ip;
      }
      if(!output) simplexTableau(D,Bindices);
    }

    rational[] Dm=D[m];
    for(int j=1; j <= n; ++j) {
      rational sum=0;
      for(int k=0; k < m; ++k)
        sum += cB[k]*D[k][j];
      Dm[j]=c[j-1]-sum;
    }

    rational sum=0;
    for(int k=0; k < m; ++k)
      sum += cB[k]*D[k][0];
    Dm[0]=-sum;

    simplexPhase2();

    case=(dual ? iterateDual : iterate)(D,n,Bindices);
    simplexTableau(D,Bindices);
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
  void operator init(rational[] c, rational[][] A, int[] s, rational[] b) {
    int m=A.length;
    if(m == 0) {case=INFEASIBLE; return;}
    int n=A[0].length;
    if(n == 0) {case=INFEASIBLE; return;}

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

    bool phase1=false;
    bool dual=count == m && all(c >= 0);

    for(int i=0; i < m; ++i) {
      rational[] ai=a[i];
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
        rational bi=b[i];
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

    rational[] C=concat(c,array(count,rational(0)));
    if(count > 0) simplexStandard(C,a,b);
    operator init(C,a,b,phase1,dual);

    if(case == OPTIMAL && count > 0)
      x.delete(n,n+count-1);
  }
}
