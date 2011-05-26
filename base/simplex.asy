/*****
 * simplex.asy
 * Andy Hammerlindl 2004/07/27
 *
 * Solves the two-variable linear programming problem using the simplex method.
 * This problem is specialized in that the second variable, "b", does not have
 * a non-negativity condition, and the first variable, "a", is the quantity
 * being maximized.
 * Correct execution of the algorithm also assumes that the coefficient of "b"
 * will be +1 or -1 in every added restriction, and that the problem can be
 * initialized to a valid state by pivoting b with one of the slack
 * variables.  This assumption may in fact be incorrect.
 *****/

private real infinity=sqrt(0.25*realMax);

struct problem {
  typedef int var;
  static var VAR_A = 0;
  static var VAR_B = 1;

  static int OPTIMAL = -1;
  static var UNBOUNDED = -2;
  static int INVALID = -3;

  struct row {
    real c, t[];
  }

  // The variables of the rows.
  // Initialized for the two variable problem.
  var[] v = {VAR_A, VAR_B};

  // The rows of equalities.
  row rowA() {
    row r = new row;
    r.c = 0;
    r.t = new real[] {1, 0};
    return r;
  }
  row rowB() {
    row r = new row;
    r.c = 0;
    r.t = new real[] {0, 1};
    return r;
  }
  row[] rows = {rowA(), rowB()};

  // The number of original variables.
  int n = rows.length;

  // Pivot the variable v[col] with vp.
  void pivot(int col, var vp)
  {
    var vc=v[col];

    // Recalculate rows v[col] and vp for the pivot-swap.
    row rvc = rows[vc], rvp = rows[vp];
    real factor=1/rvp.t[col]; // NOTE: Handle rvp.t[col] == 0 case.
    rvc.c=-rvp.c*factor;
    rvp.c=0;
    rvc.t=-rvp.t*factor;
    rvp.t *= 0;
    rvc.t[col]=factor;
    rvp.t[col]=1;
    
    var a=min(vc,vp);
    var b=max(vc,vp);
    
    // Recalculate the rows other than the two used for the above pivot.
    for (var i = 0; i < a; ++i) {
      row r=rows[i];
      real m = r.t[col];
      r.c += m*rvc.c;
      r.t += m*rvc.t;
      r.t[col]=m*factor;
    }
    for (var i = a+1; i < b; ++i) {
      row r=rows[i];
      real m = r.t[col];
      r.c += m*rvc.c;
      r.t += m*rvc.t;
      r.t[col]=m*factor;
    }
    for (var i = b+1; i < rows.length; ++i) {
      row r=rows[i];
      real m = r.t[col];
      r.c += m*rvc.c;
      r.t += m*rvc.t;
      r.t[col]=m*factor;
    }

    // Relabel the vars.
    v[col] = vp;
  }

  // As b does not have a non-negativity condition, it must initially be
  // pivoted out for a variable that does.  This selects the initial
  // variable to pivot with b.  It also assumes that there is a valid
  // solution with a == 0 to the linear programming problem, and if so, it
  // picks a pivot to get to that state.  In our case, a == 0 corresponds to
  // a picture with the user coordinates shrunk down to zero, and if that
  // doesn't fit, nothing will.
  var initVar()
  {
    real min=infinity, max=-infinity;
    var argmin=0, argmax=0;

    for (var i = 2; i < rows.length; ++i) {
      row r=rows[i];
      if (r.t[VAR_B] > 0) {
        real val=r.c/r.t[VAR_B];
        if (val < min) {
          min=val;
          argmin=i;
        }
      } else if (r.t[VAR_B] < 0) {
        real val=r.c/r.t[VAR_B];
        if (val > max) {
          max=val;
          argmax=i;
        }
      }
    }

    // If b has a minimal value, choose a pivot that will give b its minimal
    // value.  Otherwise, if b has maximal value, choose a pivot to give b its
    // maximal value.
    return argmin != 0 ? argmin :
      argmax != 0 ? argmax :
      UNBOUNDED;
  }

  // Initialize the linear program problem by moving into an acceptable state
  // this assumes that b is unrestrained and is the second variable.
  // NOTE: Works in limited cases, may be bug-ridden.
  void init()
  {
    // Find the lowest constant term in the equations.
    var lowest = 0;
    for (var i = 2; i < rows.length; ++i) {
      if (rows[i].c < rows[lowest].c)
        lowest = i;
    }

    // Pivot if necessary.
    if (lowest != 0)
      pivot(VAR_B, lowest);
  }

  // Selects a column to pivot on.  Returns OPTIMAL if the current state is
  // optimal.  Assumes we are optimizing the first row.
  int selectColumn()
  {
    int i=find(rows[0].t > 0,1);
    return (i >= 0) ? i : OPTIMAL;
  }

  // Select the new variable associated with a pivot on the column given.
  // Returns UNBOUNDED if the space is unbounded.
  var selectVar(int col)
  {
    // We assume that the first two vars (a and b) once swapped out, won't be
    // swapped back in.  This finds the variable which gives the tightest
    // non-negativity condition restricting our optimization.  This turns
    // out to be the max of c/t[col].  Note that as c is positive, and
    // t[col] is negative, all c/t[col] will be negative, so we are finding
    // the smallest in magnitude.
    var vp=UNBOUNDED;
    real max=-infinity;
    for (int i = 2; i < rows.length; ++i) {
      row r=rows[i];
      if(r.c < max*r.t[col]) {
        max=r.c/r.t[col]; vp=i;
      }
    }
    
    return vp;
  }

  // Checks that the rows are in a valid state.
  bool valid()
  {
    // Checks that constants are valid.
    bool validConstants() {
      for (int i = 0; i < rows.length; ++i)
        // Do not test the row for b, as it does not have a non-negativity
        // condition.
        if (i != VAR_B && rows[i].c < 0)
           return false;
      return true;
    }

    // Check a variable to see if its row is simple.
    // NOTE: Simple rows could be optimized out, since they are not really
    // used. 
    bool validVar(int col) {

      var vc = v[col];
      row rvc = rows[vc];

      if (rvc.c != 0)
        return false;
      for (int i = 0; i < n; ++i)
        if (rvc.t[i] != (i == col ? 1 : 0))
          return false;
      
      return true;
    }

    if (!validConstants()) {
      return false;
    }
    for (int i = 0; i < n; ++i)
      if (!validVar(i)) {
        return false;
      }

    return true;
  }


  // Perform the algorithm to find the optimal solution.  Returns OPTIMAL,
  // UNBOUNDED, or INVALID (if no solution is possible).
  int optimize()
  {
    // Put into a valid state to begin and pivot b out.
    var iv=initVar();
    if (iv == UNBOUNDED)
      return iv;
    pivot(VAR_B, iv);

    if (!valid())
      return INVALID;

    while(true) {
      int col = selectColumn();
      
      if (col == OPTIMAL)
        return col;
      var vp = selectVar(col);
      
      if (vp == UNBOUNDED)
        return vp;

      pivot(col, vp);
    }

    // Shouldn't reach here.
    return INVALID;
  }

  // Add a restriction to the problem:
  // t1*a + t2*b + c >= 0
  void addRestriction(real t1, real t2, real c)
  {
    row r = new row;
    r.c = c;
    r.t = new real[] {t1, t2};
    rows.push(r);
  }

  // Return the value of a computed.
  real a()
  {
    return rows[VAR_A].c;
  }

  // Return the value of b computed.
  real b()
  {
    return rows[VAR_B].c;
  }
}
