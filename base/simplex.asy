/*****
 * simplex.asy
 * Andy Hammerlindl 2004/07/27
 *
 * Solves the two-variable linear programming problem using the simplex method.
 * This problem is specialized in that the second variable, "b," does not have
 * a non-negativity condition, and the first variable, "a," is the quantity
 * being maximized.
 * Correct execution of the algorithm also assumes that the "b" term will be +1
 * or -1 in every added restriction, and that the problem can be initialized to
 * a valid state by pivoting b with one of the slack variables.  This
 * assumption may in fact be incorrect.
 *****/

static struct problem {
  typedef int var;
  var VAR_A = 0;
  var VAR_B = 1;

  struct row {
    public real c, t[];
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
  int n = 2;

  // Recalculate rows v[col] and vp for the pivot-swap.
  void swapPivot(int col, var vp)
  {
    row rvc = rows[v[col]], rvp = rows[vp];

    real d = rvp.t[col]; // NOTE: Handle d==0 case.
    rvc.c = -rvp.c/d;
    rvp.c = 0;
    for (int i = 0; i < n; ++i) {
      rvc.t[i] = (i==col ? 1 : -rvp.t[i])/d;
      rvp.t[i] = i==col ? 1 : 0;
    }
  }

  // Recalculate a row r other than the two above for the pivot.
  void swapOther(row r, int col)
  {
    row rvc = rows[v[col]];

    real m = r.t[col];
    r.c += m*rvc.c;
    for (int i = 0; i < n; ++i)
      if (i == col)
        r.t[i] = m*rvc.t[i];
      else
        r.t[i] += m*rvc.t[i];
  }


  // Pivot the variable v[col] with vp.
  void pivot(int col, var vp)
  {
    // Swap the two rows involved as a special case.
    swapPivot(col, vp);

    // Swap the other rows.
    for (var i = 0; i < rows.length; ++i)
      if (i != v[col] && i != vp)
        swapOther(rows[i], col);

    // Relabel the vars.
    v[col] = vp;
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
  static int OPTIMAL = -1;
  int selectColumn()
  {
    row r = rows[0];

    for (int i = 0; i < n; ++i)
      if (r.t[i] > 0)
        return i;

    return OPTIMAL;
  }

  // Select the new variable associated with a pivot on the column given.
  // Returns UNBOUNDED if the space is unbounded.
  static var UNBOUNDED = -2;
  var selectVar(int col)
  {
    // We assume that the first two vars (a and b) once swapped out, won't be
    // swapped back in.  This find the variable which gives the tightest
    // non-negativity condition restricting our optimization.  This turns
    // out to be the max of c/t[col].  Note that as c is positive, and
    // t[col] is negative, all c/t[col] will be negative, so we are finding
    // the smallest in magnitude.
    var vp = UNBOUNDED;
    row rvp = null;
    for (var i = 2; i < rows.length; ++i) {
      row r = rows[i];
      if (r.t[col] < 0 && 
          (vp == UNBOUNDED || r.c/r.t[col] > rvp.c/rvp.t[col])) {
        vp = i;
        rvp = r;
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
        if (rows[i].c < 0)
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
        if (rvc.t[i] != (i==col ? 1 : 0))
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
  static int INVALID = -3;
  int optimize()
  {
    // Put into a valid state to begin.
    if (!valid())
      init();

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
