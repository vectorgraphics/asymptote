/*****
 * knotlist.cc
 * Andy Hammerlindl 20002/08/22
 *
 * A knotlist describes a section of a path that can be solved on its
 * own.  This means all knots in it will be open except for the
 * endpoints which may have curls or directions specified.  The tension
 * in and out of each knot must also be stored.
 *****/

#include <cmath>
#include <iostream>
#include "knotlist.h"
#include "settings.h"
#include "angle.h"
#include "util.h"

#include <boost/scoped_array.hpp>

using std::cerr;
using std::endl;
using std::vector;
using boost::scoped_array;

using settings::verbose;

namespace camp {

// Bounds on tension and curl are necessary to ensure the linear
// equations for solving the path are consistent.
const double MIN_TENSION = 0.75;

void knotlist::add(pair z)
{
  assert(open);
 
  knot k;
  k.z = z;
  k.left = preknot;
  nodes.push_back(k);

  //Reset preknot
  preknot.tension = 1.0; preknot.atleast = false;

  // Close if there are sufficient restrictions
  if (rightRest.kind != rest::OPEN)
    close();
}

void knotlist::add(knotlist &splice)
{
  assert(open);
 
  // A conflict arises as preknot and the left side of the first knot of
  // splice describe the same knotside.  We will use preknot as there is
  // no way with our current syntax of describing tension before the
  // first knot.
 
  vector<knot>::iterator p = splice.nodes.begin();
  vector<knot>::iterator end = splice.nodes.end();
  if (p != end) {
    knot k = *p;
    k.left = preknot;
    nodes.push_back(k);
    ++p;

    // Reset preknot
    preknot.tension = 1.0; preknot.atleast = false;
  }

  // Do the rest normally.
  while (p != end) {
    nodes.push_back(*p);
    ++p;
  }

  // Use the right rest of the added list
  rightRest = splice.rightRest;
  open = splice.open;
}

void knotlist::tension(double l, double r, bool atleast)
{
  if (l < MIN_TENSION) l = MIN_TENSION;
  if (r < MIN_TENSION) r = MIN_TENSION;
 
  knot& k = nodes.back();
  k.right.tension = l;
  k.right.atleast = atleast;

  preknot.tension = r;
  preknot.atleast = atleast;
} 

void knotlist::clear()
{
  open = true;
  controlSpec = false;
 
  leftRest.kind = rest::OPEN;
  nodes.clear();
  rightRest.kind = rest::OPEN;

  preknot.tension = 1.0;
  preknot.atleast = false;
}

void knotlist::continuation(knotlist &last)
{
  // Copy the last knot and restriction.
  int n = (int) last.nodes.size();
  if (n == 0) {
    //cerr << "continuing empty path" << endl;
    return;
  }
  
  pair lastKnot = last.nodes.back().z;
  rest lastRest = last.rightRest;

  clear();

  leftRest = lastRest;
  add(lastKnot);
}

/***** The Linear equation solving phase *****/

/***** Constants *****/

const double VELOCITY_BOUND = 4.00;
const double TWIST_RATIO_CAP = 0.25;

/***** Auxillary computation functions *****/

// Computes the relative distance of a control point given the angles.
double velocity(double theta, double phi, double tension, bool t_atleast)
{
  const double a = 1.41421356237309504880; // sqrt(2)
  const double b = 0.0625;                 // 1/16
  const double c = 1.23606797749978969641; // sqrt(5) - 1
  const double d = 0.76393202250021030359; // 3 - sqrt(5)

  double st = sin(theta), ct = cos(theta),
         sf = sin(phi),   cf = cos(phi);

  // NOTE: Have to deal with degenerate condition theta = phi = -pi

  double r =  (2.0 + a*(st - b*sf)*(sf - b*st)*(ct-cf)) /
              (3.0 * tension * (1.0 + 0.5*c*ct + 0.5*d*cf));

  //cerr << " velocity(" << theta << "," << phi <<")= " << r << endl;
  if (r >  VELOCITY_BOUND)
    r = VELOCITY_BOUND;

  // Apply boundedness condition for tension atleast cases.
  if (t_atleast)
  {
    double sine = sin(theta + phi);
    if ((st >= 0.0 && sf >= 0.0 && sine > 0.0) ||
        (st <= 0.0 && sf <= 0.0 && sine < 0.0))
    {
      double rmax = sf / sine;
      if (r > rmax)
        r = rmax;
    }
  }

  return r;
}

// Solves a tridiagonal matrix by Gauss-Jordon elimination
void solveTriDiag(double *value,
                  double *pre,
                  double *piv,
                  double *post,
                  double *aug,
                  int n)
{
  /*printf("\nsolveTriDiag: n = %d\n", n);

  for (int k = 0; k < n; k++)
  {
    printf("%lf %lf %lf | %lf\n", pre[k], piv[k], post[k], aug[k]);
  }
  printf("\n");*/

  /* This solves the set of linear equations:

       pre[i]*value[i-1] + piv[i]*value[i] + post[i]*value[i+1] = aug[i]

     for all i = 0 to n-1 and with unknown values. It is neccessary that

       pre[0] = post[n-1] = 0

     otherwise the cyclic solution is needed.
     The values are determined and put in the array value[].
     The other arrays will be manipulated and will not preserve their data.
  */

  /* In this specific Gauss - Jordan elimination, only the given values need
     to be manipulated.  These conditions are satisfied for knots by having
     tension >= 3/4 and curl >= 0.
  */

  // Assert that it is not cyclic.
  assert(pre[0] == 0.0 && post[n-1] == 0.0);

  int i;
  for (i = 0; i < n-1; i++)
  {
    double factor;

    // Eliminate the first element of the (i+1)th row
    assert(piv[i] != 0.0);
    factor = - pre[i+1]/piv[i];
    pre[i+1]   = 0.0;
    piv[i+1]  += factor * post[i];
    aug[i+1]  += factor * aug[i];
  }

  // Find the (n-1)th value
  assert (piv[n-1] != 0.0);
  value[n-1] = aug[n-1]/piv[n-1];

  // Back substitution
  for (i = n-2; i >= 0; i--)
  {
    assert(piv[i] != 0.0);
    value[i] = (aug[i] - post[i]*value[i+1]) / piv[i];
  }

  /*for (int k = 0; k < n; k++)
  {
    printf("%lf\n", value[k]);
  }
  printf("\n"); */

}

// Solves a tridiagonal matrix with values in all corners.
void solveCyclicTriDiag(double *value,
			double *pre,
			double *piv,
			double *post,
			double *aug,
			int n)
{
  /* This solves the set of linear equations:

       pre[i]*value[i-1] + piv[i]*value[i] + post[i]*value[i+1] = aug[i]

     for all i mod n and with unknown values.
     The values are determined and put in the array value[].
     The other array will be manipulated and will not preserve their data.
  */

  /* In this specific Gauss-Jordan Elimination, the only values that need to
     be considered aside from those in the function parameters, are the
     (n-1)th column of the matrix and one value in bottom row that moves
     across the columns during the elimination.
     "last" represents the last column in A when solving Ax=B
     "bot"  represents the non-zero element in the bottom row

     Non-singular conditions are satisfied for knots by having
     tension >= 3/4 and curl >= 0.
  */
  scoped_array<double> last(new double [n]);
  double bot;
  memset(last.get(), 0, sizeof(double) * n);

  // Move the first pre and last post into their proper positions in the
  // matrix
  last[0] = pre[0];  pre[0] = 0.0;
  bot = post[n-1];   post[n-1] = 0.0;

  int i;
  for (i = 0; i < n-1; i++)
  {
    double factor;

    // Eliminate the first element of the (i+1)th row
    assert(piv[i] != 0.0);
    factor = - pre[i+1]/piv[i];
    pre[i+1]   = 0.0;
    piv[i+1]  += factor * post[i];
    last[i+1] += factor * last[i];
    aug[i+1]  += factor * aug[i];

    // Eliminate the element in the bottom row
    factor = - bot / piv[i];
    bot = factor * post[i];  // This is now in the next column
    last[n-1] += factor * last[i];
    aug[n-1]  += factor * aug[i];
  }

  // Find the (n-1)th value
  assert ((last[n-1]+piv[n-1]+bot) != 0.0);
  value[n-1] = aug[n-1]/(last[n-1]+piv[n-1]+bot);

  // Back substitution
  for (i = n-2; i >= 0; i--)
  {
    assert(piv[i] != 0.0);
    value[i] = (aug[i] - last[i]*value[n-1] - post[i]*value[i+1]) / piv[i];
  }
}

double turnAngle (pair z0, pair z1, pair z2)
{
  pair v = (z2-z1)/(z1-z0);
  return v.gety() == 0 ? v.getx() >= 0 ? 0 : PI
                       : angle(v);
}

path knotlist::solve()
{
  if (verbose > 3) cerr << "solving: " << *this << endl;

  // Deal with simple cases.
  int n = (int) nodes.size();
  if (n == 0) {
    //cerr << "solved as empty" << endl;
    return path();
  }
  if (n == 1) {
    //cerr << "solved as pair" << endl;
    return path(nodes[0].z);
  }

  // The knots with controls points as they are solved.
  solvedKnot *s = new solvedKnot[n];

  // Deal with an already solved segment,
  // ie. a .. control b and c .. d.
  if (controlSpec)
  {
    assert(n==2);
    s[0].pre = s[0].point = nodes[0].z;
    s[0].post = leftRest.control;
    s[1].pre = rightRest.control;
    s[1].point = s[1].post = nodes[1].z;
    return path(s, n);
  }

  // arg stores the heading in radians between adjacent knots.
  scoped_array<double> arg(new double[n-1]);

  // d stores the distance between adjacent knots. 
  scoped_array<double> d(new double[n-1]);

  for (int i = 0; i < n-1; i++) {
    pair dz = nodes[i+1].z - nodes[i].z;
    arg[i] = dz.angle();
    d[i] = dz.length();
  }

  // Temporary variable for the velocity at a knot.
  double v;
  
  // Check for straight line case
  if (n == 2 &&
      (leftRest.kind == rest::OPEN || leftRest.kind == rest::TWIST) &&
      (rightRest.kind == rest::OPEN || rightRest.kind == rest::TWIST))
  {
    v = velocity(0.0, 0.0, nodes[0].right.tension, nodes[0].right.atleast);
    s[0].pre = s[0].point = nodes[0].z;
    s[0].post = nodes[0].z + v * d[0] * expi(arg[0]);
    s[0].straight=true;
    
    v = velocity(0.0, 0.0, nodes[1].left.tension, nodes[1].left.atleast);
    s[1].pre = nodes[1].z - v * d[0] * expi(arg[0]);
    s[1].point = s[1].post = nodes[1].z;

    path p(s, n);

    // Debugging trace
    if(verbose > 3) std::cerr << "solved as straight: " << p << std::endl;
    
    return p;
  }

  // Calculate the turning angles at each knot.
  scoped_array<double> psi(new double[n]);
  psi[0] = 0.0; // This value should not be used.
  for (int i = 1; i < n-1; i++) {
    psi[i] = turnAngle(nodes[i-1].z,nodes[i].z,nodes[i+1].z);
  }
  psi[n-1] = 0.0;
  
  // Set up variables for linear equations
  // theta is the angle the path leaves a knot relative to the direction
  // to the next knot
  scoped_array<double> theta(new double[n]);
  // phi is the angle the path enters a knot relative to the direction
  // from the last knot
  // We have psi[k] + theta[k] + phi[k] = 0.0 for all knots
  scoped_array<double> phi(new double[n]);

  // The path angles are determined by solving linear equations in terms
  // of theta. All equations are of the form:
  //     pre[k]*theta[k-1] + piv[k}*theta[k] + post[k]*theta[k+1] = aug[k]
  scoped_array<double> pre(new double[n]);
  scoped_array<double> piv(new double[n]);
  scoped_array<double> post(new double[n]);
  scoped_array<double> aug(new double[n]);

  // If a precontrol is found here, it is the last segment's and only
  // the direction is useful.
  if (leftRest.kind == rest::CONTROL)
    leftdir(nodes[0].z - leftRest.control);
  
  // Encode the equations for the first knot.
  switch (leftRest.kind) {
    case rest::GIVEN: {
      // Simply set theta to the needs value
      pre[0] = 0.0; piv[0] = 1.0; post[0] = 0.0;
      aug[0] = leftRest.given - arg[0];
      if (aug[0] < -PI) aug[0] += 2.0*PI;
      if (aug[0] >  PI) aug[0] -= 2.0*PI;
      break;
    }
    
    case rest::OPEN: {
      leftRest.curl = 1.0;
      // let fall through and be treated as TWIST.
    }

    case rest::TWIST: {
      double alpha = 1.0/nodes[0].right.tension;
      double beta  = 1.0/nodes[1].left.tension;
      double chi   = leftRest.curl * (alpha*alpha)/(beta*beta);

      pre[0]  = 0.0;
      piv[0]  = alpha*chi + 3.0 - beta;
      post[0] = (3.0 - alpha) * chi + beta;
      aug[0]  = -post[0] * psi[1];

      // Ratio capped at four.
      if (piv[0] < TWIST_RATIO_CAP*post[0])
        piv[0] = TWIST_RATIO_CAP*post[0];
    
      break;
    }

    default: {
      assert(False);	
    }
  }

  // Encode the interior knots.
  for (int k = 1; k < n-1; k++)
  {
    double factor;
    double a_prev = 1.0/nodes[k-1].right.tension;
    double alpha  = 1.0/nodes[k].right.tension;
    double beta   = 1.0/nodes[k].left.tension;
    double b_next = 1.0/nodes[k+1].left.tension;
    double A, B, C, D;

    factor = 1.0 / (beta * beta * d[k-1]);
    A = a_prev * factor;
    B = (3.0 - a_prev) * factor;
    
    factor = 1.0 / (alpha * alpha * d[k]);
    C = (3.0 - b_next) * factor;
    D = b_next * factor;

    pre[k]  = A;
    piv[k]  = B + C;
    post[k] = D;
    aug[k]  = -B * psi[k] - D * psi[k+1];
  }

  // If a postcontrol is found here, it is the next segment's and only
  // the direction is useful.
  if (rightRest.kind == rest::CONTROL)
    rightdir(rightRest.control - nodes[n-1].z);
  
  // Encode the last knot.
  switch (rightRest.kind) {
    case rest::GIVEN: {
      // Simply set theta to the needs value
      pre[n-1] = 0.0; piv[n-1] = 1.0; post[n-1] = 0.0;
      aug[n-1] = rightRest.given - arg[n-2];
      if (aug[n-1] < -PI) aug[n-1] += 2.0*PI;
      if (aug[n-1] >  PI) aug[n-1] -= 2.0*PI;
      break;
    }
    
    case rest::OPEN: {
      rightRest.curl = 1.0;
      // let fall through and be treated as TWIST.
    }

    case rest::TWIST: {
      double alpha = 1.0/nodes[n-2].right.tension;
      double beta  = 1.0/nodes[n-1].left.tension;
      double chi   = rightRest.curl * (alpha*alpha)/(beta*beta);

      pre[n-1]  = (3.0 - beta) * chi + alpha;
      piv[n-1]  = beta * chi + 3.0 - alpha;
      post[n-1] = 0.0;
      aug[n-1]  = 0.0;

      // Ratio capped at four.
      if (piv[n-1] < TWIST_RATIO_CAP*pre[n-1])
        piv[n-1] = TWIST_RATIO_CAP*pre[n-1];
    
      break;
    }

    default: {
      assert(False);	
    }
  }

  // Debugging - print out the equations
  if (verbose > 3) {
    fprintf(stderr, "equation to solve:\n");
    fprintf(stderr, "%2s %12s %12s %12s %12s\n",
	   "k", "pre[k]", "piv[k]", "post[k]", "aug[k]");
    for (int k = 0; k < n; k++)
      {
	fprintf(stderr, "%2d %12lf %12lf %12lf %12lf\n", k, pre[k], piv[k], post[k], aug[k]);
      }
    fprintf(stderr, "\n");
  }


  // Now that the equations are setup, we solve for the thetas.
  solveTriDiag(theta.get(), pre.get(), piv.get(), post.get(), aug.get(), n);
  for (int k = 0; k < n; k++)
  {
    phi[k] = -psi[k] - theta[k];
  }

  // Debugging - print out a table of values for info
  if (verbose > 3) {
    fprintf(stderr, "final angles:\n");
    fprintf(stderr, "%2s %12s %12s %12s %12s %12s\n",
	   "k", "d[k]", "arg[k]", "psi[k]", "theta[k]", "phi[k]");
    for (int k = 0; k < n; k++)
      {
	fprintf(stderr, "%2d %12lf %12lf %12lf %12lf %12lf\n",
	       k, d[k], arg[k], psi[k], theta[k], phi[k]);
      }
    fprintf(stderr, "\n");
  }

  // Produce the control points
  v = velocity(theta[0], phi[1],
               nodes[0].right.tension, nodes[0].right.atleast);
  s[0].pre  = s[0].point = nodes[0].z;
  s[0].post = nodes[0].z + v * d[0] * expi(arg[0] + theta[0]);

  for (int k = 1; k < n-1; k++)
  {
    v = velocity(phi[k], theta[k-1],
                 nodes[k].left.tension, nodes[k].left.atleast);
    s[k].pre = nodes[k].z - v * d[k-1] * expi(arg[k-1] - phi[k]);

    s[k].point = nodes[k].z;
  
    v = velocity(theta[k], phi[k+1],
                 nodes[k].right.tension, nodes[k].right.atleast);
    s[k].post = nodes[k].z + v * d[k] * expi(arg[k] + theta[k]);
  }

  v = velocity(phi[n-1], theta[n-2],
               nodes[n-1].left.tension, nodes[n-1].left.atleast);
  s[n-1].pre = nodes[n-1].z - v * d[n-2] * expi(arg[n-2] - phi[n-1]);
  s[n-1].point = s[n-1].post = nodes[n-1].z;

  path p(s, n);

  if (verbose > 3) {
    //cerr << "solving: " << *this << endl;
    cerr << "solved to: " << p << endl;
  }

  return p; 
}

path knotlist::solveCyclic()
{
  if (verbose > 3) cerr << "solving: " << *this << endl;

  // Deal with simple cases.
  int n = (int) nodes.size();
  if (n == 0) {
    //cerr << "solved as empty" << endl;
    return path();
  }
  
  // Copy over the preknot to the first knot.
  nodes[0].left = preknot;

  if (n == 1) {
    //cerr << "solved as pair" << endl;
    return path(nodes[0].z, true);
  }

  // Special duplicate point case of the form a..b..c..a..cycle.
  if (nodes[0].z == nodes[n-1].z)
  {
    // Solve the first part as a normal non-cyclic path.
    path p = solve();

    // Copy over and make cyclic.  This will add the extra a..a segment.
    solvedKnot *nodes = new solvedKnot[n];
    for (int i = 0; i < n; i++) {
      nodes[i].pre = p.precontrol(i);
      nodes[i].point = p.point(i);
      nodes[i].post = p.postcontrol(i);
    }
    return path(nodes, n, true);
  }

  // Special case in the form a..b..c..{}cycle won't be caught at the guide
  // level.
  if (rightRest.kind != rest::OPEN)
  {
     // Translate to the form a{}..b..c..{}a and solve as non-cyclic.
     leftRest = rightRest;
     add(nodes[0].z);
     close();
     path p = solve();

    // Copy over and make cyclic.
    solvedKnot *nodes = new solvedKnot[n];
    for (int i = 0; i < n; i++) {
      nodes[i].pre = p.precontrol(i);
      nodes[i].point = p.point(i);
      nodes[i].post = p.postcontrol(i);
    }
    nodes[0].pre = p.precontrol(n);

    return path(nodes, n, true);
  }

  // The knots with control points as they are solved.
  solvedKnot *s = new solvedKnot[n];

  // arg stores the heading in radians between adjacent knots.
  scoped_array<double> arg(new double[n]);

  // d stores the distance between adjacent knots. 
  scoped_array<double> d(new double[n]);

  for (int i = 0; i < n-1; i++) {
    pair dz = nodes[i+1].z - nodes[i].z;
    arg[i] = dz.angle();
    d[i] = dz.length();
  }
  // and the loop around case.
  {
    pair dz = nodes[0].z - nodes[n-1].z;
    arg[n-1] = dz.angle();
    d[n-1] = dz.length();
  }

  // Temporary variable for the velocity at a knot.
  double v;
  
  // Calculate the turning angles at each knot.
  scoped_array<double> psi(new double[n]);
  // Wrap around case again.
  psi[0] = turnAngle(nodes[n-1].z, nodes[0].z, nodes[1].z);
  for (int i = 1; i < n-1; i++)
    psi[i] = turnAngle(nodes[i-1].z, nodes[i].z, nodes[i+1].z);
  psi[n-1] = turnAngle(nodes[n-2].z, nodes[n-1].z, nodes[0].z);
  
  // Set up variables for linear equations
  // theta is the angle the path leaves a knot relative to the direction
  // to the next knot
  scoped_array<double> theta(new double[n]);
  // phi is the angle the path enters a knot relative to the direction
  // from the last knot
  // We have psi[k] + theta[k] + phi[k] = 0.0 for all knots
  scoped_array<double> phi(new double[n]);

  // The path angles are determined by solving linear equations in terms
  // of theta. All equations are of the form:
  //     pre[k]*theta[k-1] + piv[k}*theta[k] + post[k]*theta[k+1] = aug[k]
  scoped_array<double> pre(new double[n]);
  scoped_array<double> piv(new double[n]);
  scoped_array<double> post(new double[n]);
  scoped_array<double> aug(new double[n]);

  // Encode the first knot.
  {
    double factor;
    double a_prev = 1.0/nodes[n-1].right.tension;
    double alpha  = 1.0/nodes[0].right.tension;
    double beta   = 1.0/nodes[0].left.tension;
    double b_next = 1.0/nodes[1].left.tension;
    double A, B, C, D;

    factor = 1.0 / (beta * beta * d[n-1]);
    A = a_prev * factor;
    B = (3.0 - a_prev) * factor;
    
    factor = 1.0 / (alpha * alpha * d[0]);
    C = (3.0 - b_next) * factor;
    D = b_next * factor;

    pre[0]  = A;
    piv[0]  = B + C;
    post[0] = D;
    aug[0]  = -B * psi[0] - D * psi[1];
  }
  // Encode the interior knots.
  for (int k = 1; k < n-1; k++)
  {
    double factor;
    double a_prev = 1.0/nodes[k-1].right.tension;
    double alpha  = 1.0/nodes[k].right.tension;
    double beta   = 1.0/nodes[k].left.tension;
    double b_next = 1.0/nodes[k+1].left.tension;
    double A, B, C, D;

    factor = 1.0 / (beta * beta * d[k-1]);
    A = a_prev * factor;
    B = (3.0 - a_prev) * factor;
    
    factor = 1.0 / (alpha * alpha * d[k]);
    C = (3.0 - b_next) * factor;
    D = b_next * factor;

    pre[k]  = A;
    piv[k]  = B + C;
    post[k] = D;
    aug[k]  = -B * psi[k] - D * psi[k+1];
  }
  // Encode the last knot.
  {
    double factor;
    double a_prev = 1.0/nodes[n-2].right.tension;
    double alpha  = 1.0/nodes[n-1].right.tension;
    double beta   = 1.0/nodes[n-1].left.tension;
    double b_next = 1.0/nodes[0].left.tension;
    double A, B, C, D;

    factor = 1.0 / (beta * beta * d[n-2]);
    A = a_prev * factor;
    B = (3.0 - a_prev) * factor;
    
    factor = 1.0 / (alpha * alpha * d[n-1]);
    C = (3.0 - b_next) * factor;
    D = b_next * factor;

    pre[n-1]  = A;
    piv[n-1]  = B + C;
    post[n-1] = D;
    aug[n-1]  = -B * psi[n-1] - D * psi[0];
  }

  // Debugging - print out the equations
  if (verbose > 3) {
    fprintf(stderr, "equation to solve:\n");
    fprintf(stderr, "%2s %12s %12s %12s %12s\n",
	   "k", "pre[k]", "piv[k]", "post[k]", "aug[k]");
    for (int k = 0; k < n; k++)
      {
	fprintf(stderr, "%2d %12lf %12lf %12lf %12lf\n",
	       k, pre[k], piv[k], post[k], aug[k]);
      }
    fprintf(stderr, "\n");
  }


  // Now that the equations are setup, we solve for the thetas.
  solveCyclicTriDiag(theta.get(), pre.get(), piv.get(), post.get(), aug.get(), n);
  for (int k = 0; k < n; k++)
  {
    phi[k] = -psi[k] - theta[k];
  }

  // Debugging - print out a table of values for info
  if (verbose > 3) {
    fprintf(stderr, "final angles:\n");
    fprintf(stderr, "%2s %12s %12s %12s %12s %12s\n",
	   "k", "d[k]", "arg[k]", "psi[k]", "theta[k]", "phi[k]");
    for (int k = 0; k < n; k++)
      {
	fprintf(stderr, "%2d %12lf %12lf %12lf %12lf %12lf\n",
	       k, d[k], arg[k], psi[k], theta[k], phi[k]);
      }
    fprintf(stderr, "\n");
  }

  // Produce the control points
  {
    v = velocity(phi[0], theta[n-1],
                 nodes[0].left.tension, nodes[0].left.atleast);
    s[0].pre = nodes[0].z - v * d[n-1] * expi(arg[n-1] - phi[0]);

    s[0].point = nodes[0].z;
  
    v = velocity(theta[0], phi[1],
                 nodes[0].right.tension, nodes[0].right.atleast);
    s[0].post = nodes[0].z + v * d[0] * expi(arg[0] + theta[0]);
  }

  for (int k = 1; k < n-1; k++)
  {
    v = velocity(phi[k], theta[k-1],
                 nodes[k].left.tension, nodes[k].left.atleast);
    s[k].pre = nodes[k].z - v * d[k-1] * expi(arg[k-1] - phi[k]);

    s[k].point = nodes[k].z;
  
    v = velocity(theta[k], phi[k+1],
                 nodes[k].right.tension, nodes[k].right.atleast);
    s[k].post = nodes[k].z + v * d[k] * expi(arg[k] + theta[k]);
  }

  {
    v = velocity(phi[n-1], theta[n-2],
                 nodes[n-1].left.tension, nodes[n-1].left.atleast);
    s[n-1].pre = nodes[n-1].z - v * d[n-2] * expi(arg[n-2] - phi[n-1]);

    s[n-1].point = nodes[n-1].z;
  
    v = velocity(theta[n-1], phi[0],
                 nodes[n-1].right.tension, nodes[n-1].right.atleast);
    s[n-1].post = nodes[n-1].z + v * d[n-1] * expi(arg[n-1] + theta[n-1]);
  }

  path p(s, n, true);

  if (verbose > 3) {
  //cerr << "solving: " << *this << endl;
  cerr << "solved to: " << p << endl;
  }

  return p; 
}

path knotlist::solve2(knotlist &first, knotlist &second)
{
  // Doesn't handle first == second.
  assert (&first != &second);

  if (first.isEmpty())
    return second.solve(); 
  if (second.isEmpty())
    return first.solve();

  // If second knotlist starts where the last left off, we simply bridge
  // them.
  if (first.nodes.back().z == second.nodes.front().z) {
    path p1 = first.solve();
    int n1 = p1.length();
    path p2 = second.solve();
    int n2 = p2.length();

    solvedKnot *nodes = new solvedKnot[n1+n2+2];

    int i = 0;
    nodes[0].pre = p1.point(0);
    for (int j = 0; j < n1; j++) {
      nodes[i].point = p1.point(j);
      nodes[i].post = p1.postcontrol(j);
      nodes[i+1].pre = p1.precontrol(j+1);
      i++;
    }

    // The bridge
    nodes[i].point = nodes[i].post = nodes[i+1].pre = p2.point(0);
    i++;
    
    for (int j = 0; j < n2; j++) {
      nodes[i].point = p2.point(j);
      nodes[i].post = p2.postcontrol(j);
      nodes[i+1].pre = p1.precontrol(j+1);
      i++;
    }
    nodes[i].point = nodes[i].post = p2.point(n2);

    return path(nodes, i+1);
  }
 
  // If there are no restrictions at the join point, we can solve it as
  // one big list.
  if (first.rightRest.kind == rest::OPEN) {
    if (second.leftRest.kind == rest::OPEN) {
      first.add(second);
      return first.solve();
    }
    else {
      first.rightRest = second.leftRest;
    }
  }
  else {
    if (second.leftRest.kind == rest::OPEN) {
      second.leftRest = first.rightRest;
    }
  }

  // Solve independently.
  first.add(second.nodes[0].z);
  path p1 = first.solve();
  path p2 = second.solve();
  return concat(p1, p2);
}
  

ostream& operator<< (ostream& out, const knotlist& kl)
{
  int n = (int) kl.nodes.size();
 
  if (n == 0)
    return out << "<empty>";
 
  // Print the first knot and restriction
  out << kl.nodes[0].z;
  if (kl.leftRest.kind == rest::TWIST)
    out << "{curl " << kl.leftRest.curl << "}";
  else if (kl.leftRest.kind == rest::GIVEN)
    out << "{dir(" << kl.leftRest.given << ")}";

  // Print the rest
  for (int k = 1; k < n; k++)
    // Tensions will not be printed
    out << ".." << kl.nodes[k].z;

  // Print the right restriction
  if (kl.rightRest.kind == rest::TWIST)
    out << "{curl " << kl.rightRest.curl << "}";
  else if (kl.rightRest.kind == rest::GIVEN)
    out << "{dir(" << kl.rightRest.given << ")}";

  return out;
}

} // namespace camp
