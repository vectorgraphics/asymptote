/*****
 * guide.cc
 * Andy Hammerlindl 2002/08/28
 *
 * Defines the join and cycle guides that allow robust path making
 * possible.  All joins take two guides for left and right, making each
 * guide a binary tree
 *****/

#include "guide.h"
#include "settings.h"

namespace camp {
using settings::verbose;

guide::~guide()
{}

static void
safeAddPoint(pair z, knotlist &left, pathlist &solved, knotlist &right)
{
  // Adds a pair, but in the case where there are two identical pairs 
  // in a row, it adds a special path.
  if (left.isOpen()) {
    if (left.duplicate(z)) {
      left.close();

      // Create the path in one place.
      solvedKnot *nodes = new solvedKnot[2];
      nodes[0].pre = nodes[0].point = nodes[0].post = z;
      nodes[1].pre = nodes[1].point = nodes[1].post = z;
      solved.add(path(nodes, 2));

      right.clear();
      right.add(z);
    }
    else {
      left.add(z); 
      if (!left.isOpen()) {
	right.continuation(left);
      }
    }
  }
  else {
    if (right.duplicate(z)) {
      right.close();
      if (!right.isEmpty())
        solved.add(right.solve());

      // Create the path in one place.
      solvedKnot *nodes = new solvedKnot[2];
      nodes[0].pre = nodes[0].point = nodes[0].post = z;
      nodes[1].pre = nodes[1].point = nodes[1].post = z;
      solved.add(path(nodes, 2));

      right.clear();
      right.add(z);
    }
    else {
      right.add(z);
      if (!right.isOpen()) {
	if (!right.isEmpty())
	  solved.add(right.solve());

	right.continuation(right);
      }
    }
  }
}
      
      
  

void pairguide::subsolve(knotlist &left, pathlist &solved, knotlist &right)
{
  safeAddPoint(z, left, solved, right);

  solved.setCyclic(false);
} 
  
path pairguide::solve()
{
  return path(z);
}

void pairguide::print(ostream& out) const
{
  out << z;
}

void pathguide::subsolve(knotlist &left, pathlist &solved, knotlist &right)
{
  int n = p.size();
 
  // NOTE: Decide what to do if null path goes into guide.
  // A possible option is to pretend there is nothing there. 
  assert(n >= 0);
  if(n == 0) return; // do nothing

  safeAddPoint(p.point(0), left, solved, right);

  // If path is one point, there is nothing more to add.
  if (n == 1) return;
 
  // General case.
  if (left.isOpen())
  {
    left.rightgiven(angle(p.direction(0)));
    left.close();
  
    solved.add(path(p));
    
    right.leftgiven(angle(p.direction(n-1)));
    right.add(p.point(n-1));
  }
  else
  {
    right.rightgiven(angle(p.direction(0)));
    right.close();

    if (!right.isEmpty())
      solved.add(right.solve());
    solved.add(path(p));

    right.clear();
    right.leftgiven(angle(p.direction(n-1)));
    right.add(p.point(n-1));
  }

  solved.setCyclic(false);
}

path pathguide::solve()
{
  return p;
}

void pathguide::print(ostream &out) const
{
  out << "(path)(" << p << ")";
}
  

void join::outsolve(knotlist &left, pathlist &solved, knotlist &right)
{
  if (out.kind == rest::OPEN)
    return;

  // All restrictions given to knotlists are to their rightside, as the
  // left sides are always determined by the preceding section

  if (left.isOpen()) {
    if (out.kind == rest::GIVEN)
      left.rightgiven(out.given);
    else
      left.rightcurl(out.curl);
    left.close();
    right.continuation(left);  
  }
  else
  {
    if (out.kind == rest::GIVEN)
      right.rightgiven(out.given);
    else
      right.rightcurl(out.curl);
    right.close();
    if (!right.isEmpty())
      solved.add(right.solve());
    right.continuation(right);
  }
}

void join::insolve(knotlist &left, pathlist &, knotlist &right)
{
  if (in.kind == rest::OPEN)
    return;

  if (left.isOpen()) {
    if (in.kind == rest::GIVEN)
      left.rightgiven(in.given);
    else
      left.rightcurl(in.curl);
  }
  else {
    if (in.kind == rest::GIVEN)
      right.rightgiven(in.given);
    else
      right.rightcurl(in.curl);
  }
} 

void join::subsolve(knotlist &left, pathlist &solved, knotlist &right)
{
  leftGuide->subsolve(left, solved, right);
  outsolve(left, solved, right);

  if (tensionSpec) {
    if (left.isOpen())
      left.tension(tensionLeft, tensionRight, tatleast);
    else
      right.tension(tensionLeft, tensionRight, tatleast);
  }
  
  if (controlSpec) {
    if (left.isOpen())
    {
      left.rightcontrol(controlLeft);
      left.close();

      right.continuation(left);
      right.controls(controlLeft, controlRight);
    }
    else
    {
      right.rightcontrol(controlLeft);
      right.close();

      if (!right.isEmpty())
	solved.add(right.solve());

      right.continuation(right);
      right.controls(controlLeft, controlRight);
    }
  }
  
  insolve(left, solved, right);
  rightGuide->subsolve(left, solved, right);
}

path join::solve()
{
  if (verbose > 3) std::cerr << "solving: " << *this << std::endl;
 
  knotlist left; 
  pathlist solved;
  knotlist right; 
  
  subsolve(left, solved, right);
  
  if (!left.isEmpty())
    solved.addToFront(left.solve());
  if (!right.isEmpty())
    solved.add(right.solve());

  path p = solved.solve();

  return p;
}

void join::print(ostream& out) const
{
  this->leftGuide->print(out);
  this->out.print(out);
   
  if (tensionSpec) 
    out << ".. tension " << tensionLeft << " and " << tensionRight << " ..";
  else if (controlSpec)
    out << ".. controls " << controlLeft << " and " << controlRight << " ..";
  else
    out << "..";

  this->in.print(out);
  this->rightGuide->print(out); 
}


void dirguide::subsolve(knotlist &left, pathlist &solved, knotlist &right)
{
  head->subsolve(left, solved, right);

  if (tag.kind == rest::OPEN)
    return;

  // All restrictions given to knotlists are to their rightside, as the
  // left sides are always determined by the preceding section

  if (left.isOpen()) {
    if (tag.kind == rest::GIVEN)
      left.rightgiven(tag.given);
    else
      left.rightcurl(tag.curl);
    left.close();
    right.continuation(left);  
  }
  else
  {
    if (tag.kind == rest::GIVEN)
      right.rightgiven(tag.given);
    else
      right.rightcurl(tag.curl);
    right.close();
    if (!right.isEmpty())
      solved.add(right.solve());
    right.continuation(right);
  }
}

path dirguide::solve()
{
  if (verbose > 3) std::cerr << "solving: " << *this << std::endl;
 
  knotlist left;
  pathlist solved;
  knotlist right;
  
  subsolve(left, solved, right);
  
  if (!left.isEmpty())
    solved.addToFront(left.solve());
  if (!right.isEmpty())
    solved.add(right.solve());

  path p = solved.solve();

  return p;
}

void dirguide::print(ostream& out) const
{
  this->head->print(out);
  this->tag.print(out);
}

void cycle::subsolve(knotlist &left, pathlist &solved, knotlist &right)
{
  if (!left.isEmpty()) {
    if (left.isOpen()) {
      // We have a fully cyclic path, ie. cyclic with no restrictions.
      path p = left.solveCyclic();

      left.clear();
      left.add(p.point(0));
      left.rightdir(p.direction(0));
      left.close();

      right.clear();
      right.add(p.point(0));
      right.leftdir(p.direction(0));

      solved.add(p);
    }
    else {
      int offset = (int) right.size();

      // The mechanisms of solving are complicated, and so are handled at
      // the knotlist level.
      path p = knotlist::solve2(right, left);

      left.clear();
      left.add(p.point(offset));
      left.rightdir(p.direction(offset));
      left.close();

      right.clear();
      right.add(p.point(offset));
      right.leftdir(p.direction(offset));

      solved.add(p, offset);
    }
  }

  solved.setCyclic(true);
}

path cycle::solve()
{
  // A cycle is nothing on its own.
  return path();
}

void cycle::print(ostream& out) const
{
  out << "cycle";
}

} // namespace camp
