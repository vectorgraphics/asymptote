/*****
 * guide.h
 * Andy Hammerlindl 2002/8/22
 *
 * A guide is any object that describes a path or part of a path.  In
 * this way, any pair, path, join, or cycle is technically a guide. In
 * the most common sense, an expression like
 *
 *   a..b..c
 *
 * where a, b, and c are pairs. describes a path.  This expression
 * actually contains five guides.  The three pairs are all guides.  The
 * first ".." join is a guide with children guides "a" and "b".  The second
 * ".." join is a guide with children guides "a..b" and "c".  In this
 * way any guide expression will be stored as a binary tree.
 *
 * In order to be useful, a guide must eventually be solved into a path.
 * As a guide can be cyclic, and can have direction angles or curls in its
 * expression, it is not easy to tell what sections to solve. To account for
 * this, a guide is solved by traversing the binary tree with three
 * variables.  The first, left, is a list of knots that are unsolved at
 * the start of the guide, its solution will depend on whether or not
 * the path is cyclic, so it cannot be solved till the entire tree is
 * traversed.  The solved variable holds a path that represents the
 * solved section of the guide.  The last variable, right, holds the
 * list of knots after the last solved section.  Once it gets enough
 * information, it will be solved and the solved path updated.
 *
 * The subsolve() function handles this, but the last step of the
 * solve() function is to insure that all the pieces have been solved
 * and then to return the new solved path.  Because the guide might be
 * used again as part of a larger guide, it is important that solve()
 * and subsolve() produce no side-effects on the guide.
 *****/

#ifndef GUIDE_H
#define GUIDE_H

#include <iostream>

#include "pool.h"
#include "pair.h"
#include "knotlist.h"
#include "pathlist.h"

namespace camp {

class guide : public memory::managed<guide> {
public:
  virtual void subsolve(knotlist &left, pathlist &solved, knotlist &right) = 0;
  virtual path solve() = 0;
  virtual void print(ostream& out) const = 0 ;
  virtual ~guide();
};

// Output
inline ostream& operator<< (ostream& out, const guide& g)
{
  g.print(out);
  
  return out;
}

// A guide consisting only of a pair
class pairguide : public guide {
  pair z;

public:
  pairguide(pair z)
    : z(z) {}
  virtual ~pairguide() {}

  void  subsolve(knotlist &left, pathlist &solved, knotlist &right);
  path solve();
  void print(ostream& out) const;
};

// A guide consisting only of a path
class pathguide : public guide {
  path p;

public:
  pathguide(path p)
    : p(p) {}
  virtual ~pathguide() {}

  void  subsolve(knotlist &left, pathlist &solved, knotlist &right);
  path solve();
  void print(ostream& out) const;
};

class join : public guide {

  void outsolve(knotlist &left, pathlist &solved, knotlist &right);
  void insolve(knotlist &left, pathlist &solved, knotlist &right);

  guide *leftGuide;
  rest out; // Out of the last knot (not out of the join)
  
  bool tensionSpec;
  double tensionLeft;
  double tensionRight;
  bool tatleast;

  bool controlSpec;
  pair controlLeft;
  pair controlRight;
  
  rest in; // Into the next knot (not into the join)
  guide *rightGuide;

public:
  join(guide* left, guide* right)
    : leftGuide(left),
      tensionSpec(false),
      tatleast(false),
      controlSpec(false),
      rightGuide(right) {}

  virtual ~join() {}
  
  void dirout(pair z) {
    if (z.nonZero()) {
      out.kind = rest::GIVEN;
      out.given = z.angle();
    }
    else {
      out.kind = rest::OPEN;
    }
  }

  void dirin(pair z) {
    if (z.nonZero()) {
      in.kind = rest::GIVEN;
      in.given = z.angle();
    }
    else {
      in.kind = rest::OPEN;
    }
  }

  void curlout(double c) {
    out.kind = rest::TWIST;
    out.curl = c;
  }

  void curlin(double c) {
    in.kind = rest::TWIST;
    in.curl = c;
  }

  void tension(double l, double r) {
    tensionSpec = true;
    tensionLeft = l;
    tensionRight = r;
  }

  void tensionAtleast() {
    tatleast = true;
  }

  void controls(pair l, pair r) {
    controlSpec = true;
    controlLeft = l;
    controlRight = r;
  }

  void subsolve(knotlist &left, pathlist &solved, knotlist &right); 
  path solve();
  void print(ostream& out) const;
    
};

// The case of a direction specifier tagged onto a guide such as a..b..c{}
class dirguide : public guide {
  guide *head;
  rest tag;
  
public:
  dirguide(guide *head)
    : head(head) {}

  virtual ~dirguide() {}
  
  void dir(pair z) {
    if (z.nonZero()) {
      tag.kind = rest::GIVEN;
      tag.given = z.angle();
    }
    else {
      tag.kind = rest::OPEN;
    }
  }

  void curl(double c) {
    tag.kind = rest::TWIST;
    tag.curl = c;
  }

  void subsolve(knotlist &left, pathlist &solved, knotlist &right);
  path solve();
  void print(ostream& out) const;
};

// The reserved word cycle counts as a guide of its own, even though the
// syntax does not allow the user to use the cycle keyword outside a
// join.
class cycle : public guide {
public:
  void subsolve(knotlist &left, pathlist &solved, knotlist &right);
  path solve();
  void print(ostream& out) const;
};

} //namespace camp

#endif
