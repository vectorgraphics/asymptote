/*****
 * knotlist.h 
 * Andy Hammerlindl 20002/08/22
 *
 * A knotlist describes a section of a path that can be solved on its
 * own.  This means all knots in it will be open except for the
 * endpoints which may have curls or directions specified.  The tension
 * in and out of each knot must also be stored.
 *
 * Because direction or curl information can come before or after a
 * knot, a knotlist can get a closing restriction, but still need the
 * last knot.  To store this information, we need a variable 'open' to see
 * if the knotlist is still is accepting knots.  A knotlist needs to be
 * closed to be solved.
 *
 * NOTE: Solving of fully cyclic knots still must be addressed.
 *****/

#ifndef KNOTLIST_H
#define KNOTLIST_H

#include <iostream>
#include <vector>

#include "pair.h"
#include "path.h"

namespace camp {

using std::vector;
using std::ostream;

// Restrictions need to be placed on the endpoints of a section of
// path in order to solve it.  Open is equivilent to a curl of one for
// non cyclic paths.
struct rest {
  enum {
    GIVEN,
    TWIST, // TWIST is used here as CURL is unfortunately #defined by bison 
    OPEN,
    CONTROL // Explicit controls in an adjacent segment can affect the
            // direction of the path in this segment.
  } kind;
  double curl;
  double given;  // Angle in radians
  pair control;

  rest()
    : kind(OPEN), curl(1.0), given(0.0), control(0.0,0.0) {}

  // Used for printing guides.
  void print(ostream &out) const
  {
    switch (kind) {
    case GIVEN:
      out << "{dir(" << given << ")}";
      break;
    case TWIST:
      out << "{curl " << curl << "}";
      break;
    case CONTROL:
      out << "{<CONTROL> " << control << "}";
      break;
    default:
      // To stop compiler from issuing warning.
      break;
    }
  }
};

inline ostream& operator<< (ostream& out, const rest& r) {
  r.print(out); return out;
}

class knotlist {

  struct knotside {
    // The tension on that side of the knot
    double tension;

    // If the controls should be bounded within a triangle
    bool   atleast;

    knotside()
      : tension(1.0), atleast(false) {}
  };

  struct knot {
    knotside left;
    pair z;
    knotside right;
  };

  bool open;
  
  rest leftRest; 
  vector<knot> nodes;
  rest rightRest;

  // Due to the tree structure of the guides, control specicifications
  // cannot be handled in a straight forward manner.  They are an
  // exceptional case handled here. The coordinates are stored in the
  // restrictions.
  bool controlSpec;
  

  // Tension descriptions for a yet unadded knot.
  knotside preknot;

public:

  knotlist() 
    : open(true), controlSpec(false) {}

  bool isOpen() const
  {
    return open;
  }
  
  bool isEmpty() const
  {
    return nodes.empty();
  }

  size_t size() const
  {
    return nodes.size();
  }

  void close()
  {
    open = false;
  }

  // Checks if a pair that is a candidate for adding will be a repeat of
  // the last pair added.
  bool duplicate(pair z)
  {
    if (nodes.empty()) return false;
    return nodes.back().z == z;
  }

  void add(pair z);
  
  // This is used in a cyclic knot when the front is added to the back
  void add(knotlist &splice);

  void tension(double l, double r, bool atleast = false);
  void tension(double lr, bool atleast = false)
  {
    tension(lr, lr, atleast);
  }

  void leftcurl(double curl)
  {
    if (curl < 0.0) curl = 0.0;
    leftRest.kind = rest::TWIST;
    leftRest.curl = curl;
  }

  void leftgiven(double given)
  {
    leftRest.kind = rest::GIVEN;
    leftRest.given = given;
  }

  void leftdir(pair z)
  {
    if (z.nonZero())
      leftgiven(angle(z));
    else
      leftRest.kind = rest::OPEN;
  }

  void rightcurl(double curl)
  {
    if (curl < 0.0) curl = 0.0;
    rightRest.kind = rest::TWIST;
    rightRest.curl = curl;
  }

  void rightgiven(double given)
  {
    rightRest.kind = rest::GIVEN;
    rightRest.given = given;
  }

  void rightdir(pair z)
  {
    if (z.nonZero())
      rightgiven(angle(z));
    else
      rightRest.kind = rest::OPEN;
  }

  // A control from another segment will dictate the direction of the
  // path leaving this segment.
  void rightcontrol(pair z)
  {
    rightRest.kind = rest::CONTROL;
    rightRest.control = z;
  }

  void controls(pair zl, pair zr)
  {
    // In the obscure case of a..controls b and c..{}d, the right
    // restriction will overright the CONTROL restriction, but the extra
    // controlSpec flag ensures that the right path segment will be
    // created anyway.
    controlSpec = true;

    leftRest.kind = rest::CONTROL; leftRest.control = zl;
    rightRest.kind = rest::CONTROL; rightRest.control = zr;
  }

  // Blank out knotlist to its initial state.
  void clear();

  // Replaces this knotlist with the last point of the given knotlist
  // and sets the left restriction to the specified right restriction
  void continuation(knotlist &last);

  // Solve the non-cyclic section and return the resulting section of
  // path.
  path solve();

  // Solve the path so that the finish loops around to the start.
  path solveCyclic();

  // Joins second on to the end of first as in the case of a cycle and
  // returns the path.  This respects restrictions between the splice.
  // Warning: the right and left path may be altered.
  static path solve2(knotlist &first, knotlist &second);

  // Debugging output
  friend ostream& operator<< (ostream& out, const knotlist& kl);
};
  
}

#endif
