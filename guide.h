/*****
 * guide.h
 * Andy Hammerlindl 2005/02/23
 *
 *****/

#ifndef GUIDE_H
#define GUIDE_H

#include <iostream>
#include "knot.h"
#include "flatguide.h"
#include "settings.h"

using std::cerr;
using std::endl;

namespace camp {

// Abstract base class for guides.
class guide : public gc {
public:
  // Returns the path that the guide represents.
  virtual path solve() {
    return path();
  }

  // Add the information in the guide to the flatguide, so that it can be solved
  // via the knotlist solving routines.
  virtual void flatten(flatguide&) {}

  virtual void print(ostream& out) const {
    out << "nullguide";
  }
  
  // Needed so that multiguide can know where to put in ".." symbols.
  virtual side printLocation() const {
    return END;
  }
};

inline ostream& operator<< (ostream& out, const guide& g)
{
  g.print(out);
  return out;
}

// Draws dots between two printings of guides, if their locations are such that
// the dots are necessary.
inline void adjustLocation(ostream& out, side l1, side l2)
{
  if (l1 == END)
    out << std::endl;
  if ((l1 == END || l1 == OUT) && (l2 == IN || l2 == END))
    out << "..";
}

// A guide representing a pair.
class pairguide : public guide {
  pair z;

public:
  void flatten(flatguide& g) {
    g.add(z);
  }

  pairguide(pair z)
    : z(z) {}

  path solve() {
    return path(z);
  }

  void print(ostream& out) const {
    out << z;
  }
  
  side printLocation() const {
    return END;
  }
};


// A guide representing a path.
class pathguide : public guide {
  path p;

public:
  void flatten(flatguide& g) {
    g.add(p);
  }

  pathguide(path p)
    : p(p) {}

  virtual ~pathguide() {}

  path solve() {
    return p;
  }

  void print(ostream& out) const {
    out << p;
  }
  
  side printLocation() const {
    return END;
  }
};

// A guide giving tension information (as part of a join).
class tensionguide : public guide {
  tension tout,tin;

public:
  void flatten(flatguide& g) {
    g.setTension(tin,IN);
    g.setTension(tout,OUT);
  }

  tensionguide(tension tout,tension tin)
    : tout(tout),tin(tin) {}
  tensionguide(tension t)
    : tout(t),tin(t) {}

  void print(ostream& out) const {
    out << (tout.atleast ? ".. tension atleast " : ".. tension ")
        << tout.val << " and " << tin.val << " ..";
  }
  
  side printLocation() const {
    return JOIN;
  }
};

// A guide giving a specifier.
class specguide : public guide {
  spec *p;
  side s;

public:
  void flatten(flatguide& g) {
    g.setSpec(p,s);
  }
  
  specguide(spec *p, side s)
    : p(p), s(s) {}

  void print(ostream& out) const {
    out << *p;
  }
  
  side printLocation() const {
    return s;
  }
};

// A guide for explicit control points between two knots.  This could be done
// with two specguides, instead, but this prints nicer, and is easier to encode.
class controlguide : public guide {
  pair zout, zin;

public:
  void flatten(flatguide& g) {
    g.setSpec(new controlSpec(zout), OUT);
    g.setSpec(new controlSpec(zin), IN);
  }

  controlguide(pair zout,pair zin)
    : zout(zout),zin(zin) {}
  controlguide(pair z)
    : zout(z),zin(z) {}

  void print(ostream& out) const {
    out << ".. controls "
        << zout << " and " << zin << " ..";
  }
  
  side printLocation() const {
    return JOIN;
  }
};

// A guide that is a sequence of other guide.  This is used, for instance is
// joins, where we have the left and right guide, and possibly specifiers and
// tensions in between.
typedef vector<guide *,gc_allocator<guide *> > guidevector;

class multiguide : public guide {
  guidevector v;

public:
  void flatten(flatguide& g);

  multiguide(guidevector& v)
    : v(v) {}

  virtual ~multiguide() {}

  path solve() {
    if (settings::verbose>3) {
      cerr << "solving guide:\n";
      print(cerr); cerr << "\n\n";
    }
    
    flatguide g;
    this->flatten(g);
    path p=g.solve(false);

    if (settings::verbose>3)
      cerr << "solved as:\n" << p << "\n\n";

    return p;
  }

  void print(ostream& out) const;
  
  side printLocation() const {
    return v.back()->printLocation();
  }
};

#if 0
// A wrapper around another guide that signifies that the guide should be solved
// cyclically.
class cyclicguide : public guide {
  // The guide to be solved.
  guide *core;
public:
  cyclicguide(guide *core)
    : core(core) {}

  void flatten(flatguide& g) {
    // If cycles occur in the midst of a guide, the guide up to that point
    // should be solved and added as a path.
    pathguide(this->solve()).flatten(g);
  }

  path solve() {
    if (settings::verbose>3) {
      cerr << "solving guide:\n";
      print(cerr); cerr << "\n\n";
    }
    
    flatguide g;
    core->flatten(g);
    path p=g.solve(true);

    if (settings::verbose>3)
      cerr << "solved as:\n" << p << "\n\n";

    return p;
  }

  void print(ostream& out) const {
    core->print(out);
    side loc=core->printLocation();
    adjustLocation(out,loc,this->printLocation());
    out << "cycle";
  }

  side printLocation() const {
    return END;
  }
};
#endif

// A guide representing the cycle token.
class cycletokguide : public guide {
public:
  void flatten(flatguide& g) {
    // If cycles occur in the midst of a guide, the guide up to that point
    // should be solved as a path.  Any subsequent guide will work with that
    // path locked in place.
    g.solve(true);
  }

  path solve() {
    // Just a cycle on it's own makes an empty guide.
    return path();
  }

  void print(ostream& out) const {
    out << "cycle";
  }

  side printLocation() const {
    return END;
  }
};

} // namespace camp

#endif // GUIDE_H
