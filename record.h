/*****
 * record.h
 * Andy Hammerlindl 2003/07/09
 *
 * The type for records and modules in the language.
 *****/

#ifndef RECORD_H
#define RECORD_H

#include "types.h"
#include "entry.h"
#include "frame.h"
#include "inst.h"
#include "access.h"

using trans::frame;
using trans::venv;
using trans::tenv;
using trans::varEntry;
using vm::lambda;

namespace types {

struct record : public ty {
  // The base name of this type.
  symbol *name;
  
  // The frame.  Like a frame for a function, it allocates the accesses
  // for fields and specifies the size of the record.
  frame *level;
  
  // The name bindings for fields of the record.
  tenv te;
  venv ve;

  // The runtime representation of the record used by the virtual
  // machine.
  vm::lambda *init;

public:
  record(symbol *name, frame *level, lambda *init);
  ~record();

  void addType(symbol *name, ty *desc)
  {
    te.enter(name, desc);
  }

  void addVar(symbol *name, varEntry *desc)
  {
    ve.enter(name, desc);
  }

  void list()
  {
    ve.list();
  }

  ty *lookupType(symbol *s)
  {
    return te.look(s);
  }

  varEntry *lookupExactVar(symbol *name, signature *sig)
  {
    return ve.lookExact(name, sig);
  }

  ty *varGetType(symbol *name)
  {
    return ve.getType(name);
  }

  symbol *getName()
  {
    return name;
  }

  virtual bool isReference() {
    return true;
  }

  frame *getLevel(bool statically = false)
  {
    if (statically)
      return level->getParent();
    else
      return level;
  }

  lambda *getInit()
  {
    return init;
  }

  // Allocates a new dynamic field in the record.
  trans::access *allocField(bool statically, trans::permission p)
  {
    frame *underlevel = getLevel(statically);
    assert(underlevel);
    return underlevel->allocLocal(p);
  }

  // Create a statically enclosed record from this record.
  record *newRecord(symbol *id, bool statically)
  {
    frame *underlevel = getLevel(statically);
    assert(underlevel);
    
    frame *level = new frame(underlevel, 0);

    lambda *init = new lambda;

    record *r = new record(id, level, init);
    return r;
  }

  void print(ostream& out) const
  {
    out << *name;
  }

  void debug(ostream& out) const
  {
    out << "struct " << *name << std::endl;
    out << "types:" << endl;
    out << te;
    out << "fields: " << endl;
    out << ve;
  }
};

} //namespace types

#endif  
