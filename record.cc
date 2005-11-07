/*****
 * record.cc
 * Tom Prince 2004/07/15
 *
 * The type for records and modules in the language.
 *****/

#include "record.h"
#include "inst.h"
#include "runtime.h"

namespace types {

record::record(symbol *name, frame *level)
  : ty(ty_record),
    name(name),
    level(level),
    init(new vm::lambda),
    e()
{
  assert(init);
}

record::~record()
{}

record *record::newRecord(symbol *id, bool statically)
{
  frame *underlevel = getLevel(statically);
  assert(underlevel);
    
  frame *level = new frame(underlevel, 0);

  record *r = new record(id, level);
  return r;
}

// Initialize to null by default.
trans::access *record::initializer() {
  static trans::bltinAccess a(run::pushNullRecord);
  return &a;
}

} // namespace types
