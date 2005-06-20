/*****
 * record.cc
 * Tom Prince 2004/07/15
 *
 * The type for records and modules in the language.
 *****/

#include "record.h"
#include "inst.h"

namespace types {

record::record(symbol *name, frame *level, vm::lambda *init)
  : ty(ty_record),
    name(name),
    level(level),
    te(), ve(),
    init(init)
{
  assert(level);
  assert(init);
}

record::~record()
{}

record *record::newRecord(symbol *id, bool statically)
{
  frame *underlevel = getLevel(statically);
  assert(underlevel);
    
  frame *level = new frame(underlevel, 0);

  vm::lambda *init = new vm::lambda;

  record *r = new record(id, level, init);
  return r;
}

} // namespace types
