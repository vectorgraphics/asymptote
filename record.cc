/*****
 * record.cc
 * Tom Prince 2004/07/15
 *
 * The type for records and modules in the language.
 *****/

#include "record.h"

namespace types {

record::record(symbol *name, frame *level, lambda *init)
  : ty(ty_record),
    name(name),
    level(level),
    te(), ve(),
    runtime()
{
  assert(level);
  assert(init);
  runtime.size = 0;
  runtime.init = init;
}

record::~record()
{}

} // namespace types
