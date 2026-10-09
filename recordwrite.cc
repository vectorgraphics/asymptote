//
// recordwrite.cc
//
// Implements the callRecordWriteMethod helper for writing struct values
// via their write(file, suffix) method in the heterogeneous write() var
// handler.
//

#include "stack.h"
#include "types.h"
#include "record.h"
#include "entry.h"
#include "access.h"
#include "callable.h"
#include "fileio.h"
#include "util.h"

namespace run {

// A no-op suffix: pops the file argument and does nothing.
static void noSuffixBltin(vm::stack *s)
{
  (void)vm::pop<camp::file *>(s);
}

bool isWriteSuffixType(types::ty *t)
{
  static types::function *suffixType = new types::function(
    types::primVoid(), types::formal(types::primFile()));
  return t->kind == types::ty_function && types::equivalent(t, suffixType);
}

void callRecordWriteMethod(vm::stack *s, vm::vmFrame *frame,
                           types::ty *t, camp::file *f)
{
  // Cast to record type (safe because kind is ty_record).
  types::record *recType = dynamic_cast<types::record *>(t);
  if (!recType) return;

  // Construct the expected type of the write method: void (file, void (file))
  types::function *suffixType = new types::function(
    types::primVoid(), types::formal(types::primFile()));
  types::function *writeType = new types::function(
    types::primVoid(),
    types::formal(types::primFile()),
    types::formal(suffixType));

  // Look up the "write" method in the record's protoenv.
  varEntry *ve = recType->e.lookupVarByType(symbol::trans("write"), writeType);
  if (!ve) return;

  // Get the access for the method.
  trans::access *loc = ve->getLocation();
  trans::localAccess *la = dynamic_cast<trans::localAccess *>(loc);
  if (!la) return;

  // Navigate the frame chain from the record's level to the method's level.
  // The frame chain is linked via the parentIndex of each frame level.
  trans::frame *target = la->getLevel();
  trans::frame *current = recType->getLevel();
  vm::vmFrame *targetFrame = frame;

  if (target != current) {
    // Walk up the frame chain.
    trans::frame *level = current;
    while (level && level != target) {
      targetFrame = vm::get<vm::vmFrame *>((*targetFrame)[level->parentIndex()]);
      level = level->getParent();
    }
    if (!targetFrame) return;
  }

  // Read the callable from the frame at the method's offset.
  vm::item &methodItem = (*targetFrame)[la->getOffset()];
  vm::callable *method = vm::get<vm::callable *>(methodItem);
  if (!method) return;

  // Call the method: push args (file, no-op suffix), then call.
  static vm::bfunc noSuffix(noSuffixBltin);
  s->push((vm::item)f);
  s->push((vm::item)(vm::callable *)&noSuffix);
  method->call(s);
}

} // namespace run
