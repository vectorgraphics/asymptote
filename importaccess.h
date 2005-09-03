/*****
 * importaccess.h
 * Andy Hammerlindl 2003/12/06
 *
 * When records in an import are used as variables without being
 * qualified with the name of the import, the access to the variable has
 * to know that the record should be loaded first.  This access keeps
 * location and frame info for the record, so that it can be loaded on
 * to the stack before the field.
 *****/

#ifndef IMPORTACCESS_H
#define IMPORTACCESS_H

#include "access.h"
#include "frame.h"

namespace trans {

class importAccess : public access {
private:
  // The location and frame of the record.
  access *rloc;
  frame *rlevel;

  // The location of the field (use of this access requires the loading
  // of the record).
  access *floc;

public:
  importAccess(access *rloc, frame *rlevel, access *floc)
    : rloc(rloc), rlevel(rlevel), floc(floc) {}

  void encode(action act, position pos, coder &e)
  {
    rloc->encode(READ, pos, e);
    floc->encode(act, pos, e, rlevel);
  }

  void encode(action act, position pos, coder &e, frame *top)
  {
    rloc->encode(READ, pos, e, top);
    floc->encode(act, pos, e, rlevel);
  }
};

} // namespace trans

#endif
