/*****
 * virtualfieldaccess.cc
 * Andy Hammerlindl 2009/07/23
 *
 * Implements the access subclass used to read and write virtual fields.
 *****/

#include "virtualfieldaccess.h"
#include "coder.h"

namespace trans {

inline void virtualFieldAccess::encode(action act, position pos, coder &e)
{
  switch(act) {
    case CALL:
      this->encode(READ, pos, e);
      e.encode(inst::popcall);
      return;
    case READ:
      assert(getter);
      getter->encode(CALL, pos, e);
      return;
    case WRITE:
      if (setter)
        setter->encode(CALL, pos, e);
      else {
        em.error(pos);
        em << "virtual field is read-only";
      }
      return;
  }
}

} // namespace trans
