/*****
 * camperror.cc
 * 2003/02/25 Andy Hammerlindl
 *
 * Provides a way for the classes in camp to report errors in
 * computation elegantly.  After running a method on a camp object that
 * could encounter an error, the program should call camp::errors to see
 * if any errors were encountered.
 *****/

#include <queue>
#include <cassert>

#include "camperror.h"
#include "stack.h"
#include "errormsg.h"

namespace camp {

// Used internally to report an error in an operation.
void reportError(std::string desc)
{
  em->runtime(vm::getPos());
  *em << "camp: " << desc;
  em->sync();
  throw handled_error(); 
}

} // namespace camp
