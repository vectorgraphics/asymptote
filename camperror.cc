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

namespace camp {

bool errorFlag;

std::queue<std::string> errorQueue; 

// Used internally to report an error in an operation.
void reportError(std::string desc)
{
  errorFlag = true;
  errorQueue.push(desc);
}

// Copies the description of the oldest unretrieved error into the
// buffer.  Once all errors have had their descriptions retrieve this
// way, errors() will once again return false.
std::string getError()
{
  if (!errorFlag)
    return std::string();

  assert(!errorQueue.empty());

  std::string desc = errorQueue.front();
  errorQueue.pop();
  
  if (errorQueue.empty())
    errorFlag = false;
  return desc;
}

} // namespace camp
