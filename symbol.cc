/*****
 * symbol.cc
 * Andy Hammerlindl 2002/06/18
 *
 * Creates symbols from strings so that multiple calls for a symbol of
 * the same string will return a pointer to the same object.
 *****/

#include <cstdio>
#include <string>

#include "symbol.h"

namespace sym {

std::map<std::string,symbol> symbol::dict;
  
} // namespace sym

