/*****
 * symbol.cc
 * Andy Hammerlindl 2002/06/18
 *
 * Creates symbols from strings so that multiple calls for a symbol of
 * the same string will return a pointer to the same object.
 *****/

#include <cstdio>

#include "symbol.h"

namespace sym {

std::map<string,symbol> symbol::dict;
symbol *symbol::initsym=symbol::specialTrans("operator init");
symbol *symbol::castsym=symbol::specialTrans("operator cast");
symbol *symbol::ecastsym=symbol::specialTrans("operator ecast");

} // namespace sym

