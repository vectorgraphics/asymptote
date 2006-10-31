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

GCInit symbol::initialize;
mem::map<CONST mem::string,symbol> symbol::dict;
symbol *symbol::initsym=symbol::specialTrans(mem::string("operator init"));
symbol *symbol::castsym=symbol::specialTrans(mem::string("operator cast"));
symbol *symbol::ecastsym=symbol::specialTrans(mem::string("operator ecast"));

} // namespace sym

