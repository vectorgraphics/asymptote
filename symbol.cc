/*****
 * symbol.cc
 * Andy Hammerlindl 2002/06/18
 *
 * Creates symbols from strings so that multiple calls for a symbol of
 * the same string will return a pointer to the same object.
 *****/

#include <cstdio>
#include "symbol.h"

using mem::string;

namespace sym {

GCInit symbol::initialize;
mem::map<CONST string,symbol> symbol::dict;
symbol *symbol::initsym=symbol::specialTrans(string("operator init"));
symbol *symbol::castsym=symbol::specialTrans(string("operator cast"));
symbol *symbol::ecastsym=symbol::specialTrans(string("operator ecast"));

} // namespace sym

