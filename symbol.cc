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

mem::map<CONST string,symbol> symbol::dict;
symbol *symbol::initsym=symbol::specialTrans(string("operator init"));
symbol *symbol::castsym=symbol::specialTrans(string("operator cast"));
symbol *symbol::ecastsym=symbol::specialTrans(string("operator ecast"));

} // namespace sym

#ifdef PRESYM
/* Define all of operator symbols SYM_PLUS, etc. */
#define OPSYMBOL(str, name) \
  sym::symbol *name = sym::symbol::opTrans(str)
#include "opsymbols.h"
#undef OPSYMBOL

/* Define all of the symbols of the type SYM(name) in selected files. */
#define ADDSYMBOL(name) \
  sym::symbol *PRETRANSLATED_SYMBOL_##name = sym::symbol::literalTrans(#name)
#include "allsymbols.h"
#undef ADDSYMBOL
#endif
