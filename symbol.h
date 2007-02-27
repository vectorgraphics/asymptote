/*****
 * symbol.h
 * Andy Hammerlindl 2002/06/18
 *
 * Creates symbols from strings so that multiple calls for a symbol of
 * the same string will return a pointer to the same object.
 *****/

#ifndef SYMBOL_H
#define SYMBOL_H

#include <iostream>
#include <cassert>

#include "memory.h"

using std::ostream;

namespace sym {

struct GCInit {
#ifdef _AIX
  typedef char * GC_PTR;
#endif
  
  GCInit() {
#ifdef USEGC
  GC_free_space_divisor = 2;
  GC_dont_expand = 0;
  GC_INIT();
#endif  
  }
};

struct symbol {
  static GCInit initialize;
private:
  mem::string name;

public:
  static mem::map<CONST mem::string,symbol> dict;

  static symbol *specialTrans(mem::string s) {
    assert(dict.find(s) == dict.end());
    return &(dict[s]=symbol(s,true));
  }

  symbol() : special(false) {}
  symbol(mem::string name, bool special=false)
    : name(name), special(special) {}

public:
  friend class mem::map<CONST mem::string,symbol>;
  bool special; // NOTE: make this const (later).
  
  static symbol *initsym;
  static symbol *castsym;
  static symbol *ecastsym;
  
  static symbol *literalTrans(mem::string s) {
    if (dict.find(s) != dict.end())
      return &dict[s];
    else
      return &(dict[s]=symbol(s));
  }

  static symbol *opTrans(mem::string s) {
    return literalTrans("operator "+s);
  }

  static symbol *trans(mem::string s) {
    // Figure out whether it's an operator or an identifier by looking at the
    // first character.
    char c=s[0];
    return isalpha(c) || c == '_' ? literalTrans(s) : opTrans(s);
  }

  operator mem::string () { return mem::string(name); }

  friend ostream& operator<< (ostream& out, const symbol& sym)
  { return out << sym.name; }
};

} // namespace sym

#endif
