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
#include <string>
#include <map>

using std::ostream;
using std::string;

namespace sym {

struct symbol {
private:
  string name;

  static std::map<string,symbol> dict;
  symbol() {}
  symbol(string name)
    : name(name) {}
  friend class std::map<string,symbol>;
public:
  
  static symbol *trans(string s) {
    if (dict.find(s) != dict.end())
      return &dict[s];
    else
      return &(dict[s]=symbol(s));
  }

  operator string () { return string(name); }

  friend ostream& operator<< (ostream& out, const symbol& sym)
  { return out << sym.name; }
};

} // namespace sym

#endif
