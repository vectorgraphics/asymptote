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

namespace sym {

struct symbol {
private:
  std::string name;

  static std::map<std::string,symbol> dict;
public:
  symbol() {}
  symbol(std::string name)
    : name(name) {}
  
  static symbol *trans(std::string s) {
    if (dict.find(s) != dict.end())
      return &dict[s];
    else
      return &(dict[s]=symbol(s));
  }

  operator std::string () { return std::string(name); }

  friend ostream& operator<< (ostream& out, const symbol& sym)
  { return out << sym.name; }
};

} // namespace sym

#endif
