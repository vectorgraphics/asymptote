/*****
 * envcompleter.cc
 * Andy Hammerlindl 2006/07/31
 *
 * Implements a text completion function for readline based on symbols in the
 * environment.
 *****/

#include <fstream>

#include "table.h"
#include "envcompleter.h"

namespace trans {

char *envCompleter::symbolToMallocedString(symbol *name) {
  std::string s=*name;
  size_t size=(s.size()+1)*sizeof(char);
  char *word=(char *)std::malloc(size);
  std::memcpy(word, s.c_str(), size);
  return word;
}

bool basicListLoaded=false;
envCompleter::symbol_list basicList;

static void loadBasicList() {
  assert(basicListLoaded==false);

#if 0
  // NOTE: Change, obviously, to look in the right place.
  std::ifstream in("keywords");

  if (in.is_open()) {
    while (!in.eof()) {
      string s;
      std::getline(in, s);
      if (!s.empty())
        basicList.push_back(symbol::literalTrans(s));
    }
    in.close();
  }
#endif

#if 1 
#define ADD(word) basicList.push_back(symbol::literalTrans(#word))
#include "keywords.cc"
#undef ADD
#endif

  basicListLoaded=true;
}

void envCompleter::basicCompletions(symbol_list &l, mem::string start) {
  if (!basicListLoaded)
    loadBasicList();

  for(symbol_list::iterator p = basicList.begin(); p != basicList.end(); ++p)
    if (prefix(start, **p))
      l.push_back(*p);
}

void envCompleter::makeList(const char *text) {
  l.clear();
  basicCompletions(l, text);
  e.completions(l, text);
  index=l.begin();
}

char *envCompleter::operator () (const char *text, int state) {
  if (state==0)
    makeList(text);

  if (index==l.end())
    return 0;
  else {
    symbol *name=*index;
    ++index;
    return symbolToMallocedString(name);
  }
}

} // namespace trans
