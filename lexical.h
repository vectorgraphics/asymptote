#ifndef __lexical_h__
#define __lexical_h__ 1

#include <sstream>

#include "memory.h"

namespace lexical {

class bad_cast {};

template <typename T> 
T cast(const mem::string& s) 
{
  mem::istringstream is(s);
  T value;
  if(is && is >> value && (is >> std::ws).eof()) return value;
  throw bad_cast();
} 

}

#endif
