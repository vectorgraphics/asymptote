#ifndef __lexical_h__
#define __lexical_h__ 1

#include <sstream>

#include "common.h"

namespace lexical {

class bad_cast {};

template <typename T> 
T cast(const string& s) 
{
  istringstream is(s);
  T value;
  if(is && is >> value && (is >> std::ws).eof()) return value;
  throw bad_cast();
} 

}

#endif
