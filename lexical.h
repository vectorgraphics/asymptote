#ifndef __lexical_h__
#define __lexical_h__ 1

#include <string>
#include <sstream>

namespace lexical {

class bad_cast {};

template <typename T> 
T cast(const std::string& s) 
{
  std::istringstream is(s); 
  T value;
  if(is && is >> value && (is >> std::ws).eof()) return value;
  throw bad_cast();
} 

}

#endif
