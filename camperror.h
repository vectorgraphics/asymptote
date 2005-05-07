/*****
 * camperror.h
 * 2003/02/25 Andy Hammerlindl
 *
 * Provides a way for the classes in camp to report errors in
 * computation elegantly.  After running a method on a camp object that
 * could encounter an error, the program should call camp::errors to see
 * if any errors were encountered.
 *****/

#ifndef CAMPERROR_H
#define CAMPERROR_H

#include <iostream>
#include <string>

using std::string;

namespace camp {

// Used internally to report an error in an operation.
void reportError(const string& desc);
void reportError(const std::ostringstream& desc);

inline std::ostream& newl(std::ostream& s) {s << '\n'; return s;}
  
} // namespace camp

#endif
