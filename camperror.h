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

using std::ostream;

namespace camp {

// Used internally, should use errors() to check for errors.
extern bool errorFlag;

// Used internally to report an error in an operation.
void reportError(std::string desc);

// Checks if an error has occured.
inline bool errors() {
  return errorFlag;
}

// Copies the description of the oldest unretrieved error into the
// buffer.  Once all errors have had their descriptions retrieved this
// way, errors() will once again return false.
std::string getError();

inline ostream& newl(ostream& s) {s << '\n'; return s;}
  
} // namespace camp

#endif
