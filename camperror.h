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

#include "common.h"

namespace camp {

// Used internally to report an error in an operation.
[[noreturn]] void reportError(const string& desc);
[[noreturn]] void reportError(const ostringstream& desc);

void reportWarning(const string& desc);
void reportWarning(const ostringstream& desc);

[[noreturn]] void reportFatal(const string& desc);
[[noreturn]] void reportFatal(const ostringstream& desc);

inline std::ostream& newl(std::ostream& s) {s << '\n'; return s;}

} // namespace camp

#endif
