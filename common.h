/****
 * common.h
 *
 * Definitions common to all files.
 *****/

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include "memory.h"

typedef long long Int;
//#define Int int

#define Int_MAX LLONG_MAX
#define Int_MIN LLONG_MIN
//#define Int_MAX INT_MAX
//#define Int_MIN INT_MIN

using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::istream;
using std::ostream;

using mem::string;
using mem::istringstream;
using mem::ostringstream;
using mem::stringbuf;

#endif 
