/*****
 * cast.h
 * Andy Hammerlindl 2003/07/24
 *
 * Handles the casting of types, numeric promotions, and the operators,
 * using a table of accesses for each.  An access here specifies the
 * instruction or builtin function used to handle the casting.
 *****/

#ifndef CAST_H
#define CAST_H

#include "types.h"
#include "access.h"

namespace types {

// Warning: all entries here are done based on the ty_kind of the types.
// Using user-defined types will not work here.

// Puts the default casts and operators into the tables.
void initializeCasts();

// Builds the table of initializers for primitive types.
void initializeInitializers();

// Gets the initializer for a type.
// NOTE: There may be a better place for this than with the casts.
trans::access *initializer(ty *t);

// Checks if one type can be casted into another.  Works much like the
// env::implicitCast() function but only checks for the possibility, and
// does not implement the cast.
bool castable(ty *target, ty *source);

// Given the types (possibly both overloaded), gives the resultant
// possible type (possibly overloaded) of the cast.  This is used
// primarily to detect ambiguities when assigning variables.
ty *castType(ty *target, ty *source);

// When one type is being explicitly cast to another, this determines
// the type that the castee should first be translated as.
// ex.  real f, f();
//      int x = (int)f;
// Here, this function will tell the program to first cast the
// expression 'f' to real.
// This can also return overloaded types.
ty *explicitCastType(ty *target, ty *source);

// If an explict cast is possible, it gives the corresponding access to it.
// Otherwise, null.
trans::access *explicitCast(ty *target, ty *source);

// If an implicit cast is possible, it gives the corresponding access to it.
// Otherwise, null.
trans::access *cast(ty *target, ty *source);

// Figures out what type the two operand types should be converted to
// for a binary expression.  If none is possible, it returns null.
// Currently for promoting "b" and "c" in an expression like "a ? b : c"
ty *promote(ty *t1, ty *t2);

} // namespace types

#endif
