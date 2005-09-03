/*****
 * modifier.h
 * Andy Hammerlindl 2002/08/29
 *
 * Permissions for variables.
 * PUBLIC means the variable can be read or written anywhere.
 * READONLY means it can be read anywhere, but written only in the record.
 * PRIVATE means it can only be accessed in the record.
 *
 * The modifiers static declares that variable to be allocated, are allocated in
 * the parent's frame, and code is translated into the parent's frame.
 *****/

#ifndef MODIFIER_H
#define MODIFIER_H

namespace trans {

// PUBLIC, PRIVATE, or READONLY - the permission tokens defined in
// camp.y for accessing a variable outside of its lexically enclosing
// record.
enum permission {
  READONLY,
  PUBLIC,
  PRIVATE
};

enum modifier {
  DEFAULT_STATIC,
  DEFAULT_DYNAMIC,
  EXPLICIT_STATIC,
  EXPLICIT_DYNAMIC
};

} // namespace trans

#endif
