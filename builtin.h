/*****
 * builtin.h
 * Tom Prince 2004/08/25
 *
 * Initialize builtins.
 *****/
#ifndef BUILTIN_H
#define BUILTIN_H

namespace trans {

class tenv;
class venv;
class menv;

// The base environments for built-in types and functions
void base_tenv(tenv &);
void base_venv(venv &);
void base_menv(menv &);

} //namespace trans

#endif //BUILTIN_H
