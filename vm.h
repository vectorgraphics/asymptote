/*****
 * vm.h
 * Tom Prince 2005/06/17
 * 
 * Interface to the virtual machine.
 *****/

#ifndef VM_H
#define VM_H

#include "errormsg.h"

namespace vm {

class lambda; class stack;
typedef void (*bltin)(stack *s);

void run(lambda *l);
position getPos();
void errornothrow(const char* message);
void error(const char* message);
void error(const mem::ostringstream& message);

} // namespace vm

#endif // VM_H
