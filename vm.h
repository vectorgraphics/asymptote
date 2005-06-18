/*****
 * vm.h
 * Tom Prince 2005/06/17
 * 
 * Interface to the virtual machine.
 *****/

#ifndef VM_H
#define VM_H

#include "errormsg.h"
#include "inst.h"

namespace vm {

void run(lambda *l);
position getPos();
void error(const char* message);

} // namespace vm

#endif // VM_H
