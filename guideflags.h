/*****
 * guideflags.h
 * Tom Prince 2004/5/12
 *
 * These flags are used to indicate what specifications of the join are
 * put on the stack.
 *****/
#ifndef GUIDEFLAGS_H
#define GUIDEFLAGS_H

namespace run {

const int NULL_JOIN       = 0x0000;
const int LEFT_GIVEN      = 0x0001;
const int LEFT_CURL       = 0x0002;
const int LEFT_TENSION    = 0x0004;
const int RIGHT_TENSION   = 0x0008;
const int TENSION_ATLEAST = 0x0010;
const int LEFT_CONTROL    = 0x0020;
const int RIGHT_CONTROL   = 0x0040;
const int RIGHT_GIVEN     = 0x0080;
const int RIGHT_CURL      = 0x0100;
  
}

#endif //GUIDEFLAGS_H
