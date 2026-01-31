/*****
 * guideflags.h
 * Tom Prince 2004/5/12
 *
 * These flags are used to indicate what specifications of the join are
 * put on the stack.
 *****/
#ifndef GUIDEFLAGS_H
#define GUIDEFLAGS_H

#include "asyffi.h"

namespace camp
{

#undef OUT
#undef IN

enum side: uint8_t
{
  OUT= ASY_SIDE_OUT,
  IN= ASY_SIDE_IN,
  END= ASY_SIDE_END,
  JOIN= ASY_SIDE_JOIN
};

}// namespace camp

#endif// GUIDEFLAGS_H
