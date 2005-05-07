#ifndef COLLECT_H
#define COLLECT_H

#ifdef USEGC

#include <gc_allocator.h>
#include <gc_cpp.h>

#else

class gc {};

template<class T>
class gc_allocator : public std::allocator<T> {};

#define UseGC

#endif

#endif
