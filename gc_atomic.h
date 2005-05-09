/****
 * gc_atomic.h
 * Tom Prince 2005/05/08
 *
 * Modified boehm-gc code, to allocate atomic objects automatically.
 *****/

/****************************************************************************
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 
THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY EXPRESSED
OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 
Permission is hereby granted to use or copy this program for any
purpose, provided the above notices are retained on all copies.
Permission to modify the code and to distribute modified code is
granted, provided the above notices are retained, and a notice that
the code was modified is included with the above copyright notice.
****************************************************************************/

#ifndef GC_ATOMIC_H
#define GC_ATOMIC_H

#include "gc_cpp.h"

class gc_atomic {public:
    inline void* operator new( size_t size );
    inline void* operator new( size_t size, GCPlacement gcp );
    inline void* operator new( size_t size, void *p );
    	/* Must be redefined here, since the other overloadings	*/
    	/* hide the global definition.				*/
    inline void operator delete( void* obj );
#   ifdef GC_PLACEMENT_DELETE  
      inline void operator delete( void*, void* );
#   endif

#ifdef GC_OPERATOR_NEW_ARRAY
    inline void* operator new[]( size_t size );
    inline void* operator new[]( size_t size, GCPlacement gcp );
    inline void* operator new[]( size_t size, void *p );
    inline void operator delete[]( void* obj );
#   ifdef GC_PLACEMENT_DELETE
      inline void gc_atomic::operator delete[]( void*, void* );
#   endif
#endif /* GC_OPERATOR_NEW_ARRAY */
    };

/****************************************************************************

Inline implementation

****************************************************************************/

inline void* gc_atomic::operator new( size_t size ) {
    return GC_MALLOC_ATOMIC( size );}
    
inline void* gc_atomic::operator new( size_t size, GCPlacement gcp ) {
    if (gcp == UseGC) 
        return GC_MALLOC( size );
    else if (gcp == PointerFreeGC)
	return GC_MALLOC_ATOMIC( size );
    else
        return GC_MALLOC_UNCOLLECTABLE( size );}

inline void* gc_atomic::operator new( size_t size, void *p ) {
    return p;}

inline void gc_atomic::operator delete( void* obj ) {
    GC_FREE( obj );}
    
#ifdef GC_PLACEMENT_DELETE
  inline void gc_atomic::operator delete( void*, void* ) {}
#endif

#ifdef GC_OPERATOR_NEW_ARRAY

inline void* gc_atomic::operator new[]( size_t size ) {
    return gc_atomic::operator new( size );}
    
inline void* gc_atomic::operator new[]( size_t size, GCPlacement gcp ) {
    return gc_atomic::operator new( size, gcp );}

inline void* gc_atomic::operator new[]( size_t size, void *p ) {
    return p;}

inline void gc_atomic::operator delete[]( void* obj ) {
    gc_atomic::operator delete( obj );}

#ifdef GC_PLACEMENT_DELETE
  inline void gc_atomic::operator delete[]( void*, void* ) {}
#endif
    
#endif /* GC_OPERATOR_NEW_ARRAY */

#endif /* GC_ATOMIC_H */
