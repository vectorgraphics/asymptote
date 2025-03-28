/*
 * Copyright (c) 2005 Hewlett-Packard Development Company, L.P.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif

#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifndef AO_BUILD
# define AO_BUILD
#endif

#ifndef AO_REAL_PTR_AS_MACRO
# define AO_REAL_PTR_AS_MACRO
#endif

#define AO_REQUIRE_CAS
#include "atomic_ops_stack.h"

AO_API void AO_stack_init(AO_stack_t *list)
{
  memset(list, 0, sizeof(AO_stack_t));
}

AO_API int AO_stack_is_lock_free(void)
{
# ifdef AO_USE_ALMOST_LOCK_FREE
    return 0;
# else
    return 1;
# endif
}

AO_API AO_t *AO_stack_head_ptr(const AO_stack_t *list)
{
  return AO_REAL_HEAD_PTR(*list);
}

AO_API AO_t *AO_stack_next_ptr(AO_t next)
{
  return AO_REAL_NEXT_PTR(next);
}

/* This function call must be a part of a do-while loop with a CAS      */
/* designating the condition of the loop (see the use cases below).     */
#ifdef AO_THREAD_SANITIZER
  AO_ATTR_NO_SANITIZE_THREAD
  static void store_before_cas(AO_t *addr, AO_t value)
  {
    *addr = value;
  }
#else
# define store_before_cas(addr, value) (void)(*(addr) = (value))
#endif

#ifdef AO_USE_ALMOST_LOCK_FREE

# ifdef __cplusplus
    extern "C" {
# endif
  AO_API void AO_pause(int); /* defined in atomic_ops.c */
# ifdef __cplusplus
    } /* extern "C" */
# endif

# if defined(__alpha__) && (__GNUC__ == 4)
    /* Workaround __builtin_expect bug found in         */
    /* gcc-4.6.3/alpha causing test_stack failure.      */
#   undef AO_EXPECT_FALSE
#   define AO_EXPECT_FALSE(expr) (expr)
# endif

  /* LIFO linked lists based on compare-and-swap.  We need to avoid     */
  /* the case of a node deletion and reinsertion while I'm deleting     */
  /* it, since that may cause my CAS to succeed eventhough the next     */
  /* pointer is now wrong.  Our solution is not fully lock-free, but it */
  /* is good enough for signal handlers, provided we have a suitably    */
  /* low bound on the number of recursive signal handler reentries.     */
  /* A list consists of a first pointer and a blacklist of pointer      */
  /* values that are currently being removed.  No list element on       */
  /* the blacklist may be inserted.  If we would otherwise do so, we    */
  /* are allowed to insert a variant that differs only in the least     */
  /* significant, ignored, bits.  If the list is full, we wait.         */

  /* Crucial observation: A particular padded pointer x (i.e. pointer   */
  /* plus arbitrary low order bits) can never be newly inserted into    */
  /* a list while it's in the corresponding auxiliary data structure.   */

  /* The second argument is a pointer to the link field of the element  */
  /* to be inserted.                                                    */
  /* Both list headers and link fields contain "perturbed" pointers,    */
  /* i.e. pointers with extra bits or'ed into the low order bits.       */
  AO_API void AO_stack_push_explicit_aux_release(volatile AO_t *list, AO_t *x,
                                                 AO_stack_aux *a)
  {
    AO_t x_bits = (AO_t)x;
    AO_t next;

    /* No deletions of x can start here, since x is not         */
    /* currently in the list.                                   */
  retry:
    do {
      next = AO_load_acquire(list);
      store_before_cas(x, next);
      {
#     if AO_BL_SIZE == 2
        /* Start all loads as close to concurrently as possible.        */
        AO_t entry1 = AO_load(&a->AO_stack_bl[0]);
        AO_t entry2 = AO_load(&a->AO_stack_bl[1]);
        if (AO_EXPECT_FALSE(entry1 == x_bits || entry2 == x_bits))
#     else
        int i;
        for (i = 0; i < AO_BL_SIZE; ++i)
          if (AO_EXPECT_FALSE(AO_load(&a->AO_stack_bl[i]) == x_bits))
#     endif
        {
          /* Entry is currently being removed.  Change it a little.     */
          ++x_bits;
          if ((x_bits & AO_BIT_MASK) == 0)
            /* Version count overflowed; EXTREMELY unlikely, but possible. */
            x_bits = (AO_t)x;
          goto retry;
        }
      }

      /* x_bits value is not currently being deleted.   */
    } while (AO_EXPECT_FALSE(!AO_compare_and_swap_release(list, next,
                                                          x_bits)));
  }

  /* I concluded experimentally that checking a value first before      */
  /* performing a compare-and-swap is usually beneficial on x86, but    */
  /* slows things down appreciably with contention on Itanium.          */
  /* Since the Itanium behavior makes more sense to me (more cache line */
  /* movement unless we're mostly reading, but back-off should guard    */
  /* against that), we take Itanium as the default.  Measurements on    */
  /* other multiprocessor architectures would be useful.                */
  /* (On a uniprocessor, the initial check is almost certainly a very   */
  /* small loss.) - HB                                                  */
# ifdef __i386__
#   define PRECHECK(a) (a) == 0 &&
# else
#   define PRECHECK(a)
# endif

  /* This function is used before CAS in the below AO_stack_pop and the */
  /* data race (reported by TSan) is OK because it results in a retry.  */
# ifdef AO_THREAD_SANITIZER
    AO_ATTR_NO_SANITIZE_THREAD
    static AO_t AO_load_next(const volatile AO_t *first_ptr)
    {
      /* Assuming an architecture on which loads of word type are       */
      /* atomic.  AO_load cannot be used here because it cannot be      */
      /* instructed to suppress the warning about the race.             */
      return *first_ptr;
    }
# else
#   define AO_load_next AO_load
# endif

  AO_API AO_t *AO_stack_pop_explicit_aux_acquire(volatile AO_t *list,
                                                 AO_stack_aux *a)
  {
    unsigned i;
    int j = 0;
    AO_t first;
    AO_t * first_ptr;
    AO_t next;

  retry:
    first = AO_load(list);
    if (0 == first) return 0;
    /* Insert first into aux black list.                                */
    /* This may spin if more than AO_BL_SIZE removals using auxiliary   */
    /* structure a are currently in progress.                           */
    for (i = 0; ; )
      {
        if (PRECHECK(a -> AO_stack_bl[i])
            AO_compare_and_swap_acquire(a->AO_stack_bl+i, 0, first))
          break;
        if (++i >= AO_BL_SIZE)
          {
            i = 0;
            AO_pause(++j);
          }
      }
#   ifndef AO_THREAD_SANITIZER
      assert(a -> AO_stack_bl[i] == first);
                                /* No actual race with the above CAS.   */
#   endif
    /* first is on the auxiliary black list.  It may be removed by      */
    /* another thread before we get to it, but a new insertion of x     */
    /* cannot be started here.  Only we can remove it from the black    */
    /* list.  We need to make sure that first is still the first entry  */
    /* on the list.  Otherwise it is possible that a reinsertion of it  */
    /* was already started before we added the black list entry.        */
    if (AO_EXPECT_FALSE(first != AO_load_acquire(list)))
                        /* Workaround test failure on AIX, at least, by */
                        /* using acquire ordering semantics for this    */
                        /* load.  Probably, it is not the right fix.    */
    {
      AO_store_release(a->AO_stack_bl+i, 0);
      goto retry;
    }
    first_ptr = AO_REAL_NEXT_PTR(first);
    next = AO_load_next(first_ptr);
    if (AO_EXPECT_FALSE(!AO_compare_and_swap_release(list, first, next)))
    {
      AO_store_release(a->AO_stack_bl+i, 0);
      goto retry;
    }
#   ifndef AO_THREAD_SANITIZER
      assert(*list != first); /* No actual race with the above CAS.     */
#   endif
    /* Since we never insert an entry on the black list, this cannot    */
    /* have succeeded unless first remained on the list while we were   */
    /* running.  Thus, its next link cannot have changed out from under */
    /* us, and we removed exactly one entry and preserved the rest of   */
    /* the list.  Note that it is quite possible that an additional     */
    /* entry was inserted and removed while we were running; this is OK */
    /* since the part of the list following first must have remained    */
    /* unchanged, and first must again have been at the head of the     */
    /* list when the compare_and_swap succeeded.                        */
    AO_store_release(a->AO_stack_bl+i, 0);
    return first_ptr;
  }

  AO_API void AO_stack_push_release(AO_stack_t *list, AO_t *x)
  {
    AO_stack_push_explicit_aux_release(&list->AO_pa.AO_ptr, x,
                                       &list->AO_pa.AO_aux);
  }

  AO_API AO_t *AO_stack_pop_acquire(AO_stack_t *list)
  {
    return AO_stack_pop_explicit_aux_acquire(&list->AO_pa.AO_ptr,
                                             &list->AO_pa.AO_aux);
  }

#else /* !AO_USE_ALMOST_LOCK_FREE */

  /* The functionality is the same as of AO_load_next but the atomicity */
  /* is not needed.  The usage is similar to that of store_before_cas.  */
# if defined(AO_THREAD_SANITIZER) \
     && (defined(AO_HAVE_compare_and_swap_double) \
         || defined(AO_HAVE_compare_double_and_swap_double))
    /* TODO: If compiled by Clang (as of clang-4.0) with -O3 flag,      */
    /* no_sanitize attribute is ignored unless the argument is volatile.*/
#   if defined(__clang__)
#     define LOAD_BEFORE_CAS_VOLATILE volatile
#   else
#     define LOAD_BEFORE_CAS_VOLATILE /* empty */
#   endif
    AO_ATTR_NO_SANITIZE_THREAD
    static AO_t load_before_cas(const LOAD_BEFORE_CAS_VOLATILE AO_t *addr)
    {
      return *addr;
    }
# else
#   define load_before_cas(addr) (*(addr))
# endif /* !AO_THREAD_SANITIZER */

  /* Better names for fields in AO_stack_t.     */
# define version AO_vp.AO_val1
# define ptr AO_vp.AO_val2

# if defined(AO_HAVE_compare_double_and_swap_double) \
     && !(defined(AO_STACK_PREFER_CAS_DOUBLE) \
          && defined(AO_HAVE_compare_and_swap_double))

#   ifdef LINT2
      volatile /* non-static */ AO_t AO_noop_sink;
#   endif

    AO_API void AO_stack_push_release(AO_stack_t *list, AO_t *element)
    {
      AO_t next;

      do {
        next = AO_load(&list->ptr);
        store_before_cas(element, next);
      } while (AO_EXPECT_FALSE(!AO_compare_and_swap_release(&list->ptr, next,
                                                            (AO_t)element)));
      /* This uses a narrow CAS here, an old optimization suggested     */
      /* by Treiber.  Pop is still safe, since we run into the ABA      */
      /* problem only if there were both intervening pops and pushes.   */
      /* In that case we still see a change in the version number.      */
#     ifdef LINT2
        /* Instruct static analyzer that element is not lost.   */
        AO_noop_sink = (AO_t)element;
#     endif
    }

    AO_API AO_t *AO_stack_pop_acquire(AO_stack_t *list)
    {
#     if defined(__clang__) && !AO_CLANG_PREREQ(3, 5)
        AO_t *volatile cptr;
                        /* Use volatile to workaround a bug in          */
                        /* clang-1.1/x86 causing test_stack failure.    */
#     else
        AO_t *cptr;
#     endif
      AO_t next;
      AO_t cversion;

      do {
        /* Version must be loaded first.    */
        cversion = AO_load_acquire(&list->version);
        cptr = (AO_t *)AO_load(&list->ptr);
        if (NULL == cptr)
          return NULL;
        next = load_before_cas((/* no volatile */ AO_t *)cptr);
      } while (AO_EXPECT_FALSE(!AO_compare_double_and_swap_double_release(
                                        &list->AO_vp, cversion, (AO_t)cptr,
                                        cversion+1, next)));
      return cptr;
    }

# elif defined(AO_HAVE_compare_and_swap_double)

    /* Needed for future IA64 processors.  No current clients?  */
    /* TODO: Not tested thoroughly. */

    /* We have a wide CAS, but only does an AO_t-wide comparison.       */
    /* We cannot use the Treiber optimization, since we only check      */
    /* for an unchanged version number, not an unchanged pointer.       */
    AO_API void AO_stack_push_release(AO_stack_t *list, AO_t *element)
    {
      AO_t cversion;

      do {
        AO_t next_ptr;

        /* Again version must be loaded first, for different reason.    */
        cversion = AO_load_acquire(&list->version);
        next_ptr = AO_load(&list->ptr);
        store_before_cas(element, next_ptr);
      } while (!AO_compare_and_swap_double_release(&list->AO_vp, cversion,
                                                   cversion+1, (AO_t)element));
    }

    AO_API AO_t *AO_stack_pop_acquire(AO_stack_t *list)
    {
      AO_t *cptr;
      AO_t next;
      AO_t cversion;

      do {
        cversion = AO_load_acquire(&list->version);
        cptr = (AO_t *)AO_load(&list->ptr);
        if (NULL == cptr)
          return NULL;
        next = load_before_cas(cptr);
      } while (!AO_compare_double_and_swap_double_release(&list->AO_vp,
                                cversion, (AO_t)cptr, cversion+1, next));
      return cptr;
    }
# endif /* AO_HAVE_compare_and_swap_double */

# undef ptr
# undef version
#endif /* !AO_USE_ALMOST_LOCK_FREE */
