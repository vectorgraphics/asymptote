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

#include <stdio.h>

#if defined(__vxworks)

  int main(void)
  {
    printf("test skipped\n");
    return 0;
  }

#else

#if ((defined(_WIN32) && !defined(__CYGWIN32__) && !defined(__CYGWIN__)) \
     || defined(_MSC_VER) || defined(_WIN32_WINCE)) \
    && !defined(AO_USE_WIN32_PTHREADS)
# define USE_WINTHREADS
#endif

#ifdef USE_WINTHREADS
# include <windows.h>
#else
# include <pthread.h>
#endif

#include <assert.h>
#include <stdlib.h>

#include "atomic_ops_stack.h" /* includes atomic_ops.h as well */

#if (defined(_WIN32_WCE) || defined(__MINGW32CE__)) && !defined(AO_HAVE_abort)
# define abort() _exit(-1) /* there is no abort() in WinCE */
#endif

#ifndef MAX_NTHREADS
# define MAX_NTHREADS 100
#endif

#ifndef DEFAULT_NTHREADS
# define DEFAULT_NTHREADS 16 /* must be <= MAX_NTHREADS */
#endif

#ifdef NO_TIMES
# define get_msecs() 0
#elif (defined(USE_WINTHREADS) || defined(AO_USE_WIN32_PTHREADS)) \
      && !defined(CPPCHECK)
# include <sys/timeb.h>
  unsigned long get_msecs(void)
  {
    struct timeb tb;

    ftime(&tb);
    return (unsigned long)tb.time * 1000 + tb.millitm;
  }
#else /* Unix */
# include <time.h>
# include <sys/time.h>
  unsigned long get_msecs(void)
  {
    struct timeval tv;

    gettimeofday(&tv, 0);
    return (unsigned long)tv.tv_sec * 1000 + tv.tv_usec/1000;
  }
#endif /* !NO_TIMES */

struct le {
  AO_t next; /* must be the first field */
  int data;
};

typedef union le_u {
  AO_t next;
  struct le e;
} list_element;

#if defined(CPPCHECK)
  AO_stack_t the_list; /* to test AO_stack_init() */
#else
  AO_stack_t the_list = AO_STACK_INITIALIZER;
#endif

/* Add elements from 1 to n to the list (1 is pushed first).    */
/* This is called from a single thread only.                    */
void add_elements(int n)
{
  list_element * le;
  if (n == 0) return;
  add_elements(n-1);
  le = (list_element *)malloc(sizeof(list_element));
  if (le == 0)
    {
      fprintf(stderr, "Out of memory\n");
      exit(2);
    }
# if defined(CPPCHECK)
    le->e.next = 0; /* mark field as used */
# endif
  le->e.data = n;
  AO_stack_push(&the_list, &le->next);
}

#ifdef VERBOSE_STACK
void print_list(void)
{
  AO_t *p;

  for (p = AO_REAL_HEAD_PTR(the_list);
       p != 0; p = AO_REAL_NEXT_PTR(*p))
    printf("%d\n", ((list_element*)p)->e.data);
}
#endif /* VERBOSE_STACK */

/* Check that the list contains only values from 1 to n, in any order,  */
/* w/o duplications.  Executed when the list mutation is finished.      */
void check_list(int n)
{
  AO_t *p;
  int i;
  int err_cnt = 0;
  char *marks = (char*)calloc(n + 1, 1);

  if (0 == marks)
    {
      fprintf(stderr, "Out of memory (marks)\n");
      exit(2);
    }

  for (p = AO_REAL_HEAD_PTR(the_list);
       p != 0; p = AO_REAL_NEXT_PTR(*p))
    {
      i = ((list_element*)p)->e.data;
      if (i > n || i <= 0)
        {
          fprintf(stderr, "Found erroneous list element %d\n", i);
          err_cnt++;
        }
      else if (marks[i] != 0)
        {
          fprintf(stderr, "Found duplicate list element %d\n", i);
          abort();
        }
      else marks[i] = 1;
    }

  for (i = 1; i <= n; ++i)
    if (marks[i] != 1)
      {
        fprintf(stderr, "Missing list element %d\n", i);
        err_cnt++;
      }

  free(marks);
  if (err_cnt > 0) abort();
}

volatile AO_t ops_performed = 0;

#ifndef LIMIT
        /* Total number of push/pop ops in all threads per test.    */
# ifdef AO_USE_PTHREAD_DEFS
#   define LIMIT 20000
# else
#   define LIMIT 1000000
# endif
#endif

#ifdef AO_HAVE_fetch_and_add
# define fetch_then_add(addr, val) AO_fetch_and_add(addr, val)
#else
  /* OK to perform it in two atomic steps, but really quite     */
  /* unacceptable for timing purposes.                          */
  AO_INLINE AO_t fetch_then_add(volatile AO_t * addr, AO_t val)
  {
    AO_t result = AO_load(addr);
    AO_store(addr, result + val);
    return result;
  }
#endif

#ifdef USE_WINTHREADS
  DWORD WINAPI run_one_test(LPVOID arg)
#else
  void * run_one_test(void * arg)
#endif
{
  AO_t *t[MAX_NTHREADS + 1];
  unsigned index = (unsigned)(size_t)arg;
  unsigned i;
# ifdef VERBOSE_STACK
    unsigned j = 0;

    printf("starting thread %u\n", index);
# endif
  assert(index <= MAX_NTHREADS);
  while (fetch_then_add(&ops_performed, index + 1) + index + 1 < LIMIT)
    {
      /* Pop index+1 elements (where index is the thread's one), then   */
      /* push them back (in the same order of operations).              */
      /* Note that this is done in parallel by many threads.            */
      for (i = 0; i <= index; ++i)
        {
          t[i] = AO_stack_pop(&the_list);
          if (0 == t[i])
            {
              /* This should not happen as at most n*(n+1)/2 elements   */
              /* could be popped off at a time.                         */
              fprintf(stderr, "Failed - nothing to pop\n");
              abort();
            }
        }
      for (i = 0; i <= index; ++i)
        {
          AO_stack_push(&the_list, t[i]);
        }
#     ifdef VERBOSE_STACK
        j += index + 1;
#     endif
    }
    /* Repeat until LIMIT push/pop operations are performed (by all     */
    /* the threads simultaneously).                                     */
# ifdef VERBOSE_STACK
    printf("finished thread %u: %u total ops\n", index, j);
# endif
  return 0;
}

#ifndef N_EXPERIMENTS
# define N_EXPERIMENTS 1
#endif

unsigned long times[MAX_NTHREADS + 1][N_EXPERIMENTS];

void run_one_experiment(int max_nthreads, int exper_n)
{
  int nthreads;

  assert(max_nthreads <= MAX_NTHREADS);
  assert(exper_n < N_EXPERIMENTS);
  for (nthreads = 1; nthreads <= max_nthreads; ++nthreads) {
    unsigned i;
#   ifdef USE_WINTHREADS
      DWORD thread_id;
      HANDLE thread[MAX_NTHREADS];
#   else
      pthread_t thread[MAX_NTHREADS];
#   endif
    int list_length = nthreads*(nthreads+1)/2;
    unsigned long start_time;
    AO_t *le;

#   ifdef VERBOSE_STACK
      printf("Before add_elements: exper_n=%d, nthreads=%d,"
             " max_nthreads=%d, list_length=%d\n",
             exper_n, nthreads, max_nthreads, list_length);
#   endif
    /* Create a list with n*(n+1)/2 elements.   */
    assert(0 == AO_REAL_HEAD_PTR(the_list));
    add_elements(list_length);
#   ifdef VERBOSE_STACK
      printf("Initial list (nthreads = %d):\n", nthreads);
      print_list();
#   endif
    ops_performed = 0;
    start_time = get_msecs();
    /* Start n-1 threads to run_one_test in parallel.   */
    for (i = 1; (int)i < nthreads; ++i) {
      int code;

#     ifdef USE_WINTHREADS
        thread[i] = CreateThread(NULL, 0, run_one_test, (LPVOID)(size_t)i,
                                 0, &thread_id);
        code = thread[i] != NULL ? 0 : (int)GetLastError();
#     else
        code = pthread_create(&thread[i], 0, run_one_test,
                              (void *)(size_t)i);
#     endif
      if (code != 0) {
        fprintf(stderr, "Thread creation failed %u\n", (unsigned)code);
        exit(3);
      }
    }
    /* We use the main thread to run one test.  This allows     */
    /* gprof profiling to work, for example.                    */
    run_one_test(0);
    /* Wait for all the threads to complete.    */
    for (i = 1; (int)i < nthreads; ++i) {
      int code;

#     ifdef USE_WINTHREADS
        code = WaitForSingleObject(thread[i], INFINITE) == WAIT_OBJECT_0 ?
                    0 : (int)GetLastError();
#     else
        code = pthread_join(thread[i], 0);
#     endif
      if (code != 0) {
        fprintf(stderr, "Thread join failed %u\n", (unsigned)code);
        abort();
      }
    }
    times[nthreads][exper_n] = get_msecs() - start_time;
#   ifdef VERBOSE_STACK
      printf("nthreads=%d, time_ms=%lu\n",
             nthreads, times[nthreads][exper_n]);
      printf("final list (should be reordered initial list):\n");
      print_list();
#   endif
    /* Ensure that no element is lost or duplicated.    */
    check_list(list_length);
    /* And, free the entire list.   */
    while ((le = AO_stack_pop(&the_list)) != 0)
      free(le);
    /* Retry with larger n values.      */
  }
}

void run_all_experiments(int max_nthreads)
{
  int exper_n;

  for (exper_n = 0; exper_n < N_EXPERIMENTS; ++exper_n)
    run_one_experiment(max_nthreads, exper_n);
}

/* Output the performance statistic.    */
void output_stat(int max_nthreads)
{
  int nthreads;

  assert(max_nthreads <= MAX_NTHREADS);
  for (nthreads = 1; nthreads <= max_nthreads; ++nthreads) {
#   ifndef NO_TIMES
      int exper_n;
      unsigned long sum = 0;
#   endif

    printf("About %d pushes + %d pops in %d threads:",
           LIMIT, LIMIT, nthreads);
#   ifndef NO_TIMES
      for (exper_n = 0; exper_n < N_EXPERIMENTS; ++exper_n) {
#       ifdef VERBOSE_STACK
          printf(" [%lums]", times[nthreads][exper_n]);
#       endif
        sum += times[nthreads][exper_n];
      }
      printf(" %lu msecs\n", (sum + N_EXPERIMENTS/2)/N_EXPERIMENTS);
#   else
      printf(" completed\n");
#   endif
  }
}

int main(int argc, char **argv)
{
  int max_nthreads = DEFAULT_NTHREADS;

  if (2 == argc)
    {
      max_nthreads = atoi(argv[1]);
      if (max_nthreads < 1 || max_nthreads > MAX_NTHREADS)
        {
          fprintf(stderr, "Invalid max # of threads argument\n");
          exit(1);
        }
    }
  else if (argc > 2)
    {
      fprintf(stderr, "Usage: %s [max # of threads]\n", argv[0]);
      exit(1);
    }
  if (!AO_stack_is_lock_free())
    printf("Use almost-lock-free implementation\n");
# if defined(CPPCHECK)
    AO_stack_init(&the_list);
# endif
  run_all_experiments(max_nthreads);
  output_stat(max_nthreads);
  return 0;
}

#endif
