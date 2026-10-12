#pragma once

#if defined(LSPCPP_USEGC)
#define GC_THREADS
#include <gc.h>
#endif

class GCThreadContext
{
public:
    /// Ensure Boehm GC is initialized and explicit thread registration is enabled.
    /// Safe to call repeatedly; work runs at most once per process.
    static void ensureInitialized();

    GCThreadContext();
    ~GCThreadContext();

private:
#if defined(LSPCPP_USEGC)
    GC_stack_base gsb;
    bool registered_ = false;
#endif
};
