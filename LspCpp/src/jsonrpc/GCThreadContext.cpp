#include "LibLsp/JsonRpc/GCThreadContext.h"

#if defined(LSPCPP_USEGC)
#include <mutex>
#endif

void GCThreadContext::ensureInitialized()
{
#if defined(LSPCPP_USEGC)
    static std::once_flag once;
    std::call_once(
        once,
        []
        {
            GC_INIT();
            // Required before GC_register_my_thread() on threads not created via
            // GC_pthread_create (asio::thread_pool workers fall into this category).
            GC_allow_register_threads();
        }
    );
#endif
}

GCThreadContext::GCThreadContext()
{
#if defined(LSPCPP_USEGC)
    ensureInitialized();
    if (GC_get_stack_base(&gsb) == GC_SUCCESS)
    {
        // GC_DUPLICATE means this thread is already registered; leave it alone.
        registered_ = (GC_register_my_thread(&gsb) == GC_SUCCESS);
    }
#endif
}

GCThreadContext::~GCThreadContext()
{
#if defined(LSPCPP_USEGC)
    if (registered_)
    {
        GC_unregister_my_thread();
    }
#endif
}
