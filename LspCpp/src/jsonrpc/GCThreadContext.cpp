#include "LibLsp/JsonRpc/GCThreadContext.h"
#include <iostream>

GCThreadContext::GCThreadContext()
{
#ifdef USEGC
    GC_get_stack_base(&gsb);
    GC_register_my_thread(&gsb);
#endif
}

GCThreadContext::~GCThreadContext()
{
#ifdef USEGC
    GC_unregister_my_thread();
#endif
}