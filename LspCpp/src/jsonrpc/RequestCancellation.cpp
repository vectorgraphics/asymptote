#include "LibLsp/JsonRpc/RequestCancellation.h"

#include <atomic>
#include <cassert>

namespace lsp
{
namespace
{
struct CancelState
{
    std::shared_ptr<std::atomic<int>> cancelled;
    CancelState const* parent = nullptr;
    lsRequestId id;
};

Key<CancelState> g_cancelStateKey;
} // namespace

std::pair<Context, Canceler> cancelableTask(lsRequestId const& id, int reason)
{
    assert(reason != 0 && "Can't detect cancellation if reason is zero");
    CancelState state;
    state.id = id;
    state.cancelled = std::make_shared<std::atomic<int>>(0);
    state.parent = Context::current().get(g_cancelStateKey);
    return {
        Context::current().derive(g_cancelStateKey, state),
        [reason, cancelled(state.cancelled)] { cancelled->store(reason); },
    };
}

optional<CancelMonitor> getCancelledMonitor(lsRequestId const& id, Context const& ctx)
{
    for (CancelState const* state = ctx.get(g_cancelStateKey); state != nullptr; state = state->parent)
    {
        if (id != state->id)
        {
            continue;
        }
        std::shared_ptr<std::atomic<int>> const cancelled = state->cancelled;
        return CancelMonitor([cancelled] { return cancelled->load(); });
    }
    return {};
}

bool isCancellationRequested(CancelMonitor const& monitor)
{
    return monitor && cancellationReason(monitor) != 0;
}

int cancellationReason(CancelMonitor const& monitor)
{
    return monitor ? monitor() : 0;
}

} // namespace lsp
