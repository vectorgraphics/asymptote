#pragma once

#include "LibLsp/JsonRpc/Cancellation.h"
#include "LibLsp/JsonRpc/Context.h"
#include "LibLsp/JsonRpc/lsRequestId.h"
#include "LibLsp/JsonRpc/optionalVersion.h"
#include "LibLsp/lsp/lsResponseError.h"

#include <functional>
#include <utility>

namespace lsp
{

using Canceler = std::function<void()>;

/// Defines a cancelable task scope. The returned Context must be installed with
/// WithContext while work runs; invoke the Canceler to request cancellation.
std::pair<Context, Canceler> cancelableTask(
    lsRequestId const& id,
    int reason = static_cast<int>(lsErrorCodes::RequestCancelled)
);

/// Returns a monitor for the active cancelable task with the given request id.
optional<CancelMonitor> getCancelledMonitor(lsRequestId const& id, Context const& ctx = Context::current());

/// Returns true when the monitor reports a non-zero cancellation reason.
bool isCancellationRequested(CancelMonitor const& monitor);

/// Returns the cancellation reason code, or zero when not cancelled.
int cancellationReason(CancelMonitor const& monitor);

} // namespace lsp
