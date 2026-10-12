#pragma once

#include "LibLsp/JsonRpc/RequestError.h"
#include "LibLsp/JsonRpc/lsResponseMessage.h"
#include "LibLsp/lsp/lsResponseError.h"

#include <string>
#include <utility>

namespace lsp
{

inline RequestError makeRequestCancelledError(std::string message = "Request cancelled")
{
    return RequestError(lsErrorCodes::RequestCancelled, std::move(message));
}

inline RequestError makeContentModifiedError(std::string message = "Content modified")
{
    return RequestError(lsErrorCodes::ContentModified, std::move(message));
}

inline Rsp_Error makeRequestCancelledResponse(lsRequestId const& id, std::string message = "Request cancelled")
{
    return toRspError(id, makeRequestCancelledError(std::move(message)));
}

inline Rsp_Error makeContentModifiedResponse(lsRequestId const& id, std::string message = "Content modified")
{
    return toRspError(id, makeContentModifiedError(std::move(message)));
}

inline bool isRequestCancelledCode(lsErrorCodes code)
{
    return code == lsErrorCodes::RequestCancelled;
}

inline bool isContentModifiedCode(lsErrorCodes code)
{
    return code == lsErrorCodes::ContentModified;
}

} // namespace lsp
