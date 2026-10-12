#pragma once

#include "LibLsp/JsonRpc/lsRequestId.h"
#include "LibLsp/lsp/lsResponseError.h"
#include "LibLsp/lsp/lsp_diagnostic.h"

#include <exception>
#include <string>
#include <utility>

namespace lsp
{

class RequestError : public std::exception
{
public:
    RequestError(lsErrorCodes code, std::string message) : code_(code), message_(std::move(message))
    {
    }

    lsErrorCodes code() const
    {
        return code_;
    }

    std::string const& message() const
    {
        return message_;
    }

    char const* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    lsErrorCodes code_;
    std::string message_;
};

inline Rsp_Error toRspError(lsRequestId const& id, RequestError const& error)
{
    Rsp_Error rsp;
    rsp.id = id;
    rsp.error.code = error.code();
    rsp.error.message = error.message();
    return rsp;
}

} // namespace lsp
