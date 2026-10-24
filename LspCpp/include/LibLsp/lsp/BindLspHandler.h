#pragma once

#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/JsonRpc/RequestCancellation.h"
#include "LibLsp/JsonRpc/RequestError.h"
#include "LibLsp/JsonRpc/traits.h"

namespace lsp
{
namespace detail
{

template<typename ResponseType, typename F>
ResponseOrError<ResponseType> invokeBoundHandler(F& handler, typename lsp::traits::ParameterType<F, 0> const& request)
{
    try
    {
        return handler(request);
    }
    catch (RequestError const&)
    {
        throw;
    }
    catch (std::exception const& ex)
    {
        throw RequestError(lsErrorCodes::InternalError, ex.what());
    }
}

template<typename ResponseType, typename F>
ResponseOrError<ResponseType> invokeBoundHandler(
    F& handler,
    typename lsp::traits::ParameterType<F, 0> const& request,
    CancelMonitor const& monitor
)
{
    try
    {
        return handler(request, monitor);
    }
    catch (RequestError const&)
    {
        throw;
    }
    catch (std::exception const& ex)
    {
        throw RequestError(lsErrorCodes::InternalError, ex.what());
    }
}

} // namespace detail

template<typename F, typename RequestType = lsp::traits::ParameterType<F, 0>, typename ResponseType = typename RequestType::Response>
bool BindLspHandler(RemoteEndPoint& endpoint, F&& handler)
{
    return endpoint.registerHandler(
        [handler = std::forward<F>(handler)](RequestType const& request) mutable -> ResponseOrError<ResponseType>
        {
            return detail::invokeBoundHandler<ResponseType>(handler, request);
        });
}

template<typename F, typename RequestType = lsp::traits::ParameterType<F, 0>, typename ResponseType = typename RequestType::Response>
bool BindLspHandlerWithMonitor(RemoteEndPoint& endpoint, F&& handler)
{
    return endpoint.registerHandler(
        [handler = std::forward<F>(handler)](RequestType const& request, CancelMonitor const& monitor) mutable
            -> ResponseOrError<ResponseType>
        {
            return detail::invokeBoundHandler<ResponseType>(handler, request, monitor);
        });
}

template<typename F, typename RequestType = lsp::traits::ParameterType<F, 0>, typename ResponseType = typename RequestType::Response>
bool BindLspAsyncHandler(RemoteEndPoint& endpoint, F&& handler)
{
    return endpoint.registerHandler(
        [handler = std::forward<F>(handler)](RequestType const& request) mutable -> lsp::detail::handler_return_t<F>
        {
            try
            {
                return handler(request);
            }
            catch (RequestError const&)
            {
                throw;
            }
            catch (std::exception const& ex)
            {
                throw RequestError(lsErrorCodes::InternalError, ex.what());
            }
        });
}

template<typename F, typename RequestType = lsp::traits::ParameterType<F, 0>, typename ResponseType = typename RequestType::Response>
bool BindLspAsyncHandlerWithMonitor(RemoteEndPoint& endpoint, F&& handler)
{
    return endpoint.registerHandler(
        [handler = std::forward<F>(handler)](RequestType const& request, CancelMonitor const& monitor) mutable
            -> lsp::detail::handler_return_t<F>
        {
            try
            {
                return handler(request, monitor);
            }
            catch (RequestError const&)
            {
                throw;
            }
            catch (std::exception const& ex)
            {
                throw RequestError(lsErrorCodes::InternalError, ex.what());
            }
        });
}

} // namespace lsp
