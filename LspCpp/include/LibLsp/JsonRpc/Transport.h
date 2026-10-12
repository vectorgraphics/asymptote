#pragma once

#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/JsonRpc/stream.h"

#include <memory>

namespace lsp
{

/// Optional thin facade over RemoteEndPoint for clangd-style notify/call/reply/run.
/// Does not replace RemoteEndPoint; existing construction paths stay unchanged.
class Transport
{
public:
    explicit Transport(RemoteEndPoint& endpoint) : endpoint_(endpoint)
    {
    }

    RemoteEndPoint& endpoint()
    {
        return endpoint_;
    }

    RemoteEndPoint const& endpoint() const
    {
        return endpoint_;
    }

    void notify(NotificationInMessage& msg)
    {
        endpoint_.sendNotification(msg);
    }

    void reply(ResponseInMessage& msg)
    {
        endpoint_.sendResponse(msg);
    }

    template<typename T, typename F>
    void call(T& request, F&& handler, RemoteEndPoint::RequestErrorCallback onError)
    {
        endpoint_.send(request, std::forward<F>(handler), std::move(onError));
    }

    template<typename T>
    auto call(T& request)
    {
        return endpoint_.send(request);
    }

    void run(std::shared_ptr<istream> input, std::shared_ptr<ostream> output)
    {
        endpoint_.startProcessingMessages(std::move(input), std::move(output));
    }

    /// clangd-style alias for run(); starts processing on the caller thread and returns immediately.
    void loop(std::shared_ptr<istream> input, std::shared_ptr<ostream> output)
    {
        run(std::move(input), std::move(output));
    }

    void stop()
    {
        endpoint_.stop();
    }

    bool isWorking() const
    {
        return endpoint_.isWorking();
    }

private:
    RemoteEndPoint& endpoint_;
};

} // namespace lsp
