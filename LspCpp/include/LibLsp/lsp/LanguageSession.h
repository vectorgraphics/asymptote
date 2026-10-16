#pragma once

#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/MessageIssue.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/JsonRpc/stream.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"

#include <memory>
#include <string>
#include <utility>

namespace lsp
{

struct LanguageSessionOptions
{
    JSONStreamStyle style = JSONStreamStyle::Standard;
    uint8_t max_workers = 2;
    ProtocolJsonHandlerOptions protocol;
};

class LanguageSession
{
public:
    explicit LanguageSession(Log& log, LanguageSessionOptions const& options)
        : log_(log),
          protocol_json_handler_(std::make_shared<ProtocolJsonHandler>(options.protocol)),
          endpoint_(std::make_shared<GenericEndpoint>(log_)),
          remote_endpoint_(protocol_json_handler_, endpoint_, log_, options.style, options.max_workers)
    {
    }

    explicit LanguageSession(
        Log& log, JSONStreamStyle style = JSONStreamStyle::Standard, uint8_t max_workers = 2
    )
        : LanguageSession(log, makeOptions(style, max_workers))
    {
    }

    explicit LanguageSession(LanguageSessionOptions const& options) : LanguageSession(defaultLog(), options)
    {
    }

    LanguageSession(JSONStreamStyle style = JSONStreamStyle::Standard, uint8_t max_workers = 2)
        : LanguageSession(defaultLog(), style, max_workers)
    {
    }

    template<typename F>
    void on(F&& handler)
    {
        remote_endpoint_.registerHandler(std::forward<F>(handler));
    }

    void overrideRequestParser(std::string const& method, GenericRequestJsonHandler handler)
    {
        remote_endpoint_.overrideRequestParser(method, std::move(handler));
    }

    template<typename RequestType>
    void overrideRequestParser()
    {
        remote_endpoint_.overrideRequestParser<RequestType>();
    }

    void start(std::shared_ptr<istream> input, std::shared_ptr<ostream> output)
    {
        remote_endpoint_.startProcessingMessages(std::move(input), std::move(output));
    }

    void startStdio()
    {
        start(make_stdin_stream(), make_stdout_stream());
    }

    void stop()
    {
        remote_endpoint_.stop();
    }

    RemoteEndPoint& endpoint()
    {
        return remote_endpoint_;
    }

    RemoteEndPoint const& endpoint() const
    {
        return remote_endpoint_;
    }

    std::shared_ptr<ProtocolJsonHandler> protocolJsonHandler() const
    {
        return protocol_json_handler_;
    }

    std::shared_ptr<GenericEndpoint> localEndpoint() const
    {
        return endpoint_;
    }

private:
    static LanguageSessionOptions makeOptions(JSONStreamStyle style, uint8_t max_workers)
    {
        LanguageSessionOptions options;
        options.style = style;
        options.max_workers = max_workers;
        return options;
    }

    static Log& defaultLog()
    {
        static NullLog log;
        return log;
    }

    Log& log_;
    std::shared_ptr<ProtocolJsonHandler> protocol_json_handler_;
    std::shared_ptr<GenericEndpoint> endpoint_;
    RemoteEndPoint remote_endpoint_;
};

using LanguageServer = LanguageSession;

} // namespace lsp
