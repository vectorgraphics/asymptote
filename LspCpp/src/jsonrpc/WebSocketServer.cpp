#include "LibLsp/JsonRpc/MessageIssue.h"
#include "LibLsp/JsonRpc/WebSocketServer.h"
#include <cstdio>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <utility>
#include "LibLsp/JsonRpc/stream.h"
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>

// namespace beast = boost::beast; // from <boost/beast.hpp>
// namespace http = beast::http; // from <boost/beast/http.hpp>
// namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = asio;
using tcp = asio::ip::tcp;
namespace lsp
{

//------------------------------------------------------------------------------

struct WebSocketServer::Data {
    ix::WebSocketServer server;
    std::shared_ptr<MessageJsonHandler> handler;
    std::shared_ptr<Endpoint> endpoint;
    RemoteEndPoint& point;
    lsp::Log& log;
    std::mutex connection_mutex;
    std::shared_ptr<websocket_stream_wrapper> active_stream;

    Data(const std::string& ua,
       const std::string& addr,
         int port,
         std::shared_ptr<MessageJsonHandler> h,
         std::shared_ptr<Endpoint> ep,
         lsp::Log& lg,
         RemoteEndPoint& remote_endpoint)
        : server(port, addr)
          , handler(std::move(h))
          , endpoint(std::move(ep))
          , point(remote_endpoint)
          , log(lg)
    {
        server.setOnConnectionCallback(
            [this, ua](std::weak_ptr<ix::WebSocket> wp, std::shared_ptr<ix::ConnectionState>) {
                // wrap and start processing
                auto ws = wp.lock();
                if(!ws) return;
                auto wrapper = std::make_shared<websocket_stream_wrapper>(ws);
                wrapper->bindMessageCallback();
                {
                    std::lock_guard<std::mutex> lock(connection_mutex);
                    if (active_stream)
                    {
                        active_stream->interrupt();
                        point.stop();
                    }
                    active_stream = wrapper;
                }
                point.startProcessingMessages(wrapper, wrapper);
            }
        );
    }
};

websocket_stream_wrapper::websocket_stream_wrapper(std::shared_ptr<ix::WebSocket> ws)
    : ws_(std::move(ws)), request_waiter(new MultiQueueWaiter()), on_request(request_waiter)
{
}

websocket_stream_wrapper::~websocket_stream_wrapper()
{
    if (ws_)
    {
        ws_->setOnMessageCallback([](ix::WebSocketMessagePtr const&) {});
    }
    interrupt();
}

void websocket_stream_wrapper::bindMessageCallback()
{
    std::weak_ptr<websocket_stream_wrapper> weak_self = std::static_pointer_cast<websocket_stream_wrapper>(shared_from_this());
    // incoming messages -> queue
    ws_->setOnMessageCallback(
        [weak_self](ix::WebSocketMessagePtr const& msg)
        {
            auto self = weak_self.lock();
            if (!self)
            {
                return;
            }
            if (msg->type == ix::WebSocketMessageType::Message)
            {
                const auto& s = msg->str;
                self->on_request.EnqueueAll(std::vector<char>(s.begin(), s.end()), false);
            }
            else if (msg->type == ix::WebSocketMessageType::Error)
            {
                self->error_message = msg->str;
                self->interrupt();
            }
            else if (msg->type == ix::WebSocketMessageType::Close)
            {
                self->interrupt();
            }
        }
    );
}

bool websocket_stream_wrapper::fail()
{
    return bad();
}

bool websocket_stream_wrapper::eof()
{
    return bad();
}

bool websocket_stream_wrapper::good()
{
    return !bad();
}

websocket_stream_wrapper& websocket_stream_wrapper::read(char* str, std::streamsize count)
{
    auto some = on_request.TryDequeueSome(static_cast<size_t>(count));
    memcpy(str, some.data(), some.size());
    for (std::streamsize i = some.size(); i < count; ++i)
    {
        int const c = get();
        if (c == EOF)
        {
            break;
        }
        str[i] = static_cast<char>(c);
    }
    return *this;
}

int websocket_stream_wrapper::get()
{
    if (request_waiter->Wait(quit, &on_request))
    {
        return EOF;
    }
    auto item = on_request.TryDequeue(false);
    return item ? static_cast<unsigned char>(*item) : EOF;
}

bool websocket_stream_wrapper::bad()
{
    return quit.load(std::memory_order_relaxed);
}

websocket_stream_wrapper& websocket_stream_wrapper::write(std::string const& c)
{
    ws_->send(c);
    return *this;
}

websocket_stream_wrapper& websocket_stream_wrapper::write(std::streamsize _s)
{
    std::ostringstream temp;
    temp << _s;
    ws_->send(temp.str());
    return *this;
}

websocket_stream_wrapper& websocket_stream_wrapper::flush()
{
    return *this;
}

void websocket_stream_wrapper::clear()
{
}

void websocket_stream_wrapper::interrupt()
{
    quit.store(true, std::memory_order_relaxed);
    request_waiter->cv.notify_all();
}

std::string websocket_stream_wrapper::what()
{
    if (!error_message.empty())
    {
        return error_message;
    }

    auto const state = ws_->getReadyState();
    if (state == ix::ReadyState::Closing || state == ix::ReadyState::Closed)
    {
        return "Socket is not open.";
    }
    return {};
}

WebSocketServer::~WebSocketServer()
{
    delete d_ptr;
}

WebSocketServer::WebSocketServer(
    std::string const& user_agent, std::string const& address, std::string const& port,
    std::shared_ptr<MessageJsonHandler> json_handler, std::shared_ptr<Endpoint> localEndPoint, lsp::Log& log,
    uint32_t _max_workers
)
    : point(json_handler, localEndPoint, log, lsp::JSONStreamStyle::Standard, static_cast<uint8_t>(_max_workers)),
      d_ptr(new Data(user_agent, address, std::stoi(port), json_handler, localEndPoint, log, point))

{

}

void WebSocketServer::run()
{
    ix::initNetSystem();
    auto const listen_result = d_ptr->server.listen();
    if (!listen_result.first)
    {
        d_ptr->log.log(lsp::Log::Level::SEVERE, listen_result.second);
        return;
    }
    d_ptr->server.start();
}

void WebSocketServer::stop()
{
    d_ptr->server.stop();
    {
        std::lock_guard<std::mutex> lock(d_ptr->connection_mutex);
        if (d_ptr->active_stream)
        {
            d_ptr->active_stream->interrupt();
            d_ptr->active_stream.reset();
        }
    }
    point.stop();
}

} // namespace lsp
