#include "LibLsp/JsonRpc/MessageIssue.h"
#include "LibLsp/JsonRpc/WebSocketServer.h"
#include <iostream>
#include <signal.h>
#include <utility>
#include "LibLsp/JsonRpc/stream.h"
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>

// namespace beast = boost::beast; // from <boost/beast.hpp>
// namespace http = beast::http; // from <boost/beast/http.hpp>
// namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = asio; // from <boost/asio.hpp>
using tcp = asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
namespace lsp
{

//------------------------------------------------------------------------------

struct WebSocketServer::Data {
    ix::WebSocketServer server;
    std::shared_ptr<MessageJsonHandler> handler;
    std::shared_ptr<Endpoint> endpoint;
    RemoteEndPoint point;
    lsp::Log& log;

    Data(const std::string& ua,
       const std::string& addr,
         int port,
         std::shared_ptr<MessageJsonHandler> h,
         std::shared_ptr<Endpoint> ep,
         lsp::Log& lg,
         uint32_t max_workers)
        : server(port)
          , handler(std::move(h))
          , endpoint(std::move(ep))
          , point(handler, endpoint, lg, lsp::JSONStreamStyle::Standard, static_cast<uint8_t>(max_workers))
          , log(lg)
    {
        server.setOnConnectionCallback(
            [this, ua](std::weak_ptr<ix::WebSocket> wp, std::shared_ptr<ix::ConnectionState>) {
                // wrap and start processing
                auto ws = wp.lock();
                if(!ws) return;
                auto wrapper = std::make_shared<websocket_stream_wrapper>(ws);
                point.startProcessingMessages(wrapper, wrapper);
            }
        );
    }
};

websocket_stream_wrapper::websocket_stream_wrapper(std::shared_ptr<ix::WebSocket> ws)
    : ws_(std::move(ws)), request_waiter(new MultiQueueWaiter()), on_request(request_waiter)
{
    // incoming messages → queue
    ws_->setOnMessageCallback(
        [this](const ix::WebSocketMessagePtr& msg) {
            if (msg->type == ix::WebSocketMessageType::Message) {
                const auto& s = msg->str;
                on_request.EnqueueAll(std::vector<char>(s.begin(), s.end()), false);
            }
            else if (msg->type == ix::WebSocketMessageType::Error) {
                error_message = msg->str;
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
        str[i] = static_cast<char>(get());
    }
    return *this;
}

int websocket_stream_wrapper::get()
{
    return on_request.Dequeue();
}

bool websocket_stream_wrapper::bad()
{
    return ws_->getReadyState() != ix::ReadyState::Open;
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

std::string websocket_stream_wrapper::what()
{
    if (!error_message.empty())
    {
        return error_message;
    }

    if (ws_->getReadyState() != ix::ReadyState::Open)
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
      d_ptr(new Data(user_agent, address, std::stoi(port), json_handler, localEndPoint, log, _max_workers))

{

}

void WebSocketServer::run()
{
    ix::initNetSystem();
    d_ptr->server.listen();
    d_ptr->server.start();
}

void WebSocketServer::stop()
{
    d_ptr->server.stop();
}

} // namespace lsp
