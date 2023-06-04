
#include "LibLsp/JsonRpc/WebSocketServer.h"
#include "LibLsp/lsp/textDocument/signature_help.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/textDocument/typeHierarchy.h"
#include "LibLsp/lsp/AbsolutePath.h"
#include "LibLsp/lsp/textDocument/resolveCompletionItem.h"
#include <network/uri.hpp>


#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/stream.h"
#include "LibLsp/JsonRpc/TcpServer.h"
#include "LibLsp/lsp/textDocument/document_symbol.h"
#include "LibLsp/lsp/workspace/execute_command.h"

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <boost/beast/version.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/dispatch.hpp>
#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/JsonRpc/stream.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

//------------------------------------------------------------------------------

std::string _address = "127.0.0.1";
std::string _port = "9333";




using namespace std;
class DummyLog :public lsp::Log
{
public:

    void log(Level level, std::wstring&& msg)
    {
        std::wcout << msg << std::endl;
    };
    void log(Level level, const std::wstring& msg)
    {
        std::wcout << msg << std::endl;
    };
    void log(Level level, std::string&& msg)
    {
        std::cout << msg << std::endl;
    };
    void log(Level level, const std::string& msg)
    {
        std::cout << msg << std::endl;
    };
};



// Sends a WebSocket message and prints the response
class Client : public std::enable_shared_from_this<Client>
{
    net::io_context ioc;
    tcp::resolver resolver_;
    websocket::stream<beast::tcp_stream> ws_;
    beast::flat_buffer buffer_;
    std::string host_;
    std::string user_agent_;
    std::shared_ptr < lsp::ProtocolJsonHandler >  protocol_json_handler = std::make_shared< lsp::ProtocolJsonHandler>();
    DummyLog _log;

    std::shared_ptr<GenericEndpoint>  endpoint = std::make_shared<GenericEndpoint>(_log);

    std::shared_ptr<lsp::websocket_stream_wrapper>  proxy_;
public:
   RemoteEndPoint point;

public:
    // Resolver and socket require an io_context
    explicit
        Client()
        : resolver_(net::make_strand(ioc))
        , ws_(net::make_strand(ioc)),point(protocol_json_handler, endpoint, _log)
    {
        proxy_ = std::make_shared<lsp::websocket_stream_wrapper>(ws_);

    }

    // Start the asynchronous operation
    void
        run(
            char const* host,
            char const* port, char const* user_agent)
    {
        // Save these for later
        host_ = host;
        user_agent_ = user_agent;
        // Look up the domain name
        resolver_.async_resolve(
            host,
            port,
            beast::bind_front_handler(
                &Client::on_resolve,
                shared_from_this()));
        std::thread([&]
        {
               ioc.run();
        }).detach();
        while (!point.isWorking())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds (50));
        }
    }

    void
        on_resolve(
            beast::error_code ec,
            tcp::resolver::results_type results)
    {
        if (ec)
            return;

        // Set the timeout for the operation
        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));

        // Make the connection on the IP address we get from a lookup
        beast::get_lowest_layer(ws_).async_connect(
            results,
            beast::bind_front_handler(
                &Client::on_connect,
                shared_from_this()));
    }

    void
        on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type)
    {
        if (ec)
            return;

        // Turn off the timeout on the tcp_stream, because
        // the websocket stream has its own timeout system.
        beast::get_lowest_layer(ws_).expires_never();

        // Set suggested timeout settings for the websocket
        ws_.set_option(
            websocket::stream_base::timeout::suggested(
                beast::role_type::client));

        // Set a decorator to change the User-Agent of the handshake
        ws_.set_option(websocket::stream_base::decorator(
            [=](websocket::request_type& req)
            {
                req.set(http::field::user_agent,
                    user_agent_.c_str());
            }));

        // Perform the websocket handshake
        ws_.async_handshake(host_, "/",
            beast::bind_front_handler(
                &Client::on_handshake,
                shared_from_this()));
    }

    void
        on_handshake(beast::error_code ec)
    {
        if (ec)
            return;

        // Send the message


        point.startProcessingMessages(proxy_, proxy_);
        // Read a message into our buffer
        ws_.async_read(
            buffer_,
            beast::bind_front_handler(
                &Client::on_read,
                shared_from_this()));
    }


    void
        on_read(
            beast::error_code ec,
            std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if (ec)
            return;

        char* data = reinterpret_cast<char*>(buffer_.data().data());
        std::vector<char> elements(data, data + bytes_transferred);
        buffer_.clear();
        proxy_->on_request.EnqueueAll(std::move(elements), false);

        ws_.async_read(
            buffer_,
            beast::bind_front_handler(
                &Client::on_read,
                shared_from_this()));
    }

    void
        on_close(beast::error_code ec)
    {
        if (ec)
            return;

        // If we get here then the connection is closed gracefully

        // The make_printable() function helps print a ConstBufferSequence
        std::cout << beast::make_printable(buffer_.data()) << std::endl;
    }
};

class Server
{
public:
    Server(const std::string& user_agent) : server(user_agent,_address, _port, protocol_json_handler, endpoint, _log)
    {
        server.point.registerHandler(
            [&](const td_initialize::request& req)
            {
                td_initialize::response rsp;
                CodeLensOptions code_lens_options;
                code_lens_options.resolveProvider = true;
                rsp.result.capabilities.codeLensProvider = code_lens_options;
                return rsp;
                });
        std::thread([&]()
            {
                server.run();
            }).detach();
    }
    ~Server()
    {
        server.stop();
    }
    std::shared_ptr <  lsp::ProtocolJsonHandler >  protocol_json_handler = std::make_shared < lsp::ProtocolJsonHandler >();
    DummyLog _log;

    std::shared_ptr < GenericEndpoint >  endpoint = std::make_shared<GenericEndpoint>(_log);
    lsp::WebSocketServer server;

};

int main()
{
    std::string user_agent = std::string(BOOST_BEAST_VERSION_STRING) +" websocket-server-async";
    Server server(user_agent);

    auto client = std::make_shared<Client>();
    user_agent = std::string(BOOST_BEAST_VERSION_STRING) + " websocket-client-async";
    client->run(_address.c_str(), _port.c_str(), user_agent.c_str());

    td_initialize::request req;

    auto rsp = client->point.waitResponse(req);
    if (rsp)
    {
        std::cout << rsp->ToJson() << std::endl;
    }
    return 0;
}









