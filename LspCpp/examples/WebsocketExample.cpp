#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/WebSocketServer.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace
{
std::string const kAddress = "127.0.0.1";
std::string const kPort = "9334";

class DummyLog : public lsp::Log
{
public:
    void log(Level, std::wstring&& msg) override
    {
        std::wcout << msg << std::endl;
    }

    void log(Level, const std::wstring& msg) override
    {
        std::wcout << msg << std::endl;
    }

    void log(Level, std::string&& msg) override
    {
        std::cout << msg << std::endl;
    }

    void log(Level, const std::string& msg) override
    {
        std::cout << msg << std::endl;
    }
};

class Server
{
public:
    Server()
        : server(
              "lspcpp-websocket-example",
              kAddress,
              kPort,
              protocol_json_handler,
              endpoint,
              log)
    {
        server.point.registerHandler(
            [](const td_initialize::request& req) -> lsp::ResponseOrError<td_initialize::response> {
                td_initialize::response rsp;
                rsp.id = req.id;
                CodeLensOptions code_lens_options;
                code_lens_options.resolveProvider = true;
                rsp.result.capabilities.codeLensProvider = code_lens_options;
                return rsp;
            });

        server.point.registerHandler(
            [](const td_definition::request& req,
               const CancelMonitor&) -> lsp::ResponseOrError<td_definition::response> {
                td_definition::response rsp;
                rsp.id = req.id;
                rsp.result.first = std::vector<lsLocation>();
                return rsp;
            });

        server.point.registerHandler([](Notify_Exit::notify&) {
        });
    }

    ~Server()
    {
        stop();
    }

    void start()
    {
        if (started_)
        {
            return;
        }
        started_ = true;
        thread = std::thread([this]() {
            server.run();
        });
    }

    void stop()
    {
        if (stopped_)
        {
            return;
        }
        stopped_ = true;
        server.stop();
        if (thread.joinable())
        {
            thread.join();
        }
    }

    bool ready() const
    {
        return server.point.isWorking();
    }

private:
    bool started_ = false;
    bool stopped_ = false;
    std::shared_ptr<lsp::ProtocolJsonHandler> protocol_json_handler =
        std::make_shared<lsp::ProtocolJsonHandler>();
    DummyLog log;
    std::shared_ptr<GenericEndpoint> endpoint = std::make_shared<GenericEndpoint>(log);
    lsp::WebSocketServer server;
    std::thread thread;
};

class Client
{
private:
    std::shared_ptr<lsp::ProtocolJsonHandler> protocol_json_handler =
        std::make_shared<lsp::ProtocolJsonHandler>();
    DummyLog log;
    std::shared_ptr<GenericEndpoint> endpoint = std::make_shared<GenericEndpoint>(log);
    std::shared_ptr<ix::WebSocket> websocket;
    std::shared_ptr<lsp::websocket_stream_wrapper> stream;
    std::thread websocket_thread;

public:
    Client()
        : websocket(std::make_shared<ix::WebSocket>()),
          point(protocol_json_handler, endpoint, log)
    {
    }

    ~Client()
    {
        stop();
    }

    bool connect()
    {
        websocket->setUrl("ws://" + kAddress + ":" + kPort + "/");
        websocket->disableAutomaticReconnection();
        websocket->setOnMessageCallback([](const ix::WebSocketMessagePtr&) {});
        auto const result = websocket->connect(5);
        if (!result.success)
        {
            std::cerr << result.errorStr << std::endl;
            return false;
        }

        stream = std::make_shared<lsp::websocket_stream_wrapper>(websocket);
        stream->bindMessageCallback();
        websocket_thread = std::thread([this]() {
            websocket->run();
        });
        point.startProcessingMessages(stream, stream);
        return true;
    }

    void stop()
    {
        if (stopped_)
        {
            return;
        }
        stopped_ = true;

        if (stream)
        {
            stream->interrupt();
        }
        if (websocket)
        {
            websocket->stop();
        }
        if (websocket_thread.joinable())
        {
            websocket_thread.join();
        }
        point.stop();
    }

    RemoteEndPoint point;

private:
    bool stopped_ = false;
};

} // namespace

int main()
{
    ix::initNetSystem();

    Server server;
    server.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    Client client;
    if (!client.connect())
    {
        std::cerr << "websocket client connect time out" << std::endl;
        return 1;
    }

    bool server_ready = false;
    for (int i = 0; i < 100; ++i)
    {
        if (server.ready())
        {
            server_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!server_ready)
    {
        std::cerr << "websocket server endpoint not ready" << std::endl;
        return 1;
    }

    td_initialize::request initialize_req;
    auto initialize_rsp = client.point.waitResponse(initialize_req, 5000);
    if (!initialize_rsp || initialize_rsp->is_error)
    {
        std::cerr << "initialize response failed" << std::endl;
        return 1;
    }
    std::cout << initialize_rsp->response.ToJson() << std::endl;

    td_definition::request definition_req;
    auto definition_rsp = client.point.waitResponse(definition_req, 5000);
    if (!definition_rsp || definition_rsp->is_error)
    {
        std::cerr << "definition response failed" << std::endl;
        return 1;
    }
    std::cout << definition_rsp->response.ToJson() << std::endl;

    Notify_Exit::notify notify;
    client.point.send(notify);
    client.stop();
    server.stop();

    return 0;
}
