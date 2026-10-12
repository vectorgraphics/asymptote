#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/ScopeExit.h"
#include "LibLsp/JsonRpc/Transport.h"
#include "LibLsp/JsonRpc/WebSocketServer.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/windows/MessageNotify.h"
#include "test_helpers.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

namespace
{
using test::Expect;

int PickPort()
{
    auto const now = std::chrono::steady_clock::now().time_since_epoch().count();
    return 20000 + static_cast<int>(now % 20000);
}

bool WaitForServerConnection(RemoteEndPoint const& endpoint)
{
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        if (endpoint.isWorking())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

bool ExtractLspBody(std::string const& frame, std::string& body)
{
    size_t content_length = 0;
    bool saw_content_length = false;
    size_t pos = 0;
    while (pos < frame.size())
    {
        auto const line_end = frame.find('\n', pos);
        if (line_end == std::string::npos)
        {
            return false;
        }
        std::string line = frame.substr(pos, line_end - pos);
        pos = line_end + 1;
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            break;
        }
        std::string const prefix = "Content-Length:";
        if (line.compare(0, prefix.size(), prefix) == 0)
        {
            content_length = static_cast<size_t>(std::atoi(line.substr(prefix.size()).c_str()));
            saw_content_length = true;
        }
    }
    if (!saw_content_length || pos + content_length > frame.size())
    {
        return false;
    }
    body.assign(frame.data() + pos, content_length);
    return true;
}

class WebSocketTestClient
{
public:
    explicit WebSocketTestClient(int port)
    {
        websocket_ = std::make_shared<ix::WebSocket>();
        websocket_->setUrl("ws://127.0.0.1:" + std::to_string(port) + "/");
        websocket_->disableAutomaticReconnection();
        websocket_->setOnMessageCallback(
            [this](ix::WebSocketMessagePtr const& msg)
            {
                if (msg->type != ix::WebSocketMessageType::Message)
                {
                    return;
                }
                std::lock_guard<std::mutex> lock(mutex_);
                messages_.push_back(msg->str);
                cv_.notify_all();
            });
    }

    ~WebSocketTestClient()
    {
        stop();
    }

    bool connect()
    {
        auto const result = websocket_->connect(5);
        if (!result.success)
        {
            return false;
        }
        websocket_thread_ = std::thread(
            [this]()
            {
                websocket_->run();
            });
        return true;
    }

    void stop()
    {
        if (stopped_)
        {
            return;
        }
        stopped_ = true;
        websocket_->stop();
        if (websocket_thread_.joinable())
        {
            websocket_thread_.join();
        }
    }

    void send(std::string const& frame)
    {
        websocket_->send(frame);
    }

    bool waitForMessageCount(size_t count, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000))
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(
            lock,
            timeout,
            [&]()
            {
                return messages_.size() >= count;
            });
    }

    std::string popMessage()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (messages_.empty())
        {
            return {};
        }
        std::string message = std::move(messages_.front());
        messages_.pop_front();
        return message;
    }

private:
    std::shared_ptr<ix::WebSocket> websocket_;
    std::thread websocket_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> messages_;
    bool stopped_ = false;
};

void RegisterInitializeHandler(lsp::WebSocketServer& server)
{
    server.point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities.hoverProvider = true;
            return rsp;
        });
}

bool SendInitializeAndReadResponse(WebSocketTestClient& client, int id, std::string& body)
{
    std::string const request =
        std::string(R"({"jsonrpc":"2.0","id":)") + std::to_string(id) + R"(,"method":"initialize","params":{}})";
    client.send(test::MakeLspFrame(request));
    if (!client.waitForMessageCount(1))
    {
        return false;
    }
    return ExtractLspBody(client.popMessage(), body);
}

void TestWebSocketWriteQueuePreservesLargeNotificationOrder()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-write-queue-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    WebSocketTestClient client(port);
    bool const connected = client.connect();
    Expect(connected, "WebSocket write queue test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "WebSocketServer must install RemoteEndPoint streams after accept");
    if (!server_ready)
    {
        return;
    }

    std::vector<std::string> expected_markers;
    std::vector<std::string> expected_payloads;
    for (int i = 0; i < 6; ++i)
    {
        auto const marker = "ws-message-" + std::to_string(i) + ":";
        auto payload = marker + std::string(5000 + static_cast<size_t>(i), static_cast<char>('a' + i));
        expected_markers.push_back(marker);
        expected_payloads.push_back(payload);

        Notify_LogMessage::notify notify;
        notify.params.type = lsMessageType::Log;
        notify.params.message = payload;
        server.point.send(notify);
    }

    bool const got_all = client.waitForMessageCount(expected_payloads.size());
    Expect(got_all, "WebSocket client must receive every large notification");
    if (!got_all)
    {
        return;
    }

    for (size_t i = 0; i < expected_payloads.size(); ++i)
    {
        std::string const frame = client.popMessage();
        std::string body;
        bool const got_frame = ExtractLspBody(frame, body);
        Expect(got_frame, "WebSocket client must receive a complete LSP frame");
        if (!got_frame)
        {
            return;
        }

        Expect(
            body.find("\"method\":\"window/logMessage\"") != std::string::npos,
            "WebSocket frame must be a window/logMessage notification");
        Expect(
            body.find(expected_markers[i]) != std::string::npos,
            "WebSocket write queue must preserve notification order");
        Expect(
            body.find(expected_payloads[i]) != std::string::npos,
            "WebSocket write queue must preserve large notification body bytes");
    }
}

void TestWebSocketInboundRequestReceivesResponse()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-inbound-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    RegisterInitializeHandler(server);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    WebSocketTestClient client(port);
    bool const connected = client.connect();
    Expect(connected, "WebSocket inbound test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "WebSocketServer must install streams before reading inbound requests");
    if (!server_ready)
    {
        return;
    }

    std::string const request =
        std::string(R"({"jsonrpc":"2.0","id":)") + std::to_string(5101) + R"(,"method":"initialize","params":{}})";
    client.send(test::MakeLspFrame(request));

    bool const got_response = client.waitForMessageCount(1);
    Expect(got_response, "WebSocket inbound request must receive a complete response frame");
    if (!got_response)
    {
        return;
    }

    std::string body;
    bool const parsed = ExtractLspBody(client.popMessage(), body);
    Expect(parsed, "WebSocket inbound response must be a valid LSP frame");
    Expect(body.find("\"id\":5101") != std::string::npos, "WebSocket inbound response must preserve request id");
    Expect(
        body.find("\"hoverProvider\":true") != std::string::npos,
        "WebSocket inbound response must include handler result");
}

void TestWebSocketWriteQueuePreservesConcurrentLargeNotifications()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-concurrent-write-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    WebSocketTestClient client(port);
    bool const connected = client.connect();
    Expect(connected, "concurrent WebSocket write queue test client must connect");
    if (!connected || !WaitForServerConnection(server.point))
    {
        return;
    }

    std::vector<std::string> expected_markers;
    std::vector<std::string> expected_payloads;
    for (int i = 0; i < 12; ++i)
    {
        auto const marker = "ws-concurrent-" + std::to_string(i) + ":";
        expected_markers.push_back(marker);
        expected_payloads.push_back(marker + std::string(4096 + static_cast<size_t>(i), static_cast<char>('A' + i)));
    }

    std::vector<std::thread> senders;
    for (size_t i = 0; i < expected_payloads.size(); ++i)
    {
        senders.emplace_back(
            [&, i]()
            {
                Notify_LogMessage::notify notify;
                notify.params.type = lsMessageType::Log;
                notify.params.message = expected_payloads[i];
                server.point.send(notify);
            });
    }
    for (auto& sender : senders)
    {
        sender.join();
    }

    bool const got_all = client.waitForMessageCount(expected_payloads.size());
    Expect(got_all, "WebSocket client must receive every concurrent large notification");
    if (!got_all)
    {
        return;
    }

    std::set<std::string> seen_markers;
    for (size_t frame = 0; frame < expected_payloads.size(); ++frame)
    {
        std::string body;
        bool const got_frame = ExtractLspBody(client.popMessage(), body);
        Expect(got_frame, "WebSocket client must receive every concurrent frame completely");
        if (!got_frame)
        {
            return;
        }
        Expect(
            body.find("\"method\":\"window/logMessage\"") != std::string::npos,
            "concurrent WebSocket frame must be a complete logMessage notification");

        bool matched_payload = false;
        for (size_t i = 0; i < expected_payloads.size(); ++i)
        {
            if (body.find(expected_payloads[i]) != std::string::npos)
            {
                seen_markers.insert(expected_markers[i]);
                matched_payload = true;
                break;
            }
        }
        Expect(matched_payload, "concurrent WebSocket frame must contain exactly one expected large payload");
    }

    Expect(
        seen_markers.size() == expected_markers.size(),
        "concurrent WebSocket writes must preserve every notification without frame interleaving");
}

void TestWebSocketSecondClientPreemptsFirstConnection()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-second-client-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    RegisterInitializeHandler(server);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    WebSocketTestClient first_client(port);
    bool const first_connected = first_client.connect();
    Expect(first_connected, "first WebSocket client must connect");
    if (!first_connected || !WaitForServerConnection(server.point))
    {
        return;
    }

    WebSocketTestClient second_client(port);
    bool const second_connected = second_client.connect();
    Expect(second_connected, "second WebSocket client must connect");
    if (!second_connected)
    {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::string body;
    bool const got_response = SendInitializeAndReadResponse(second_client, 5202, body);
    Expect(got_response, "second WebSocket client must become the active connection and receive responses");
    Expect(body.find("\"id\":5202") != std::string::npos, "second WebSocket client response must preserve request id");
}

void TestWebSocketDisconnectAllowsNewClient()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-disconnect-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    RegisterInitializeHandler(server);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    {
        WebSocketTestClient first_client(port);
        bool const first_connected = first_client.connect();
        Expect(first_connected, "first WebSocket disconnect test client must connect");
        if (!first_connected)
        {
            return;
        }
        first_client.stop();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    WebSocketTestClient second_client(port);
    bool const second_connected = second_client.connect();
    Expect(second_connected, "WebSocketServer must accept a new client after disconnect");
    if (!second_connected)
    {
        return;
    }

    std::string body;
    bool const got_response = SendInitializeAndReadResponse(second_client, 5303, body);
    Expect(got_response, "new WebSocket client after disconnect must receive responses");
    Expect(body.find("\"id\":5303") != std::string::npos, "new WebSocket client response must preserve request id");
}

void TestWebSocketStopWithActiveConnectionReturns()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-active-stop-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });

    WebSocketTestClient client(port);
    bool const connected = client.connect();
    Expect(connected, "WebSocket active-stop test client must connect");
    if (connected)
    {
        Expect(WaitForServerConnection(server.point), "WebSocketServer must install streams before active-stop test");
    }

    server.stop();
    if (server_thread.joinable())
    {
        server_thread.join();
    }

    Expect(!server.point.isWorking(), "WebSocketServer stop with an active client must stop the RemoteEndPoint");
}

void TestWebSocketTransportFacadeNotify()
{
    ix::initNetSystem();
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::WebSocketServer server(
        "lspcpp-websocket-transport-facade-test",
        "127.0.0.1",
        std::to_string(port),
        protocol_json_handler,
        endpoint,
        log);
    RegisterInitializeHandler(server);
    std::thread server_thread(
        [&]()
        {
            server.run();
        });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    WebSocketTestClient client(port);
    bool const connected = client.connect();
    Expect(connected, "WebSocket transport facade test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "WebSocketServer must install RemoteEndPoint streams before facade notify");
    if (!server_ready)
    {
        return;
    }

    std::string initialize_body;
    bool const got_initialize = SendInitializeAndReadResponse(client, 4404, initialize_body);
    Expect(got_initialize, "WebSocket transport facade test must establish bidirectional LSP traffic first");
    if (!got_initialize)
    {
        return;
    }
    Expect(
        initialize_body.find("\"id\":4404") != std::string::npos,
        "WebSocket transport facade initialize response must preserve request id");

    lsp::Transport transport(server.point);
    Notify_LogMessage::notify notify;
    notify.params.type = lsMessageType::Log;
    notify.params.message = "transport-facade-websocket";
    transport.notify(notify);

    bool const got_message = client.waitForMessageCount(1);
    Expect(got_message, "Transport facade must emit a complete LSP frame over WebSocket");
    if (!got_message)
    {
        return;
    }

    std::string body;
    bool const got_frame = ExtractLspBody(client.popMessage(), body);
    Expect(got_frame, "Transport facade must preserve WebSocket framing");
    Expect(
        body.find("\"method\":\"window/logMessage\"") != std::string::npos,
        "Transport facade notify must serialize window/logMessage over WebSocket");
    Expect(
        body.find("transport-facade-websocket") != std::string::npos,
        "Transport facade notify must preserve payload over WebSocket");
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestWebSocketWriteQueuePreservesLargeNotificationOrder);
    RUN_TEST(TestWebSocketWriteQueuePreservesConcurrentLargeNotifications);
    RUN_TEST(TestWebSocketInboundRequestReceivesResponse);
    RUN_TEST(TestWebSocketSecondClientPreemptsFirstConnection);
    RUN_TEST(TestWebSocketDisconnectAllowsNewClient);
    RUN_TEST(TestWebSocketStopWithActiveConnectionReturns);
    RUN_TEST(TestWebSocketTransportFacadeNotify);
    return test::Failures() == 0 ? 0 : 1;
}
