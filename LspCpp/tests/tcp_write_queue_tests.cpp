#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/TcpServer.h"
#include "LibLsp/JsonRpc/ScopeExit.h"
#include "LibLsp/JsonRpc/Transport.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/asio.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/windows/MessageNotify.h"
#include "test_helpers.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
using test::Expect;

int PickPort()
{
    auto const now = std::chrono::steady_clock::now().time_since_epoch().count();
    return 20000 + static_cast<int>(now % 20000);
}

bool ConnectWithRetry(asio::ip::tcp::iostream& socket, int port)
{
    asio::ip::tcp::endpoint endpoint(asio::ip::make_address("127.0.0.1"), static_cast<unsigned short>(port));
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        socket.close();
        socket.clear();
        socket.connect(endpoint);
        if (socket.good())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
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

bool ReadLspFrame(std::istream& input, std::string& body)
{
    std::string line;
    size_t content_length = 0;
    bool saw_content_length = false;
    while (std::getline(input, line))
    {
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
    if (!saw_content_length)
    {
        return false;
    }

    body.assign(content_length, '\0');
    input.read(&body[0], static_cast<std::streamsize>(content_length));
    return static_cast<size_t>(input.gcount()) == content_length;
}

void RegisterInitializeHandler(lsp::TcpServer& server)
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

bool SendInitializeAndReadResponse(asio::ip::tcp::iostream& client, int id, std::string& body)
{
    std::string const request =
        std::string(R"({"jsonrpc":"2.0","id":)") + std::to_string(id) + R"(,"method":"initialize","params":{}})";
    client << test::MakeLspFrame(request) << std::flush;
    return ReadLspFrame(client, body);
}

void TestTcpWriteQueuePreservesLargeNotificationOrder()
{
    // Sends oversized notifications through the real TCP transport; frame order
    // and payload checks catch interleaved async writes.
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    std::thread server_thread([&]() { server.run(); });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    asio::ip::tcp::iostream client;
    bool const connected = ConnectWithRetry(client, port);
    Expect(connected, "TCP write queue test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "TcpServer must install RemoteEndPoint streams after accept");
    if (!server_ready)
    {
        return;
    }

    std::vector<std::string> expected_markers;
    std::vector<std::string> expected_payloads;
    for (int i = 0; i < 6; ++i)
    {
        auto const marker = "message-" + std::to_string(i) + ":";
        auto payload = marker + std::string(5000 + static_cast<size_t>(i), static_cast<char>('a' + i));
        expected_markers.push_back(marker);
        expected_payloads.push_back(payload);

        Notify_LogMessage::notify notify;
        notify.params.type = lsMessageType::Log;
        notify.params.message = payload;
        server.point.send(notify);
    }

    for (size_t i = 0; i < expected_payloads.size(); ++i)
    {
        std::string body;
        bool const got_frame = ReadLspFrame(client, body);
        Expect(got_frame, "client must read a complete LSP frame from TcpServer");
        if (!got_frame)
        {
            return;
        }

        Expect(
            body.find("\"method\":\"window/logMessage\"") != std::string::npos,
            "TcpServer frame must be a window/logMessage notification");
        Expect(
            body.find(expected_markers[i]) != std::string::npos,
            "TcpServer write queue must preserve notification order");
        Expect(
            body.find(expected_payloads[i]) != std::string::npos,
            "TcpServer write queue must preserve large notification body bytes");
    }
}

void TestTcpWriteQueuePreservesConcurrentLargeNotifications()
{
    // Calls RemoteEndPoint::send() from multiple threads and verifies each
    // received LSP frame remains complete and unmixed.
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    std::thread server_thread([&]() { server.run(); });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    asio::ip::tcp::iostream client;
    bool const connected = ConnectWithRetry(client, port);
    Expect(connected, "concurrent TCP write queue test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "TcpServer must install streams before concurrent sends");
    if (!server_ready)
    {
        return;
    }

    std::vector<std::string> expected_markers;
    std::vector<std::string> expected_payloads;
    for (int i = 0; i < 12; ++i)
    {
        auto const marker = "concurrent-" + std::to_string(i) + ":";
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

    std::set<std::string> seen_markers;
    for (size_t frame = 0; frame < expected_payloads.size(); ++frame)
    {
        std::string body;
        bool const got_frame = ReadLspFrame(client, body);
        Expect(got_frame, "client must read every concurrent TCP frame completely");
        if (!got_frame)
        {
            return;
        }
        Expect(
            body.find("\"method\":\"window/logMessage\"") != std::string::npos,
            "concurrent TcpServer frame must be a complete logMessage notification");

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
        Expect(matched_payload, "concurrent TcpServer frame must contain exactly one expected large payload");
    }

    Expect(
        seen_markers.size() == expected_markers.size(),
        "concurrent TcpServer writes must preserve every notification without frame interleaving");
}

void TestTcpInboundRequestReceivesResponse()
{
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    RegisterInitializeHandler(server);
    std::thread server_thread([&]() { server.run(); });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    asio::ip::tcp::iostream client;
    bool const connected = ConnectWithRetry(client, port);
    Expect(connected, "TCP inbound test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "TcpServer must install streams before reading inbound requests");
    if (!server_ready)
    {
        return;
    }

    std::string body;
    bool const got_response = SendInitializeAndReadResponse(client, 4101, body);
    Expect(got_response, "TCP inbound request must receive a complete response frame");
    Expect(body.find("\"id\":4101") != std::string::npos, "TCP inbound response must preserve request id");
    Expect(body.find("\"hoverProvider\":true") != std::string::npos, "TCP inbound response must include handler result");
}

void TestTcpSecondClientPreemptsFirstConnection()
{
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    RegisterInitializeHandler(server);
    std::thread server_thread([&]() { server.run(); });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    asio::ip::tcp::iostream first_client;
    bool const first_connected = ConnectWithRetry(first_client, port);
    Expect(first_connected, "first TCP client must connect");
    if (!first_connected || !WaitForServerConnection(server.point))
    {
        return;
    }

    asio::ip::tcp::iostream second_client;
    bool const second_connected = ConnectWithRetry(second_client, port);
    Expect(second_connected, "second TCP client must connect");
    if (!second_connected)
    {
        return;
    }

    std::string body;
    bool const got_response = SendInitializeAndReadResponse(second_client, 4202, body);
    Expect(got_response, "second TCP client must become the active connection and receive responses");
    Expect(body.find("\"id\":4202") != std::string::npos, "second TCP client response must preserve request id");
}

void TestTcpDisconnectAllowsNewClient()
{
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    RegisterInitializeHandler(server);
    std::thread server_thread([&]() { server.run(); });
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
        asio::ip::tcp::iostream first_client;
        bool const first_connected = ConnectWithRetry(first_client, port);
        Expect(first_connected, "first TCP disconnect test client must connect");
        if (!first_connected)
        {
            return;
        }
        first_client.close();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    asio::ip::tcp::iostream second_client;
    bool const second_connected = ConnectWithRetry(second_client, port);
    Expect(second_connected, "TcpServer must accept a new client after disconnect");
    if (!second_connected)
    {
        return;
    }

    std::string body;
    bool const got_response = SendInitializeAndReadResponse(second_client, 4303, body);
    Expect(got_response, "new TCP client after disconnect must receive responses");
    Expect(body.find("\"id\":4303") != std::string::npos, "new TCP client response must preserve request id");
}

void TestTcpStopWithActiveConnectionReturns()
{
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    std::thread server_thread([&]() { server.run(); });

    asio::ip::tcp::iostream client;
    bool const connected = ConnectWithRetry(client, port);
    Expect(connected, "TCP active-stop test client must connect");
    if (connected)
    {
        Expect(WaitForServerConnection(server.point), "TcpServer must install streams before active-stop test");
    }

    server.stop();
    if (server_thread.joinable())
    {
        server_thread.join();
    }

    Expect(!server.point.isWorking(), "TcpServer stop with an active client must stop the RemoteEndPoint");
}

void TestTcpTransportFacadeNotify()
{
    lsp::NullLog log;
    auto protocol_json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    int const port = PickPort();
    lsp::TcpServer server("127.0.0.1", std::to_string(port), protocol_json_handler, endpoint, log);
    std::thread server_thread([&]() { server.run(); });
    auto cleanup = lsp::make_scope_exit(
        [&]()
        {
            server.stop();
            if (server_thread.joinable())
            {
                server_thread.join();
            }
        });

    asio::ip::tcp::iostream client;
    bool const connected = ConnectWithRetry(client, port);
    Expect(connected, "TCP transport facade test client must connect");
    if (!connected)
    {
        return;
    }

    bool const server_ready = WaitForServerConnection(server.point);
    Expect(server_ready, "TcpServer must install RemoteEndPoint streams before facade notify");
    if (!server_ready)
    {
        return;
    }

    lsp::Transport transport(server.point);
    Notify_LogMessage::notify notify;
    notify.params.type = lsMessageType::Log;
    notify.params.message = "transport-facade-tcp";
    transport.notify(notify);

    std::string body;
    bool const got_frame = ReadLspFrame(client, body);
    Expect(got_frame, "Transport facade must emit a complete LSP frame over TCP");
    Expect(
        body.find("\"method\":\"window/logMessage\"") != std::string::npos,
        "Transport facade notify must serialize window/logMessage over TCP");
    Expect(
        body.find("transport-facade-tcp") != std::string::npos,
        "Transport facade notify must preserve payload over TCP");
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestTcpWriteQueuePreservesLargeNotificationOrder);
    RUN_TEST(TestTcpWriteQueuePreservesConcurrentLargeNotifications);
    RUN_TEST(TestTcpInboundRequestReceivesResponse);
    RUN_TEST(TestTcpSecondClientPreemptsFirstConnection);
    RUN_TEST(TestTcpDisconnectAllowsNewClient);
    RUN_TEST(TestTcpStopWithActiveConnectionReturns);
    RUN_TEST(TestTcpTransportFacadeNotify);
    return test::Failures() == 0 ? 0 : 1;
}
