#include "LibLsp/LspCpp.h"
#include "LibLsp/JsonRpc/RequestError.h"
#include "LibLsp/lsp/BindLspHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/shutdown.h"
#include "LibLsp/lsp/windows/MessageNotify.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace
{
using test::Expect;
using test::FeedableIStream;
using test::StringOStream;
using test::StringIStream;
using test::WaitForOutputContaining;

bool WaitUntil(std::function<bool()> const& predicate, int attempts = 100)
{
    for (int i = 0; i < attempts; ++i)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
}

void TestLanguageSessionInitializeRoundTrip()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> handled {false};

    session.on(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handled.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities.hoverProvider = true;
            return rsp;
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"init-session","method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, R"("id":"init-session")");
    session.stop();

    Expect(handled.load(std::memory_order_relaxed), "LanguageSession must dispatch initialize through on()");
    Expect(response.find(R"("id":"init-session")") != std::string::npos, "LanguageSession response must preserve id");
    Expect(
        response.find("\"hoverProvider\":true") != std::string::npos,
        "LanguageSession response must include handler result");
}

void TestLanguageSessionNotificationHandler()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> notified {false};

    session.on(
        [&](Notify_Exit::notify const&)
        {
            notified.store(true, std::memory_order_relaxed);
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
    for (int i = 0; i < 100 && !notified.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    session.stop();

    Expect(notified.load(std::memory_order_relaxed), "LanguageSession must dispatch notifications through on()");
}

void TestLanguageSessionEndpointCanSendNotifications()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    session.start(input, output);

    Notify_LogMessage::notify notify;
    notify.params.type = lsMessageType::Log;
    notify.params.message = "session-log";
    session.endpoint().send(notify);

    std::string const sent = WaitForOutputContaining(output, "session-log");
    session.stop();

    Expect(
        sent.find(R"("method":"window/logMessage")") != std::string::npos,
        "LanguageSession endpoint() must expose outbound notification sends");
    Expect(sent.find("session-log") != std::string::npos, "LanguageSession outbound notification must include payload");
}

void TestLanguageSessionCanOverrideBuiltinRequestParser()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> parser_used {false};
    std::atomic<bool> handled {false};

    session.overrideRequestParser(
        td_initialize::request::kMethodInfo,
        [&](Reader& visitor)
        {
            parser_used.store(true, std::memory_order_relaxed);
            return td_initialize::request::ReflectReader(visitor);
        });
    session.on(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handled.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    session.start(input, output);

    input->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"custom-parser","method":"initialize","params":{}})")
    );
    std::string const response = WaitForOutputContaining(output, R"("id":"custom-parser")");
    session.stop();

    Expect(parser_used.load(std::memory_order_relaxed), "custom request parser must override the built-in parser");
    Expect(handled.load(std::memory_order_relaxed), "overridden request parser must still dispatch to the handler");
    Expect(
        response.find(R"("id":"custom-parser")") != std::string::npos,
        "response must preserve the request id parsed by the custom parser");
}

void TestLanguageSessionOverrideRequestParserTemplate()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> handled {false};

    session.overrideRequestParser<td_initialize::request>();
    session.on(
        [&](td_initialize::request const& req) -> td_initialize::response
        {
            handled.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"template-parser","method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, R"("id":"template-parser")");
    session.stop();

    Expect(handled.load(std::memory_order_relaxed), "template overrideRequestParser must leave initialize dispatch usable");
    Expect(
        response.find(R"("id":"template-parser")") != std::string::npos,
        "template overrideRequestParser response must preserve request id");
}

void TestLanguageServerAliasConstructsUsableSession()
{
    lsp::LanguageServer server;

    Expect(server.protocolJsonHandler() != nullptr, "LanguageServer alias must expose ProtocolJsonHandler");
    Expect(server.localEndpoint() != nullptr, "LanguageServer alias must expose GenericEndpoint");
    Expect(
        server.protocolJsonHandler()->GetRequestJsonHandler(td_initialize::request::kMethodInfo) != nullptr,
        "LanguageServer alias must keep standard protocol registrations");
}

void TestLanguageSessionShutdownSequence()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> shutdown_handled {false};
    std::atomic<bool> exit_notified {false};

    session.on(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    session.on(
        [&](td_shutdown::request const& req) -> lsp::ResponseOrError<td_shutdown::response>
        {
            shutdown_handled.store(true, std::memory_order_relaxed);
            td_shutdown::response rsp;
            rsp.id = req.id;
            lsp::Any result;
            result.SetJsonString("null", lsp::Any::kNullType);
            rsp.result = result;
            return rsp;
        });
    session.on(
        [&](Notify_Exit::notify const&)
        {
            exit_notified.store(true, std::memory_order_relaxed);
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})"));
    std::string const init_response = WaitForOutputContaining(output, "\"id\":1");

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2,"method":"shutdown","params":null})"));
    std::string const shutdown_response = WaitForOutputContaining(output, "\"id\":2");

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
    for (int i = 0; i < 100 && !exit_notified.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    session.stop();

    Expect(init_response.find("\"id\":1") != std::string::npos, "initialize must receive a response before shutdown");
    Expect(shutdown_handled.load(std::memory_order_relaxed), "LanguageSession must dispatch shutdown request handler");
    Expect(
        shutdown_response.find("\"id\":2") != std::string::npos,
        "LanguageSession shutdown response must preserve request id");
    Expect(exit_notified.load(std::memory_order_relaxed), "LanguageSession must dispatch exit after shutdown");
}

void TestLanguageSessionThrowsRequestError()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    session.on(
        [](td_initialize::request const&) -> td_initialize::response
        {
            throw lsp::RequestError(lsErrorCodes::ServerNotInitialized, "server not initialized");
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"session-error","method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"code\":-32002");
    session.stop();

    Expect(
        response.find(R"("id":"session-error")") != std::string::npos,
        "LanguageSession RequestError response must preserve request id");
    Expect(
        response.find("server not initialized") != std::string::npos,
        "LanguageSession RequestError response must serialize error message");
}

void TestLanguageSessionAsyncHandlerCompletesLater()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    session.on(
        [](td_initialize::request const& req) -> lsp::future<td_initialize::response>
        {
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            auto future = promise->get_future();
            std::thread(
                [promise, id = req.id]()
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    td_initialize::response rsp;
                    rsp.id = id;
                    rsp.result.capabilities.hoverProvider = true;
                    promise->set_value(std::move(rsp));
                }
            ).detach();
            return future;
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"session-async","method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"hoverProvider\":true");
    session.stop();

    Expect(
        response.find(R"("id":"session-async")") != std::string::npos,
        "LanguageSession async handler response must preserve request id");
    Expect(
        response.find("\"hoverProvider\":true") != std::string::npos,
        "LanguageSession async handler response must include handler result");
}

void TestLanguageSessionRunningRequestObservesCancellation()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log, lsp::JSONStreamStyle::Standard, 2);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> handler_entered {false};
    std::atomic<bool> handler_saw_cancel {false};

    session.on(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            handler_entered.store(true, std::memory_order_release);
            for (int i = 0; i < 100; ++i)
            {
                if (monitor && monitor())
                {
                    handler_saw_cancel.store(true, std::memory_order_relaxed);
                    Rsp_Error error;
                    error.id = req.id;
                    error.error.code = lsErrorCodes::RequestCancelled;
                    error.error.message = "cancelled";
                    return error;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":620,"method":"initialize","params":{}})"));
    for (int i = 0; i < 100 && !handler_entered.load(std::memory_order_acquire); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Expect(handler_entered.load(std::memory_order_acquire), "LanguageSession request handler must start before cancel");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":620}})"));
    std::string const response = WaitForOutputContaining(output, "\"code\":-32800");
    session.stop();

    Expect(handler_saw_cancel.load(std::memory_order_relaxed), "LanguageSession handler must observe cancellation");
    Expect(response.find("\"code\":-32800") != std::string::npos, "LanguageSession cancelled request must report -32800");
}

void TestLanguageSessionDelimitedRoundTrip()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log, lsp::JSONStreamStyle::Delimited);
    auto input = std::make_shared<StringIStream>(
        "{\"jsonrpc\":\"2.0\",\"id\":\"delimited\",\"method\":\"initialize\",\"params\":{}}\n"
        "// -----\n");
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> handled {false};

    session.on(
        [&](td_initialize::request const& req) -> td_initialize::response
        {
            handled.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities.hoverProvider = true;
            return rsp;
        });
    session.start(input, output);

    std::string const response = WaitForOutputContaining(output, R"("id":"delimited")");
    session.stop();

    Expect(handled.load(std::memory_order_relaxed), "LanguageSession delimited stream must dispatch requests");
    Expect(response.find(R"("id":"delimited")") != std::string::npos, "Delimited LanguageSession response must preserve id");
    Expect(
        response.find("\"hoverProvider\":true") != std::string::npos,
        "Delimited LanguageSession response must include handler result");
}

void TestLanguageSessionClientRequestRoundTrip()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    session.start(input, output);

    td_initialize::request req = session.endpoint().createRequest<td_initialize::request>();
    auto response_future = session.endpoint().send(req);
    std::string const request_json = WaitForOutputContaining(output, "\"method\":\"initialize\"");
    Expect(
        request_json.find("\"method\":\"initialize\"") != std::string::npos,
        "LanguageSession endpoint must write outbound client-role requests");

    input->append(test::MakeLspFrame(
        std::string(R"({"jsonrpc":"2.0","id":)") + std::to_string(req.id.value) +
        R"(,"result":{"capabilities":{"hoverProvider":true}}})"));
    auto const ready = response_future.wait_for(std::chrono::milliseconds(1000));
    session.stop();

    Expect(ready == lsp::future_status::ready, "LanguageSession outbound request future must resolve");
    if (ready == lsp::future_status::ready)
    {
        auto response = response_future.get();
        Expect(!response.IsError(), "LanguageSession outbound request must receive a success response");
        if (!response.IsError())
        {
            Expect(
                response.response.result.capabilities.hoverProvider &&
                    *response.response.result.capabilities.hoverProvider,
                "LanguageSession outbound response must parse initialize capabilities");
        }
    }
}

void TestLanguageSessionExitAfterShutdownOnly()
{
    // Adapted from clangd/test/exit-with-shutdown.test
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> exited {false};

    session.on([](td_initialize::request const& req) -> td_initialize::response
               {
                   td_initialize::response rsp;
                   rsp.id = req.id;
                   return rsp;
               });
    session.on([](td_shutdown::request const& req) -> td_shutdown::response
               {
                   td_shutdown::response rsp;
                   rsp.id = req.id;
                   lsp::Any result;
                   result.SetJsonString("null", lsp::Any::kNullType);
                   rsp.result = result;
                   return rsp;
               });
    session.on(
        [&](Notify_Exit::notify const&)
        {
            exited.store(true, std::memory_order_relaxed);
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}})"));
    Expect(WaitForOutputContaining(output, "\"id\":0").find("\"id\":0") != std::string::npos, "shutdown sequence must initialize");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":3,"method":"shutdown","params":null})"));
    Expect(WaitForOutputContaining(output, "\"id\":3").find("\"id\":3") != std::string::npos, "shutdown sequence must shutdown");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
    Expect(WaitUntil([&] { return exited.load(std::memory_order_relaxed); }), "exit notification must run after shutdown");
    session.stop();
}

// Adapted from clangd/test/exit-eof.test — records current behavior when stdin closes with no frames.
void TestLanguageSessionEofWithoutMessages()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<StringIStream>("");
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> exited {false};

    session.on(
        [&](Notify_Exit::notify const&)
        {
            exited.store(true, std::memory_order_relaxed);
        });
    session.start(input, output);
    session.stop();

    Expect(!exited.load(std::memory_order_relaxed), "EOF without LSP frames must not dispatch exit notification");
    Expect(output->snapshot().empty(), "EOF without LSP frames must not write responses");
}

// Adapted from clangd/test/exit-without-shutdown.test — records current LspCpp behavior.
void TestLanguageSessionExitWithoutPriorShutdown()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> exited {false};

    session.on([](td_initialize::request const& req) -> td_initialize::response
               {
                   td_initialize::response rsp;
                   rsp.id = req.id;
                   return rsp;
               });
    session.on(
        [&](Notify_Exit::notify const&)
        {
            exited.store(true, std::memory_order_relaxed);
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}})"));
    Expect(WaitForOutputContaining(output, "\"id\":0").find("\"id\":0") != std::string::npos, "initialize must succeed");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
    Expect(WaitUntil([&] { return exited.load(std::memory_order_relaxed); }), "exit without shutdown must still dispatch exit");
    session.stop();
}

void TestLanguageSessionCanEnableJdtlsExtensions()
{
    lsp::LanguageSessionOptions options;
    options.protocol.enableJdtlsExtensions = true;
    lsp::LanguageSession session(options);

    Expect(
        session.protocolJsonHandler()->GetResponseJsonHandler("java/classFileContents") != nullptr,
        "LanguageSessionOptions must allow opt-in JDTLS response parsers");
    Expect(
        session.protocolJsonHandler()->GetNotificationJsonHandler("java/projectConfigurationUpdate") != nullptr,
        "LanguageSessionOptions must allow opt-in JDTLS notification parsers");
}

void TestBindLspHandlerRegistersTypedHandler()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    bool bound = lsp::BindLspHandler(
        session.endpoint(),
        [](td_initialize::request const& req) -> td_initialize::response
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    Expect(bound, "BindLspHandler must register before startProcessingMessages");
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestLanguageSessionInitializeRoundTrip);
    RUN_TEST(TestLanguageSessionNotificationHandler);
    RUN_TEST(TestLanguageSessionEndpointCanSendNotifications);
    RUN_TEST(TestLanguageSessionCanOverrideBuiltinRequestParser);
    RUN_TEST(TestLanguageSessionOverrideRequestParserTemplate);
    RUN_TEST(TestLanguageServerAliasConstructsUsableSession);
    RUN_TEST(TestLanguageSessionShutdownSequence);
    RUN_TEST(TestLanguageSessionExitAfterShutdownOnly);
    RUN_TEST(TestLanguageSessionEofWithoutMessages);
    RUN_TEST(TestLanguageSessionExitWithoutPriorShutdown);
    RUN_TEST(TestLanguageSessionThrowsRequestError);
    RUN_TEST(TestLanguageSessionAsyncHandlerCompletesLater);
    RUN_TEST(TestLanguageSessionRunningRequestObservesCancellation);
    RUN_TEST(TestLanguageSessionDelimitedRoundTrip);
    RUN_TEST(TestLanguageSessionClientRequestRoundTrip);
    RUN_TEST(TestLanguageSessionCanEnableJdtlsExtensions);
    RUN_TEST(TestBindLspHandlerRegistersTypedHandler);
    return test::Failures() == 0 ? 0 : 1;
}
