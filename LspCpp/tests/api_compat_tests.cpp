#include "LibLsp/LspCpp.h"
#include "LibLsp/JsonRpc/Error.h"
#include "LibLsp/JsonRpc/RequestCancellation.h"
#include "LibLsp/JsonRpc/Transport.h"
#include "LibLsp/lsp/BindLspHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/path_mapping.h"
#include "LibLsp/lsp/AbsolutePath.h"
#include "LibLsp/lsp/request_context.h"
#include "LibLsp/lsp/uri.h"
#include "LibLsp/lsp/working_files.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

namespace
{
using test::Expect;
using test::FeedableIStream;
using test::StringOStream;
using test::WaitForOutputContaining;

void TestLegacyPublicApiStillConstructs()
{
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    lsp::NullLog log;
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint remote(json_handler, endpoint, log);
    WorkingFiles files;
    lsp::LanguageSession session;
    lsDocumentUri uri = lsDocumentUri::FromPath(AbsolutePath::FromNormalized("/tmp/example.cpp"));

    Expect(json_handler->GetRequestJsonHandler("initialize") != nullptr, "default ProtocolJsonHandler must stay registered");
    Expect(files.GetFileByFilename(uri.GetAbsolutePath()) == nullptr, "WorkingFiles default constructor must work");
    Expect(session.protocolJsonHandler() != nullptr, "LanguageSession default constructor must work");
    Expect(remote.getNextRequestId() >= 0, "RemoteEndPoint legacy constructor must work");
}

void TestRequestCancellationHelpersObserveCancel()
{
    lsRequestId id;
    id.set(42);

    auto task = lsp::cancelableTask(id, static_cast<int>(lsErrorCodes::RequestCancelled));
    lsp::WithContext scope(std::move(task.first));

    auto monitor = lsp::getCancelledMonitor(id);
    Expect(monitor.has_value(), "cancelable task must expose a monitor");
    Expect(!lsp::isCancellationRequested(*monitor), "task must start uncancelled");

    task.second();
    Expect(lsp::isCancellationRequested(*monitor), "canceler must mark task cancelled");
    Expect(
        lsp::cancellationReason(*monitor) == static_cast<int>(lsErrorCodes::RequestCancelled),
        "cancel reason must match RequestCancelled");
}

void TestGetCancelledMonitorAbsentOutsideContext()
{
    lsRequestId id;
    id.set(99);
    Expect(!lsp::getCancelledMonitor(id).has_value(), "ambient monitor lookup must be empty outside cancelable scope");
}

void TestNestedCancellationContextsOnlyCancelMatchingId()
{
    lsRequestId id_a;
    lsRequestId id_b;
    id_a.set(1);
    id_b.set(2);

    auto task_a = lsp::cancelableTask(id_a);
    lsp::WithContext outer(std::move(task_a.first));
    {
        auto task_b = lsp::cancelableTask(id_b);
        lsp::WithContext inner(std::move(task_b.first));
        task_b.second();

        auto monitor_a = lsp::getCancelledMonitor(id_a);
        auto monitor_b = lsp::getCancelledMonitor(id_b);
        Expect(monitor_a.has_value() && monitor_b.has_value(), "nested cancelable scopes must expose monitors");
        Expect(!lsp::isCancellationRequested(*monitor_a), "outer task must remain active when inner task is cancelled");
        Expect(lsp::isCancellationRequested(*monitor_b), "inner task must observe cancellation");
    }
}

void TestCancelableTaskContentModifiedReason()
{
    lsRequestId id;
    id.set(55);
    auto task = lsp::cancelableTask(id, static_cast<int>(lsErrorCodes::ContentModified));
    lsp::WithContext scope(std::move(task.first));
    auto monitor = lsp::getCancelledMonitor(id);
    Expect(monitor.has_value(), "content-modified task must expose a monitor");
    task.second();
    Expect(
        lsp::cancellationReason(*monitor) == static_cast<int>(lsErrorCodes::ContentModified),
        "custom cancel reason must be preserved");
}

void TestBindLspHandlerUsesAmbientCancellationMonitor()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    RemoteEndPoint remote(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    Expect(
        lsp::BindLspHandler(
            remote,
            [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
            {
                for (int i = 0; i < 200; ++i)
                {
                    if (auto monitor = lsp::getCancelledMonitor(req.id))
                    {
                        if (lsp::isCancellationRequested(*monitor))
                        {
                            throw lsp::makeRequestCancelledError();
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                td_initialize::response rsp;
                rsp.id = req.id;
                return rsp;
            }),
        "BindLspHandler must register ambient cancellation polling");

    remote.startProcessingMessages(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":901,"method":"initialize","params":{}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":901}})"));

    std::string const response = WaitForOutputContaining(output, "\"code\":-32800");
    remote.stop();

    Expect(response.find("\"code\":-32800") != std::string::npos, "ambient cancellation must emit RequestCancelled");
    auto const bodies = test::ExtractLspFrameBodies(response);
    Expect(!bodies.empty(), "ambient cancellation must produce a response frame");
    if (!bodies.empty())
    {
        test::ExpectJsonFixture(
            bodies.back(),
            "request_cancelled_error_response.json",
            "ambient cancellation response must match fixture");
    }
}

void TestBindLspHandlerWithMonitorUsesPublicCancellationHelpers()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    RemoteEndPoint remote(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    Expect(
        lsp::BindLspHandlerWithMonitor(
            remote,
            [&](td_initialize::request const&, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
            {
                for (int i = 0; i < 200; ++i)
                {
                    if (lsp::isCancellationRequested(monitor))
                    {
                        throw lsp::makeRequestCancelledError();
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                td_initialize::response rsp;
                return rsp;
            }),
        "BindLspHandlerWithMonitor must register");

    remote.startProcessingMessages(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":903,"method":"initialize","params":{}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":903}})"));

    std::string const response = WaitForOutputContaining(output, "\"code\":-32800");
    remote.stop();
    Expect(response.find("\"code\":-32800") != std::string::npos, "monitor-based handler must emit RequestCancelled");
}

void TestErrorHelpersBuildExpectedCodes()
{
    auto cancelled = lsp::makeRequestCancelledError("cancelled by test");
    auto modified = lsp::makeContentModifiedError("modified by test");
    lsRequestId id;
    id.set(7);

    Expect(cancelled.code() == lsErrorCodes::RequestCancelled, "RequestCancelled helper must set code");
    Expect(modified.code() == lsErrorCodes::ContentModified, "ContentModified helper must set code");
    Expect(lsp::isRequestCancelledCode(lsErrorCodes::RequestCancelled), "isRequestCancelledCode must match");
    Expect(lsp::isContentModifiedCode(lsErrorCodes::ContentModified), "isContentModifiedCode must match");

    auto rsp = lsp::makeRequestCancelledResponse(id, "cancelled by test");
    Expect(rsp.error.code == lsErrorCodes::RequestCancelled, "response helper must preserve code");
    Expect(rsp.id == id, "response helper must preserve id");
}

void TestErrorHelpersSerializeContentModifiedResponse()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    RemoteEndPoint remote(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    remote.registerHandler(
        [](td_initialize::request const&) -> lsp::ResponseOrError<td_initialize::response>
        {
            throw lsp::makeContentModifiedError();
        });
    remote.startProcessingMessages(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":902,"method":"initialize","params":{}})"));

    std::string const response = WaitForOutputContaining(output, "\"code\":-32801");
    remote.stop();

    auto const bodies = test::ExtractLspFrameBodies(response);
    Expect(!bodies.empty(), "ContentModified handler must produce a response frame");
    if (!bodies.empty())
    {
        test::ExpectJsonFixture(
            bodies.back(),
            "content_modified_error_response.json",
            "ContentModified response must match fixture");
    }
}

void TestTransportFacadeDelegatesToRemoteEndPoint()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint remote(json_handler, endpoint, log);
    lsp::Transport transport(remote);

    remote.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities.hoverProvider = true;
            return rsp;
        });

    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    transport.run(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":55,"method":"initialize","params":{}})"));

    std::string const response = WaitForOutputContaining(output, "\"hoverProvider\":true");
    transport.stop();
    Expect(response.find("\"hoverProvider\":true") != std::string::npos, "Transport must preserve responses");
}

void TestTransportLoopAliasMatchesRun()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint remote(json_handler, endpoint, log);
    lsp::Transport transport(remote);

    remote.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    transport.loop(input, output);
    Expect(transport.isWorking(), "Transport loop must start endpoint processing");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":56,"method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"id\":56");
    transport.stop();
    Expect(!transport.isWorking(), "Transport stop must end endpoint processing");
    Expect(response.find("\"id\":56") != std::string::npos, "Transport loop must dispatch requests");
}

void TestTransportNotifyReplyAndCall()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint remote(json_handler, endpoint, log);
    lsp::Transport transport(remote);

    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();
    transport.run(input, output);

    Notify_Exit::notify exit_notify;
    transport.notify(exit_notify);

    td_initialize::request client_req;
    client_req.id.set(710);
    auto future = transport.call(client_req);

    std::string const outbound = WaitForOutputContaining(output, "\"id\":710");
    Expect(
        outbound.find("\"method\":\"initialize\"") != std::string::npos,
        "Transport call must send server-initiated request");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":710,"result":{"capabilities":{"hoverProvider":true}}})"));

    Expect(future.wait_for(std::chrono::seconds(2)) == lsp::future_status::ready, "Transport call future must complete");
    auto const result = future.get();
    Expect(!result.is_error, "Transport call must receive success response");
    Expect(result.response.result.capabilities.hoverProvider == true, "Transport call must preserve response payload");

    td_initialize::request callback_req;
    callback_req.id.set(712);
    auto callback_result = std::make_shared<std::promise<bool>>();
    auto callback_done = callback_result->get_future();
    transport.call(
        callback_req,
        [callback_result](td_initialize::response const& response)
        {
            callback_result->set_value(response.result.capabilities.hoverProvider == true);
        },
        [callback_result](Rsp_Error const&)
        {
            callback_result->set_value(false);
        });
    std::string const callback_outbound = WaitForOutputContaining(output, "\"id\":712");
    Expect(
        callback_outbound.find("\"method\":\"initialize\"") != std::string::npos,
        "Transport callback call must send server-initiated request");
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":712,"result":{"capabilities":{"hoverProvider":true}}})"));
    Expect(
        callback_done.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
        "Transport callback call future must complete");
    if (callback_done.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
    {
        Expect(callback_done.get(), "Transport callback call must preserve response payload");
    }

    lsRequestId reply_id;
    reply_id.set(711);
    auto cancelled = lsp::makeRequestCancelledResponse(reply_id, "cancelled by transport test");
    transport.reply(cancelled);

    std::string const output_snapshot = WaitForOutputContaining(output, "\"code\":-32800");
    transport.stop();
    Expect(output_snapshot.find("\"method\":\"exit\"") != std::string::npos, "Transport notify must emit notification");
    Expect(output_snapshot.find("\"code\":-32800") != std::string::npos, "Transport reply must emit error response");
}

void TestRequestContextOffsetEncodingDiscriminatesMultibyteLine()
{
    std::string const content = "/*\xc3\xb6*/int x;\nint y=x;";
    lsPosition shared_pos;
    shared_pos.line = 0;
    shared_pos.character = 9;

    int const default_offset = lsp::GetOffsetForPositionInContext(shared_pos, content);
    Expect(
        default_offset == lsp::GetOffsetForPositionWithEncoding(shared_pos, content, lsp::OffsetEncoding::Utf16),
        "default request context must keep UTF-16 semantics");

    lsp::WithContext scope(lsp::withOffsetEncodingContext(lsp::OffsetEncoding::Utf8));
    int const utf8_offset = lsp::GetOffsetForPositionInContext(shared_pos, content);
    Expect(utf8_offset != default_offset, "UTF-8 context must change offset conversion on multibyte lines");
    Expect(
        utf8_offset == lsp::GetOffsetForPositionWithEncoding(shared_pos, content, lsp::OffsetEncoding::Utf8),
        "UTF-8 context must match explicit UTF-8 helper");
}

void TestRequestContextOffsetEncodingRoundTripAndRestoresDefault()
{
    std::string const content = "/*\xc3\xb6*/int x;\nint y=x;";
    lsPosition utf8_pos;
    utf8_pos.line = 0;
    utf8_pos.character = 10;

    {
        lsp::WithContext scope(lsp::withOffsetEncodingContext(lsp::OffsetEncoding::Utf8));
        Expect(lsp::currentOffsetEncoding() == lsp::OffsetEncoding::Utf8, "active context must expose UTF-8");
        int const offset = lsp::GetOffsetForPositionInContext(utf8_pos, content);
        Expect(offset == 10, "UTF-8 context must treat character as byte offset");
        lsPosition const round_trip = lsp::GetPositionForOffsetInContext(
            static_cast<size_t>(offset),
            content,
            lsp::Context::current());
        Expect(round_trip.line == utf8_pos.line, "UTF-8 round-trip must preserve line");
        Expect(round_trip.character == utf8_pos.character, "UTF-8 round-trip must preserve character");
        Expect(
            lsp::GetOffsetForPositionInContext(round_trip, content) == offset,
            "UTF-8 position/offset round-trip must be stable");
    }
    Expect(
        lsp::currentOffsetEncoding() == lsp::OffsetEncoding::Utf16,
        "offset encoding context must restore UTF-16 default after scope exit");
    lsPosition const default_position = lsp::GetPositionForOffsetInContext(10, content);
    Expect(default_position.line == 0, "default UTF-16 restoration must preserve line conversion");
    Expect(default_position.character != utf8_pos.character, "default UTF-16 restoration must change multibyte character conversion");
}

void TestRequestContextPathMappingIsOptIn()
{
    auto mapping = std::make_shared<lsp::PathMapping>();
    mapping->add("/client/project", "/server/project");

    lsp::WithContext scope(lsp::withPathMapping(mapping));
    auto active = lsp::currentPathMapping();
    Expect(active != nullptr, "path mapping context must be installed");
    Expect(
        active->client_to_server("/client/project/file.cpp") == "/server/project/file.cpp",
        "path mapping context must delegate to configured mapping");
}

void TestBindLspAsyncHandlerReturnsResponseOrError()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    RemoteEndPoint remote(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    Expect(
        lsp::BindLspAsyncHandler(
            remote,
            [](td_initialize::request const& req) -> lsp::future<lsp::ResponseOrError<td_initialize::response>>
            {
                auto promise = std::make_shared<lsp::promise<lsp::ResponseOrError<td_initialize::response>>>();
                auto future = promise->get_future();
                std::thread(
                    [promise, id = req.id]()
                    {
                        Rsp_Error error;
                        error.id = id;
                        error.error.code = lsErrorCodes::InvalidParams;
                        error.error.message = "async invalid params";
                        promise->set_value(lsp::ResponseOrError<td_initialize::response>(std::move(error)));
                    })
                    .detach();
                return future;
            }),
        "BindLspAsyncHandler must accept future<ResponseOrError>");

    remote.startProcessingMessages(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1503,"method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"code\":-32602");
    remote.stop();
    Expect(response.find("\"id\":1503") != std::string::npos, "async ResponseOrError binder must preserve request id");
    Expect(response.find("\"code\":-32602") != std::string::npos, "async ResponseOrError binder must serialize error code");
}

void TestBindLspAsyncHandlerWithMonitorCompletesLater()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    RemoteEndPoint remote(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    Expect(
        lsp::BindLspAsyncHandlerWithMonitor(
            remote,
            [](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::future<td_initialize::response>
            {
                auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
                auto future = promise->get_future();
                std::thread(
                    [promise, id = req.id, monitor]()
                    {
                        (void)monitor;
                        td_initialize::response rsp;
                        rsp.id = id;
                        promise->set_value(std::move(rsp));
                    })
                    .detach();
                return future;
            }),
        "BindLspAsyncHandlerWithMonitor must register");

    remote.startProcessingMessages(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1504,"method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"id\":1504");
    remote.stop();
    Expect(response.find("\"id\":1504") != std::string::npos, "async monitor binder must produce a response");
}

void TestBindLspAsyncHandlerWithMonitorObservesCancellation()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    RemoteEndPoint remote(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    Expect(
        lsp::BindLspAsyncHandlerWithMonitor(
            remote,
            [](td_initialize::request const& req, CancelMonitor const& monitor)
                -> lsp::future<lsp::ResponseOrError<td_initialize::response>>
            {
                auto promise = std::make_shared<lsp::promise<lsp::ResponseOrError<td_initialize::response>>>();
                auto future = promise->get_future();
                std::thread(
                    [promise, id = req.id, monitor]()
                    {
                        for (int i = 0; i < 200; ++i)
                        {
                            if (lsp::isCancellationRequested(monitor))
                            {
                                promise->set_value(lsp::ResponseOrError<td_initialize::response>(
                                    lsp::makeRequestCancelledResponse(id)
                                ));
                                return;
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }

                        td_initialize::response rsp;
                        rsp.id = id;
                        promise->set_value(lsp::ResponseOrError<td_initialize::response>(std::move(rsp)));
                    })
                    .detach();
                return future;
            }),
        "BindLspAsyncHandlerWithMonitor must accept future<ResponseOrError>");

    remote.startProcessingMessages(input, output);
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1505,"method":"initialize","params":{}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":1505}})"));

    std::string const response = WaitForOutputContaining(output, "\"code\":-32800");
    remote.stop();
    Expect(response.find("\"id\":1505") != std::string::npos, "async monitor cancellation must preserve request id");
    Expect(response.find("\"code\":-32800") != std::string::npos, "async monitor cancellation must emit RequestCancelled");
}

void TestLegacyBindLspHandlerStillRegisters()
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
    Expect(bound, "legacy BindLspHandler signature must remain available");
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
    RUN_TEST(TestLegacyPublicApiStillConstructs);
    RUN_TEST(TestRequestCancellationHelpersObserveCancel);
    RUN_TEST(TestGetCancelledMonitorAbsentOutsideContext);
    RUN_TEST(TestNestedCancellationContextsOnlyCancelMatchingId);
    RUN_TEST(TestCancelableTaskContentModifiedReason);
    RUN_TEST(TestBindLspHandlerUsesAmbientCancellationMonitor);
    RUN_TEST(TestBindLspHandlerWithMonitorUsesPublicCancellationHelpers);
    RUN_TEST(TestErrorHelpersBuildExpectedCodes);
    RUN_TEST(TestErrorHelpersSerializeContentModifiedResponse);
    RUN_TEST(TestTransportFacadeDelegatesToRemoteEndPoint);
    RUN_TEST(TestTransportLoopAliasMatchesRun);
    RUN_TEST(TestTransportNotifyReplyAndCall);
    RUN_TEST(TestRequestContextOffsetEncodingDiscriminatesMultibyteLine);
    RUN_TEST(TestRequestContextOffsetEncodingRoundTripAndRestoresDefault);
    RUN_TEST(TestRequestContextPathMappingIsOptIn);
    RUN_TEST(TestBindLspAsyncHandlerReturnsResponseOrError);
    RUN_TEST(TestBindLspAsyncHandlerWithMonitorCompletesLater);
    RUN_TEST(TestBindLspAsyncHandlerWithMonitorObservesCancellation);
    RUN_TEST(TestLegacyBindLspHandlerStillRegisters);
    return test::Failures() == 0 ? 0 : 1;
}
