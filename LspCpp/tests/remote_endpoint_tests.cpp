#include "LibLsp/JsonRpc/MessageJsonHandler.h"
#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/progress.h"
#include "LibLsp/lsp/textDocument/completion.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include "LibLsp/lsp/workspace/applyEdit.h"
#include "LibLsp/JsonRpc/json.h"
#include "lsp_client_test_helpers.h"
#include "protocol_test_helpers.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

struct ScalarParams
{
    int value = 0;
    MAKE_SWAP_METHOD(ScalarParams, value);
};
MAKE_REFLECT_STRUCT(ScalarParams, value);

struct ScalarResult
{
    int value = 0;
    MAKE_SWAP_METHOD(ScalarResult, value);
};
MAKE_REFLECT_STRUCT(ScalarResult, value);
DEFINE_REQUEST_RESPONSE_TYPE(test_scalar, ScalarParams, ScalarResult, "test/scalar");

struct OrderedNotificationParams
{
    int value = 0;
    MAKE_SWAP_METHOD(OrderedNotificationParams, value);
};
MAKE_REFLECT_STRUCT(OrderedNotificationParams, value);
DEFINE_NOTIFICATION_TYPE(test_ordered_notification, OrderedNotificationParams, "test/orderedNotification");

namespace
{
using test::DummyLog;
using test::Expect;
using test::FeedableIStream;
using test::StringIStream;
using test::StringOStream;

static_assert(
    lsp::detail::is_valid_async_request_return<lsp::future<td_initialize::response>, td_initialize::response>::value,
    "request handlers must accept future<Response>");
static_assert(
    lsp::detail::is_valid_async_request_return<
        lsp::future<lsp::ResponseOrError<td_initialize::response>>,
        td_initialize::response>::value,
    "request handlers must accept future<ResponseOrError<Response>>");
static_assert(
    !lsp::detail::is_valid_async_request_return<lsp::future<int>, td_initialize::response>::value,
    "request handlers must reject futures with unrelated value types");

struct MoveOnlyFutureValue
{
    MoveOnlyFutureValue() = default;
    explicit MoveOnlyFutureValue(int value) : value(value)
    {
    }
    MoveOnlyFutureValue(MoveOnlyFutureValue const&) = delete;
    MoveOnlyFutureValue& operator=(MoveOnlyFutureValue const&) = delete;
    MoveOnlyFutureValue(MoveOnlyFutureValue&& other) noexcept : value(other.value)
    {
        other.value = 0;
    }
    MoveOnlyFutureValue& operator=(MoveOnlyFutureValue&& other) noexcept
    {
        value = other.value;
        other.value = 0;
        return *this;
    }

    int value = 0;
};

std::string WaitForOutputContainingAll(std::shared_ptr<StringOStream> const& output_stream, std::vector<std::string> const& needles)
{
    std::string output;
    for (int i = 0; i < 100; ++i)
    {
        output = output_stream->snapshot();
        bool found_all = true;
        for (auto const& needle : needles)
        {
            if (output.find(needle) == std::string::npos)
            {
                found_all = false;
                break;
            }
        }
        if (found_all)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return output;
}

std::string MakeDelimitedInput(std::string const& body)
{
    return body + "\n// -----\n";
}

std::string MakeOrderedNotificationBody(int value, size_t payload_size = 0)
{
    std::string body = std::string(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":)") +
        std::to_string(value);
    if (payload_size != 0)
    {
        body += R"(,"payload":")";
        body.append(payload_size, 'x');
        body += "\"";
    }
    body += "}}";
    return body;
}

std::vector<std::string> ExtractFrameBodies(std::string const& output)
{
    std::vector<std::string> bodies;
    size_t pos = 0;
    while (pos < output.size())
    {
        size_t const header_end = output.find("\r\n\r\n", pos);
        if (header_end == std::string::npos)
        {
            break;
        }
        size_t const length_start = output.find("Content-Length:", pos);
        if (length_start == std::string::npos || length_start > header_end)
        {
            break;
        }
        size_t const value_start = length_start + std::string("Content-Length:").size();
        int const length = std::atoi(output.substr(value_start, header_end - value_start).c_str());
        size_t const body_start = header_end + 4;
        if (length < 0 || body_start + static_cast<size_t>(length) > output.size())
        {
            break;
        }
        bodies.push_back(output.substr(body_start, static_cast<size_t>(length)));
        pos = body_start + static_cast<size_t>(length);
    }
    return bodies;
}

bool WaitUntil(std::function<bool()> const& predicate, int attempts = 100, int delay_ms = 10)
{
    for (int i = 0; i < attempts; ++i)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return predicate();
}

std::string ProcessSingleInput(std::string const& input, std::string const& wait_for = "")
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    json_handler->SetRequestJsonHandler(
        td_initialize::request::kMethodInfo,
        [](Reader& visitor)
        {
            return td_initialize::request::ReflectReader(visitor);
        });
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<StringIStream>(input);
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    std::string output;
    if (!wait_for.empty())
    {
        output = WaitForOutputContaining(output_stream, wait_for);
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        output = output_stream->snapshot();
    }
    point.stop();
    return output;
}

void TestDuplicateRequestIdReturnsError()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    StringOStream output;
    StringIStream input("");
    auto input_stream = std::make_shared<StringIStream>("");
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request first;
    first.id.set(7);
    auto first_future = point.send(first);

    td_initialize::request duplicate;
    duplicate.id.set(7);
    auto duplicate_future = point.send(duplicate);
    auto duplicate_result = duplicate_future.get();

    Expect(duplicate_result.is_error, "duplicate request id must return an error response");
    Expect(
        duplicate_result.error.error.message == "Failed to send request.",
        "duplicate request id must report failed send");

    point.stop();
}

void TestWaitResponseTimeoutClearsPendingRequest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<StringIStream>("");
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request timed_out;
    timed_out.id.set(99);
    auto const response = point.waitResponse(timed_out, 200);
    Expect(response == nullptr, "waitResponse must return null on timeout");

    td_initialize::request retry;
    retry.id.set(99);
    auto retry_future = point.send(retry);
    auto const status = retry_future.wait_for(std::chrono::milliseconds(50));
    Expect(
        status == lsp::future_status::timeout,
        "request id must be reusable after timeout cleanup");

    point.stop();
}

void TestBadOutputStreamReturnsError()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto output_stream = std::make_shared<StringOStream>();
    output_stream->set_bad(true);

    RemoteEndPoint point(json_handler, endpoint, log);

    td_initialize::request req;
    req.id.set(3);
    auto future = point.send(req);
    auto result = future.get();

    Expect(result.is_error, "bad output stream must return an error response");
    Expect(
        result.error.error.message == "Failed to send request.",
        "bad output stream must report failed send");
}

void TestStopCompletesPendingRequestWithError()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<StringIStream>("");
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    req.id.set(123);
    auto future = point.send(req);

    point.stop();

    auto const status = future.wait_for(std::chrono::milliseconds(50));
    Expect(status == lsp::future_status::ready, "stop must complete pending request futures");
    if (status != lsp::future_status::ready)
    {
        return;
    }

    auto result = future.get();
    Expect(result.is_error, "stopped pending request must return an error response");
    Expect(
        result.error.error.message == "Remote endpoint stopped.",
        "stopped pending request must report endpoint stop");
}

void TestStopCancelsRunningRequestHandlers()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> handler_entered {false};
    std::atomic<bool> handler_saw_cancel {false};
    std::atomic<bool> handler_exited {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            handler_entered.store(true, std::memory_order_relaxed);
            for (int i = 0; i < 200; ++i)
            {
                if (monitor && monitor())
                {
                    handler_saw_cancel.store(true, std::memory_order_relaxed);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            handler_exited.store(true, std::memory_order_relaxed);

            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1701,"method":"initialize","params":{}})"));
    Expect(
        WaitUntil([&]() { return handler_entered.load(std::memory_order_relaxed); }),
        "request handler must start before stop is called");

    point.stop();

    Expect(
        WaitUntil([&]() { return handler_exited.load(std::memory_order_relaxed); }),
        "stop must let a running handler observe cancellation and exit");
    Expect(
        handler_saw_cancel.load(std::memory_order_relaxed),
        "stop must cancel running request handlers through their CancelMonitor");
}

void TestStopFromNotificationHandlerDoesNotDeadlock()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> handler_entered {false};
    std::atomic<bool> stop_returned {false};

    auto point = std::make_unique<RemoteEndPoint>(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    RemoteEndPoint* point_ptr = point.get();
    point->registerHandler(
        [&](Notify_Exit::notify const&)
        {
            handler_entered.store(true, std::memory_order_relaxed);
            point_ptr->stop();
            stop_returned.store(true, std::memory_order_relaxed);
        });
    point->startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));

    bool const stopped = WaitUntil([&]() { return stop_returned.load(std::memory_order_relaxed); }, 200, 10);
    Expect(handler_entered.load(std::memory_order_relaxed), "exit notification handler must run");
    Expect(stopped, "stop called from a notification handler must not deadlock");
    if (stopped)
    {
        point.reset();
    }
    else
    {
        // Avoid calling the destructor if the failure mode is a stuck self-stop.
        point.release();
    }
}

void TestAsyncFutureCompletedAfterStopIsDroppedSafely()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex promise_mutex;
    std::shared_ptr<lsp::promise<td_initialize::response>> pending_promise;
    std::atomic<bool> handler_started {false};

    auto point = new RemoteEndPoint(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point->registerHandler(
        [&](td_initialize::request const&) -> lsp::future<td_initialize::response>
        {
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            auto future = promise->get_future();
            {
                std::lock_guard<std::mutex> lock(promise_mutex);
                pending_promise = promise;
            }
            handler_started.store(true, std::memory_order_relaxed);
            return future;
        });
    point->startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1703,"method":"initialize","params":{}})"));
    Expect(
        WaitUntil([&]() { return handler_started.load(std::memory_order_relaxed); }),
        "async request handler must start before stop");

    std::shared_ptr<lsp::promise<td_initialize::response>> promise;
    {
        std::lock_guard<std::mutex> lock(promise_mutex);
        promise = pending_promise;
    }
    Expect(promise != nullptr, "async request handler must expose its pending promise");

    point->stop();
    std::string const output_before_completion = output_stream->snapshot();
    delete point;

    if (promise)
    {
        td_initialize::response rsp;
        rsp.id.set(1703);
        promise->set_value(std::move(rsp));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    Expect(
        output_stream->snapshot() == output_before_completion,
        "async future completed after stop/destruction must not write a late response");
}

void TestUnregisteredRequestReturnsMethodNotFound()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    json_handler->SetRequestJsonHandler(
        td_initialize::request::kMethodInfo,
        [](Reader& visitor)
        {
            return td_initialize::request::ReflectReader(visitor);
        });
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":77,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32601");
    point.stop();

    Expect(
        output.find("\"id\":77") != std::string::npos,
        "MethodNotFound response must keep the original request id");
    Expect(
        output.find("\"code\":-32601") != std::string::npos,
        "unregistered request must return MethodNotFound");
    Expect(
        output.find("Method not found: initialize") != std::string::npos,
        "MethodNotFound response must include the method name");
}

void TestUnregisteredRequestPreservesStringIdInMethodNotFound()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    json_handler->SetRequestJsonHandler(
        td_initialize::request::kMethodInfo,
        [](Reader& visitor)
        {
            return td_initialize::request::ReflectReader(visitor);
        });
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":"abc-77","method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32601");
    point.stop();

    Expect(
        output.find(R"("id":"abc-77")") != std::string::npos,
        "MethodNotFound response must preserve string request ids");
    Expect(
        output.find("\"code\":-32601") != std::string::npos,
        "string-id unregistered request must return MethodNotFound");
}

void TestRegisteredRequestDoesNotReturnMethodNotFound()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":88,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":88");
    point.stop();

    Expect(output.find("\"id\":88") != std::string::npos, "registered request must produce a response");
    Expect(
        output.find("\"code\":-32601") == std::string::npos,
        "registered request must not produce MethodNotFound");
}

void TestMalformedJsonDoesNotCrash()
{
    auto const output = ProcessSingleInput("Content-Length: 5\r\n\r\n{bad}");

    Expect(output.empty(), "malformed JSON must not produce a response");
}

void TestWrongJsonRpcVersionIsRejected()
{
    std::string const request = R"({"jsonrpc":"1.0","id":1,"method":"initialize","params":{}})";
    auto const output = ProcessSingleInput(test::MakeLspFrame(request));

    Expect(output.empty(), "wrong jsonrpc version must not produce a response");
}

void TestNonStringJsonRpcVersionIsRejected()
{
    std::string const request = R"({"jsonrpc":2.0,"id":1,"method":"initialize","params":{}})";
    auto const output = ProcessSingleInput(test::MakeLspFrame(request));

    Expect(output.empty(), "non-string jsonrpc version must be rejected without producing a response");
}

void TestNonStringMethodIsRejected()
{
    std::string const request = R"({"jsonrpc":"2.0","id":1,"method":123,"params":{}})";
    auto const request_output = ProcessSingleInput(test::MakeLspFrame(request));
    std::string const notification = R"({"jsonrpc":"2.0","method":123,"params":{}})";
    auto const notification_output = ProcessSingleInput(test::MakeLspFrame(notification));

    Expect(request_output.empty(), "request with non-string method must be rejected without producing a response");
    Expect(
        notification_output.empty(),
        "notification with non-string method must be rejected without producing a response");
}

void TestRequestMissingParamsDoesNotCrashEndpoint()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled_count {0};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handled_count.fetch_add(1, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1901,"method":"initialize"})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1902,"method":"initialize","params":{}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1902");
    point.stop();

    Expect(output.find("\"id\":1902") != std::string::npos, "endpoint must still handle requests after missing params");
    Expect(
        handled_count.load(std::memory_order_relaxed) >= 1,
        "endpoint must stay alive after a request that omits params");
}

void TestNullAndZeroParamsDoNotDropValidScalarValues()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled_count {0};
    std::atomic<int> last_value {-1};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            handled_count.fetch_add(1, std::memory_order_relaxed);
            last_value.store(req.params.value, std::memory_order_relaxed);
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1910,"method":"test/scalar","params":null})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1911,"method":"test/scalar","params":{"value":0}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1911");
    point.stop();

    Expect(output.find("\"id\":1911") != std::string::npos, "endpoint must recover after null params");
    Expect(output.find("\"value\":0") != std::string::npos, "scalar zero params must round-trip");
    Expect(handled_count.load(std::memory_order_relaxed) >= 1, "valid scalar request must be handled");
    Expect(last_value.load(std::memory_order_relaxed) == 0, "zero must not be treated as a missing parameter");
}

void TestWorkDoneTokenParamsPassThroughToHandler()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    optional<lsProgressToken> seen_token;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                seen_token = req.params.workDoneToken;
            }
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","id":1930,"method":"initialize","params":{"capabilities":{},"workDoneToken":"token"}})"));
    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1930");
    point.stop();

    Expect(output.find("\"id\":1930") != std::string::npos, "initialize with workDoneToken must get a response");
    std::lock_guard<std::mutex> lock(mutex);
    Expect(
        seen_token && seen_token->first && *seen_token->first == "token",
        "workDoneToken must pass through to the request handler unchanged");
}

void TestPositionalArrayParamsAreHandledWithoutStoppingEndpoint()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled_count {0};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            handled_count.fetch_add(1, std::memory_order_relaxed);
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1920,"method":"test/scalar","params":[0]})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1921,"method":"test/scalar","params":{"value":7}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1921");
    point.stop();

    Expect(output.find("\"id\":1920") != std::string::npos, "single positional param must produce a response");
    Expect(output.find("\"id\":1921") != std::string::npos, "endpoint must recover after positional params");
    Expect(output.find("\"value\":7") != std::string::npos, "named params after positional params must still work");
    Expect(handled_count.load(std::memory_order_relaxed) == 2, "single positional params must dispatch predictably");
}

void TestResponseWithNoMatchingRequestIsIgnored()
{
    std::string const response = R"({"jsonrpc":"2.0","id":999,"result":{}})";
    auto const output = ProcessSingleInput(test::MakeLspFrame(response));

    Expect(output.empty(), "unknown response id must not produce output");
}

void TestMessageWithIdAndMethodUsesRequestPath()
{
    std::string const request = R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})";
    auto const output = ProcessSingleInput(test::MakeLspFrame(request), "\"code\":-32601");

    Expect(
        output.find("\"code\":-32601") != std::string::npos,
        "message with id and method must be treated as a request");
}

void TestMessageWithNoMethodNoResultIsRejected()
{
    std::string const message = R"({"jsonrpc":"2.0","id":1})";
    auto const output = ProcessSingleInput(test::MakeLspFrame(message));

    Expect(output.empty(), "message without method/result/error must be rejected");
}

void TestSendRequestAndReceiveSuccessResponse()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    auto future = point.send(req);
    WaitForOutputContaining(output_stream, "\"method\":\"initialize\"");

    std::string const response =
        std::string(R"({"jsonrpc":"2.0","id":)") + ToString(req.id) + R"(,"result":{"capabilities":{}}})";
    input_stream->append(test::MakeLspFrame(response));

    auto const status = future.wait_for(std::chrono::milliseconds(1000));
    Expect(status == lsp::future_status::ready, "success response must complete the request future");
    if (status == lsp::future_status::ready)
    {
        auto result = future.get();
        Expect(!result.is_error, "success response must produce a non-error result");
    }
    point.stop();
}

void TestSendRequestAndReceiveErrorResponse()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    auto future = point.send(req);
    WaitForOutputContaining(output_stream, "\"method\":\"initialize\"");

    std::string const response = std::string(R"({"jsonrpc":"2.0","id":)") + ToString(req.id) +
                                 R"(,"error":{"code":-32600,"message":"test error"}})";
    input_stream->append(test::MakeLspFrame(response));

    auto const status = future.wait_for(std::chrono::milliseconds(1000));
    Expect(status == lsp::future_status::ready, "error response must complete the request future");
    if (status == lsp::future_status::ready)
    {
        auto result = future.get();
        Expect(result.is_error, "error response must produce an error result");
        Expect(result.error.error.code == lsErrorCodes::InvalidRequest, "error code must round-trip");
        Expect(result.error.error.message == "test error", "error message must round-trip");
    }
    point.stop();
}

void TestDuplicateResponseCompletesFutureOnce()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    std::atomic<int> success_calls {0};
    std::atomic<int> error_calls {0};
    td_initialize::request req;
    req.id.set(1801);
    point.send(
        req,
        [&](td_initialize::response const& rsp)
        {
            Expect(rsp.id == req.id, "duplicate response test must receive the matching response id");
            success_calls.fetch_add(1, std::memory_order_relaxed);
        },
        [&](Rsp_Error const&)
        {
            error_calls.fetch_add(1, std::memory_order_relaxed);
        });
    WaitForOutputContaining(output_stream, "\"id\":1801");

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1801,"result":{"capabilities":{}}})"));
    Expect(
        WaitUntil([&]() { return success_calls.load(std::memory_order_relaxed) == 1; }),
        "first response must complete the request callback");
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1801,"result":{"capabilities":{"hoverProvider":true}}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    point.stop();

    Expect(success_calls.load(std::memory_order_relaxed) == 1, "duplicate response must not complete a request twice");
    Expect(error_calls.load(std::memory_order_relaxed) == 0, "duplicate response must not report an error");
}

void TestSendRequestAndReceiveSuccessResponseWithStringId()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    req.id.set("client-alpha");
    auto future = point.send(req);
    WaitForOutputContaining(output_stream, R"("id":"client-alpha")");

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"client-alpha","result":{"capabilities":{}}})"));

    auto const status = future.wait_for(std::chrono::milliseconds(1000));
    Expect(status == lsp::future_status::ready, "string-id success response must complete the request future");
    if (status == lsp::future_status::ready)
    {
        auto result = future.get();
        Expect(!result.is_error, "string-id success response must produce a non-error result");
        Expect(result.response.id == req.id, "string-id response must preserve the original request id");
    }
    point.stop();
}

void TestLateResponseAfterTimeoutIsIgnored()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request timed_out;
    timed_out.id.set(404);
    auto const timed_out_response = point.waitResponse(timed_out, 20);
    Expect(timed_out_response == nullptr, "waitResponse must time out before the late response arrives");

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":404,"result":{"capabilities":{}}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    td_initialize::request retry;
    retry.id.set(404);
    auto retry_future = point.send(retry);
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":404,"result":{"capabilities":{"hoverProvider":true}}})"));

    auto const status = retry_future.wait_for(std::chrono::milliseconds(1000));
    Expect(status == lsp::future_status::ready, "request id must remain reusable after a late timed-out response");
    if (status == lsp::future_status::ready)
    {
        auto result = retry_future.get();
        Expect(!result.is_error, "retry after late response must receive the new success response");
        Expect(
            result.response.result.capabilities.hoverProvider &&
                *result.response.result.capabilities.hoverProvider == true,
            "retry response must be the second response, not the ignored late response");
    }
    point.stop();
}

void TestParsedLateResponseDoesNotCompleteReusedRequestId()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool parser_entered = false;
    bool release_first_parse = false;
    bool block_first_parse = true;

    json_handler->SetResponseJsonHandler(
        td_initialize::request::kMethodInfo,
        [&](Reader& visitor)
        {
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (block_first_parse)
                {
                    parser_entered = true;
                    cv.notify_all();
                    cv.wait(lock, [&] { return release_first_parse; });
                    block_first_parse = false;
                }
            }
            return td_initialize::response::ReflectReader(visitor);
        });

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 2);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request timed_out;
    timed_out.id.set(2404);
    std::unique_ptr<lsp::ResponseOrError<td_initialize::response>> timed_out_response;
    std::thread waiter(
        [&]()
        {
            timed_out_response = point.waitResponse(timed_out, 50);
        });
    WaitForOutputContaining(output_stream, "\"id\":2404");
    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","id":2404,"result":{"capabilities":{"hoverProvider":false}}})"));
    {
        std::unique_lock<std::mutex> lock(mutex);
        bool const entered = cv.wait_for(lock, std::chrono::milliseconds(1000), [&] { return parser_entered; });
        Expect(entered, "first response parser must capture the original pending request before timeout");
    }
    waiter.join();
    Expect(timed_out_response == nullptr, "first request must time out while its response is still parsing");

    td_initialize::request retry;
    retry.id.set(2404);
    auto retry_future = point.send(retry);
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_first_parse = true;
    }
    cv.notify_all();

    auto const stale_status = retry_future.wait_for(std::chrono::milliseconds(100));
    Expect(
        stale_status == lsp::future_status::timeout,
        "parsed stale response must not complete a later request that reused the same id");

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","id":2404,"result":{"capabilities":{"hoverProvider":true}}})"));
    auto const retry_status = retry_future.wait_for(std::chrono::milliseconds(1000));
    Expect(retry_status == lsp::future_status::ready, "reused id must complete from its own response");
    if (retry_status == lsp::future_status::ready)
    {
        auto result = retry_future.get();
        Expect(!result.is_error, "reused id response must be successful");
        Expect(
            result.response.result.capabilities.hoverProvider &&
                *result.response.result.capabilities.hoverProvider == true,
            "reused id must receive the second response, not the parsed stale response");
    }
    point.stop();
}

void TestWaitResponseInsideSingleWorkerHandlerDoesNotDeadlock()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            td_initialize::request outbound;
            outbound.id.set(2501);
            auto response = point.waitResponse(outbound, 5000);
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = response ? 1 : -1;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2500,"method":"test/scalar","params":{"value":0}})"));
    WaitForOutputContaining(output_stream, "\"id\":2501");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2501,"result":{"capabilities":{}}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":2500", 200);
    point.stop();

    Expect(output.find("\"id\":2500") != std::string::npos, "handler response must be written");
    Expect(
        output.find("\"value\":1") != std::string::npos,
        "nested waitResponse must complete even when the only handler worker is blocked");
}

void TestStoppedSynchronousHandlerDoesNotWriteToRestartedOutput()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    auto restart_input = std::make_shared<FeedableIStream>();
    auto restart_output = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool release_handler = false;
    std::atomic<bool> handler_entered {false};
    std::atomic<bool> restart_returned {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handler_entered.store(true, std::memory_order_relaxed);
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return release_handler; });
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2600,"method":"initialize","params":{}})"));
    Expect(
        WaitUntil([&] { return handler_entered.load(std::memory_order_relaxed); }),
        "old synchronous handler must enter before stop");
    point.stop();

    std::thread restarter(
        [&]()
        {
            point.startProcessingMessages(restart_input, restart_output);
            restart_returned.store(true, std::memory_order_relaxed);
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_handler = true;
    }
    cv.notify_all();
    restarter.join();
    Expect(restart_returned.load(std::memory_order_relaxed), "restart must complete after old handler exits");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::string const restarted_output_before_request = restart_output->snapshot();
    restart_input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2601,"method":"initialize","params":{}})"));
    std::string const restarted_output = WaitForOutputContaining(restart_output, "\"id\":2601");
    point.stop();

    Expect(
        restarted_output_before_request.find("\"id\":2600") == std::string::npos,
        "stopped handler must not write its old response to the restarted output");
    Expect(restarted_output.find("\"id\":2601") != std::string::npos, "restarted endpoint must still handle new requests");
}

void TestStartProcessingMessagesWhileWorkingDoesNotTerminate()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto first_input = std::make_shared<FeedableIStream>();
    auto second_input = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(first_input, output_stream);
    point.startProcessingMessages(second_input, output_stream);

    first_input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2700,"method":"initialize","params":{}})"));
    std::string const output = WaitForOutputContaining(output_stream, "\"id\":2700");
    point.stop();

    Expect(output.find("\"id\":2700") != std::string::npos, "duplicate start must leave existing session usable");
}

void TestResponseParseFailureCompletesFutureAndClearsPendingRequest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    json_handler->SetResponseJsonHandler(
        td_initialize::request::kMethodInfo,
        [](Reader&) -> std::unique_ptr<LspMessage>
        {
            return {};
        });
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    req.id.set(2800);
    auto future = point.send(req);
    WaitForOutputContaining(output_stream, "\"id\":2800");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2800,"result":{"capabilities":{}}})"));

    auto const failed_status = future.wait_for(std::chrono::milliseconds(1000));
    Expect(failed_status == lsp::future_status::ready, "response parse failure must complete the pending future");
    if (failed_status == lsp::future_status::ready)
    {
        auto result = future.get();
        Expect(result.is_error, "response parse failure must complete with an error result");
    }

    json_handler->SetResponseJsonHandler(
        td_initialize::request::kMethodInfo,
        [](Reader& visitor)
        {
            return td_initialize::response::ReflectReader(visitor);
        });
    td_initialize::request retry;
    retry.id.set(2800);
    auto retry_future = point.send(retry);
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":2800,"result":{"capabilities":{}}})"));
    auto const retry_status = retry_future.wait_for(std::chrono::milliseconds(1000));
    point.stop();

    Expect(retry_status == lsp::future_status::ready, "id must be reusable after response parse failure");
}

void TestCreateRequestAssignsMonotonicIds()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint point(json_handler, endpoint, log);

    auto first = point.createRequest<td_initialize::request>();
    auto second = point.createRequest<td_initialize::request>();
    auto third = point.createRequest<td_initialize::request>();

    Expect(first.id.type == lsRequestId::kInt, "createRequest must assign an integer id");
    Expect(first.id.value < second.id.value, "second generated id must be greater than first");
    Expect(second.id.value < third.id.value, "third generated id must be greater than second");
}

void TestCancelRequestSendsCancelNotification()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    auto future = point.send(req);
    WaitForOutputContaining(output_stream, "\"method\":\"initialize\"");

    bool const cancelled = point.cancelRequest(req.id);
    std::string const output = WaitForOutputContaining(output_stream, "$/cancelRequest");

    Expect(cancelled, "cancelRequest must return true for a pending request");
    Expect(
        output.find(R"("method":"$/cancelRequest")") != std::string::npos,
        "cancelRequest must send a cancel notification");
    Expect(output.find("\"id\":" + ToString(req.id)) != std::string::npos, "cancel notification must include id");
    point.stop();
}

void TestPendingCancelArrivingBeforeRequestIsNotLost()
{
    // Covers the scheduling race where $/cancelRequest is dispatched before the
    // target request has registered its canceler.
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    std::string const cancel = R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":501}})";
    std::string const request = R"({"jsonrpc":"2.0","id":501,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(cancel) + test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    bool handler_saw_cancel = false;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            handler_saw_cancel = monitor && monitor();
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":501");
    point.stop();

    Expect(handler_saw_cancel, "cancel notification that arrives before request registration must be applied");
    Expect(output.find("\"id\":501") != std::string::npos, "request with pending cancel must still receive a response");
}

void TestLateCancelAfterCompletedRequestDoesNotCancelReusedId()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<bool> saw_cancel;

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 2);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                saw_cancel.push_back(monitor && monitor());
            }
            cv.notify_all();
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":710,"method":"initialize","params":{}})"));
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(
            lock,
            std::chrono::milliseconds(1000),
            [&]()
            {
                return saw_cancel.size() >= 1;
            });
    }

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":710}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":710,"method":"initialize","params":{}})"));
    {
        std::unique_lock<std::mutex> lock(mutex);
        bool const handled_reused_id = cv.wait_for(
            lock,
            std::chrono::milliseconds(1000),
            [&]()
            {
                return saw_cancel.size() >= 2;
            });
        Expect(handled_reused_id, "reused request id must be handled after a late cancel");
    }
    point.stop();

    Expect(saw_cancel.size() == 2, "both original and reused request ids must be handled");
    if (saw_cancel.size() >= 2)
    {
        Expect(!saw_cancel[0], "original request must not be pre-cancelled");
        Expect(!saw_cancel[1], "late cancel for completed request must not cancel reused id");
    }
}

void TestOutOfOrderPendingCancelAppliesToReusedRequestId()
{
    // Scenario A: cancel is earlier on the wire than a reused request id. Sequential
    // routing gives the reused request a higher sequence, so the pending cancel must
    // not pre-cancel the reused request.
    {
        DummyLog log;
        auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
        auto endpoint = std::make_shared<GenericEndpoint>(log);
        auto input_stream = std::make_shared<FeedableIStream>();
        auto output_stream = std::make_shared<StringOStream>();

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<bool> saw_cancel;

        RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
        point.registerHandler(
            [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
            {
                bool const cancelled = monitor && monitor();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    saw_cancel.push_back(cancelled);
                }
                cv.notify_all();
                td_initialize::response rsp;
                rsp.id = req.id;
                return rsp;
            });
        point.startProcessingMessages(input_stream, output_stream);

        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1750,"method":"initialize","params":{}})"));
        Expect(
            WaitUntil(
                [&]()
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    return saw_cancel.size() >= 1;
                }),
            "first request must complete before testing pending cancel");

        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":1750}})"));
        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1750,"method":"initialize","params":{}})"));
        Expect(
            WaitUntil(
                [&]()
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    return saw_cancel.size() >= 2;
                }),
            "reused request id must be handled after pending cancel on wire");
        point.stop();

        Expect(saw_cancel.size() == 2, "both original and reused requests must be handled");
        if (saw_cancel.size() >= 2)
        {
            Expect(!saw_cancel[0], "first request must not be pre-cancelled");
            Expect(!saw_cancel[1], "pending cancel before reused request must not pre-cancel reuse");
        }
    }

    // Scenario B: cancel is later on the wire than a reused request id, but the
    // reused request is still in flight when the cancel is routed. Sequential routing
    // must still deliver the cancel to the active request via the live canceler.
    {
        DummyLog log;
        auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
        auto endpoint = std::make_shared<GenericEndpoint>(log);
        auto input_stream = std::make_shared<FeedableIStream>();
        auto output_stream = std::make_shared<StringOStream>();

        std::mutex mutex;
        std::condition_variable cv;
        bool release_reused = false;
        std::atomic<int> handled_count {0};
        std::atomic<bool> reused_entered {false};
        std::atomic<bool> reused_saw_cancel {false};
        std::vector<bool> saw_cancel;

        RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
        point.registerHandler(
            [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
            {
                int const count = handled_count.fetch_add(1, std::memory_order_relaxed) + 1;
                bool cancelled = false;
                if (count == 2)
                {
                    reused_entered.store(true, std::memory_order_relaxed);
                    std::unique_lock<std::mutex> lock(mutex);
                    for (int i = 0; i < 200; ++i)
                    {
                        if (monitor && monitor())
                        {
                            cancelled = true;
                            reused_saw_cancel.store(true, std::memory_order_relaxed);
                            break;
                        }
                        cv.wait_for(lock, std::chrono::milliseconds(5), [&] { return release_reused; });
                        if (release_reused)
                        {
                            break;
                        }
                    }
                    if (!cancelled)
                    {
                        cancelled = monitor && monitor();
                        reused_saw_cancel.store(cancelled, std::memory_order_relaxed);
                    }
                }
                else
                {
                    cancelled = monitor && monitor();
                }

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    saw_cancel.push_back(cancelled);
                }
                cv.notify_all();
                td_initialize::response rsp;
                rsp.id = req.id;
                return rsp;
            });
        point.startProcessingMessages(input_stream, output_stream);

        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1750,"method":"initialize","params":{}})"));
        Expect(
            WaitUntil(
                [&]()
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    return saw_cancel.size() >= 1;
                }),
            "first request must complete before reused request blocks");

        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1750,"method":"initialize","params":{}})"));
        Expect(
            WaitUntil([&] { return reused_entered.load(std::memory_order_relaxed); }),
            "reused request must start before cancel is routed");
        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":1750}})"));

        Expect(
            WaitUntil([&] { return reused_saw_cancel.load(std::memory_order_relaxed); }),
            "cancel routed after reused request must cancel the in-flight request");
        {
            std::lock_guard<std::mutex> lock(mutex);
            release_reused = true;
        }
        cv.notify_all();
        Expect(
            WaitUntil(
                [&]()
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    return saw_cancel.size() >= 2;
                }),
            "reused request must finish after cancel");
        point.stop();

        Expect(saw_cancel.size() == 2, "both original and reused requests must be handled");
        if (saw_cancel.size() >= 2)
        {
            Expect(!saw_cancel[0], "first request must not be pre-cancelled");
            Expect(saw_cancel[1], "wire-order cancel after reused request must apply while request is active");
        }
    }

    // Scenario C: a reused request is earlier on the wire than its cancel, but is
    // parked behind a prior ordered notification. The cancel bypasses the blocked
    // notification queue and becomes pending before the reused request registers
    // its canceler, so pending->sequence > request->sequence must pre-cancel it.
    {
        DummyLog log;
        auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
        auto endpoint = std::make_shared<GenericEndpoint>(log);
        auto input_stream = std::make_shared<FeedableIStream>();
        auto output_stream = std::make_shared<StringOStream>();

        std::mutex mutex;
        std::condition_variable cv;
        bool release_notification = false;
        std::atomic<bool> notification_entered {false};
        std::atomic<bool> cancel_routed {false};
        std::vector<bool> saw_cancel;

        RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
        point.allowConcurrentNotification<Notify_Exit::notify>();
        point.registerHandler(
            [&](test_ordered_notification::notify const&)
            {
                notification_entered.store(true, std::memory_order_relaxed);
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return release_notification; });
            });
        point.registerHandler(
            [&](Notify_Exit::notify const&)
            {
                cancel_routed.store(true, std::memory_order_relaxed);
            });
        point.registerHandler(
            [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
            {
                bool const cancelled = monitor && monitor();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    saw_cancel.push_back(cancelled);
                }
                cv.notify_all();
                td_initialize::response rsp;
                rsp.id = req.id;
                return rsp;
            });
        point.startProcessingMessages(input_stream, output_stream);

        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1750,"method":"initialize","params":{}})"));
        Expect(
            WaitUntil(
                [&]()
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    return saw_cancel.size() >= 1;
                }),
            "first request must complete before parking reused request");

        input_stream->append(test::MakeLspFrame(MakeOrderedNotificationBody(1)));
        Expect(
            WaitUntil([&] { return notification_entered.load(std::memory_order_relaxed); }),
            "ordered notification must block following request");
        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1750,"method":"initialize","params":{}})"));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        {
            std::lock_guard<std::mutex> lock(mutex);
            Expect(saw_cancel.size() == 1, "reused request must remain parked behind ordered notification");
        }

        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":1750}})"));
        input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
        Expect(
            WaitUntil([&] { return cancel_routed.load(std::memory_order_relaxed); }),
            "concurrent notification after cancel must prove cancel was routed");

        {
            std::lock_guard<std::mutex> lock(mutex);
            release_notification = true;
        }
        cv.notify_all();
        Expect(
            WaitUntil(
                [&]()
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    return saw_cancel.size() >= 2;
                }),
            "parked reused request must run after ordered notification completes");
        point.stop();

        Expect(saw_cancel.size() == 2, "both original and parked reused requests must be handled");
        if (saw_cancel.size() >= 2)
        {
            Expect(!saw_cancel[0], "first request must not be pre-cancelled");
            Expect(saw_cancel[1], "pending cancel with higher sequence must apply to parked reused request");
        }
    }
}

void TestStopClearsPendingCancelRequests()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool handler_called = false;
    bool handler_saw_cancel = false;

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                handler_called = true;
                handler_saw_cancel = monitor && monitor();
            }
            cv.notify_all();
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    auto first_input = std::make_shared<StringIStream>(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":720}})"));
    point.startProcessingMessages(first_input, output_stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    point.stop();

    auto second_input = std::make_shared<StringIStream>(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":720,"method":"initialize","params":{}})"));
    point.startProcessingMessages(second_input, output_stream);
    {
        std::unique_lock<std::mutex> lock(mutex);
        bool const handled = cv.wait_for(
            lock,
            std::chrono::milliseconds(1000),
            [&]()
            {
                return handler_called;
            });
        Expect(handled, "request after restart must be handled");
    }
    point.stop();

    Expect(!handler_saw_cancel, "pending cancel from a previous run must be cleared on stop");
}

void TestRunningRequestObservesCancellation()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool handler_entered = false;
    std::atomic<bool> handler_saw_cancel {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 2);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                handler_entered = true;
            }
            cv.notify_all();

            for (int i = 0; i < 200; ++i)
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
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":620,"method":"initialize","params":{}})"));
    {
        std::unique_lock<std::mutex> lock(mutex);
        bool const entered = cv.wait_for(
            lock,
            std::chrono::milliseconds(1000),
            [&]()
            {
                return handler_entered;
            });
        Expect(entered, "request handler must start before cancellation is sent");
    }

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":620}})"));
    std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32800");
    point.stop();

    Expect(handler_saw_cancel.load(std::memory_order_relaxed), "running request handler must observe cancellation");
    Expect(output.find("\"code\":-32800") != std::string::npos, "cancelled request must report RequestCancelled");
}

void TestBurstRequestsAreAllDispatchedAndResponded()
{
    // Feeds a burst of framed requests at once to guard the producer-to-worker
    // scheduling path against dropped dispatches.
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    std::string input;
    std::vector<std::string> response_ids;
    for (int i = 0; i < 24; ++i)
    {
        int const id = 800 + i;
        input += test::MakeLspFrame(
            std::string(R"({"jsonrpc":"2.0","id":)") + std::to_string(id) + R"(,"method":"initialize","params":{}})");
        response_ids.push_back("\"id\":" + std::to_string(id));
    }

    auto input_stream = std::make_shared<StringIStream>(input);
    auto output_stream = std::make_shared<StringOStream>();
    std::mutex mutex;
    std::set<int> handled_ids;

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                handled_ids.insert(req.id.value);
            }
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContainingAll(output_stream, response_ids);
    point.stop();

    Expect(handled_ids.size() == response_ids.size(), "RemoteEndPoint must dispatch every request in a burst");
    for (auto const& id : response_ids)
    {
        Expect(output.find(id) != std::string::npos, "RemoteEndPoint burst dispatch must produce every response");
    }
}

void TestMultipleWorkersProcessRequestsConcurrently()
{
    // Confirms max_workers is effective by blocking handlers until at least two
    // have entered concurrently.
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const first = R"({"jsonrpc":"2.0","id":201,"method":"initialize","params":{}})";
    std::string const second = R"({"jsonrpc":"2.0","id":202,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(first) + test::MakeLspFrame(second));
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    int active_handlers = 0;
    int max_active_handlers = 0;
    int entered_handlers = 0;
    bool release_handlers = false;

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++active_handlers;
                ++entered_handlers;
                if (max_active_handlers < active_handlers)
                {
                    max_active_handlers = active_handlers;
                }
                cv.notify_all();
                cv.wait(
                    lock,
                    [&]()
                    {
                        return release_handlers;
                    });
                --active_handlers;
            }

            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    {
        std::unique_lock<std::mutex> lock(mutex);
        bool const both_entered = cv.wait_for(
            lock,
            std::chrono::milliseconds(1000),
            [&]()
            {
                return entered_handlers >= 2;
            });
        Expect(both_entered, "both requests must enter handlers");
        Expect(max_active_handlers >= 2, "multiple workers must run request handlers concurrently");
        release_handlers = true;
    }
    cv.notify_all();

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":202");
    point.stop();

    Expect(output.find("\"id\":201") != std::string::npos, "first concurrent request must get a response");
    Expect(output.find("\"id\":202") != std::string::npos, "second concurrent request must get a response");
}

void TestSingleWorkerProcessesRequestsSerially()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const first = R"({"jsonrpc":"2.0","id":211,"method":"initialize","params":{}})";
    std::string const second = R"({"jsonrpc":"2.0","id":212,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(first) + test::MakeLspFrame(second));
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::vector<std::string> events;

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                events.push_back(std::to_string(req.id.value) + "-start");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            {
                std::lock_guard<std::mutex> lock(mutex);
                events.push_back(std::to_string(req.id.value) + "-end");
            }
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":212");
    point.stop();

    Expect(output.find("\"id\":211") != std::string::npos, "single-worker first request must get a response");
    Expect(output.find("\"id\":212") != std::string::npos, "single-worker second request must get a response");
    Expect(events.size() == 4, "single-worker test must record four handler events");
    if (events.size() == 4)
    {
        Expect(events[0] == "211-start", "single worker must start first request first");
        Expect(events[1] == "211-end", "single worker must finish first request before second starts");
        Expect(events[2] == "212-start", "single worker must start second request after first completes");
        Expect(events[3] == "212-end", "single worker must finish second request last");
    }
}

void TestSendRequestWritesExactContentLengthFrame()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerResponseParser<test_scalar::request>();
    point.startProcessingMessages(input_stream, output_stream);

    test_scalar::request req;
    req.id.set(7001);
    req.params.value = 0;
    auto future = point.send(req);

    std::string const expected_body = R"({"jsonrpc":"2.0","id":7001,"method":"test/scalar","params":{"value":0}})";
    std::string const output = WaitForOutputContaining(output_stream, expected_body);
    point.stop();

    (void)future;
    Expect(output == test::MakeLspFrame(expected_body), "send(request) must write exact Content-Length frame bytes");
}

void TestConcurrentSendWritesCompleteFrames()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    std::vector<lsp::future<lsp::ResponseOrError<td_initialize::response>>> futures;
    std::mutex futures_mutex;
    std::vector<std::thread> senders;
    for (int i = 0; i < 16; ++i)
    {
        senders.emplace_back(
            [&, i]()
            {
                td_initialize::request req;
                req.id.set(3000 + i);
                auto future = point.send(req);
                std::lock_guard<std::mutex> lock(futures_mutex);
                futures.push_back(std::move(future));
            });
    }
    for (auto& sender : senders)
    {
        sender.join();
    }

    std::vector<std::string> needles;
    for (int i = 0; i < 16; ++i)
    {
        needles.push_back("\"id\":" + std::to_string(3000 + i));
    }
    std::string const output = WaitForOutputContainingAll(output_stream, needles);
    auto const bodies = ExtractFrameBodies(output);
    point.stop();

    Expect(bodies.size() == 16, "concurrent send must produce one complete frame per request");
    for (int i = 0; i < 16; ++i)
    {
        bool found = false;
        for (auto const& body : bodies)
        {
            if (body.find("\"id\":" + std::to_string(3000 + i)) != std::string::npos &&
                body.find(R"("method":"initialize")") != std::string::npos)
            {
                found = true;
                break;
            }
        }
        Expect(found, "concurrent send output must contain every request body exactly as a frame");
    }
}

void TestNotificationHandlerReceivesNotification()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<bool> notified {false};

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](Notify_Exit::notify const&)
        {
            notified.store(true, std::memory_order_relaxed);
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
    for (int i = 0; i < 100 && !notified.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    point.stop();

    Expect(notified.load(std::memory_order_relaxed), "registered notification handler must receive notifications");
}

void TestNotificationsRunInWireOrderWithMultipleWorkers()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::vector<int> observed;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const& notification)
        {
            // If notifications are allowed to run concurrently, later messages
            // can finish first. The endpoint must serialize them by wire order.
            std::this_thread::sleep_for(std::chrono::milliseconds(20 - notification.params.value));
            std::lock_guard<std::mutex> lock(mutex);
            observed.push_back(notification.params.value);
        });
    point.startProcessingMessages(input_stream, output_stream);

    for (int i = 0; i < 10; ++i)
    {
        input_stream->append(test::MakeLspFrame(
            std::string(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":)") +
            std::to_string(i) + R"(}})"));
    }

    bool const saw_all = WaitUntil(
        [&]()
        {
            std::lock_guard<std::mutex> lock(mutex);
            return observed.size() == 10;
        });
    point.stop();

    Expect(saw_all, "all ordered notifications must be delivered");
    if (saw_all)
    {
        for (int i = 0; i < 10; ++i)
        {
            Expect(observed[static_cast<size_t>(i)] == i, "notifications must run in wire order with multiple workers");
        }
    }
}

void TestParallelParsedMessagesRouteInWireOrder()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::vector<int> observed;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const& notification)
        {
            std::lock_guard<std::mutex> lock(mutex);
            observed.push_back(notification.params.value);
        });
    point.startProcessingMessages(input_stream, output_stream);

    for (int i = 0; i < 16; ++i)
    {
        size_t const payload_size = i == 0 ? 256 * 1024 : 0;
        input_stream->append(test::MakeLspFrame(MakeOrderedNotificationBody(i, payload_size)));
    }

    bool const saw_all = WaitUntil(
        [&]()
        {
            std::lock_guard<std::mutex> lock(mutex);
            return observed.size() == 16;
        });
    point.stop();

    Expect(saw_all, "all parallel-parsed notifications must be delivered");
    if (saw_all)
    {
        for (int i = 0; i < 16; ++i)
        {
            Expect(observed[static_cast<size_t>(i)] == i, "parallel parsed messages must route in wire order");
        }
    }
}

void TestConcurrentNotificationOptOutDoesNotGateRequests()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool release_first = false;
    std::vector<int> observed_notifications;
    std::atomic<bool> first_notification_entered {false};
    std::atomic<bool> request_ran {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.allowConcurrentNotification<test_ordered_notification::notify>();
    point.registerHandler(
        [&](test_ordered_notification::notify const& notification)
        {
            if (notification.params.value == 1)
            {
                first_notification_entered.store(true, std::memory_order_relaxed);
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return release_first; });
            }
            std::lock_guard<std::mutex> lock(mutex);
            observed_notifications.push_back(notification.params.value);
        });
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            request_ran.store(true, std::memory_order_relaxed);
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(MakeOrderedNotificationBody(1)));
    Expect(
        WaitUntil([&] { return first_notification_entered.load(std::memory_order_relaxed); }),
        "first opt-out notification must start");
    input_stream->append(test::MakeLspFrame(MakeOrderedNotificationBody(2)));
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4210,"method":"test/scalar","params":{"value":9}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":4210");
    bool const second_notification_finished = WaitUntil(
        [&]()
        {
            std::lock_guard<std::mutex> lock(mutex);
            return observed_notifications.size() == 1 && observed_notifications[0] == 2;
        });
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_first = true;
    }
    cv.notify_all();
    bool const saw_both_notifications = WaitUntil(
        [&]()
        {
            std::lock_guard<std::mutex> lock(mutex);
            return observed_notifications.size() == 2;
        });
    point.stop();

    Expect(output.find("\"id\":4210") != std::string::npos, "request behind opt-out notification must get a response");
    Expect(request_ran.load(std::memory_order_relaxed), "opt-out notification must not gate following request");
    Expect(second_notification_finished, "later opt-out notification must be able to finish before earlier one");
    Expect(saw_both_notifications, "all opt-out notifications must eventually complete");
}

void TestRequestWaitsForPriorNotificationWithMultipleWorkers()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> notification_applied {false};
    std::atomic<bool> request_saw_notification {false};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const&)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            notification_applied.store(true, std::memory_order_release);
        });
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            request_saw_notification.store(notification_applied.load(std::memory_order_acquire), std::memory_order_relaxed);
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":1}})"));
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4201,"method":"test/scalar","params":{"value":3}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":4201");
    point.stop();

    Expect(output.find("\"id\":4201") != std::string::npos, "request after notification must get a response");
    Expect(request_saw_notification.load(std::memory_order_relaxed), "request must start after prior notification completes");
}

void TestProgressNotificationsDispatchThroughStandardProtocolHandler()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::vector<std::string> events;
    endpoint->registerNotifyHandler(
        Notify_Progress::notify::kMethodInfo,
        [&](std::unique_ptr<LspMessage> msg)
        {
            auto* progress = dynamic_cast<Notify_Progress::notify*>(msg.get());
            if (progress == nullptr)
            {
                return false;
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (progress->params.token.first)
            {
                events.push_back(*progress->params.token.first + ":" + progress->params.value.Data());
            }
            else if (progress->params.token.second)
            {
                events.push_back(std::to_string(*progress->params.token.second) + ":" + progress->params.value.Data());
            }
            return true;
        });

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"$/progress","params":{"token":"work","value":{"kind":"begin","title":"Index"}}})"));
    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"$/progress","params":{"token":"work","value":{"kind":"report","message":"Half"}}})"));
    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"$/progress","params":{"token":99,"value":[{"uri":"file:///tmp/a.cpp"}]}})"));

    bool const saw_all = WaitUntil(
        [&]()
        {
            std::lock_guard<std::mutex> lock(mutex);
            return events.size() == 3;
        });
    point.stop();

    Expect(saw_all, "$/progress notifications must dispatch through ProtocolJsonHandler");
    if (saw_all)
    {
        Expect(events[0].find(R"("kind":"begin")") != std::string::npos, "work-done begin payload must be preserved");
        Expect(events[1].find(R"("kind":"report")") != std::string::npos, "work-done report payload must be preserved");
        Expect(events[2].find("99:") == 0, "numeric partial result token must parse");
        Expect(events[2].find("file:///tmp/a.cpp") != std::string::npos, "partial result payload must be preserved");
    }
}

void TestNotificationWaitResponseDoesNotDeadlockWithParkedRequest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::unique_ptr<RemoteEndPoint> point;
    std::atomic<bool> request_after_notification_ran {false};

    point.reset(new RemoteEndPoint(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1));
    point->registerHandler(
        [&](test_ordered_notification::notify const&)
        {
            td_initialize::request req;
            req.id.set(9100);
            auto response = point->waitResponse(req, 2000);
            Expect(response != nullptr && !response->IsError(), "notification waitResponse must complete");
        });
    point->registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            request_after_notification_ran.store(true, std::memory_order_relaxed);
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point->startProcessingMessages(input_stream, output_stream);

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":1}})"));
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9101,"method":"test/scalar","params":{"value":4}})"));

    std::string const request_output = WaitForOutputContaining(output_stream, "\"id\":9100");
    Expect(request_output.find("\"id\":9100") != std::string::npos, "notification must send nested request");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9100,"result":{"capabilities":{}}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":9101");
    point->stop();

    Expect(output.find("\"id\":9101") != std::string::npos, "parked request must run after notification waitResponse");
    Expect(request_after_notification_ran.load(std::memory_order_relaxed), "request parked behind notification must eventually run");
}

void TestCancelRequestBypassesBlockedNotificationQueue()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool release_notification = false;
    std::atomic<bool> request_entered {false};
    std::atomic<bool> handler_saw_cancel {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const&)
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return release_notification; });
        });
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor)
            -> lsp::ResponseOrError<td_initialize::response>
        {
            request_entered.store(true, std::memory_order_relaxed);
            for (int i = 0; i < 200; ++i)
            {
                if (monitor() != 0)
                {
                    handler_saw_cancel.store(true, std::memory_order_relaxed);
                    Rsp_Error error;
                    error.id = req.id;
                    error.error.code = lsErrorCodes::RequestCancelled;
                    error.error.message = "cancelled";
                    return lsp::ResponseOrError<td_initialize::response>(std::move(error));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4300,"method":"initialize","params":{}})"));
    Expect(WaitUntil([&] { return request_entered.load(std::memory_order_relaxed); }), "request must start before cancel");

    input_stream->append(test::MakeLspFrame(MakeOrderedNotificationBody(1, 256 * 1024)));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":4300}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32800");
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_notification = true;
    }
    cv.notify_all();
    point.stop();

    Expect(handler_saw_cancel.load(std::memory_order_relaxed), "cancel must bypass a blocked notification queue");
    Expect(output.find("\"code\":-32800") != std::string::npos, "cancelled request must report RequestCancelled");
}

void TestFailedNotificationDoesNotBlockFollowingRequest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const&)
        {
            throw std::runtime_error("ordered notification failed");
        });
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"unknown/notification","params":{}})"));
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":1}})"));
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4400,"method":"test/scalar","params":{"value":5}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":4400");
    point.stop();

    Expect(output.find("\"id\":4400") != std::string::npos, "failed or unknown notifications must not block requests");
    Expect(output.find("\"value\":5") != std::string::npos, "request after failed notification must still run");
}

void TestStopWithParkedRequestDoesNotDeadlockAndCanRestart()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool release_notification = false;
    std::atomic<bool> notification_entered {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const&)
        {
            notification_entered.store(true, std::memory_order_relaxed);
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return release_notification; });
        });
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":1}})"));
    Expect(WaitUntil([&] { return notification_entered.load(std::memory_order_relaxed); }), "slow notification must start");
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4500,"method":"test/scalar","params":{"value":6}})"));

    std::atomic<bool> stopped {false};
    std::thread stopper(
        [&]()
        {
            point.stop();
            stopped.store(true, std::memory_order_relaxed);
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_notification = true;
    }
    cv.notify_all();
    stopper.join();
    Expect(stopped.load(std::memory_order_relaxed), "stop with parked request must not deadlock");

    auto restart_input = std::make_shared<FeedableIStream>();
    auto restart_output = std::make_shared<StringOStream>();
    point.startProcessingMessages(restart_input, restart_output);
    restart_input->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4501,"method":"test/scalar","params":{"value":7}})"));
    std::string const output = WaitForOutputContaining(restart_output, "\"id\":4501");
    point.stop();

    Expect(output.find("\"value\":7") != std::string::npos, "endpoint must restart after stop with parked request");
}

void TestStopAndRestartFromNotificationHandlerKeepsDispatcherAlive()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    auto restart_input = std::make_shared<FeedableIStream>();
    auto restart_output = std::make_shared<StringOStream>();

    std::atomic<bool> restarted {false};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4);
    point.registerHandler(
        [&](test_ordered_notification::notify const&)
        {
            point.stop();
            point.startProcessingMessages(restart_input, restart_output);
            restarted.store(true, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](test_scalar::request const& req) -> lsp::ResponseOrError<test_scalar::response>
        {
            test_scalar::response rsp;
            rsp.id = req.id;
            rsp.result.value = req.params.value;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":1}})"));
    Expect(
        WaitUntil([&] { return restarted.load(std::memory_order_relaxed); }),
        "notification handler must be able to stop and restart the endpoint");

    restart_input->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":4510,"method":"test/scalar","params":{"value":8}})"));
    std::string const output = WaitForOutputContaining(restart_output, "\"id\":4510");
    point.stop();

    Expect(
        output.find("\"value\":8") != std::string::npos,
        "self-restarted endpoint must keep ordered dispatcher alive for new messages");
}

void TestThrowingNotificationHandlerDoesNotStopEndpoint()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<bool> notification_seen {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](Notify_Exit::notify const&)
        {
            notification_seen.store(true, std::memory_order_relaxed);
            throw std::runtime_error("notification handler failed");
        });
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"exit","params":{}})"));
    Expect(
        WaitUntil([&]() { return notification_seen.load(std::memory_order_relaxed); }),
        "throwing notification handler must run");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1950,"method":"initialize","params":{}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1950");
    point.stop();

    Expect(output.find("\"id\":1950") != std::string::npos, "endpoint must handle requests after notification throw");
}

void TestConditionTimedWaitReturnsNotifiedValue()
{
    Condition<LspMessage> condition;
    std::thread notifier(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::unique_ptr<LspMessage> msg(new Rsp_Error());
            condition.notify(std::move(msg));
        });

    auto msg = condition.wait(1000);
    notifier.join();

    Expect(msg != nullptr, "timed Condition wait must return the notified value");
}

void TestConditionTimedWaitStillTimesOut()
{
    Condition<LspMessage> condition;

    auto msg = condition.wait(10);

    Expect(msg == nullptr, "timed Condition wait must still return null on timeout");
}

void TestConditionWaitZeroBlocksUntilNotify()
{
    // RemoteEndPoint::internalWaitResponse uses wait(0) for indefinite blocking.
    Condition<LspMessage> condition;
    std::atomic<bool> wait_completed {false};
    std::unique_ptr<LspMessage> received;

    std::thread waiter(
        [&]()
        {
            received = condition.wait(0);
            wait_completed.store(true, std::memory_order_release);
        });

    for (int i = 0; i < 50 && !wait_completed.load(std::memory_order_acquire); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Expect(!wait_completed.load(std::memory_order_acquire), "Condition wait(0) must block until notify");

    std::unique_ptr<LspMessage> msg(new Rsp_Error());
    condition.notify(std::move(msg));
    waiter.join();

    Expect(wait_completed.load(std::memory_order_acquire), "Condition wait(0) must return after notify");
    Expect(received != nullptr, "Condition wait(0) must deliver the notified value");
}

void TestInternalWaitResponseSuccessTimeoutAndSendFailure()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request success_req;
    success_req.id.set(901);
    std::unique_ptr<LspMessage> success_response;
    std::thread success_waiter(
        [&]()
        {
            success_response = point.internalWaitResponse(success_req, 5000);
        });
    WaitForOutputContaining(output_stream, "\"id\":901");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":901,"result":{"capabilities":{}}})"));
    success_waiter.join();
    Expect(success_response != nullptr, "internalWaitResponse must return a response on success");
    if (success_response)
    {
        Expect(
            !static_cast<ResponseInMessage*>(success_response.get())->IsErrorType(),
            "internalWaitResponse success must deliver a non-error response");
    }

    td_initialize::request timed_out_req;
    timed_out_req.id.set(902);
    auto const timed_out_response = point.internalWaitResponse(timed_out_req, 50);
    Expect(timed_out_response == nullptr, "internalWaitResponse must return null on timeout");

    td_initialize::request retry_req;
    retry_req.id.set(902);
    auto retry_future = point.send(retry_req);
    auto const retry_status = retry_future.wait_for(std::chrono::milliseconds(50));
    Expect(
        retry_status == lsp::future_status::timeout,
        "request id must be reusable after internalWaitResponse timeout");

    point.stop();

    auto bad_output = std::make_shared<StringOStream>();
    bad_output->set_bad(true);
    RemoteEndPoint bad_point(json_handler, endpoint, log);
    td_initialize::request send_failed_req;
    send_failed_req.id.set(903);
    auto const send_failed_response = bad_point.internalWaitResponse(send_failed_req, 0);
    Expect(send_failed_response == nullptr, "internalWaitResponse must return null when send fails");
}

void TestSendWithOnErrorCallback()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request success_req;
    success_req.id.set(1000);
    std::atomic<bool> success_response_called {false};
    std::atomic<bool> unexpected_error_called {false};
    point.send(
        success_req,
        [&](td_initialize::response const& rsp)
        {
            success_response_called.store(true, std::memory_order_relaxed);
            Expect(rsp.id == success_req.id, "send(onError) success handler must receive the matching response id");
        },
        [&](Rsp_Error const&)
        {
            unexpected_error_called.store(true, std::memory_order_relaxed);
        });
    WaitForOutputContaining(output_stream, "\"id\":1000");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1000,"result":{"capabilities":{}}})"));
    for (int i = 0; i < 100 && !success_response_called.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Expect(success_response_called.load(std::memory_order_relaxed), "send(onError) must invoke success handler for success responses");
    Expect(!unexpected_error_called.load(std::memory_order_relaxed), "send(onError) must not invoke onError for success responses");

    td_initialize::request req;
    req.id.set(1001);
    std::atomic<bool> success_called {false};
    std::atomic<bool> error_called {false};
    lsErrorCodes captured_code = lsErrorCodes::UnknownErrorCode;
    std::string captured_message;

    point.send(
        req,
        [&](td_initialize::response const&)
        {
            success_called.store(true, std::memory_order_relaxed);
        },
        [&](Rsp_Error const& err)
        {
            error_called.store(true, std::memory_order_relaxed);
            captured_code = err.error.code;
            captured_message = err.error.message;
        });
    WaitForOutputContaining(output_stream, "\"id\":1001");
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1001,"error":{"code":-32600,"message":"callback error"}})"));
    for (int i = 0; i < 100 && !error_called.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    point.stop();

    Expect(error_called.load(std::memory_order_relaxed), "send(onError) must invoke onError for error responses");
    Expect(!success_called.load(std::memory_order_relaxed), "send(onError) must not invoke success handler on error");
    Expect(captured_code == lsErrorCodes::InvalidRequest, "send(onError) must deliver the response error code");
    Expect(captured_message == "callback error", "send(onError) must deliver the response error message");

    auto bad_output = std::make_shared<StringOStream>();
    bad_output->set_bad(true);
    RemoteEndPoint bad_point(json_handler, endpoint, log);
    td_initialize::request failed_send_req;
    failed_send_req.id.set(1002);
    std::atomic<bool> send_failure_called {false};
    std::string send_failure_message;
    bad_point.send(
        failed_send_req,
        [&](td_initialize::response const&)
        {
        },
        [&](Rsp_Error const& err)
        {
            send_failure_called.store(true, std::memory_order_relaxed);
            send_failure_message = err.error.message;
        });
    Expect(send_failure_called.load(std::memory_order_relaxed), "send(onError) must invoke onError when send fails");
    Expect(
        send_failure_message == "Failed to send request.",
        "send(onError) must report failed send through onError");
}

void TestSelectiveCancelAmongConcurrentInFlightRequests()
{
    // Production defaults to max_workers=2. Two overlapping requests must be able to run
    // concurrently, and cancel must affect only the targeted request id.
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    int entered_handlers = 0;
    std::atomic<bool> release_handlers {false};
    std::atomic<bool> cancelled_handler_saw_cancel {false};
    std::atomic<bool> live_handler_saw_cancel {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 2);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor)
            -> lsp::ResponseOrError<td_initialize::response>
        {
            int const id = static_cast<int>(req.id.value);
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++entered_handlers;
            }
            cv.notify_all();

            for (int i = 0; i < 400; ++i)
            {
                if (monitor && monitor())
                {
                    if (id == 630)
                    {
                        cancelled_handler_saw_cancel.store(true, std::memory_order_relaxed);
                    }
                    else
                    {
                        live_handler_saw_cancel.store(true, std::memory_order_relaxed);
                    }
                    Rsp_Error error;
                    error.id = req.id;
                    error.error.code = lsErrorCodes::RequestCancelled;
                    error.error.message = "cancelled";
                    return error;
                }
                if (release_handlers.load(std::memory_order_relaxed))
                {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":630,"method":"initialize","params":{}})"));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":631,"method":"initialize","params":{}})"));
    {
        std::unique_lock<std::mutex> lock(mutex);
        bool const both_entered = cv.wait_for(
            lock,
            std::chrono::milliseconds(1000),
            [&]()
            {
                return entered_handlers >= 2;
            });
        Expect(both_entered, "both concurrent requests must enter handlers before selective cancel");
    }

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":630}})"));
    Expect(
        WaitUntil([&]() { return cancelled_handler_saw_cancel.load(std::memory_order_relaxed); }, 200, 10),
        "targeted synchronous in-flight request must observe cancellation even when handlers fill the worker pool");
    release_handlers.store(true, std::memory_order_relaxed);

    std::string const output = WaitForOutputContainingAll(output_stream, {"\"id\":630", "\"id\":631"});
    point.stop();

    Expect(entered_handlers >= 2, "both concurrent requests must enter handlers");
    Expect(cancelled_handler_saw_cancel.load(std::memory_order_relaxed), "cancelled request must observe cancellation");
    Expect(!live_handler_saw_cancel.load(std::memory_order_relaxed), "non-cancelled concurrent request must not observe cancellation");
    Expect(output.find("\"id\":630") != std::string::npos, "cancelled request must produce a response frame");
    Expect(output.find("\"id\":631") != std::string::npos, "non-cancelled request must still complete");
    Expect(output.find("\"code\":-32800") != std::string::npos, "cancelled request must report RequestCancelled");
}

void TestDelimitedTransportPreservesOrderedDispatchAndCancelBypass()
{
    {
        DummyLog log;
        auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
        auto endpoint = std::make_shared<GenericEndpoint>(log);
        auto input_stream = std::make_shared<FeedableIStream>();
        auto output_stream = std::make_shared<StringOStream>();

        std::mutex mutex;
        std::vector<int> observed;
        RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Delimited, 4);
        point.registerHandler(
            [&](test_ordered_notification::notify const& notification)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20 - notification.params.value));
                std::lock_guard<std::mutex> lock(mutex);
                observed.push_back(notification.params.value);
            });
        point.startProcessingMessages(input_stream, output_stream);

        for (int i = 0; i < 3; ++i)
        {
            input_stream->append(MakeDelimitedInput(
                std::string(R"({"jsonrpc":"2.0","method":"test/orderedNotification","params":{"value":)") +
                std::to_string(i) + R"(}})"));
        }

        bool const saw_ordered = WaitUntil(
            [&]()
            {
                std::lock_guard<std::mutex> lock(mutex);
                return observed.size() == 3;
            });
        point.stop();

        Expect(saw_ordered, "delimited transport must preserve ordered notification delivery");
        if (saw_ordered)
        {
            for (int i = 0; i < 3; ++i)
            {
                Expect(
                    observed[static_cast<size_t>(i)] == i,
                    "delimited transport notifications must run in wire order");
            }
        }
    }

    {
        DummyLog log;
        auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
        auto endpoint = std::make_shared<GenericEndpoint>(log);
        auto input_stream = std::make_shared<FeedableIStream>();
        auto output_stream = std::make_shared<StringOStream>();

        std::mutex mutex;
        std::condition_variable cv;
        bool release_notification = false;
        std::atomic<bool> request_entered {false};
        std::atomic<bool> handler_saw_cancel {false};

        RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Delimited, 4);
        point.registerHandler(
            [&](test_ordered_notification::notify const&)
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return release_notification; });
            });
        point.registerHandler(
            [&](td_initialize::request const& req, CancelMonitor const& monitor)
                -> lsp::ResponseOrError<td_initialize::response>
            {
                request_entered.store(true, std::memory_order_relaxed);
                for (int i = 0; i < 200; ++i)
                {
                    if (monitor() != 0)
                    {
                        handler_saw_cancel.store(true, std::memory_order_relaxed);
                        Rsp_Error error;
                        error.id = req.id;
                        error.error.code = lsErrorCodes::RequestCancelled;
                        error.error.message = "cancelled";
                        return lsp::ResponseOrError<td_initialize::response>(std::move(error));
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                td_initialize::response rsp;
                rsp.id = req.id;
                return rsp;
            });
        point.startProcessingMessages(input_stream, output_stream);

        input_stream->append(MakeDelimitedInput(R"({"jsonrpc":"2.0","id":4400,"method":"initialize","params":{}})"));
        Expect(
            WaitUntil([&] { return request_entered.load(std::memory_order_relaxed); }),
            "delimited request must start before cancel");
        input_stream->append(MakeDelimitedInput(MakeOrderedNotificationBody(99, 256 * 1024)));
        input_stream->append(MakeDelimitedInput(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":4400}})"));

        std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32800");
        {
            std::lock_guard<std::mutex> lock(mutex);
            release_notification = true;
        }
        cv.notify_all();
        point.stop();

        Expect(
            handler_saw_cancel.load(std::memory_order_relaxed),
            "delimited cancel must bypass a blocked notification queue");
        Expect(
            output.find("\"code\":-32800") != std::string::npos,
            "delimited cancelled request must report RequestCancelled");
    }
}

void TestRejectedCompletionConsumesResponseAndClearsPendingRequest()
{
    // When a pending completion handler rejects a response, mainLoop clears the pending
    // entry without routing the consumed message to GenericEndpoint::onResponse.
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> pending_rejected {false};
    std::atomic<bool> retry_completed {false};
    std::atomic<bool> endpoint_received {false};

    endpoint->method2response["initialize"] =
        [&](std::unique_ptr<LspMessage> msg)
        {
            endpoint_received.store(msg != nullptr, std::memory_order_relaxed);
            return true;
        };

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    req.id.set(9401);
    Expect(
        point.internalSendRequest(
            req,
            [&](std::unique_ptr<LspMessage>)
            {
                pending_rejected.store(true, std::memory_order_relaxed);
                return false;
            }),
        "internalSendRequest must accept a rejecting completion handler");

    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9401,"result":{"capabilities":{}}})"));
    Expect(
        WaitUntil([&]() { return pending_rejected.load(std::memory_order_relaxed); }),
        "pending completion handler must reject the inbound response");
    Expect(
        !endpoint_received.load(std::memory_order_relaxed),
        "rejected pending completion must not route to endpoint onResponse");

    Expect(
        point.internalSendRequest(
            req,
            [&](std::unique_ptr<LspMessage> msg)
            {
                retry_completed.store(msg != nullptr, std::memory_order_relaxed);
                return true;
            }),
        "rejected completion must clear pending state so the request id can be reused");
    input_stream->append(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9401,"result":{"capabilities":{}}})"));
    Expect(
        WaitUntil([&]() { return retry_completed.load(std::memory_order_relaxed); }),
        "reused request id must complete after rejected completion clears pending state");
    point.stop();
}

void TestDelimitedRemoteEndPointRoundTrip()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1101,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(MakeDelimitedInput(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Delimited);
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1101");
    point.stop();

    Expect(output.find("\"id\":1101") != std::string::npos, "delimited RemoteEndPoint must respond to delimited input");
    Expect(
        output.find("Content-Length:") != std::string::npos,
        "delimited RemoteEndPoint must still write standard framed output");
}

void TestDispatchUnknownNotificationReturnsFalse()
{
    std::string const notification = R"({"jsonrpc":"2.0","method":"unknown/notification","params":{}})";
    auto const output = ProcessSingleInput(test::MakeLspFrame(notification));

    Expect(output.empty(), "unknown notification must not produce output");
}

void TestDispatchResponseParseFailure()
{
    DummyLog log;
    auto json_handler = std::make_shared<MessageJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.overrideResponseParser(
        td_initialize::request::kMethodInfo,
        [](Reader&) -> std::unique_ptr<LspMessage>
        {
            return {};
        });
    point.startProcessingMessages(input_stream, output_stream);

    td_initialize::request req;
    req.id.set(1201);
    std::unique_ptr<LspMessage> response;
    std::thread waiter(
        [&]()
        {
            response = point.internalWaitResponse(req, 500);
        });
    WaitForOutputContaining(output_stream, "\"id\":1201");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1201,"result":{"capabilities":{}}})"));
    waiter.join();
    point.stop();

    Expect(response != nullptr, "response parse failure must complete the pending request");
    if (response)
    {
        auto* error = static_cast<ResponseInMessage*>(response.get());
        Expect(error->IsErrorType(), "response parse failure must complete with an error response");
    }
}

void TestRegisterHandlerReturnsErrorResponse()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1301,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            Rsp_Error error;
            error.id = req.id;
            error.error.code = lsErrorCodes::InternalError;
            error.error.message = "handler returned error";
            return error;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32603");
    point.stop();

    Expect(output.find("\"id\":1301") != std::string::npos, "handler error response must preserve request id");
    Expect(output.find("\"code\":-32603") != std::string::npos, "handler error response must serialize error code");
    Expect(
        output.find("handler returned error") != std::string::npos,
        "handler error response must serialize error message");
}

void TestCancelRequestWhenNotWorkingOrUnknownId()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint point(json_handler, endpoint, log);

    lsRequestId unknown_id;
    unknown_id.set(404);
    Expect(!point.cancelRequest(unknown_id), "cancelRequest must return false when endpoint is not working");

    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    point.startProcessingMessages(input_stream, output_stream);

    lsRequestId not_pending_id;
    not_pending_id.set(405);
    Expect(!point.cancelRequest(not_pending_id), "cancelRequest must return false for unknown request id");

    point.stop();
    Expect(!point.cancelRequest(not_pending_id), "cancelRequest must return false after stop");
}

void TestSendNotificationBeforeStartDoesNotCrash()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint point(json_handler, endpoint, log);

    Notify_Exit::notify notification;
    point.send(notification);

    Expect(!point.isWorking(), "sending a notification before start must not mark endpoint as working");
}

// Adapted from clangd/test/request-reply.test
void TestServerInitiatedApplyEditRoundTrip()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    json_handler->SetResponseJsonHandler(
        WorkspaceApply::request::kMethodInfo,
        [](Reader& visitor) -> std::unique_ptr<LspMessage>
        {
            if (visitor.HasMember("error"))
            {
                return Rsp_Error::ReflectReader(visitor);
            }
            return WorkspaceApply::response::ReflectReader(visitor);
        });
    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    WorkspaceApply::request req;
    req.id.set(5501);
    lsTextEdit edit;
    edit.range.start.line = 0;
    edit.range.start.character = 0;
    edit.range.end.line = 0;
    edit.range.end.character = 4;
    edit.newText = "int";
    req.params.edit.changes = std::map<std::string, std::vector<lsTextEdit>>();
    req.params.edit.changes->emplace("file:///fixture.cpp", std::vector<lsTextEdit> {edit});

    std::atomic<bool> completed {false};
    std::atomic<bool> applied {false};
    std::thread waiter(
        [&]
        {
            auto future = point.send(req);
            auto result = future.get();
            completed.store(true, std::memory_order_relaxed);
            Expect(!result.IsError(), "workspace/applyEdit round trip must succeed");
            if (!result.IsError())
            {
                applied.store(result.response.result.applied, std::memory_order_relaxed);
            }
        });

    std::string const output = WaitForOutputContaining(output_stream, "workspace/applyEdit");
    Expect(output.find("\"id\":5501") != std::string::npos, "server-initiated applyEdit must preserve request id");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":"invalid","result":{"applied":true}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Expect(!completed.load(std::memory_order_relaxed), "workspace/applyEdit invalid reply id must be ignored");
    input_stream->append(test::MakeLspFrame(test::ReadFixture("workspace_apply_edit_response.json")));
    waiter.join();
    point.stop();

    Expect(completed.load(std::memory_order_relaxed), "workspace/applyEdit client reply must complete pending request");
    Expect(applied.load(std::memory_order_relaxed), "workspace/applyEdit client reply must deserialize applied=true");
}

void TestServerInitiatedApplyEditFixtureParse()
{
    rapidjson::Document request_document;
    request_document.Parse(test::ReadFixture("workspace_apply_edit_request.json").c_str());
    JsonReader request_reader(&request_document);
    Expect(
        WorkspaceApply::request::ReflectReader(request_reader) != nullptr,
        "clangd-derived workspace/applyEdit request fixture must parse");

    rapidjson::Document response_document;
    response_document.Parse(test::ReadFixture("workspace_apply_edit_response.json").c_str());
    JsonReader response_reader(&response_document);
    Expect(
        WorkspaceApply::response::ReflectReader(response_reader) != nullptr,
        "clangd-derived workspace/applyEdit response fixture must parse");
}

void TestRemoteEndPointRestartTwice()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled_count {0};
    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handled_count.fetch_add(1, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    Expect(!point.isWorking(), "RemoteEndPoint must not be working before start");
    auto first_input = std::make_shared<StringIStream>(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1401,"method":"initialize","params":{}})"));
    point.startProcessingMessages(first_input, output_stream);
    Expect(point.isWorking(), "RemoteEndPoint must report working after start");
    WaitForOutputContaining(output_stream, "\"id\":1401");
    point.stop();
    Expect(!point.isWorking(), "RemoteEndPoint must not report working after stop");

    auto second_input = std::make_shared<StringIStream>(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1402,"method":"initialize","params":{}})"));
    point.startProcessingMessages(second_input, output_stream);
    Expect(point.isWorking(), "RemoteEndPoint must report working after restart");
    WaitForOutputContaining(output_stream, "\"id\":1402");
    point.stop();
    Expect(!point.isWorking(), "RemoteEndPoint must not report working after second stop");

    Expect(handled_count.load(std::memory_order_relaxed) == 2, "RemoteEndPoint must handle requests across two restarts");
}

void TestRegisterHandlerThrowsRequestError()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1501,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [](td_initialize::request const&) -> td_initialize::response
        {
            throw lsp::RequestError(lsErrorCodes::InvalidParams, "invalid initialize params");
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"code\":-32602");
    point.stop();

    Expect(output.find("\"id\":1501") != std::string::npos, "RequestError response must preserve request id");
    Expect(
        output.find("invalid initialize params") != std::string::npos,
        "RequestError response must serialize error message");
}

void TestRegisterHandlerThrowsStdException()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1505,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [](td_initialize::request const&) -> td_initialize::response
        {
            throw std::runtime_error("sync handler failed");
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "sync handler failed");
    point.stop();

    Expect(output.find("\"id\":1505") != std::string::npos, "std::exception response must preserve request id");
    Expect(output.find("\"code\":-32603") != std::string::npos, "std::exception must become InternalError");
    Expect(output.find("sync handler failed") != std::string::npos, "std::exception message must be serialized");
}

void TestAsyncRegisterHandlerCompletesLater()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1502,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
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
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"hoverProvider\":true");
    point.stop();

    Expect(output.find("\"id\":1502") != std::string::npos, "async handler response must preserve request id");
    Expect(
        output.find("\"hoverProvider\":true") != std::string::npos,
        "async handler response must include handler result");
}

void TestAsyncRegisterHandlerReturnsErrorResult()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1503,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [](td_initialize::request const&) -> lsp::future<lsp::ResponseOrError<td_initialize::response>>
        {
            auto promise = std::make_shared<lsp::promise<lsp::ResponseOrError<td_initialize::response>>>();
            auto future = promise->get_future();
            std::thread(
                [promise]()
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    Rsp_Error error;
                    error.error.code = lsErrorCodes::InvalidParams;
                    error.error.message = "async invalid params";
                    promise->set_value(lsp::ResponseOrError<td_initialize::response>(std::move(error)));
                }
            ).detach();
            return future;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "async invalid params");
    point.stop();

    Expect(output.find("\"id\":1503") != std::string::npos, "async error response must preserve request id");
    Expect(
        output.find("\"code\":-32602") != std::string::npos,
        "async error response must serialize error code");
}

void TestAsyncRegisterHandlerWithMonitorCompletesLater()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string const request = R"({"jsonrpc":"2.0","id":1504,"method":"initialize","params":{}})";
    auto input_stream = std::make_shared<StringIStream>(test::MakeLspFrame(request));
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> monitor_seen {false};
    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::future<td_initialize::response>
        {
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            auto future = promise->get_future();
            std::thread(
                [promise, id = req.id, monitor, &monitor_seen]()
                {
                    monitor_seen.store(monitor && monitor(), std::memory_order_relaxed);
                    td_initialize::response rsp;
                    rsp.id = id;
                    promise->set_value(std::move(rsp));
                }
            ).detach();
            return future;
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1504");
    point.stop();

    Expect(output.find("\"id\":1504") != std::string::npos, "async cancellable handler must produce a response");
    Expect(!monitor_seen.load(std::memory_order_relaxed), "async handler must receive a live cancel monitor");
}

void TestAsyncHandlerDoesNotBlockDispatchWorker()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> handled_count {0};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::future<td_initialize::response>
        {
            int const count = handled_count.fetch_add(1, std::memory_order_relaxed) + 1;
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            auto future = promise->get_future();
            if (count == 2)
            {
                td_initialize::response rsp;
                rsp.id = req.id;
                promise->set_value(std::move(rsp));
            }
            return future;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1601,"method":"initialize","params":{}})"));
    for (int i = 0; i < 100 && handled_count.load(std::memory_order_relaxed) < 1; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1602,"method":"initialize","params":{}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1602");
    point.stop();

    Expect(handled_count.load(std::memory_order_relaxed) >= 2, "pending async future must not block dispatch worker");
    Expect(output.find("\"id\":1602") != std::string::npos, "second request must complete while first future is pending");
}

void TestMultiplePendingAsyncFuturesBothComplete()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> release {false};
    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::future<td_initialize::response>
        {
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            auto future = promise->get_future();
            std::thread(
                [promise, id = req.id, &release]()
                {
                    for (int i = 0; i < 200 && !release.load(std::memory_order_relaxed); ++i)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                    td_initialize::response rsp;
                    rsp.id = id;
                    promise->set_value(std::move(rsp));
                }
            ).detach();
            return future;
        });
    point.startProcessingMessages(input_stream, output_stream);

    // Queue two unready futures at the same time, then release both. The
    // completion loop must keep rotating through pending futures and answer
    // each request once its future becomes ready.
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1605,"method":"initialize","params":{}})"));
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1606,"method":"initialize","params":{}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    release.store(true, std::memory_order_relaxed);

    std::string const output = WaitForOutputContainingAll(output_stream, {"\"id\":1605", "\"id\":1606"});
    point.stop();

    Expect(output.find("\"id\":1605") != std::string::npos, "first pending async future must complete");
    Expect(output.find("\"id\":1606") != std::string::npos, "second pending async future must complete");
}

void TestAsyncCancellationAfterHandlerReturns()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> handler_started {false};
    std::atomic<bool> saw_cancel {false};
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::future<td_initialize::response>
        {
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            auto future = promise->get_future();
            handler_started.store(true, std::memory_order_relaxed);
            std::thread(
                [promise, id = req.id, monitor, &saw_cancel]()
                {
                    for (int i = 0; i < 100; ++i)
                    {
                        if (monitor && monitor())
                        {
                            saw_cancel.store(true, std::memory_order_relaxed);
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    td_initialize::response rsp;
                    rsp.id = id;
                    promise->set_value(std::move(rsp));
                }
            ).detach();
            return future;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1603,"method":"initialize","params":{}})"));
    for (int i = 0; i < 100 && !handler_started.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":1603}})"));

    std::string const output = WaitForOutputContaining(output_stream, "\"id\":1603");
    point.stop();

    Expect(output.find("\"id\":1603") != std::string::npos, "cancelled async request must still be able to respond");
    Expect(saw_cancel.load(std::memory_order_relaxed), "async handler monitor must observe cancel after handler returns");
}

void TestStopDoesNotWaitForUnfinishedAsyncFuture()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> handler_started {false};
    auto point = new RemoteEndPoint(json_handler, endpoint, log);
    point->registerHandler(
        [&](td_initialize::request const&) -> lsp::future<td_initialize::response>
        {
            auto promise = std::make_shared<lsp::promise<td_initialize::response>>();
            handler_started.store(true, std::memory_order_relaxed);
            return promise->get_future();
        });
    point->startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1604,"method":"initialize","params":{}})"));
    for (int i = 0; i < 100 && !handler_started.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::atomic<bool> stopped {false};
    std::thread stopper(
        [&]()
        {
            point->stop();
            stopped.store(true, std::memory_order_relaxed);
        });
    for (int i = 0; i < 100 && !stopped.load(std::memory_order_relaxed); ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    Expect(stopped.load(std::memory_order_relaxed), "stop must not wait for an unfinished async handler future");
    if (stopped.load(std::memory_order_relaxed))
    {
        stopper.join();
        delete point;
    }
    else
    {
        stopper.detach();
    }
}

void TestFutureGetSupportsMoveOnlyValue()
{
    lsp::promise<MoveOnlyFutureValue> promise;
    auto future = promise.get_future();

    promise.set_value(MoveOnlyFutureValue(42));
    MoveOnlyFutureValue value = future.get();

    Expect(value.value == 42, "future::get must move move-only values out of the shared state");
}

void TestFrameSizeLimitStopsEndpointBeforeDispatch()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<StringIStream>(
        test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9101,"method":"initialize","params":{}})")
    );
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> handler_called {false};
    RemoteEndPointLimits limits;
    limits.max_frame_size = 8;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1, limits);
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handler_called.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    Expect(
        WaitUntil([&] { return !point.isWorking(); }),
        "oversized frame must stop the endpoint");
    Expect(!handler_called.load(std::memory_order_relaxed), "oversized frame must not dispatch a handler");
    point.stop();
}

void TestParseQueueLimitStopsEndpoint()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::condition_variable cv;
    bool release_parser = false;
    std::atomic<bool> parser_entered {false};

    RemoteEndPointLimits limits;
    limits.max_parse_queue_size = 1;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1, limits);
    point.overrideRequestParser(
        td_initialize::request::kMethodInfo,
        [&](Reader& visitor)
        {
            parser_entered.store(true, std::memory_order_relaxed);
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return release_parser; });
            return td_initialize::request::ReflectReader(visitor);
        });
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9102,"method":"initialize","params":{}})"));
    Expect(
        WaitUntil([&] { return parser_entered.load(std::memory_order_relaxed); }),
        "first parse task must occupy the parse queue slot");
    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9103,"method":"initialize","params":{}})"));
    Expect(
        WaitUntil([&] { return !point.isWorking(); }),
        "parse queue overload must stop the endpoint");

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_parser = true;
    }
    cv.notify_all();
    point.stop();
}

void TestPendingCancelRequestsArePrunedByLimit()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    std::string input;
    input += test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":9201}})");
    input += test::MakeLspFrame(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":9202}})");
    input += test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9201,"method":"initialize","params":{}})");
    input += test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9202,"method":"initialize","params":{}})");
    auto input_stream = std::make_shared<StringIStream>(input);
    auto output_stream = std::make_shared<StringOStream>();

    std::mutex mutex;
    std::map<int, bool> cancelled_by_id;
    RemoteEndPointLimits limits;
    limits.max_pending_cancel_requests = 1;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1, limits);
    point.registerHandler(
        [&](td_initialize::request const& req, CancelMonitor const& monitor) -> lsp::ResponseOrError<td_initialize::response>
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                cancelled_by_id[req.id.value] = monitor && monitor();
            }
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    WaitForOutputContaining(output_stream, "\"id\":9202");
    point.stop();

    std::lock_guard<std::mutex> lock(mutex);
    Expect(cancelled_by_id[9201] == false, "oldest pending cancel must be pruned when the limit is exceeded");
    Expect(cancelled_by_id[9202] == true, "newest pending cancel must be retained within the limit");
}

void TestMaxPendingOutgoingRequestsLimitsServerInitiatedRpc()
{
    DummyLog log;
    lsp::ProtocolJsonHandlerOptions options;
    options.enableExperimentalStandardRequests = true;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>(options);
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPointLimits limits;
    limits.max_pending_outgoing_requests = 1;
    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1, limits);
    point.startProcessingMessages(input_stream, output_stream);

    WorkspaceApply::request first;
    first.id.set(9601);
    first.params.edit.changes = std::map<std::string, std::vector<lsTextEdit>>();

    WorkspaceApply::request second;
    second.id.set(9602);
    second.params.edit.changes = std::map<std::string, std::vector<lsTextEdit>>();

    auto first_future = point.send(first);
    auto second_future = point.send(second);
    auto second_result = second_future.get();

    Expect(second_result.IsError(), "second outgoing request must fail when pending cap is reached");
    if (second_result.IsError())
    {
        Expect(
            second_result.error.error.code == lsErrorCodes::InternalError,
            "pending outgoing cap must surface InternalError on send failure");
    }

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9601,"result":{"applied":true}})"));
    auto first_result = first_future.get();
    Expect(!first_result.IsError(), "first outgoing request must still complete after client reply");
    point.stop();
}

void TestLSPClientHelperExpectsServerCall()
{
    DummyLog log;
    lsp::ProtocolJsonHandlerOptions options;
    options.enableExperimentalStandardRequests = true;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>(options);
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);

    test::LSPClient client;
    client.attach(output_stream, input_stream);

    WorkspaceApply::request req;
    req.id.set(9701);
    req.params.edit.changes = std::map<std::string, std::vector<lsTextEdit>>();
    auto future = point.send(req);

    Expect(client.expectServerCall("workspace/applyEdit"), "LSPClient helper must observe server-initiated RPC");
    auto const call_body = client.takeLastServerCallBody("workspace/applyEdit");
    Expect(call_body.find("\"id\":9701") != std::string::npos, "LSPClient helper must capture server call body");
    auto const params = client.takeCallParams("workspace/applyEdit");
    Expect(params.IsObject() && params.HasMember("edit"), "LSPClient helper must extract server call params");
    client.reply(R"({"jsonrpc":"2.0","id":9701,"result":{"applied":true}})");

    auto result = future.get();
    Expect(!result.IsError(), "LSPClient helper reply injection must complete pending request");
    point.stop();
}

void TestRegisterHandlerAfterStartIsIgnored()
{
#ifndef NDEBUG
    return;
#endif
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<bool> handler_called {false};
    RemoteEndPoint point(json_handler, endpoint, log);
    point.startProcessingMessages(input_stream, output_stream);
    bool const registered = point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            handler_called.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    input_stream->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":9301,"method":"initialize","params":{}})"));
    std::string const output = WaitForOutputContaining(output_stream, "Method not found");
    point.stop();

    Expect(!handler_called.load(std::memory_order_relaxed), "handler registered after start must not run");
    Expect(!registered, "registerHandler after start must report failure");
    Expect(output.find("Method not found") != std::string::npos, "late handler registration must leave request unhandled");
}

void TestLspSessionInitDidOpenDidChangeCompletion()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();

    std::atomic<int> stage {0};
    std::vector<std::string> notification_order;
    std::mutex order_mutex;

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities = lsServerCapabilities();
            return rsp;
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidOpen::notify const&)
        {
            std::lock_guard<std::mutex> lock(order_mutex);
            notification_order.push_back("didOpen");
            stage.store(1, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidChange::notify const&)
        {
            std::lock_guard<std::mutex> lock(order_mutex);
            notification_order.push_back("didChange");
            stage.store(2, std::memory_order_relaxed);
        });
    point.registerHandler(
        [](td_completion::request const& req) -> lsp::ResponseOrError<td_completion::response>
        {
            td_completion::response rsp;
            rsp.id = req.id;
            rsp.result.isIncomplete = false;
            lsCompletionItem item;
            item.label = "fixtureItem";
            item.kind = lsCompletionItemKind::Function;
            item.detail = "fixture detail";
            rsp.result.items.push_back(item);
            return rsp;
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(test::ReadFixture("initialize_request.json")));
    Expect(
        !test::WaitForOutputContaining(output_stream, "\"id\":100").empty(),
        "initialize request must receive a response");

    input_stream->append(test::MakeLspFrame(test::ReadFixture("did_open_notification.json")));
    Expect(WaitUntil([&] { return stage.load(std::memory_order_relaxed) >= 1; }), "didOpen must be handled");

    input_stream->append(test::MakeLspFrame(test::ReadFixture("did_change_notification.json")));
    Expect(WaitUntil([&] { return stage.load(std::memory_order_relaxed) >= 2; }), "didChange must be handled");

    input_stream->append(test::MakeLspFrame(test::ReadFixture("completion_request.json")));
    std::string const output = test::WaitForOutputContaining(output_stream, "\"id\":104");
    Expect(output.find("\"id\":104") != std::string::npos, "completion request must receive a response");

    std::string completion_body;
    for (auto const& body : test::ExtractLspFrameBodies(output))
    {
        if (body.find("\"id\":104") != std::string::npos)
        {
            completion_body = body;
        }
    }
    Expect(!completion_body.empty(), "completion response frame must be present");
    test::ExpectJsonFixture(
        completion_body,
        "completion_list_response.json",
        "completion response must match golden fixture");

    for (auto const& body : test::ExtractLspFrameBodies(output_stream->snapshot()))
    {
        if (body.find("\"id\":100") != std::string::npos)
        {
            test::ExpectJsonFixture(
                body,
                "initialize_response.json",
                "initialize response must match golden fixture");
            break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(order_mutex);
        Expect(notification_order.size() == 2, "session must observe didOpen then didChange");
        if (notification_order.size() >= 2)
        {
            Expect(notification_order[0] == "didOpen", "didOpen must run before didChange");
            Expect(notification_order[1] == "didChange", "didChange must follow didOpen");
        }
    }
    point.stop();
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestDuplicateRequestIdReturnsError);
    RUN_TEST(TestWaitResponseTimeoutClearsPendingRequest);
    RUN_TEST(TestBadOutputStreamReturnsError);
    RUN_TEST(TestStopCompletesPendingRequestWithError);
    RUN_TEST(TestStopCancelsRunningRequestHandlers);
    RUN_TEST(TestStopFromNotificationHandlerDoesNotDeadlock);
    RUN_TEST(TestAsyncFutureCompletedAfterStopIsDroppedSafely);
    RUN_TEST(TestUnregisteredRequestReturnsMethodNotFound);
    RUN_TEST(TestUnregisteredRequestPreservesStringIdInMethodNotFound);
    RUN_TEST(TestRegisteredRequestDoesNotReturnMethodNotFound);
    RUN_TEST(TestMalformedJsonDoesNotCrash);
    RUN_TEST(TestWrongJsonRpcVersionIsRejected);
    RUN_TEST(TestNonStringJsonRpcVersionIsRejected);
    RUN_TEST(TestNonStringMethodIsRejected);
    RUN_TEST(TestRequestMissingParamsDoesNotCrashEndpoint);
    RUN_TEST(TestNullAndZeroParamsDoNotDropValidScalarValues);
    RUN_TEST(TestWorkDoneTokenParamsPassThroughToHandler);
    RUN_TEST(TestPositionalArrayParamsAreHandledWithoutStoppingEndpoint);
    RUN_TEST(TestResponseWithNoMatchingRequestIsIgnored);
    RUN_TEST(TestMessageWithIdAndMethodUsesRequestPath);
    RUN_TEST(TestMessageWithNoMethodNoResultIsRejected);
    RUN_TEST(TestSendRequestAndReceiveSuccessResponse);
    RUN_TEST(TestSendRequestAndReceiveErrorResponse);
    RUN_TEST(TestDuplicateResponseCompletesFutureOnce);
    RUN_TEST(TestRejectedCompletionConsumesResponseAndClearsPendingRequest);
    RUN_TEST(TestSendRequestAndReceiveSuccessResponseWithStringId);
    RUN_TEST(TestLateResponseAfterTimeoutIsIgnored);
    RUN_TEST(TestParsedLateResponseDoesNotCompleteReusedRequestId);
    RUN_TEST(TestWaitResponseInsideSingleWorkerHandlerDoesNotDeadlock);
    RUN_TEST(TestStoppedSynchronousHandlerDoesNotWriteToRestartedOutput);
    RUN_TEST(TestStartProcessingMessagesWhileWorkingDoesNotTerminate);
    RUN_TEST(TestResponseParseFailureCompletesFutureAndClearsPendingRequest);
    RUN_TEST(TestCreateRequestAssignsMonotonicIds);
    RUN_TEST(TestCancelRequestSendsCancelNotification);
    RUN_TEST(TestPendingCancelArrivingBeforeRequestIsNotLost);
    RUN_TEST(TestLateCancelAfterCompletedRequestDoesNotCancelReusedId);
    RUN_TEST(TestOutOfOrderPendingCancelAppliesToReusedRequestId);
    RUN_TEST(TestStopClearsPendingCancelRequests);
    RUN_TEST(TestRunningRequestObservesCancellation);
    RUN_TEST(TestSelectiveCancelAmongConcurrentInFlightRequests);
    RUN_TEST(TestBurstRequestsAreAllDispatchedAndResponded);
    RUN_TEST(TestMultipleWorkersProcessRequestsConcurrently);
    RUN_TEST(TestSingleWorkerProcessesRequestsSerially);
    RUN_TEST(TestSendRequestWritesExactContentLengthFrame);
    RUN_TEST(TestConcurrentSendWritesCompleteFrames);
    RUN_TEST(TestNotificationHandlerReceivesNotification);
    RUN_TEST(TestNotificationsRunInWireOrderWithMultipleWorkers);
    RUN_TEST(TestParallelParsedMessagesRouteInWireOrder);
    RUN_TEST(TestConcurrentNotificationOptOutDoesNotGateRequests);
    RUN_TEST(TestRequestWaitsForPriorNotificationWithMultipleWorkers);
    RUN_TEST(TestProgressNotificationsDispatchThroughStandardProtocolHandler);
    RUN_TEST(TestNotificationWaitResponseDoesNotDeadlockWithParkedRequest);
    RUN_TEST(TestCancelRequestBypassesBlockedNotificationQueue);
    RUN_TEST(TestFailedNotificationDoesNotBlockFollowingRequest);
    RUN_TEST(TestStopWithParkedRequestDoesNotDeadlockAndCanRestart);
    RUN_TEST(TestStopAndRestartFromNotificationHandlerKeepsDispatcherAlive);
    RUN_TEST(TestThrowingNotificationHandlerDoesNotStopEndpoint);
    RUN_TEST(TestConditionTimedWaitReturnsNotifiedValue);
    RUN_TEST(TestConditionTimedWaitStillTimesOut);
    RUN_TEST(TestConditionWaitZeroBlocksUntilNotify);
    RUN_TEST(TestInternalWaitResponseSuccessTimeoutAndSendFailure);
    RUN_TEST(TestSendWithOnErrorCallback);
    RUN_TEST(TestDelimitedTransportPreservesOrderedDispatchAndCancelBypass);
    RUN_TEST(TestDelimitedRemoteEndPointRoundTrip);
    RUN_TEST(TestDispatchUnknownNotificationReturnsFalse);
    RUN_TEST(TestDispatchResponseParseFailure);
    RUN_TEST(TestRegisterHandlerReturnsErrorResponse);
    RUN_TEST(TestRegisterHandlerThrowsRequestError);
    RUN_TEST(TestRegisterHandlerThrowsStdException);
    RUN_TEST(TestAsyncRegisterHandlerCompletesLater);
    RUN_TEST(TestAsyncRegisterHandlerReturnsErrorResult);
    RUN_TEST(TestAsyncRegisterHandlerWithMonitorCompletesLater);
    RUN_TEST(TestAsyncHandlerDoesNotBlockDispatchWorker);
    RUN_TEST(TestMultiplePendingAsyncFuturesBothComplete);
    RUN_TEST(TestAsyncCancellationAfterHandlerReturns);
    RUN_TEST(TestStopDoesNotWaitForUnfinishedAsyncFuture);
    RUN_TEST(TestFutureGetSupportsMoveOnlyValue);
    RUN_TEST(TestFrameSizeLimitStopsEndpointBeforeDispatch);
    RUN_TEST(TestParseQueueLimitStopsEndpoint);
    RUN_TEST(TestPendingCancelRequestsArePrunedByLimit);
    RUN_TEST(TestMaxPendingOutgoingRequestsLimitsServerInitiatedRpc);
    RUN_TEST(TestLSPClientHelperExpectsServerCall);
    RUN_TEST(TestRegisterHandlerAfterStartIsIgnored);
    RUN_TEST(TestCancelRequestWhenNotWorkingOrUnknownId);
    RUN_TEST(TestSendNotificationBeforeStartDoesNotCrash);
    RUN_TEST(TestServerInitiatedApplyEditRoundTrip);
    RUN_TEST(TestServerInitiatedApplyEditFixtureParse);
    RUN_TEST(TestRemoteEndPointRestartTwice);
    RUN_TEST(TestLspSessionInitDidOpenDidChangeCompletion);

    return test::Failures() == 0 ? 0 : 1;
}
