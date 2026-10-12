#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/MessageJsonHandler.h"
#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/lsServerCapabilities.h"
#include "test_helpers.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using test::DummyLog;
using test::Expect;

JsonReader MakeReader(rapidjson::Document& document, char const* json)
{
    document.Parse(json);
    return JsonReader(&document);
}

template<typename Fn>
bool ThrowsInvalidArgument(Fn fn)
{
    try
    {
        fn();
    }
    catch (std::invalid_argument const&)
    {
        return true;
    }
    catch (...)
    {
    }
    return false;
}

void TestGenericEndpointDispatchesRegisteredHandlers()
{
    DummyLog log;
    GenericEndpoint endpoint(log);

    bool request_called = false;
    endpoint.registerRequestHandler(
        "initialize",
        [&](std::unique_ptr<LspMessage> msg)
        {
            request_called = msg && std::string(msg->GetMethodType()) == "initialize";
            return true;
        });

    std::unique_ptr<LspMessage> request(new td_initialize::request());
    bool const request_result = endpoint.onRequest(std::move(request));
    Expect(request_result, "registered request handler must return true");
    Expect(request_called, "registered request handler must be called");

    bool notify_called = false;
    endpoint.registerNotifyHandler(
        "exit",
        [&](std::unique_ptr<LspMessage> msg)
        {
            notify_called = msg && std::string(msg->GetMethodType()) == "exit";
            return true;
        });

    std::unique_ptr<LspMessage> notify(new Notify_Exit::notify());
    bool const notify_result = endpoint.notify(std::move(notify));
    Expect(notify_result, "registered notification handler must return true");
    Expect(notify_called, "registered notification handler must be called");

    bool response_called = false;
    endpoint.method2response["initialize"] = [&](std::unique_ptr<LspMessage> msg) {
        response_called = msg != nullptr;
        return true;
    };

    std::unique_ptr<LspMessage> response(new td_initialize::response());
    bool const response_result = endpoint.onResponse("initialize", std::move(response));
    Expect(response_result, "registered response handler must return true");
    Expect(response_called, "registered response handler must be called");
}

void TestGenericEndpointRejectsMissingHandlers()
{
    DummyLog log;
    GenericEndpoint endpoint(log);

    std::unique_ptr<LspMessage> request(new td_initialize::request());
    Expect(!endpoint.onRequest(std::move(request)), "missing request handler must return false");

    std::unique_ptr<LspMessage> notify(new Notify_Exit::notify());
    Expect(!endpoint.notify(std::move(notify)), "missing notification handler must return false");

    std::unique_ptr<LspMessage> response(new td_initialize::response());
    Expect(!endpoint.onResponse("initialize", std::move(response)), "missing response handler must return false");
}

void TestMessageJsonHandlerParsesRegisteredMethods()
{
    MessageJsonHandler handler;
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})");

    handler.SetRequestJsonHandler(
        "initialize",
        [](Reader&)
        {
            return std::unique_ptr<LspMessage>(new td_initialize::request());
        });
    auto request = handler.parseRequstMessage("initialize", reader);
    Expect(request != nullptr, "registered request parser must return a message");
    Expect(
        handler.parseRequstMessage("unknown", reader) == nullptr,
        "unknown request parser must return null");

    handler.SetNotificationJsonHandler(
        "exit",
        [](Reader&)
        {
            return std::unique_ptr<LspMessage>(new Notify_Exit::notify());
        });
    auto notify = handler.parseNotificationMessage("exit", reader);
    Expect(notify != nullptr, "registered notification parser must return a message");
    Expect(
        handler.parseNotificationMessage("unknown", reader) == nullptr,
        "unknown notification parser must return null");

    handler.SetResponseJsonHandler(
        "initialize",
        [](Reader&)
        {
            return std::unique_ptr<LspMessage>(new td_initialize::response());
        });
    auto response = handler.parseResponseMessage("initialize", reader);
    Expect(response != nullptr, "registered response parser must return a message");
    Expect(
        handler.parseResponseMessage("unknown", reader) == nullptr,
        "unknown response parser must return null");
}

void TestMessageJsonHandlerResolveResponseSkipsThrowingParsers()
{
    MessageJsonHandler handler;
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"({"jsonrpc":"2.0","id":1,"result":{}})");

    handler.SetResponseJsonHandler(
        "a.throwing",
        [](Reader&) -> std::unique_ptr<LspMessage>
        {
            throw std::invalid_argument("expected test exception");
        });
    handler.SetResponseJsonHandler(
        "b.success",
        [](Reader&)
        {
            return std::unique_ptr<LspMessage>(new td_initialize::response());
        });

    std::pair<std::string, std::unique_ptr<LspMessage>> resolved;
    bool const result = handler.resovleResponseMessage(reader, resolved);
    Expect(result, "response resolver must succeed when a later parser accepts the message");
    Expect(resolved.first == "b.success", "response resolver must report the successful method");
    Expect(resolved.second != nullptr, "response resolver must return parsed message");
}

void TestPrimitiveReaderRejectsWrongTypes()
{
    rapidjson::Document string_document;
    JsonReader string_reader = MakeReader(string_document, R"("not-an-int")");
    int int_value = 0;
    Expect(
        ThrowsInvalidArgument([&]() { Reflect(string_reader, int_value); }),
        "int reader must reject JSON string values");

    rapidjson::Document int_document;
    JsonReader int_reader = MakeReader(int_document, "123");
    std::string string_value;
    Expect(
        ThrowsInvalidArgument([&]() { Reflect(int_reader, string_value); }),
        "string reader must reject JSON integer values");

    rapidjson::Document bool_document;
    JsonReader bool_reader = MakeReader(bool_document, "true");
    double double_value = 0.0;
    Expect(
        ThrowsInvalidArgument([&]() { Reflect(bool_reader, double_value); }),
        "double reader must reject JSON bool values");
}

template<typename T>
std::string SerializeJson(T value)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    JsonWriter json_writer(&writer);
    Reflect(json_writer, value);
    return buffer.GetString();
}

template<typename T>
T ParseJson(char const* json)
{
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, json);
    T value;
    Reflect(reader, value);
    return value;
}

void TestCapabilityUnionBoolRoundTrip()
{
    using Provider = std::pair<optional<bool>, optional<WorkDoneProgressOptions>>;

    Provider const parsed = ParseJson<Provider>("true");
    Expect(parsed.first && *parsed.first == true, "bool capability union must parse JSON true");
    Expect(!parsed.second, "bool capability union must leave object side unset");

    Provider provider;
    provider.first = true;
    Expect(SerializeJson(provider) == "true", "bool capability union must serialize as JSON true");
}

void TestCapabilityUnionObjectRoundTrip()
{
    using Provider = std::pair<optional<bool>, optional<CodeActionOptions>>;

    Provider const parsed = ParseJson<Provider>(R"({"codeActionKinds":["quickfix","refactor"]})");
    Expect(!parsed.first, "object capability union must leave bool side unset");
    Expect(
        parsed.second && parsed.second->codeActionKinds.size() == 2 &&
            parsed.second->codeActionKinds[0] == "quickfix" &&
            parsed.second->codeActionKinds[1] == "refactor",
        "object capability union must parse provider options");

    Provider provider;
    CodeActionOptions options;
    options.codeActionKinds = std::vector<std::string>{"quickfix"};
    provider.second = options;
    std::string const json = SerializeJson(provider);
    Expect(
        json.find("\"codeActionKinds\"") != std::string::npos,
        "object capability union must serialize provider options");
    Expect(json.find("\"quickfix\"") != std::string::npos, "object capability union must preserve option values");
}

void TestJsonReaderIterArrayRejectsNonArray()
{
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"({"not":"array"})");
    Expect(
        ThrowsInvalidArgument([&]() { reader.IterArray([](Reader&) {}); }),
        "IterArray must reject non-array JSON values");

    rapidjson::Document array_document;
    JsonReader array_reader = MakeReader(array_document, "[1,2,3]");
    std::vector<int> values;
    array_reader.IterArray([&](Reader& child) {
        int value = 0;
        Reflect(child, value);
        values.push_back(value);
    });
    Expect(values.size() == 3, "IterArray must iterate JSON arrays");
    Expect(values[0] == 1 && values[1] == 2 && values[2] == 3, "IterArray must preserve element order");
}

void TestJsonReaderDoMemberSkipsMissingKey()
{
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"({"present":7})");

    bool missing_called = false;
    reader.DoMember(
        "missing",
        [&](Reader&)
        {
            missing_called = true;
        });
    Expect(!missing_called, "DoMember must skip missing keys without invoking callback");

    int present_value = 0;
    reader.DoMember(
        "present",
        [&](Reader& child)
        {
            Reflect(child, present_value);
        });
    Expect(present_value == 7, "DoMember must invoke callback for present keys");
}

void TestJsonReaderHasMemberAndNull()
{
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"({"known":null,"nested":{"child":1}})");

    Expect(reader.HasMember("known"), "HasMember must find existing object keys");
    Expect(!reader.HasMember("missing"), "HasMember must return false for missing keys");

    reader.DoMember(
        "known",
        [&](Reader& child)
        {
            Expect(child.IsNull(), "reader positioned at null must report IsNull");
        });

    rapidjson::Document string_document;
    JsonReader string_reader = MakeReader(string_document, R"("not-an-object")");
    Expect(!string_reader.HasMember("anything"), "HasMember must return false for non-object values");
}

void TestJsonWriterNullRoundTrip()
{
    optional<int> value;
    std::string const json = SerializeJson(value);
    Expect(json == "null", "unset optional must serialize as null");

    optional<int> const parsed = ParseJson<optional<int>>("null");
    Expect(!parsed, "null optional must deserialize as unset");
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestGenericEndpointDispatchesRegisteredHandlers);
    RUN_TEST(TestGenericEndpointRejectsMissingHandlers);
    RUN_TEST(TestMessageJsonHandlerParsesRegisteredMethods);
    RUN_TEST(TestMessageJsonHandlerResolveResponseSkipsThrowingParsers);
    RUN_TEST(TestPrimitiveReaderRejectsWrongTypes);
    RUN_TEST(TestCapabilityUnionBoolRoundTrip);
    RUN_TEST(TestCapabilityUnionObjectRoundTrip);
    RUN_TEST(TestJsonReaderIterArrayRejectsNonArray);
    RUN_TEST(TestJsonReaderDoMemberSkipsMissingKey);
    RUN_TEST(TestJsonReaderHasMemberAndNull);
    RUN_TEST(TestJsonWriterNullRoundTrip);

    return test::Failures() == 0 ? 0 : 1;
}
