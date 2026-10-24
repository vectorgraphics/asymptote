#include "LibLsp/JsonRpc/StreamMessageProducer.h"
#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/JsonRpc/lsRequestId.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/lsDocumentUri.h"
#include "LibLsp/lsp/workspace/workspaceFolders.h"
#include "test_helpers.h"

#include <cstdint>
#include <map>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <sstream>
#include <string>

namespace
{
using test::Expect;

lsRequestId ParseRequestIdFromJson(char const* json)
{
    rapidjson::Document document;
    document.Parse(json);
    JsonReader reader(&document);
    lsRequestId id;
    Reflect(reader, id);
    return id;
}

std::string SerializeRequestIdToJson(lsRequestId const& id)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    JsonWriter json_writer(&writer);
    lsRequestId copy = id;
    Reflect(json_writer, copy);
    return buffer.GetString();
}

void TestRequestIdOrderingKeepsTypesDistinct()
{
    lsRequestId int_id;
    int_id.set(1);

    lsRequestId string_id;
    string_id.set("1");

    std::map<lsRequestId, std::string> ids;
    ids[int_id] = "int";
    ids[string_id] = "string";

    Expect(ids.size() == 2, "int and string request ids must be distinct map keys");
    Expect(ids[int_id] == "int", "integer request id lookup returned wrong value");
    Expect(ids[string_id] == "string", "string request id lookup returned wrong value");
}

void TestRequestIdOrderingKeepsLargeIntegersDistinct()
{
    lsRequestId small;
    small.set(1);

    lsRequestId large;
    large.type = lsRequestId::kInt;
    large.value = static_cast<int64_t>(1) << 40;

    std::map<lsRequestId, std::string> ids;
    ids[small] = "small";
    ids[large] = "large";

    Expect(ids.size() == 2, "large integer request ids must not collide");
    Expect(ids[large] == "large", "large integer request id lookup returned wrong value");
}

void TestRequestIdIntegerRoundTrip()
{
    lsRequestId const id = ParseRequestIdFromJson("42");

    Expect(id.type == lsRequestId::kInt, "integer request id must stay integer type");
    Expect(id.value == 42, "integer request id value mismatch");
    Expect(SerializeRequestIdToJson(id) == "42", "integer request id must serialize as JSON number");
}

void TestRequestIdLargeIntegerRoundTrip()
{
    int64_t const large = static_cast<int64_t>(1) << 40;
    lsRequestId const id = ParseRequestIdFromJson("1099511627776");

    Expect(id.type == lsRequestId::kInt, "large integer request id must stay integer type");
    Expect(id.value == large, "large integer request id value mismatch");
    Expect(
        SerializeRequestIdToJson(id) == "1099511627776",
        "large integer request id must serialize as JSON number");
}

void TestRequestIdIntegerReflectRoundTrip()
{
    lsRequestId id;
    id.set(9001);

    lsRequestId const round_trip = ParseRequestIdFromJson(SerializeRequestIdToJson(id).c_str());
    Expect(round_trip == id, "integer request id must round-trip through Reflect");
}

void TestRequestIdStringRoundTrip()
{
    lsRequestId const id = ParseRequestIdFromJson("\"abc-123\"");

    Expect(id.type == lsRequestId::kString, "string request id must stay string type");
    Expect(id.k_string == "abc-123", "string request id value mismatch");
    Expect(id.value == -1, "string request id must not derive a numeric value");
    Expect(SerializeRequestIdToJson(id) == "\"abc-123\"", "string request id must serialize as JSON string");
}

void TestNumericStringRequestIdDoesNotBecomeInteger()
{
    lsRequestId const id = ParseRequestIdFromJson("\"123\"");

    Expect(id.type == lsRequestId::kString, "numeric-looking string request id must stay string type");
    Expect(id.k_string == "123", "numeric-looking string request id text mismatch");
    Expect(id.value == -1, "numeric-looking string request id must not populate integer value");
    Expect(SerializeRequestIdToJson(id) == "\"123\"", "numeric-looking string id must serialize as JSON string");
}

void TestRequestIdNullAndInvalid()
{
    lsRequestId const null_id = ParseRequestIdFromJson("null");
    Expect(null_id.type == lsRequestId::kNone, "null request id must become kNone");
    Expect(SerializeRequestIdToJson(null_id) == "null", "kNone request id must serialize as null");

    lsRequestId const bool_id = ParseRequestIdFromJson("true");
    Expect(bool_id.type == lsRequestId::kNone, "non-int/non-string request id must become kNone");
}

void TestLspMessageWriteFraming()
{
    td_initialize::request req;
    req.id.set(42);

    std::ostringstream out;
    req.Write(out);
    std::string const framed = out.str();

    Expect(framed.find("Content-Length: ") == 0, "framed message must start with Content-Length header");

    auto const header_end = framed.find("\r\n\r\n");
    Expect(header_end != std::string::npos, "framed message must contain header/body separator");

    std::string const body = framed.substr(header_end + 4);
    int const content_length = test::ParseContentLength(framed);
    Expect(content_length >= 0, "Content-Length header must be parseable");
    Expect(static_cast<size_t>(content_length) == body.size(), "Content-Length must match JSON body size");
    Expect(body.find("\"id\":42") != std::string::npos, "framed body must contain serialized request id");
}

void TestRequestSwapSwapsMethod()
{
    td_initialize::request first;
    td_initialize::request second;
    first.SetMethodType("first");
    second.SetMethodType("second");

    first.swap(second);

    Expect(first.method == "second", "request swap must take the other method");
    Expect(second.method == "first", "request swap must give the original method to the other request");
}

void TestHandleMessageReadsContentLengthBody()
{
    std::string const body = R"({"jsonrpc":"2.0","id":42,"method":"test"})";
    auto input = std::make_shared<test::StringIStream>(body);
    test::CollectingIssueHandler issues;
    LSPStreamMessageProducer producer(issues, input);

    LSPStreamMessageProducer::Headers headers;
    headers.contentLength = static_cast<int>(body.size());

    std::string received;
    bool const ok = producer.handleMessage(
        headers,
        [&](std::string&& content)
        {
            received = std::move(content);
        });

    Expect(ok, "handleMessage must succeed when the stream has enough bytes");
    Expect(received == body, "handleMessage must deliver the full JSON body");
}

void TestHandleMessageReturnsFalseOnBadStream()
{
    auto input = std::make_shared<test::StringIStream>("short");
    input->set_bad(true);
    test::CollectingIssueHandler issues;
    LSPStreamMessageProducer producer(issues, input);

    LSPStreamMessageProducer::Headers headers;
    headers.contentLength = 10;

    bool callback_called = false;
    bool const ok = producer.handleMessage(
        headers,
        [&](std::string&&)
        {
            callback_called = true;
        });

    Expect(!ok, "handleMessage must fail when the input stream is bad");
    Expect(!callback_called, "handleMessage must not invoke callback on bad stream");
    Expect(!issues.issues.empty(), "handleMessage must report stream issues");
}

void TestHandleMessageReturnsFalseOnEof()
{
    std::string const partial = R"({"jsonrpc":"2.0","id":1})";
    auto input = std::make_shared<test::StringIStream>(partial);
    test::CollectingIssueHandler issues;
    LSPStreamMessageProducer producer(issues, input);

    LSPStreamMessageProducer::Headers headers;
    headers.contentLength = static_cast<int>(partial.size() + 32);

    bool callback_called = false;
    bool const ok = producer.handleMessage(
        headers,
        [&](std::string&&)
        {
            callback_called = true;
        });

    Expect(!ok, "handleMessage must fail when the input stream hits eof before the body is complete");
    Expect(!callback_called, "handleMessage must not invoke callback on eof");
    Expect(!issues.issues.empty(), "handleMessage must report eof while reading content body");
    Expect(
        issues.issues[0].text.find("No more input when reading content body") != std::string::npos,
        "handleMessage must report missing body bytes on eof");
}

void TestHandleMessageReturnsFalseOnFail()
{
    auto input = std::make_shared<test::StringIStream>(R"({"jsonrpc":"2.0","id":1})");
    input->set_fail(true);
    test::CollectingIssueHandler issues;
    LSPStreamMessageProducer producer(issues, input);

    LSPStreamMessageProducer::Headers headers;
    headers.contentLength = 4;

    bool callback_called = false;
    bool const ok = producer.handleMessage(
        headers,
        [&](std::string&&)
        {
            callback_called = true;
        });

    Expect(!ok, "handleMessage must fail when the input stream reports fail");
    Expect(!callback_called, "handleMessage must not invoke callback on fail");
    Expect(!issues.issues.empty(), "handleMessage must report stream fail issues");
}

struct OptionalFieldStruct
{
    optional<std::string> field;
};
MAKE_REFLECT_STRUCT(OptionalFieldStruct, field);

struct NullableFieldStruct
{
    Nullable<std::string> field;
};
MAKE_REFLECT_STRUCT(NullableFieldStruct, field);

template<typename T>
std::string SerializeReflectValue(T& value)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    JsonWriter json_writer(&writer);
    Reflect(json_writer, value);
    return buffer.GetString();
}

template<typename T>
T ParseReflectValue(char const* json)
{
    rapidjson::Document document;
    document.Parse(json);
    JsonReader reader(&document);
    T value;
    Reflect(reader, value);
    return value;
}

void TestNullableObjectMemberSerializesNull()
{
    NullableFieldStruct value;
    value.field = nullptr;
    std::string const json = SerializeReflectValue(value);
    Expect(json.find("\"field\":null") != std::string::npos, "Nullable member must serialize explicit JSON null");
}

void TestOptionalObjectMemberOmitsWhenDisengaged()
{
    OptionalFieldStruct value;
    std::string const json = SerializeReflectValue(value);
    Expect(json.find("field") == std::string::npos, "optional member must omit key when disengaged");
}

void TestNullableRoundTripNullAndValue()
{
    Nullable<std::string> null_value = nullptr;
    Nullable<std::string> parsed_null = ParseReflectValue<Nullable<std::string>>("null");
    Expect(parsed_null.isNull(), "Nullable must deserialize JSON null as null");
    Expect(SerializeReflectValue(null_value) == "null", "Nullable null must serialize as JSON null");

    Nullable<std::string> text;
    text = std::string("hello");
    Nullable<std::string> parsed_text = ParseReflectValue<Nullable<std::string>>("\"hello\"");
    Expect(!parsed_text.isNull(), "Nullable must deserialize JSON string value");
    Expect(parsed_text.value() == "hello", "Nullable string value must round-trip");
    Expect(SerializeReflectValue(text) == "\"hello\"", "Nullable value must serialize as JSON string");
}

void TestWorkspaceFoldersResponseNullableRoundTrip()
{
    WorkspaceFolders::response null_rsp;
    null_rsp.id.set(1);
    null_rsp.result = nullptr;
    std::string const null_json = SerializeReflectValue(null_rsp);
    Expect(null_json.find("\"result\":null") != std::string::npos, "workspaceFolders null response must serialize result:null");
    WorkspaceFolders::response const parsed_null_rsp =
        ParseReflectValue<WorkspaceFolders::response>(null_json.c_str());
    Expect(parsed_null_rsp.result.isNull(), "workspaceFolders result:null must parse as Nullable null");

    WorkspaceFolders::response empty_rsp;
    empty_rsp.id.set(2);
    empty_rsp.result.emplace();
    std::string const empty_json = SerializeReflectValue(empty_rsp);
    Expect(empty_json.find("\"result\":[]") != std::string::npos, "workspaceFolders empty response must serialize result:[]");
    WorkspaceFolders::response const parsed_empty_rsp =
        ParseReflectValue<WorkspaceFolders::response>(empty_json.c_str());
    Expect(parsed_empty_rsp.result.has_value(), "workspaceFolders result:[] must parse as a Nullable value");
    Expect(parsed_empty_rsp.result.value().empty(), "workspaceFolders result:[] must parse as an empty folder list");

    WorkspaceFolders::response folder_rsp;
    folder_rsp.id.set(3);
    WorkspaceFolder folder;
    folder.uri = lsDocumentUri();
    folder.uri.raw_uri_ = "file:///tmp/project";
    folder.name = "project";
    folder_rsp.result = std::vector<WorkspaceFolder> {folder};
    std::string const folder_json = SerializeReflectValue(folder_rsp);
    Expect(folder_json.find("file:///tmp/project") != std::string::npos, "workspaceFolders folder response must serialize uri");
    WorkspaceFolders::response const parsed_folder_rsp =
        ParseReflectValue<WorkspaceFolders::response>(folder_json.c_str());
    Expect(parsed_folder_rsp.result.has_value(), "workspaceFolders folder response must parse as a Nullable value");
    Expect(parsed_folder_rsp.result.value().size() == 1, "workspaceFolders folder response must parse one folder");
    if (parsed_folder_rsp.result.has_value() && parsed_folder_rsp.result.value().size() == 1)
    {
        Expect(
            parsed_folder_rsp.result.value()[0].uri.raw_uri_ == "file:///tmp/project",
            "workspaceFolders folder response must round-trip uri");
        Expect(parsed_folder_rsp.result.value()[0].name == "project", "workspaceFolders folder response must round-trip name");
    }
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestRequestIdOrderingKeepsTypesDistinct);
    RUN_TEST(TestRequestIdOrderingKeepsLargeIntegersDistinct);
    RUN_TEST(TestRequestIdIntegerRoundTrip);
    RUN_TEST(TestRequestIdLargeIntegerRoundTrip);
    RUN_TEST(TestRequestIdIntegerReflectRoundTrip);
    RUN_TEST(TestRequestIdStringRoundTrip);
    RUN_TEST(TestNumericStringRequestIdDoesNotBecomeInteger);
    RUN_TEST(TestRequestIdNullAndInvalid);
    RUN_TEST(TestLspMessageWriteFraming);
    RUN_TEST(TestRequestSwapSwapsMethod);
    RUN_TEST(TestHandleMessageReadsContentLengthBody);
    RUN_TEST(TestHandleMessageReturnsFalseOnBadStream);
    RUN_TEST(TestHandleMessageReturnsFalseOnEof);
    RUN_TEST(TestHandleMessageReturnsFalseOnFail);
    RUN_TEST(TestNullableObjectMemberSerializesNull);
    RUN_TEST(TestOptionalObjectMemberOmitsWhenDisengaged);
    RUN_TEST(TestNullableRoundTripNullAndValue);
    RUN_TEST(TestWorkspaceFoldersResponseNullableRoundTrip);

    return test::Failures() == 0 ? 0 : 1;
}
