#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/LspCpp.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/shutdown.h"
#include "LibLsp/lsp/textDocument/SemanticTokens.h"
#include "LibLsp/lsp/textDocument/completion.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include "LibLsp/lsp/textDocument/hover.h"
#include "LibLsp/lsp/textDocument/references.h"
#include "LibLsp/lsp/textDocument/signature_help.h"
#include "LibLsp/lsp/textDocument/code_action.h"
#include "LibLsp/lsp/textDocument/rename.h"
#include "LibLsp/lsp/textDocument/foldingRange.h"
#include "LibLsp/lsp/textDocument/selectionRange.h"
#include "LibLsp/lsp/textDocument/inlayHint.h"
#include "LibLsp/lsp/workspace/symbol.h"
#include "LibLsp/lsp/workspace/did_change_configuration.h"
#include "protocol_test_helpers.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <rapidjson/document.h>
#include <string>
#include <thread>
#include <vector>

namespace
{
using test::DummyLog;
using test::Expect;
using test::FeedableIStream;
using test::StringIStream;
using test::StringOStream;
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

std::string ExtractRequestId(std::string const& json)
{
    rapidjson::Document document;
    document.Parse(json.c_str());
    if (!document.IsObject() || !document.HasMember("id"))
    {
        return {};
    }
    auto const& id = document["id"];
    if (id.IsInt())
    {
        return std::to_string(id.GetInt());
    }
    if (id.IsInt64())
    {
        return std::to_string(id.GetInt64());
    }
    if (id.IsString())
    {
        return std::string("\"") + id.GetString() + "\"";
    }
    return {};
}

struct SessionStep
{
    std::string step;
    std::string file;
    std::string kind;
    std::string response_fixture;
};

td_shutdown::response MakeNullShutdownResponse(lsRequestId id)
{
    td_shutdown::response rsp;
    rsp.id = id;
    lsp::Any result;
    result.SetJsonString("null", lsp::Any::kNullType);
    rsp.result = result;
    return rsp;
}

std::vector<SessionStep> LoadSessionSteps(char const* manifest_file)
{
    std::vector<SessionStep> steps;
    rapidjson::Document document;
    document.Parse(test::ReadFixture(manifest_file).c_str());
    Expect(document.IsArray(), "session manifest must be a JSON array");
    if (!document.IsArray())
    {
        return steps;
    }

    for (auto const& entry : document.GetArray())
    {
        if (!entry.IsObject())
        {
            continue;
        }
        SessionStep step;
        if (entry.HasMember("step") && entry["step"].IsString())
        {
            step.step = entry["step"].GetString();
        }
        if (entry.HasMember("file") && entry["file"].IsString())
        {
            step.file = entry["file"].GetString();
        }
        if (entry.HasMember("kind") && entry["kind"].IsString())
        {
            step.kind = entry["kind"].GetString();
        }
        if (entry.HasMember("response_fixture") && entry["response_fixture"].IsString())
        {
            step.response_fixture = entry["response_fixture"].GetString();
        }
        steps.push_back(std::move(step));
    }
    return steps;
}

void RegisterFixtureSessionHandlers(RemoteEndPoint& point, std::atomic<int>& notification_count)
{
    point.registerHandler(
        [](td_shutdown::request const& req) -> lsp::ResponseOrError<td_shutdown::response>
        {
            return MakeNullShutdownResponse(req.id);
        });
    point.registerHandler(
        [](td_semanticTokens_full::request const& req) -> lsp::ResponseOrError<td_semanticTokens_full::response>
        {
            td_semanticTokens_full::response rsp;
            rsp.id = req.id;
            SemanticTokens tokens;
            tokens.data = {0, 4, 1, 0, 131075};
            tokens.resultId = "1";
            rsp.result = tokens;
            return rsp;
        });
    point.registerHandler(
        [](td_rename::request const& req) -> lsp::ResponseOrError<td_rename::response>
        {
            td_rename::response rsp;
            rsp.id = req.id;
            lsTextEdit edit;
            edit.range.start.line = 0;
            edit.range.start.character = 0;
            edit.range.end.line = 0;
            edit.range.end.character = 3;
            edit.newText = "renamed";
            rsp.result.changes = std::map<std::string, std::vector<lsTextEdit>>();
            rsp.result.changes->emplace("file:///fixture.cpp", std::vector<lsTextEdit> {edit});
            return rsp;
        });
    point.registerHandler(
        [](td_codeAction::request const& req) -> lsp::ResponseOrError<td_codeAction::response>
        {
            td_codeAction::response rsp;
            rsp.id = req.id;
            lsCommandWithAny action;
            action.title = "Fixture action";
            rsp.result.push_back(action);
            return rsp;
        });
    point.registerHandler(
        [](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities = lsServerCapabilities();
            return rsp;
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
    point.registerHandler(
        [](td_hover::request const& req) -> lsp::ResponseOrError<td_hover::response>
        {
            td_hover::response rsp;
            rsp.id = req.id;
            TextDocumentHover::Result result;
            MarkupContent content;
            content.kind = "plaintext";
            content.value = "function foo\n\n-> void\n\nvoid foo()";
            result.contents = TextDocumentHover::Either {TextDocumentHover::Left {}, content};
            lsRange range;
            range.start.line = 0;
            range.start.character = 25;
            range.end.line = 0;
            range.end.character = 28;
            result.range = range;
            rsp.result = result;
            return rsp;
        });
    point.registerHandler(
        [](td_references::request const& req) -> lsp::ResponseOrError<td_references::response>
        {
            td_references::response rsp;
            rsp.id = req.id;
            lsLocation location;
            lsDocumentUri uri;
            uri.raw_uri_ = "file:///fixture.cpp";
            location.uri = uri;
            location.range.start.line = 0;
            location.range.start.character = 15;
            location.range.end.line = 0;
            location.range.end.character = 16;
            rsp.result.push_back(location);
            return rsp;
        });
    point.registerHandler(
        [](td_signatureHelp::request const& req) -> lsp::ResponseOrError<td_signatureHelp::response>
        {
            td_signatureHelp::response rsp;
            rsp.id = req.id;
            lsSignatureHelp help;
            help.activeParameter = 0;
            help.activeSignature = 0;
            lsSignatureInformation signature;
            signature.label = "x(int) -> void";
            MarkupContent documentation;
            documentation.kind = "markdown";
            documentation.value = "comment `markdown` \\_escape\\_";
            signature.documentation = std::make_pair(optional<std::string>(), documentation);
            lsParameterInformation parameter;
            parameter.label = "int";
            signature.parameters.push_back(parameter);
            help.signatures.push_back(signature);
            rsp.result = help;
            return rsp;
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidOpen::notify const&)
        {
            notification_count.fetch_add(1, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidChange::notify const&)
        {
            notification_count.fetch_add(1, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](Notify_WorkspaceDidChangeConfiguration::notify const& notify)
        {
            Expect(
                !notify.params.settings.Data().empty(),
                "didChangeConfiguration settings must parse as object");
            notification_count.fetch_add(1, std::memory_order_relaxed);
        });
}

void RegisterFeatureSessionHandlers(RemoteEndPoint& point, std::atomic<int>& notification_count)
{
    RegisterFixtureSessionHandlers(point, notification_count);
    point.registerHandler(
        [](td_foldingRange::request const& req) -> lsp::ResponseOrError<td_foldingRange::response>
        {
            td_foldingRange::response rsp;
            rsp.id = req.id;
            FoldingRange range {};
            range.startLine = 0;
            range.endLine = 1;
            range.kind = "region";
            rsp.result.push_back(range);
            return rsp;
        });
    point.registerHandler(
        [](td_selectionRange::request const& req) -> lsp::ResponseOrError<td_selectionRange::response>
        {
            td_selectionRange::response rsp;
            rsp.id = req.id;
            SelectionRange selection;
            selection.range.start.line = 0;
            selection.range.start.character = 0;
            selection.range.end.line = 0;
            selection.range.end.character = 3;
            rsp.result.push_back(selection);
            return rsp;
        });
    point.registerHandler(
        [](td_inlayHint::request const& req) -> lsp::ResponseOrError<td_inlayHint::response>
        {
            td_inlayHint::response rsp;
            rsp.id = req.id;
            lsInlayHint hint;
            hint.position.line = 0;
            hint.position.character = 5;
            hint.label = "int";
            rsp.result.push_back(hint);
            return rsp;
        });
}

void RegisterLifecycleSessionHandlers(RemoteEndPoint& point, std::atomic<bool>& initialized)
{
    point.registerHandler(
        [&](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response>
        {
            if (initialized.load(std::memory_order_relaxed))
            {
                throw lsp::RequestError(lsErrorCodes::InvalidRequest, "server already initialized");
            }
            initialized.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            rsp.result.capabilities = lsServerCapabilities();
            return rsp;
        });
    point.registerHandler(
        [&](wp_symbol::request const& req) -> lsp::ResponseOrError<wp_symbol::response>
        {
            if (!initialized.load(std::memory_order_relaxed))
            {
                throw lsp::RequestError(lsErrorCodes::ServerNotInitialized, "server not initialized");
            }
            wp_symbol::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    point.registerHandler(
        [](td_shutdown::request const& req) -> lsp::ResponseOrError<td_shutdown::response>
        {
            return MakeNullShutdownResponse(req.id);
        });
}

void RunFixtureDrivenSession(
    char const* manifest_file,
    std::shared_ptr<FeedableIStream> const& input_stream,
    std::shared_ptr<StringOStream> const& output_stream,
    std::atomic<int>& notification_count)
{
    auto const steps = LoadSessionSteps(manifest_file);
    Expect(steps.size() >= 2, "session manifest must describe at least two steps");

    int expected_notifications = 0;
    for (auto const& step : steps)
    {
        std::string const payload = test::ReadFixture(step.file.c_str());
        Expect(!payload.empty(), "session step fixture must be readable");
        input_stream->append(test::MakeLspFrame(payload));

        if (step.kind == "notification")
        {
            ++expected_notifications;
            Expect(
                WaitUntil([&] { return notification_count.load(std::memory_order_relaxed) >= expected_notifications; }),
                "fixture-driven session must dispatch notification steps");
            continue;
        }

        if (step.kind != "request")
        {
            continue;
        }

        std::string const request_id = ExtractRequestId(payload);
        Expect(!request_id.empty(), "request step fixture must include an id");
        std::string const output = WaitForOutputContaining(output_stream, "\"id\":" + request_id);
        Expect(output.find("\"id\":" + request_id) != std::string::npos, "request step must receive a response");

        if (!step.response_fixture.empty())
        {
            std::string response_body;
            for (auto const& body : test::ExtractLspFrameBodies(output_stream->snapshot()))
            {
                if (body.find("\"id\":" + request_id) != std::string::npos)
                {
                    response_body = body;
                }
            }
            Expect(!response_body.empty(), "request step response frame must be present");
            test::ExpectJsonFixture(
                response_body,
                step.response_fixture.c_str(),
                "fixture-driven session response must match golden fixture");
        }
    }
}

void TestClangdDerivedGoldenFixturesParse()
{
    lsp::ProtocolJsonHandler handler;

    test::ExpectParsesRequest(
        handler, td_hover::request::kMethodInfo, test::ReadFixture("hover_request.json").c_str(),
        "clangd-derived hover request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_references::request::kMethodInfo, test::ReadFixture("references_request.json").c_str(),
        "clangd-derived references request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_signatureHelp::request::kMethodInfo, test::ReadFixture("signature_help_request.json").c_str(),
        "clangd-derived signatureHelp request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_initialize::request::kMethodInfo, test::ReadFixture("initialize_root_uri_request.json").c_str(),
        "clangd-derived initialize rootUri request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_initialize::request::kMethodInfo,
        test::ReadFixture("initialize_position_encodings_request.json").c_str(),
        "clangd-derived initialize positionEncodings request fixture must parse");
    test::ExpectParsesRequest(
        handler, wp_symbol::request::kMethodInfo, test::ReadFixture("initialize_before_init_request.json").c_str(),
        "clangd-derived workspace/symbol request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_semanticTokens_full::request::kMethodInfo,
        test::ReadFixture("semantic_tokens_full_request.json").c_str(),
        "clangd-derived semanticTokens/full request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_rename::request::kMethodInfo, test::ReadFixture("rename_request.json").c_str(),
        "clangd-derived rename request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_codeAction::request::kMethodInfo, test::ReadFixture("code_action_request.json").c_str(),
        "clangd-derived codeAction request fixture must parse");
    test::ExpectParsesRequest(
        handler, td_shutdown::request::kMethodInfo, test::ReadFixture("shutdown_request.json").c_str(),
        "clangd-derived shutdown request fixture must parse");

    test::ExpectParsesResponse(
        handler, td_hover::request::kMethodInfo, test::ReadFixture("hover_markup_content_response.json").c_str(),
        "clangd-derived hover MarkupContent response fixture must parse");
    test::ExpectParsesResponse(
        handler, td_hover::request::kMethodInfo, test::ReadFixture("hover_null_response.json").c_str(),
        "clangd-derived hover null response fixture must parse");
    test::ExpectParsesResponse(
        handler, td_signatureHelp::request::kMethodInfo, test::ReadFixture("signature_help_response.json").c_str(),
        "clangd-derived signatureHelp response fixture must parse");
    test::ExpectParsesResponse(
        handler, td_references::request::kMethodInfo, test::ReadFixture("references_locations_response.json").c_str(),
        "clangd-derived references response fixture must parse");
    test::ExpectParsesResponse(
        handler, td_definition::request::kMethodInfo, test::ReadFixture("definition_locations_response.json").c_str(),
        "clangd-derived definition array response fixture must parse");
    test::ExpectParsesResponse(
        handler, td_semanticTokens_full::request::kMethodInfo,
        test::ReadFixture("semantic_tokens_full_response.json").c_str(),
        "clangd-derived semanticTokens/full response fixture must parse");

    test::ExpectParsesNotification(
        handler, Notify_WorkspaceDidChangeConfiguration::notify::kMethodInfo,
        test::ReadFixture("did_change_configuration_notification.json").c_str(),
        "clangd-derived didChangeConfiguration notification fixture must parse");

    auto const method_not_found = test::ParseProtocolResponse(
        handler, td_initialize::request::kMethodInfo,
        test::ReadFixture("method_not_found_error_response.json").c_str());
    Expect(method_not_found != nullptr, "method not found error fixture must parse");
    auto* not_found_error = dynamic_cast<Rsp_Error*>(method_not_found.get());
    Expect(not_found_error != nullptr, "method not found fixture must deserialize as Rsp_Error");
    if (not_found_error != nullptr)
    {
        Expect(
            not_found_error->error.code == lsErrorCodes::MethodNotFound,
            "method not found fixture must preserve -32601");
    }

    auto const not_initialized = test::ParseProtocolResponse(
        handler, td_initialize::request::kMethodInfo,
        test::ReadFixture("server_not_initialized_error_response.json").c_str());
    auto* init_error = dynamic_cast<Rsp_Error*>(not_initialized.get());
    Expect(init_error != nullptr, "server not initialized fixture must deserialize as Rsp_Error");
    if (init_error != nullptr)
    {
        Expect(
            init_error->error.code == lsErrorCodes::ServerNotInitialized,
            "server not initialized fixture must preserve -32002");
    }

    auto const invalid_request = test::ParseProtocolResponse(
        handler, td_initialize::request::kMethodInfo,
        test::ReadFixture("invalid_request_error_response.json").c_str());
    auto* invalid_error = dynamic_cast<Rsp_Error*>(invalid_request.get());
    Expect(invalid_error != nullptr, "invalid request fixture must deserialize as Rsp_Error");
    if (invalid_error != nullptr)
    {
        Expect(
            invalid_error->error.code == lsErrorCodes::InvalidRequest,
            "invalid request fixture must preserve -32600");
    }

    auto const invalid_params = test::ParseProtocolResponse(
        handler, td_initialize::request::kMethodInfo,
        test::ReadFixture("invalid_params_error_response.json").c_str());
    auto* params_error = dynamic_cast<Rsp_Error*>(invalid_params.get());
    Expect(params_error != nullptr, "invalid params fixture must deserialize as Rsp_Error");
    if (params_error != nullptr)
    {
        Expect(
            params_error->error.code == lsErrorCodes::InvalidParams,
            "invalid params fixture must preserve -32602");
    }

    auto const already_initialized = test::ParseProtocolResponse(
        handler, td_initialize::request::kMethodInfo,
        test::ReadFixture("already_initialized_error_response.json").c_str());
    auto* duplicate_init_error = dynamic_cast<Rsp_Error*>(already_initialized.get());
    Expect(duplicate_init_error != nullptr, "already initialized fixture must deserialize as Rsp_Error");
    if (duplicate_init_error != nullptr)
    {
        Expect(
            duplicate_init_error->error.code == lsErrorCodes::InvalidRequest,
            "already initialized fixture must preserve -32600");
    }
}

void TestFixtureDrivenSessionFromStepsManifest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<int> notification_count {0};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    RegisterFixtureSessionHandlers(point, notification_count);
    point.startProcessingMessages(input_stream, output_stream);
    RunFixtureDrivenSession("lsp_session_steps.json", input_stream, output_stream, notification_count);
    point.stop();
}

// Adapted from clangd/test/initialize-sequence.test
void TestFixtureDrivenLifecycleFromStepsManifest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<int> notification_count {0};
    std::atomic<bool> initialized {false};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    RegisterLifecycleSessionHandlers(point, initialized);
    point.startProcessingMessages(input_stream, output_stream);
    RunFixtureDrivenSession("lsp_lifecycle_steps.json", input_stream, output_stream, notification_count);
    point.stop();
}

void TestFixtureDrivenExtendedSessionFromStepsManifest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<int> notification_count {0};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    RegisterFixtureSessionHandlers(point, notification_count);
    point.startProcessingMessages(input_stream, output_stream);
    RunFixtureDrivenSession("lsp_extended_session_steps.json", input_stream, output_stream, notification_count);
    point.stop();
}

// Adapted from clangd/test/folding-range.test, selection-range.test, inlayHints.test
void TestFixtureDrivenFeatureSessionFromStepsManifest()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<int> notification_count {0};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    RegisterFeatureSessionHandlers(point, notification_count);
    point.startProcessingMessages(input_stream, output_stream);
    RunFixtureDrivenSession("lsp_feature_steps.json", input_stream, output_stream, notification_count);
    point.stop();
}

// Adapted from clangd/test/unsupported-method.test
void TestFixtureDrivenUnsupportedMethodSession()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<int> notification_count {0};

    RemoteEndPoint point(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 1);
    RegisterFixtureSessionHandlers(point, notification_count);
    point.startProcessingMessages(input_stream, output_stream);
    RunFixtureDrivenSession("lsp_error_steps.json", input_stream, output_stream, notification_count);
    point.stop();
}

void TestLanguageSessionRejectsUnsupportedMethod()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    session.on(
        [](td_initialize::request const& req) -> td_initialize::response
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(test::ReadFixture("initialize_request.json")));
    Expect(!WaitForOutputContaining(output, "\"id\":100").empty(), "initialize must succeed before unsupported method");

    input->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","id":111,"method":"textDocument/jumpInTheAirLikeYouJustDontCare","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"code\":-32601");
    session.stop();

    Expect(response.find("\"id\":111") != std::string::npos, "unsupported method response must preserve request id");
    Expect(response.find("\"code\":-32601") != std::string::npos, "unsupported method must return MethodNotFound");
    auto const bodies = test::ExtractLspFrameBodies(response);
    Expect(!bodies.empty(), "unsupported method response frame must be present");
    if (!bodies.empty())
    {
        rapidjson::Document document;
        document.Parse(bodies.back().c_str());
        Expect(!document.HasParseError(), "unsupported method response must be valid JSON");
        Expect(document.HasMember("error"), "unsupported method response must include error object");
        if (document.HasMember("error") && document["error"].IsObject())
        {
            Expect(
                document["error"]["code"].GetInt() == static_cast<int>(lsErrorCodes::MethodNotFound),
                "unsupported method must return MethodNotFound");
            Expect(document["error"]["message"].IsString(), "unsupported method error must include message");
        }
    }
}

void TestDelimitedSessionWithBlankLines()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log, lsp::JSONStreamStyle::Delimited);
    auto input = std::make_shared<StringIStream>(
        "\n\n"
        "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{}}\n"
        "\n\n"
        "// -----\n"
        "\n"
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"shutdown\",\"params\":null}\n"
        "// -----\n"
        "{\"jsonrpc\":\"2.0\",\"method\":\"exit\",\"params\":{}}\n");
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> initialized {false};
    std::atomic<bool> shutdown {false};
    std::atomic<bool> exited {false};

    session.on(
        [&](td_initialize::request const& req) -> td_initialize::response
        {
            initialized.store(true, std::memory_order_relaxed);
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });
    session.on(
        [&](td_shutdown::request const& req) -> td_shutdown::response
        {
            shutdown.store(true, std::memory_order_relaxed);
            return MakeNullShutdownResponse(req.id);
        });
    session.on(
        [&](Notify_Exit::notify const&)
        {
            exited.store(true, std::memory_order_relaxed);
        });
    session.start(input, output);

    Expect(WaitForOutputContaining(output, "\"id\":0").find("\"id\":0") != std::string::npos, "blank lines must not break initialize");
    Expect(WaitUntil([&] { return initialized.load(std::memory_order_relaxed); }), "initialize must run with blank delimited input");
    Expect(WaitUntil([&] { return shutdown.load(std::memory_order_relaxed); }), "shutdown must run with blank delimited input");
    session.stop();
}

void TestRequestErrorFixturesMatchClangdLifecycle()
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

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":112,"method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"code\":-32002");
    session.stop();

    auto const bodies = test::ExtractLspFrameBodies(response);
    Expect(!bodies.empty(), "RequestError response frame must be present");
    if (!bodies.empty())
    {
        test::ExpectJsonFixture(
            bodies.back(),
            "server_not_initialized_error_response.json",
            "RequestError response must match clangd server-not-initialized fixture");
    }
}

// Adapted from clangd/test/delimited-input-comment-at-the-end.test
void TestDelimitedSessionIgnoresCommentLines()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log, lsp::JSONStreamStyle::Delimited);
    auto input = std::make_shared<StringIStream>(
        "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{}}\n"
        "// -----\n"
        "// trailing comment must be ignored\n"
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"shutdown\",\"params\":null}\n"
        "// -----\n"
        "{\"jsonrpc\":\"2.0\",\"method\":\"exit\",\"params\":{}}\n");
    auto output = std::make_shared<StringOStream>();
    std::atomic<bool> shutdown {false};

    session.on([](td_initialize::request const& req) -> td_initialize::response
               {
                   td_initialize::response rsp;
                   rsp.id = req.id;
                   return rsp;
               });
    session.on(
        [&](td_shutdown::request const& req) -> td_shutdown::response
        {
            shutdown.store(true, std::memory_order_relaxed);
            return MakeNullShutdownResponse(req.id);
        });
    session.start(input, output);

    Expect(
        WaitForOutputContaining(output, "\"id\":0").find("\"id\":0") != std::string::npos,
        "comment lines must not break initialize");
    Expect(WaitUntil([&] { return shutdown.load(std::memory_order_relaxed); }), "comment lines must not break shutdown");
    session.stop();
}

void TestRequestInvalidParamsFixtureParse()
{
    lsp::NullLog log;
    lsp::LanguageSession session(log);
    auto input = std::make_shared<FeedableIStream>();
    auto output = std::make_shared<StringOStream>();

    session.on(
        [](td_initialize::request const&) -> td_initialize::response
        {
            throw lsp::RequestError(lsErrorCodes::InvalidParams, "invalid params");
        });
    session.start(input, output);

    input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":114,"method":"initialize","params":{}})"));
    std::string const response = WaitForOutputContaining(output, "\"code\":-32602");
    session.stop();

    auto const bodies = test::ExtractLspFrameBodies(response);
    Expect(!bodies.empty(), "InvalidParams response frame must be present");
    if (!bodies.empty())
    {
        test::ExpectJsonFixture(
            bodies.back(),
            "invalid_params_error_response.json",
            "InvalidParams response must match clangd-derived fixture");
    }
}

void TestDidChangeConfigurationNotificationRoundTrip()
{
    DummyLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<bool> handled {false};

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](Notify_WorkspaceDidChangeConfiguration::notify const& notify)
        {
            handled.store(true, std::memory_order_relaxed);
            Expect(
                !notify.params.settings.Data().empty(),
                "didChangeConfiguration settings must remain object-shaped");
        });
    point.startProcessingMessages(input_stream, output_stream);

    input_stream->append(test::MakeLspFrame(test::ReadFixture("did_change_configuration_notification.json")));
    Expect(WaitUntil([&] { return handled.load(std::memory_order_relaxed); }), "didChangeConfiguration must dispatch");
    point.stop();
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
    RUN_TEST(TestClangdDerivedGoldenFixturesParse);
    RUN_TEST(TestFixtureDrivenSessionFromStepsManifest);
    RUN_TEST(TestFixtureDrivenLifecycleFromStepsManifest);
    RUN_TEST(TestFixtureDrivenExtendedSessionFromStepsManifest);
    RUN_TEST(TestFixtureDrivenFeatureSessionFromStepsManifest);
    RUN_TEST(TestFixtureDrivenUnsupportedMethodSession);
    RUN_TEST(TestLanguageSessionRejectsUnsupportedMethod);
    RUN_TEST(TestDelimitedSessionWithBlankLines);
    RUN_TEST(TestDelimitedSessionIgnoresCommentLines);
    RUN_TEST(TestRequestErrorFixturesMatchClangdLifecycle);
    RUN_TEST(TestRequestInvalidParamsFixtureParse);
    RUN_TEST(TestDidChangeConfigurationNotificationRoundTrip);
    return test::Failures() == 0 ? 0 : 1;
}
