#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/progress.h"
#include "LibLsp/lsp/general/shutdown.h"
#include "LibLsp/lsp/textDocument/code_action.h"
#include "LibLsp/lsp/textDocument/completion.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include "LibLsp/lsp/textDocument/hover.h"
#include "LibLsp/lsp/textDocument/publishDiagnostics.h"
#include "LibLsp/lsp/textDocument/prepareRename.h"
#include "LibLsp/lsp/textDocument/rename.h"
#include "LibLsp/lsp/textDocument/selectionRange.h"
#include "LibLsp/lsp/textDocument/type_definition.h"
#include "LibLsp/lsp/workspace/applyEdit.h"
#include "LibLsp/lsp/protocol_3_18.h"
#include "protocol_test_helpers.h"
#include "test_helpers.h"

#include <fstream>
#include <iostream>
#include <map>
#include <rapidjson/document.h>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using test::Expect;

std::string JsonString(std::string const& value)
{
    std::string out = "\"";
    for (char c : value)
    {
        if (c == '"' || c == '\\')
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

char const* CommonProtocolParams()
{
    return R"({
        "textDocument":{"uri":"file:///a.cpp","languageId":"cpp","version":1,"text":"int main(){}"},
        "uri":"file:///a.cpp",
        "targetUri":"file:///target.cpp",
        "version":1,
        "position":{"line":0,"character":0},
        "range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},
        "ranges":[{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}],
        "context":{
            "triggerKind":1,
            "diagnostics":[],
            "includeDeclaration":true,
            "frameId":1,
            "stoppedLocation":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}
        },
        "contentChanges":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"text":"x"}],
        "diagnostics":[],
        "lines":[],
        "type":3,
        "message":"hello",
        "settings":{},
        "event":{"added":[],"removed":[]},
        "changes":[],
        "files":[{"oldUri":"file:///old.cpp","newUri":"file:///new.cpp"}],
        "registrations":[],
        "unregisterations":[],
        "items":[],
        "query":"symbol",
        "command":"cmd.fixture",
        "arguments":[],
        "newName":"renamed",
        "options":{"tabSize":4,"insertSpaces":true},
        "previousResultId":"old",
        "previousResultIds":[],
        "identifier":"cpp",
        "label":"hint",
        "insertText":"x",
        "title":"Fixture action",
        "kind":"quickfix",
        "edit":{"changes":{}},
        "data":{},
        "text":"text",
        "name":"symbol",
        "location":{"uri":"file:///a.cpp","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}},
        "eventType":1
    })";
}

std::string RequestEnvelope(std::string const& method, char const* params)
{
    return std::string(R"({"jsonrpc":"2.0","id":1,"method":)") + JsonString(method) + R"(,"params":)" + params + "}";
}

std::vector<char const*> RequestParamsForMethod(std::string const& method)
{
    if (method == "codeAction/resolve")
    {
        return {
            R"({"title":"Fixture action","command":{"title":"Run","command":"cmd.fixture"}})",
        };
    }
    return {CommonProtocolParams(), "{}", "null"};
}

std::string NotificationEnvelope(std::string const& method, char const* params)
{
    return std::string(R"({"jsonrpc":"2.0","method":)") + JsonString(method) + R"(,"params":)" + params + "}";
}

std::string ErrorResponseEnvelope()
{
    return R"({"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"Request failed"}})";
}

void ExpectParsed(bool parsed, char const* kind, std::string const& method)
{
    if (!parsed)
    {
        std::cerr << kind << " parser failed for method: " << method << std::endl;
    }
    Expect(parsed, "registered protocol parser must parse its guard fixture");
}

std::string ReadTextFile(std::vector<std::string> const& candidates)
{
    return test::ReadTextFile(candidates);
}

std::map<std::string, std::vector<std::string>> LoadAllowlist()
{
    std::string const json = ReadTextFile(
        {
            "tools/lsp-metamodel-allowlist.json",
            "../tools/lsp-metamodel-allowlist.json",
            "../../tools/lsp-metamodel-allowlist.json",
            "../../../tools/lsp-metamodel-allowlist.json",
        });
    Expect(!json.empty(), "protocol handler test must locate lsp metamodel allowlist");

    std::map<std::string, std::vector<std::string>> result;
    if (json.empty())
    {
        return result;
    }

    rapidjson::Document document;
    document.Parse(json.c_str());
    Expect(document.IsObject(), "lsp metamodel allowlist must be a JSON object");
    if (!document.IsObject())
    {
        return result;
    }

    for (auto const& member : document.GetObject())
    {
        if (!member.value.IsArray())
        {
            continue;
        }
        auto& values = result[member.name.GetString()];
        for (auto const& item : member.value.GetArray())
        {
            if (item.IsString())
            {
                values.push_back(item.GetString());
            }
        }
    }
    return result;
}

std::string ReadFixture(char const* name)
{
    return test::ReadFixture(name);
}

bool Contains(std::vector<std::string> const& values, std::string const& needle)
{
    for (auto const& value : values)
    {
        if (value == needle)
        {
            return true;
        }
    }
    return false;
}

void TestEveryRegisteredRequestParserIsUsable()
{
    lsp::ProtocolJsonHandler handler;

    Expect(!handler.method2request.empty(), "ProtocolJsonHandler must register request parsers");
    for (auto const& entry : handler.method2request)
    {
        bool parsed = false;
        for (char const* param : RequestParamsForMethod(entry.first))
        {
            try
            {
                std::string const json = RequestEnvelope(entry.first, param);
                parsed = test::ParseProtocolRequest(handler, entry.first.c_str(), json.c_str()) != nullptr;
            }
            catch (...)
            {
                parsed = false;
            }
            if (parsed)
            {
                break;
            }
        }
        ExpectParsed(parsed, "request", entry.first);
    }
}

void TestEveryRegisteredNotificationParserIsUsable()
{
    lsp::ProtocolJsonHandler handler;
    std::vector<char const*> const params = {CommonProtocolParams(), "{}", "null"};

    Expect(!handler.method2notification.empty(), "ProtocolJsonHandler must register notification parsers");
    for (auto const& entry : handler.method2notification)
    {
        bool parsed = false;
        for (char const* param : params)
        {
            try
            {
                std::string const json = NotificationEnvelope(entry.first, param);
                parsed = test::ParseProtocolNotification(handler, entry.first.c_str(), json.c_str()) != nullptr;
            }
            catch (...)
            {
                parsed = false;
            }
            if (parsed)
            {
                break;
            }
        }
        ExpectParsed(parsed, "notification", entry.first);
    }
}

void TestEveryRegisteredResponseParserIsUsable()
{
    lsp::ProtocolJsonHandler handler;
    std::string const error = ErrorResponseEnvelope();

    Expect(!handler.method2response.empty(), "ProtocolJsonHandler must register response parsers");
    for (auto const& entry : handler.method2response)
    {
        bool parsed = false;
        try
        {
            parsed = test::ParseProtocolResponse(handler, entry.first.c_str(), error.c_str()) != nullptr;
        }
        catch (...)
        {
            parsed = false;
        }
        ExpectParsed(parsed, "response", entry.first);
    }
}

void TestRegisteredMethodsAreNotMarkedMissingInAllowlist()
{
    lsp::ProtocolJsonHandler handler;
    auto const allowlist = LoadAllowlist();

    auto const missing_requests = allowlist.find("missing_request_parser");
    auto const missing_responses = allowlist.find("missing_response_parser");
    auto const missing_notifications = allowlist.find("missing_notification_parser");

    for (auto const& entry : handler.method2request)
    {
        Expect(
            missing_requests == allowlist.end() || !Contains(missing_requests->second, entry.first),
            "registered request parser must not stay listed as missing in allowlist");
    }
    for (auto const& entry : handler.method2response)
    {
        Expect(
            missing_responses == allowlist.end() || !Contains(missing_responses->second, entry.first),
            "registered response parser must not stay listed as missing in allowlist");
    }
    for (auto const& entry : handler.method2notification)
    {
        Expect(
            missing_notifications == allowlist.end() || !Contains(missing_notifications->second, entry.first),
            "registered notification parser must not stay listed as missing in allowlist");
    }
}

void TestProgressNotificationIsRegistered()
{
    lsp::ProtocolJsonHandler handler;
    std::unique_ptr<LspMessage> parsed = test::ParseProtocolNotification(
        handler,
        Notify_Progress::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"$/progress","params":{"token":"work","value":{"kind":"begin","title":"Index"}}})");

    auto* progress = dynamic_cast<Notify_Progress::notify*>(parsed.get());
    Expect(progress != nullptr, "$/progress must be registered as a standard notification");
    if (progress != nullptr)
    {
        Expect(progress->params.token.first && *progress->params.token.first == "work", "$/progress string token must parse");
        Expect(
            progress->params.value.Data().find(R"("kind":"begin")") != std::string::npos,
            "$/progress value payload must be preserved as Any JSON");
    }
}

void TestMalformedCoreRequestParamsDoNotEscape()
{
    lsp::ProtocolJsonHandler handler;
    std::vector<std::string> const malformed = {
        R"({"jsonrpc":"2.0","id":1,"method":"textDocument/completion","params":{"textDocument":{},"position":{},"context":{}}})",
        R"({"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{"textDocument":{},"position":{}}})",
        R"({"jsonrpc":"2.0","id":3,"method":"initialize","params":{"capabilities":{}}})",
    };

    for (auto const& json : malformed)
    {
        try
        {
            rapidjson::Document document;
            document.Parse(json.c_str());
            JsonReader reader(&document);
            auto const method = document["method"].GetString();
            (void)handler.parseRequstMessage(method, reader);
        }
        catch (...)
        {
            Expect(false, "malformed core request params must not escape protocol parser");
        }
    }
}

void TestJdtlsExtensionsAreOptIn()
{
    lsp::ProtocolJsonHandler standard_handler;

    Expect(
        standard_handler.GetResponseJsonHandler("java/classFileContents") == nullptr,
        "JDTLS response parsers must be disabled by default");
    Expect(
        standard_handler.GetRequestJsonHandler("java/didRenameFiles") == nullptr,
        "JDTLS request parsers must be disabled by default");
    Expect(
        standard_handler.GetNotificationJsonHandler("java/projectConfigurationUpdate") == nullptr,
        "JDTLS notification parsers must be disabled by default");

    lsp::ProtocolJsonHandlerOptions options;
    options.enableJdtlsExtensions = true;
    lsp::ProtocolJsonHandler jdtls_handler(options);

    Expect(
        jdtls_handler.GetResponseJsonHandler("java/classFileContents") != nullptr,
        "JDTLS response parsers must be registered when configured");
    Expect(
        jdtls_handler.GetRequestJsonHandler("java/didRenameFiles") != nullptr,
        "JDTLS request parsers must be registered when configured");
    Expect(
        jdtls_handler.GetNotificationJsonHandler("java/projectConfigurationUpdate") != nullptr,
        "JDTLS notification parsers must be registered when configured");

    Expect(
        standard_handler.GetRequestJsonHandler("workspace/configuration") != nullptr,
        "standard workspace/configuration client request parser must remain enabled");
    Expect(
        standard_handler.GetRequestJsonHandler("workspace/workspaceFolders") != nullptr,
        "standard workspace/workspaceFolders client request parser must remain enabled");
}

void TestExperimentalStandardRequestsAreOptIn()
{
    lsp::ProtocolJsonHandler standard_handler;
    Expect(
        standard_handler.GetRequestJsonHandler(td_prepareRename::request::kMethodInfo) == nullptr,
        "prepareRename request parser must stay disabled by default");
    Expect(
        standard_handler.GetRequestJsonHandler(WorkspaceApply::request::kMethodInfo) == nullptr,
        "workspace/applyEdit request parser must stay disabled by default");
    Expect(
        standard_handler.GetResponseJsonHandler(WorkspaceApply::request::kMethodInfo) == nullptr,
        "workspace/applyEdit response parser must stay disabled by default");

    lsp::ProtocolJsonHandlerOptions options;
    options.enableExperimentalStandardRequests = true;
    lsp::ProtocolJsonHandler experimental_handler(options);

    Expect(
        experimental_handler.GetRequestJsonHandler(td_prepareRename::request::kMethodInfo) != nullptr,
        "prepareRename request parser must register when configured");
    Expect(
        experimental_handler.GetRequestJsonHandler(td_selectionRange::request::kMethodInfo) != nullptr,
        "selectionRange request parser must register when configured");
    Expect(
        experimental_handler.GetRequestJsonHandler(td_typeDefinition::request::kMethodInfo) != nullptr,
        "typeDefinition request parser must register when configured");
    Expect(
        experimental_handler.GetRequestJsonHandler(WorkspaceApply::request::kMethodInfo) != nullptr,
        "workspace/applyEdit request parser must register when configured");
    Expect(
        experimental_handler.GetResponseJsonHandler(WorkspaceApply::request::kMethodInfo) != nullptr,
        "workspace/applyEdit response parser must register when configured");
}

void TestServerRefreshRequestsAreOptIn()
{
    lsp::ProtocolJsonHandler standard_handler;
    Expect(
        standard_handler.GetRequestJsonHandler(workspace_semanticTokens_refresh::request::kMethodInfo) == nullptr,
        "semanticTokens refresh request parser must stay disabled by default");
    Expect(
        standard_handler.GetResponseJsonHandler(workspace_inlayHint_refresh::request::kMethodInfo) == nullptr,
        "inlayHint refresh response parser must stay disabled by default");

    lsp::ProtocolJsonHandlerOptions options;
    options.enableServerRefreshRequests = true;
    lsp::ProtocolJsonHandler refresh_handler(options);

    Expect(
        refresh_handler.GetRequestJsonHandler(workspace_semanticTokens_refresh::request::kMethodInfo) != nullptr,
        "semanticTokens refresh request parser must register when configured");
    Expect(
        refresh_handler.GetResponseJsonHandler(workspace_diagnostic_refresh::request::kMethodInfo) != nullptr,
        "diagnostic refresh response parser must register when configured");
}

void TestGoldenLspFixturesParse()
{
    lsp::ProtocolJsonHandler handler;

    std::string const initialize = ReadFixture("initialize_request.json");
    Expect(
        test::ParseProtocolRequest(handler, td_initialize::request::kMethodInfo, initialize.c_str()) != nullptr,
        "golden initialize request fixture must parse");

    std::string const did_change = ReadFixture("did_change_notification.json");
    Expect(
        test::ParseProtocolNotification(
            handler, Notify_TextDocumentDidChange::notify::kMethodInfo, did_change.c_str()) != nullptr,
        "golden didChange notification fixture must parse");

    std::string const did_open = ReadFixture("did_open_notification.json");
    Expect(
        test::ParseProtocolNotification(
            handler, Notify_TextDocumentDidOpen::notify::kMethodInfo, did_open.c_str()) != nullptr,
        "golden didOpen notification fixture must parse");

    std::string const completion_request = ReadFixture("completion_request.json");
    Expect(
        test::ParseProtocolRequest(handler, td_completion::request::kMethodInfo, completion_request.c_str()) !=
            nullptr,
        "golden completion request fixture must parse");

    std::string const diagnostics = ReadFixture("publish_diagnostics_notification.json");
    Expect(
        test::ParseProtocolNotification(
            handler, Notify_TextDocumentPublishDiagnostics::notify::kMethodInfo, diagnostics.c_str()) != nullptr,
        "golden publishDiagnostics notification fixture must parse");

    std::string const hover = ReadFixture("hover_marked_string_response.json");
    Expect(
        test::ParseProtocolResponse(handler, td_hover::request::kMethodInfo, hover.c_str()) != nullptr,
        "golden hover response fixture must parse");

    std::string const code_action = ReadFixture("code_action_title_response.json");
    Expect(
        test::ParseProtocolResponse(handler, td_codeAction::request::kMethodInfo, code_action.c_str()) != nullptr,
        "golden codeAction response fixture must parse");

    std::string const completion = ReadFixture("completion_list_response.json");
    Expect(
        test::ParseProtocolResponse(handler, td_completion::request::kMethodInfo, completion.c_str()) != nullptr,
        "golden completion response fixture must parse");

    std::string const definition = ReadFixture("definition_single_location_response.json");
    Expect(
        test::ParseProtocolResponse(handler, td_definition::request::kMethodInfo, definition.c_str()) != nullptr,
        "golden single-location definition response fixture must parse");

    std::string const rename = ReadFixture("rename_workspace_edit_response.json");
    Expect(
        test::ParseProtocolResponse(handler, td_rename::request::kMethodInfo, rename.c_str()) != nullptr,
        "golden rename workspace edit response fixture must parse");

    std::string const shutdown = ReadFixture("shutdown_null_response.json");
    Expect(
        test::ParseProtocolResponse(handler, td_shutdown::request::kMethodInfo, shutdown.c_str()) != nullptr,
        "golden shutdown null response fixture must parse");
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
    RUN_TEST(TestEveryRegisteredRequestParserIsUsable);
    RUN_TEST(TestEveryRegisteredNotificationParserIsUsable);
    RUN_TEST(TestEveryRegisteredResponseParserIsUsable);
    RUN_TEST(TestRegisteredMethodsAreNotMarkedMissingInAllowlist);
    RUN_TEST(TestProgressNotificationIsRegistered);
    RUN_TEST(TestMalformedCoreRequestParamsDoNotEscape);
    RUN_TEST(TestJdtlsExtensionsAreOptIn);
    RUN_TEST(TestExperimentalStandardRequestsAreOptIn);
    RUN_TEST(TestServerRefreshRequestsAreOptIn);
    RUN_TEST(TestGoldenLspFixturesParse);

    return test::Failures() == 0 ? 0 : 1;
}
