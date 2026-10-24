#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/lsp_completion.h"
#include "LibLsp/lsp/protocol_3_18.h"
#include "LibLsp/lsp/textDocument/SemanticTokens.h"
#include "LibLsp/lsp/textDocument/inlayHint.h"
#include "LibLsp/lsp/textDocument/signature_help.h"
#include "LibLsp/lsp/workspace/applyEdit.h"
#include "protocol_test_helpers.h"
#include "test_helpers.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <string>

namespace
{
using test::Expect;
using test::ExpectParsesRequest;
using test::ExpectParsesResponse;

template<typename T>
std::string SerializeJson(T value)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    JsonWriter json_writer(&writer);
    Reflect(json_writer, value);
    return buffer.GetString();
}

JsonReader MakeReader(rapidjson::Document& document, char const* json)
{
    document.Parse(json);
    return JsonReader(&document);
}

void TestProtocolJsonHandlerRegistersExistingFeatureRequests()
{
    lsp::ProtocolJsonHandler handler;

    rapidjson::Document inlay_document;
    JsonReader inlay_reader = MakeReader(
        inlay_document,
        R"({"jsonrpc":"2.0","id":1,"method":"textDocument/inlayHint","params":{"textDocument":{"uri":"file:///a.cpp"},"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}})");
    Expect(
        handler.parseRequstMessage(td_inlayHint::request::kMethodInfo, inlay_reader) != nullptr,
        "default handler must register textDocument/inlayHint requests");

    rapidjson::Document semantic_document;
    JsonReader semantic_reader = MakeReader(
        semantic_document,
        R"({"jsonrpc":"2.0","id":2,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///a.cpp"}}})");
    Expect(
        handler.parseRequstMessage(td_semanticTokens_full::request::kMethodInfo, semantic_reader) != nullptr,
        "default handler must register textDocument/semanticTokens/full requests");
}

void TestProtocolJsonHandlerRegistersNew318Requests()
{
    lsp::ProtocolJsonHandler handler;

    rapidjson::Document diagnostic_document;
    JsonReader diagnostic_reader = MakeReader(
        diagnostic_document,
        R"({"jsonrpc":"2.0","id":3,"method":"textDocument/diagnostic","params":{"textDocument":{"uri":"file:///a.cpp"},"previousResultId":"old"}})");
    Expect(
        handler.parseRequstMessage(td_diagnostic::request::kMethodInfo, diagnostic_reader) != nullptr,
        "default handler must register textDocument/diagnostic requests");

    rapidjson::Document inline_document;
    JsonReader inline_reader = MakeReader(
        inline_document,
        R"({"jsonrpc":"2.0","id":4,"method":"textDocument/inlineCompletion","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2},"context":{"triggerKind":1}}})");
    Expect(
        handler.parseRequstMessage(td_inlineCompletion::request::kMethodInfo, inline_reader) != nullptr,
        "default handler must register textDocument/inlineCompletion requests");

    ExpectParsesRequest(
        handler,
        td_inlineValue::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":8,"method":"textDocument/inlineValue","params":{"textDocument":{"uri":"file:///a.cpp"},"range":{"start":{"line":0,"character":0},"end":{"line":1,"character":0}},"context":{"frameId":1,"stoppedLocation":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}}})",
        "default handler must parse textDocument/inlineValue requests");

    rapidjson::Document ranges_document;
    JsonReader ranges_reader = MakeReader(
        ranges_document,
        R"({"jsonrpc":"2.0","id":5,"method":"textDocument/rangesFormatting","params":{"textDocument":{"uri":"file:///a.cpp"},"ranges":[{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}],"options":{"tabSize":4,"insertSpaces":true}}})");
    Expect(
        handler.parseRequstMessage(td_rangesFormatting::request::kMethodInfo, ranges_reader) != nullptr,
        "default handler must register textDocument/rangesFormatting requests");

    rapidjson::Document content_document;
    JsonReader content_reader = MakeReader(
        content_document,
        R"({"jsonrpc":"2.0","id":6,"method":"workspace/textDocumentContent","params":{"uri":"lspcpp:///generated.cpp"}})");
    Expect(
        handler.parseRequstMessage(workspace_textDocumentContent::request::kMethodInfo, content_reader) != nullptr,
        "default handler must register workspace/textDocumentContent requests");

    ExpectParsesRequest(
        handler,
        workspace_diagnostic::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":7,"method":"workspace/diagnostic","params":{"identifier":"cpp"}})",
        "default handler must parse workspace/diagnostic requests");
    ExpectParsesRequest(
        handler,
        workspace_textDocumentContent_refresh::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":9,"method":"workspace/textDocumentContent/refresh","params":null})",
        "default handler must parse workspace/textDocumentContent/refresh requests");
    ExpectParsesRequest(
        handler,
        workspace_inlineValue_refresh::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":10,"method":"workspace/inlineValue/refresh","params":null})",
        "default handler must parse workspace/inlineValue/refresh requests");
    ExpectParsesRequest(
        handler,
        workspace_foldingRange_refresh::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":11,"method":"workspace/foldingRange/refresh","params":null})",
        "default handler must parse workspace/foldingRange/refresh requests");
}

void TestProtocolJsonHandlerParses318Responses()
{
    lsp::ProtocolJsonHandler handler;

    ExpectParsesResponse(
        handler,
        td_diagnostic::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":30,"result":{"kind":"full","items":[]}})",
        "default handler must parse textDocument/diagnostic responses");
    ExpectParsesResponse(
        handler,
        workspace_diagnostic::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":31,"result":{"items":[]}})",
        "default handler must parse workspace/diagnostic responses");
    ExpectParsesResponse(
        handler,
        td_inlineCompletion::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":32,"result":{"items":[{"insertText":"x"}]}})",
        "default handler must parse textDocument/inlineCompletion responses");
    ExpectParsesResponse(
        handler,
        td_inlineValue::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":33,"result":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"text":"value"}]})",
        "default handler must parse textDocument/inlineValue responses");
    ExpectParsesResponse(
        handler,
        workspace_textDocumentContent_refresh::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":34,"result":null})",
        "default handler must parse workspace/textDocumentContent/refresh responses");
}

void TestNew318ModelsSerializeExpectedFields()
{
    InlineCompletionList list;
    InlineCompletionItem item;
    item.insertText = "hello";
    list.items.push_back(item);
    std::string const inline_json = SerializeJson(list);
    Expect(inline_json.find("\"items\"") != std::string::npos, "inline completion list must serialize items");
    Expect(inline_json.find("\"insertText\":\"hello\"") != std::string::npos, "inline completion item must serialize insertText");

    DiagnosticOptions diagnostic_options;
    diagnostic_options.identifier = std::string("cpp");
    diagnostic_options.interFileDependencies = true;
    diagnostic_options.workspaceDiagnostics = true;
    std::string const diagnostic_json = SerializeJson(diagnostic_options);
    Expect(
        diagnostic_json.find("\"interFileDependencies\":true") != std::string::npos,
        "diagnostic options must serialize interFileDependencies");
    Expect(
        diagnostic_json.find("\"workspaceDiagnostics\":true") != std::string::npos,
        "diagnostic options must serialize workspaceDiagnostics");

    TextDocumentContentResult content_result;
    content_result.text = "generated text";
    Expect(
        SerializeJson(content_result).find("\"text\":\"generated text\"") != std::string::npos,
        "text document content result must serialize text");
}

void TestCapabilitiesSerialize318Fields()
{
    lsServerCapabilities capabilities;
    capabilities.diagnosticProvider = DiagnosticOptions();
    capabilities.diagnosticProvider->workspaceDiagnostics = true;
    capabilities.inlineCompletionProvider = std::make_pair(optional<bool>(true), optional<InlineCompletionOptions>());
    capabilities.inlineValueProvider = std::make_pair(optional<bool>(true), optional<InlineValueOptions>());
    DocumentRangeFormattingOptions range_options;
    range_options.rangesSupport = true;
    capabilities.documentRangeFormattingProvider =
        std::make_pair(optional<bool>(), optional<DocumentRangeFormattingOptions>(range_options));
    capabilities.workspace = WorkspaceServerCapabilities();
    capabilities.workspace->textDocumentContentProvider = TextDocumentContentOptions();
    capabilities.workspace->textDocumentContentProvider->schemes.push_back("lspcpp");

    std::string const json = SerializeJson(capabilities);
    Expect(json.find("\"diagnosticProvider\"") != std::string::npos, "server capabilities must expose diagnosticProvider");
    Expect(
        json.find("\"inlineCompletionProvider\"") != std::string::npos,
        "server capabilities must expose inlineCompletionProvider");
    Expect(json.find("\"inlineValueProvider\"") != std::string::npos, "server capabilities must expose inlineValueProvider");
    Expect(json.find("\"rangesSupport\":true") != std::string::npos, "range formatting provider must expose rangesSupport");
    Expect(
        json.find("\"textDocumentContentProvider\"") != std::string::npos,
        "workspace server capabilities must expose textDocumentContentProvider");
}

void TestExisting318FieldExtensionsSerialize()
{
    CompletionList completions;
    completions.applyKind = 1;
    Expect(
        SerializeJson(completions).find("\"applyKind\":1") != std::string::npos,
        "completion list must serialize applyKind");

    lsSignatureInformation signature;
    signature.label = "fn";
    signature.activeParameter = 0;
    Expect(
        SerializeJson(signature).find("\"activeParameter\":0") != std::string::npos,
        "signature information must serialize activeParameter");

    ApplyWorkspaceEditParams apply_edit;
    apply_edit.label = "refactor";
    apply_edit.metadata = lsWorkspaceEditMetadata();
    apply_edit.metadata->isRefactoring = true;
    Expect(
        SerializeJson(apply_edit).find("\"isRefactoring\":true") != std::string::npos,
        "apply edit params must serialize workspace edit metadata");
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestProtocolJsonHandlerRegistersExistingFeatureRequests);
    RUN_TEST(TestProtocolJsonHandlerRegistersNew318Requests);
    RUN_TEST(TestProtocolJsonHandlerParses318Responses);
    RUN_TEST(TestNew318ModelsSerializeExpectedFields);
    RUN_TEST(TestCapabilitiesSerialize318Fields);
    RUN_TEST(TestExisting318FieldExtensionsSerialize);

    return test::Failures() == 0 ? 0 : 1;
}
