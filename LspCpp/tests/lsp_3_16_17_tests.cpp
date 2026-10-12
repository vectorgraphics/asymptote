#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/ResourceOperation.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/lsTextDocumentClientCapabilities.h"
#include "LibLsp/lsp/general/lsWorkspaceClientCapabilites.h"
#include "LibLsp/lsp/lsp_diagnostic.h"
#include "LibLsp/lsp/lsTextEdit.h"
#include "LibLsp/lsp/lsWorkspaceEdit.h"
#include "LibLsp/lsp/textDocument/SemanticTokens.h"
#include "LibLsp/lsp/textDocument/callHierarchy.h"
#include "LibLsp/lsp/textDocument/foldingRange.h"
#include "LibLsp/lsp/textDocument/inlayHint.h"
#include "LibLsp/lsp/textDocument/linkedEditingRange.h"
#include "LibLsp/lsp/textDocument/resolveTypeHierarchy.h"
#include "LibLsp/lsp/textDocument/selectionRange.h"
#include "LibLsp/lsp/textDocument/typeHierarchy.h"
#include "LibLsp/lsp/textDocument/hover.h"
#include "LibLsp/lsp/textDocument/completion.h"
#include "LibLsp/lsp/textDocument/code_lens.h"
#include "LibLsp/lsp/textDocument/code_action.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
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

lsDocumentUri MakeDocumentUri(char const* uri)
{
    lsDocumentUri document_uri;
    document_uri.raw_uri_ = uri;
    return document_uri;
}

void TestProtocolJsonHandlerRegisters316317Requests()
{
    lsp::ProtocolJsonHandler handler;

    rapidjson::Document semantic_document;
    JsonReader semantic_reader = MakeReader(
        semantic_document,
        R"({"jsonrpc":"2.0","id":1,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///a.cpp"}}})");
    Expect(
        handler.parseRequstMessage(td_semanticTokens_full::request::kMethodInfo, semantic_reader) != nullptr,
        "default handler must register textDocument/semanticTokens/full requests");

    rapidjson::Document semantic_delta_document;
    JsonReader semantic_delta_reader = MakeReader(
        semantic_delta_document,
        R"({"jsonrpc":"2.0","id":2,"method":"textDocument/semanticTokens/full/delta","params":{"textDocument":{"uri":"file:///a.cpp"},"previousResultId":"prev-1"}})");
    Expect(
        handler.parseRequstMessage(td_semanticTokens_full_delta::request::kMethodInfo, semantic_delta_reader) != nullptr,
        "default handler must register textDocument/semanticTokens/full/delta requests");

    rapidjson::Document inlay_document;
    JsonReader inlay_reader = MakeReader(
        inlay_document,
        R"({"jsonrpc":"2.0","id":3,"method":"textDocument/inlayHint","params":{"textDocument":{"uri":"file:///a.cpp"},"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}})");
    Expect(
        handler.parseRequstMessage(td_inlayHint::request::kMethodInfo, inlay_reader) != nullptr,
        "default handler must register textDocument/inlayHint requests");

    rapidjson::Document inlay_resolve_document;
    JsonReader inlay_resolve_reader = MakeReader(
        inlay_resolve_document,
        R"({"jsonrpc":"2.0","id":4,"method":"inlayHint/resolve","params":{"position":{"line":1,"character":2},"label":"x"}})");
    Expect(
        handler.parseRequstMessage(td_inlayHintResolve::request::kMethodInfo, inlay_resolve_reader) != nullptr,
        "default handler must register inlayHint/resolve requests");

    ExpectParsesRequest(
        handler,
        td_codeLens::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":5,"method":"textDocument/codeLens","params":{"textDocument":{"uri":"file:///a.cpp"}}})",
        "default handler must parse textDocument/codeLens requests");
    ExpectParsesRequest(
        handler,
        td_declaration::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":6,"method":"textDocument/declaration","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":0,"character":0}}})",
        "default handler must parse textDocument/declaration requests");
    ExpectParsesRequest(
        handler,
        td_foldingRange::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":7,"method":"textDocument/foldingRange","params":{"textDocument":{"uri":"file:///a.cpp"}}})",
        "default handler must parse textDocument/foldingRange requests");
}

void TestProtocolJsonHandlerParses316317Responses()
{
    lsp::ProtocolJsonHandler handler;

    rapidjson::Document folding_document;
    JsonReader folding_reader = MakeReader(
        folding_document,
        R"({"jsonrpc":"2.0","id":10,"result":[{"startLine":0,"endLine":2}]})");
    Expect(
        handler.parseResponseMessage(td_foldingRange::request::kMethodInfo, folding_reader) != nullptr,
        "default handler must parse textDocument/foldingRange responses");

    rapidjson::Document selection_document;
    JsonReader selection_reader = MakeReader(
        selection_document,
        R"({"jsonrpc":"2.0","id":11,"result":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":5}}}]})");
    Expect(
        handler.parseResponseMessage(td_selectionRange::request::kMethodInfo, selection_reader) != nullptr,
        "default handler must parse textDocument/selectionRange responses");

    rapidjson::Document type_hierarchy_document;
    JsonReader type_hierarchy_reader = MakeReader(
        type_hierarchy_document,
        R"({"jsonrpc":"2.0","id":12,"result":{"name":"Base","kind":5,"uri":"file:///a.cpp","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":4}},"selectionRange":{"start":{"line":0,"character":0},"end":{"line":0,"character":4}}}})");
    Expect(
        handler.parseResponseMessage(td_typeHierarchy::request::kMethodInfo, type_hierarchy_reader) != nullptr,
        "default handler must parse textDocument/typeHierarchy responses");

    rapidjson::Document resolve_document;
    JsonReader resolve_reader = MakeReader(
        resolve_document,
        R"({"jsonrpc":"2.0","id":13,"result":{"name":"Derived","kind":5,"uri":"file:///a.cpp","range":{"start":{"line":1,"character":0},"end":{"line":1,"character":7}},"selectionRange":{"start":{"line":1,"character":0},"end":{"line":1,"character":7}}}})");
    Expect(
        handler.parseResponseMessage(typeHierarchy_resolve::request::kMethodInfo, resolve_reader) != nullptr,
        "default handler must parse typeHierarchy/resolve responses");

    ExpectParsesResponse(
        handler,
        td_semanticTokens_full::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":14,"result":{"data":[0,1,2,3,4]}})",
        "default handler must parse textDocument/semanticTokens/full responses");
    ExpectParsesResponse(
        handler,
        td_inlayHint::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":15,"result":[{"position":{"line":0,"character":0},"label":"hint"}]})",
        "default handler must parse textDocument/inlayHint responses");
    ExpectParsesResponse(
        handler,
        td_codeLens::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":16,"result":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}]})",
        "default handler must parse textDocument/codeLens responses");
    ExpectParsesResponse(
        handler,
        td_completion::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":17,"result":{"isIncomplete":false,"items":[]}})",
        "default handler must parse textDocument/completion responses");
    ExpectParsesResponse(
        handler,
        td_hover::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":18,"result":{"contents":"doc"}})",
        "default handler must parse textDocument/hover responses");
    ExpectParsesResponse(
        handler,
        td_codeAction::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":19,"result":[{"title":"Fix","command":"cmd.fix"}]})",
        "default handler must parse textDocument/codeAction responses");
    ExpectParsesResponse(
        handler,
        td_declaration::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":20,"result":[{"uri":"file:///a.cpp","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}]})",
        "default handler must parse textDocument/declaration responses");
}

void Test316ModelsSerializeExpectedFields()
{
    lsChangeAnnotation annotation;
    annotation.label = "Rename symbol";
    annotation.needsConfirmation = true;
    annotation.description = "Updates all references";
    std::string const annotation_json = SerializeJson(annotation);
    Expect(annotation_json.find("\"label\":\"Rename symbol\"") != std::string::npos, "change annotation must serialize label");
    Expect(
        annotation_json.find("\"needsConfirmation\":true") != std::string::npos,
        "change annotation must serialize needsConfirmation");

    lsWorkspaceEdit workspace_edit;
    workspace_edit.changeAnnotations = lsChangeAnnotations();
    workspace_edit.changeAnnotations->id.label = "Grouped edit";
    std::string const workspace_edit_json = SerializeJson(workspace_edit);
    Expect(
        workspace_edit_json.find("\"changeAnnotations\"") != std::string::npos,
        "workspace edit must serialize changeAnnotations");

    lsTextEdit text_edit;
    text_edit.range.start.line = 0;
    text_edit.range.start.character = 0;
    text_edit.range.end.line = 0;
    text_edit.range.end.character = 1;
    text_edit.newText = "x";
    text_edit.annotationId = std::string("edit-1");
    Expect(
        SerializeJson(text_edit).find("\"annotationId\":\"edit-1\"") != std::string::npos,
        "text edit must serialize annotationId");

    lsDiagnostic diagnostic;
    diagnostic.range.start.line = 1;
    diagnostic.range.start.character = 2;
    diagnostic.range.end.line = 1;
    diagnostic.range.end.character = 5;
    diagnostic.message = "unused variable";
    diagnostic.codeDescription = DiagnosticCodeDescription();
    diagnostic.codeDescription->href = "https://example.com/diag";
    diagnostic.data = lsp::Any();
    diagnostic.data->SetJsonString(R"({"owner":"test"})", lsp::Any::kObjectType);
    std::string const diagnostic_json = SerializeJson(diagnostic);
    Expect(
        diagnostic_json.find("\"codeDescription\"") != std::string::npos,
        "diagnostic must serialize codeDescription");
    Expect(diagnostic_json.find("\"data\"") != std::string::npos, "diagnostic must serialize data");

    SemanticTokens tokens;
    tokens.resultId = std::string("tok-42");
    tokens.data = {0, 1, 2, 3, 4};
    std::string const tokens_json = SerializeJson(tokens);
    Expect(tokens_json.find("\"resultId\":\"tok-42\"") != std::string::npos, "semantic tokens must serialize resultId");
    Expect(tokens_json.find("\"data\"") != std::string::npos, "semantic tokens must serialize data");

    SemanticTokensDeltaParams delta_params;
    delta_params.textDocument.uri = MakeDocumentUri("file:///a.cpp");
    delta_params.previousResultId = "tok-42";
    Expect(
        SerializeJson(delta_params).find("\"previousResultId\":\"tok-42\"") != std::string::npos,
        "semantic tokens delta params must serialize previousResultId");

    CallHierarchyItem call_item;
    call_item.name = "foo";
    call_item.kind = SymbolKind::Function;
    call_item.uri = MakeDocumentUri("file:///a.cpp");
    call_item.range.start.line = 0;
    call_item.range.end.line = 0;
    call_item.selectionRange.start.line = 0;
    call_item.selectionRange.end.line = 0;
    call_item.tags = std::vector<SymbolTag>{SymbolTag::Deprecated};
    Expect(
        SerializeJson(call_item).find("\"tags\"") != std::string::npos,
        "call hierarchy item must serialize tags");

    LinkedEditingRanges linked_ranges;
    linked_ranges.ranges.push_back(lsRange());
    linked_ranges.wordPattern = std::string("[a-z]+");
    Expect(
        SerializeJson(linked_ranges).find("\"wordPattern\":\"[a-z]+\"") != std::string::npos,
        "linked editing ranges must serialize wordPattern");

    FoldingRange folding_range;
    folding_range.startLine = 0;
    folding_range.endLine = 3;
    folding_range.kind = "comment";
    Expect(
        SerializeJson(folding_range).find("\"kind\":\"comment\"") != std::string::npos,
        "folding range must serialize kind");

    lsCreateFile create_file;
    create_file.uri = MakeDocumentUri("file:///new.cpp");
    create_file.annotationId = std::string("create-1");
    Expect(
        SerializeJson(create_file).find("\"annotationId\":\"create-1\"") != std::string::npos,
        "create file operation must serialize annotationId");
}

void Test317ModelsSerializeExpectedFields()
{
    lsInlayHintLabelPart label_part;
    label_part.value = "count";
    label_part.tooltip = std::string("parameter name");
    Expect(
        SerializeJson(label_part).find("\"tooltip\":\"parameter name\"") != std::string::npos,
        "inlay hint label part must serialize tooltip");

    lsInlayHint hint;
    hint.position.line = 2;
    hint.position.character = 4;
    hint.label = "count";
    hint.kind = lsInlayHintKind::Parameter;
    hint.paddingLeft = true;
    hint.paddingRight = false;
    std::string const hint_json = SerializeJson(hint);
    Expect(hint_json.find("\"kind\":2") != std::string::npos, "inlay hint must serialize kind");
    Expect(hint_json.find("\"paddingLeft\":true") != std::string::npos, "inlay hint must serialize paddingLeft");
    Expect(hint_json.find("\"paddingRight\":false") != std::string::npos, "inlay hint must serialize paddingRight");
}

void TestCapabilitiesSerialize316317Fields()
{
    lsServerCapabilities server_capabilities;
    server_capabilities.semanticTokensProvider = SemanticTokensWithRegistrationOptions();
    server_capabilities.semanticTokensProvider->legend.tokenTypes.push_back("function");
    server_capabilities.semanticTokensProvider->legend.tokenModifiers.push_back("deprecated");
    server_capabilities.callHierarchyProvider =
        std::make_pair(optional<bool>(true), optional<StaticRegistrationOptions>());
    server_capabilities.linkedEditingRangeProvider =
        std::make_pair(optional<bool>(true), optional<StaticRegistrationOptions>());
    InlayHintOptions inlay_hint_options;
    inlay_hint_options.resolveProvider = true;
    server_capabilities.inlayHintProvider =
        std::make_pair(optional<bool>(), optional<InlayHintOptions>(inlay_hint_options));
    server_capabilities.workspace = WorkspaceServerCapabilities();
    server_capabilities.workspace->fileOperations = WorkspaceServerCapabilities::lsFileOperations();
    server_capabilities.workspace->fileOperations->willCreate = lsFileOperationRegistrationOptions();

    std::string const server_json = SerializeJson(server_capabilities);
    Expect(
        server_json.find("\"semanticTokensProvider\"") != std::string::npos,
        "server capabilities must expose semanticTokensProvider");
    Expect(
        server_json.find("\"callHierarchyProvider\"") != std::string::npos,
        "server capabilities must expose callHierarchyProvider");
    Expect(
        server_json.find("\"linkedEditingRangeProvider\"") != std::string::npos,
        "server capabilities must expose linkedEditingRangeProvider");
    Expect(server_json.find("\"inlayHintProvider\"") != std::string::npos, "server capabilities must expose inlayHintProvider");
    Expect(server_json.find("\"resolveProvider\":true") != std::string::npos, "inlay hint provider must expose resolveProvider");
    Expect(server_json.find("\"fileOperations\"") != std::string::npos, "workspace capabilities must expose fileOperations");

    lsTextDocumentClientCapabilities text_document_capabilities;
    text_document_capabilities.semanticTokens = SemanticTokensClientCapabilities();
    text_document_capabilities.semanticTokens->tokenTypes.push_back("variable");
    text_document_capabilities.semanticTokens->multilineTokenSupport = true;
    text_document_capabilities.inlayHint = InlayHintClientCapabilities();
    text_document_capabilities.inlayHint->resolveSupport = InlayHintLazyProperties();
    text_document_capabilities.inlayHint->resolveSupport->properties = std::vector<std::string>{"tooltip"};
    text_document_capabilities.foldingRange = FoldingRangeCapabilities();
    text_document_capabilities.foldingRange->lineFoldingOnly = true;
    text_document_capabilities.publishDiagnostics = PublishDiagnosticsClientCapabilities();
    text_document_capabilities.publishDiagnostics->codeDescriptionSupport = true;
    text_document_capabilities.publishDiagnostics->dataSupport = true;

    std::string const client_json = SerializeJson(text_document_capabilities);
    Expect(client_json.find("\"semanticTokens\"") != std::string::npos, "text document capabilities must expose semanticTokens");
    Expect(
        client_json.find("\"multilineTokenSupport\":true") != std::string::npos,
        "semantic tokens client capabilities must expose multilineTokenSupport");
    Expect(client_json.find("\"inlayHint\"") != std::string::npos, "text document capabilities must expose inlayHint");
    Expect(client_json.find("\"resolveSupport\"") != std::string::npos, "inlay hint client capabilities must expose resolveSupport");
    Expect(client_json.find("\"lineFoldingOnly\":true") != std::string::npos, "folding range capabilities must expose lineFoldingOnly");
    Expect(
        client_json.find("\"codeDescriptionSupport\":true") != std::string::npos,
        "publish diagnostics capabilities must expose codeDescriptionSupport");
    Expect(
        client_json.find("\"dataSupport\":true") != std::string::npos,
        "publish diagnostics capabilities must expose dataSupport");
}

void TestClangdDerived316RequestFixturesParse()
{
    lsp::ProtocolJsonHandler handler;

    test::ExpectParsesRequest(
        handler, td_foldingRange::request::kMethodInfo, test::ReadFixture("folding_range_request.json").c_str(),
        "clangd-derived foldingRange request fixture must parse");

    rapidjson::Document selection_document;
    selection_document.Parse(test::ReadFixture("selection_range_request.json").c_str());
    JsonReader selection_reader(&selection_document);
    Expect(
        td_selectionRange::request::ReflectReader(selection_reader) != nullptr,
        "clangd-derived selectionRange request fixture must parse");
}
} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestProtocolJsonHandlerRegisters316317Requests);
    RUN_TEST(TestProtocolJsonHandlerParses316317Responses);
    RUN_TEST(Test316ModelsSerializeExpectedFields);
    RUN_TEST(Test317ModelsSerializeExpectedFields);
    RUN_TEST(TestCapabilitiesSerialize316317Fields);
    RUN_TEST(TestClangdDerived316RequestFixturesParse);

    return test::Failures() == 0 ? 0 : 1;
}
