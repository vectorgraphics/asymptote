#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/AbsolutePath.h"
#include "LibLsp/lsp/Directory.h"
#include "LibLsp/lsp/lsDocumentUri.h"
#include "LibLsp/lsp/utils.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/lsServerCapabilities.h"
#include "LibLsp/lsp/general/lsTextDocumentClientCapabilities.h"
#include "LibLsp/lsp/location_type.h"
#include "LibLsp/lsp/lsp_completion.h"
#include "LibLsp/lsp/lsp_diagnostic.h"
#include "LibLsp/lsp/lsTextEdit.h"
#include "LibLsp/lsp/lsWorkspaceEdit.h"
#include "LibLsp/lsp/textDocument/code_action.h"
#include "LibLsp/lsp/textDocument/completion.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_close.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include "LibLsp/lsp/textDocument/document_symbol.h"
#include "LibLsp/lsp/textDocument/formatting.h"
#include "LibLsp/lsp/textDocument/hover.h"
#include "LibLsp/lsp/textDocument/publishDiagnostics.h"
#include "LibLsp/lsp/textDocument/references.h"
#include "LibLsp/lsp/textDocument/rename.h"
#include "LibLsp/lsp/textDocument/signature_help.h"
#include "LibLsp/lsp/workspace/execute_command.h"
#include "LibLsp/lsp/workspace/symbol.h"
#include "LibLsp/lsp/out_list.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/shutdown.h"
#include "LibLsp/lsp/general/initialized.h"
#include "LibLsp/lsp/lsCodeAction.h"
#include "LibLsp/lsp/ResourceOperation.h"
#include "LibLsp/lsp/lsTextDocumentEdit.h"
#include "LibLsp/lsp/lsp_diagnostic.h"
#include "LibLsp/lsp/textDocument/code_lens.h"
#include "LibLsp/lsp/textDocument/highlight.h"
#include "LibLsp/lsp/textDocument/did_save.h"
#include "LibLsp/lsp/textDocument/willSave.h"
#include "LibLsp/lsp/windows/MessageNotify.h"
#include "protocol_test_helpers.h"
#include "test_helpers.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace
{
using test::Expect;
using test::ExpectParsesErrorResponse;
using test::ExpectParsesNotification;
using test::ExpectParsesRequest;
using test::ExpectParsesResponse;

#ifndef _WIN32
struct ScopedTempDir
{
    ScopedTempDir()
    {
        char path_template[] = "/tmp/lspcpp-absolute-path-XXXXXX";
        char* created = mkdtemp(path_template);
        if (created != nullptr)
        {
            path = created;
        }
    }

    ~ScopedTempDir()
    {
        if (path.empty())
        {
            return;
        }

        std::remove((path + "/link/child.cpp").c_str());
        unlink((path + "/link").c_str());
        std::remove((path + "/real/child.cpp").c_str());
        rmdir((path + "/real").c_str());
        std::remove((path + "/dir/file.cpp").c_str());
        rmdir((path + "/dir").c_str());
        rmdir(path.c_str());
    }

    std::string path;
};
#endif

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
T RoundTrip(T value)
{
    std::string const json = SerializeJson(value);
    rapidjson::Document document;
    document.Parse(json.c_str());
    JsonReader reader(&document);
    T result;
    Reflect(reader, result);
    return result;
}

JsonReader MakeReader(rapidjson::Document& document, char const* json)
{
    document.Parse(json);
    return JsonReader(&document);
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

template<typename T>
bool TryParseJson(char const* json, T& value)
{
    try
    {
        rapidjson::Document document;
        JsonReader reader = MakeReader(document, json);
        Reflect(reader, value);
        return true;
    }
    catch (std::exception const&)
    {
        return false;
    }
    catch (...)
    {
        return false;
    }
}

lsDocumentUri MakeDocumentUri(char const* uri)
{
    lsDocumentUri document_uri;
    document_uri.raw_uri_ = uri;
    return document_uri;
}

void TestPositionRangeLocationAndTextEditRoundTrip()
{
    lsPosition const position = RoundTrip(lsPosition(3, 14));
    Expect(position.line == 3 && position.character == 14, "lsPosition must round-trip line and character");

    lsRange range(lsPosition(1, 2), lsPosition(3, 4));
    lsRange const range_copy = RoundTrip(range);
    Expect(range_copy == range, "lsRange must round-trip start and end positions");

    lsLocation location(MakeDocumentUri("file:///tmp/main.cpp"), range);
    lsLocation const location_copy = RoundTrip(location);
    Expect(location_copy == location, "lsLocation must round-trip URI and range");

    lsTextEdit edit;
    edit.range = range;
    edit.newText = "replacement";
    edit.annotationId = std::string("edit-1");
    lsTextEdit const edit_copy = RoundTrip(edit);
    Expect(edit_copy.range == edit.range, "lsTextEdit range must round-trip");
    Expect(edit_copy.newText == edit.newText, "lsTextEdit newText must round-trip");
    Expect(edit_copy.annotationId && *edit_copy.annotationId == "edit-1", "text edit annotationId must survive round-trip");
}

void TestWorkspaceEditAndDiagnosticRoundTrip()
{
    lsTextEdit edit;
    edit.range = lsRange(lsPosition(0, 0), lsPosition(0, 3));
    edit.newText = "bar";

    lsWorkspaceEdit workspace_edit;
    workspace_edit.changes = std::map<std::string, std::vector<lsTextEdit>>();
    (*workspace_edit.changes)["file:///tmp/a.cpp"].push_back(edit);
    workspace_edit.metadata = lsWorkspaceEditMetadata();
    workspace_edit.metadata->isRefactoring = true;

    lsWorkspaceEdit const workspace_copy = RoundTrip(workspace_edit);
    Expect(workspace_copy.changes.has_value(), "workspace edit changes must survive round-trip");
    Expect(
        workspace_copy.changes && workspace_copy.changes->at("file:///tmp/a.cpp")[0].range == edit.range &&
            workspace_copy.changes->at("file:///tmp/a.cpp")[0].newText == edit.newText,
        "workspace edit text changes must preserve nested edits");
    Expect(
        workspace_copy.metadata && workspace_copy.metadata->isRefactoring && *workspace_copy.metadata->isRefactoring,
        "workspace edit metadata must survive round-trip");

    lsDiagnostic diagnostic;
    diagnostic.range = lsRange(lsPosition(4, 1), lsPosition(4, 8));
    diagnostic.severity = lsDiagnosticSeverity::Warning;
    diagnostic.source = std::string("lspcpp-test");
    diagnostic.message = "warning text";
    diagnostic.tags = std::vector<DiagnosticTag> {DiagnosticTag::Unnecessary};

    lsDiagnostic const diagnostic_copy = RoundTrip(diagnostic);
    Expect(diagnostic_copy.range == diagnostic.range, "diagnostic range must round-trip");
    Expect(
        diagnostic_copy.severity && *diagnostic_copy.severity == lsDiagnosticSeverity::Warning,
        "diagnostic severity must round-trip");
    Expect(diagnostic_copy.source && *diagnostic_copy.source == "lspcpp-test", "diagnostic source must round-trip");
    Expect(diagnostic_copy.message == "warning text", "diagnostic message must round-trip");
    Expect(
        diagnostic_copy.tags && diagnostic_copy.tags->size() == 1 &&
            (*diagnostic_copy.tags)[0] == DiagnosticTag::Unnecessary,
        "diagnostic tags must round-trip");
}

void TestCompletionHoverAndInitializeRoundTrip()
{
    lsCompletionItem item;
    item.label = "printf";
    item.kind = lsCompletionItemKind::Function;
    item.detail = std::string("int printf(const char*, ...)");
    item.insertText = std::string("printf($1)");
    item.insertTextFormat = lsInsertTextFormat::Snippet;
    item.commitCharacters = std::vector<std::string> {"(", "."};

    lsCompletionItem const item_copy = RoundTrip(item);
    Expect(item_copy.label == "printf", "completion item label must round-trip");
    Expect(item_copy.kind && *item_copy.kind == lsCompletionItemKind::Function, "completion item kind must round-trip");
    Expect(item_copy.insertText && *item_copy.insertText == "printf($1)", "completion insertText must round-trip");
    Expect(
        item_copy.commitCharacters && item_copy.commitCharacters->size() == 2,
        "completion commit characters must round-trip");

    TextDocumentHover::Result hover;
    MarkupContent markdown;
    markdown.kind = "markdown";
    markdown.value = "**symbol**";
    hover.contents.second = markdown;
    hover.range = lsRange(lsPosition(2, 0), lsPosition(2, 6));
    TextDocumentHover::Result const hover_copy = RoundTrip(hover);
    Expect(hover_copy.contents.second.has_value(), "hover markup content must round-trip");
    Expect(
        hover_copy.contents.second && hover_copy.contents.second->value == "**symbol**",
        "hover markup value must round-trip");
    Expect(hover_copy.range && *hover_copy.range == *hover.range, "hover range must round-trip");

    td_initialize::request initialize_request;
    initialize_request.id.set("init-1");
    initialize_request.params.processId = 1234;
    initialize_request.params.clientInfo = ClientInfo();
    initialize_request.params.clientInfo->name = "roundtrip-client";
    initialize_request.params.rootUri = MakeDocumentUri("file:///tmp/project");
    td_initialize::request const request_copy = RoundTrip(initialize_request);
    Expect(request_copy.id == initialize_request.id, "initialize request id must round-trip");
    Expect(request_copy.method == "initialize", "initialize request method must round-trip");
    Expect(
        request_copy.params.clientInfo && request_copy.params.clientInfo->name == "roundtrip-client",
        "initialize request clientInfo must round-trip");

    td_initialize::response initialize_response;
    initialize_response.id.set("init-1");
    initialize_response.result.capabilities.hoverProvider = true;
    td_initialize::response const response_copy = RoundTrip(initialize_response);
    Expect(response_copy.id == initialize_response.id, "initialize response id must round-trip");
    Expect(
        response_copy.result.capabilities.hoverProvider &&
            *response_copy.result.capabilities.hoverProvider == true,
        "initialize response capabilities must round-trip");
}

void TestTextDocumentSyncUnionRoundTrip()
{
    lsServerCapabilities kind_caps;
    kind_caps.textDocumentSync = std::make_pair(
        optional<lsTextDocumentSyncKind>(lsTextDocumentSyncKind::Incremental), optional<lsTextDocumentSyncOptions>());
    lsServerCapabilities const kind_copy = RoundTrip(kind_caps);
    Expect(
        kind_copy.textDocumentSync && kind_copy.textDocumentSync->first &&
            *kind_copy.textDocumentSync->first == lsTextDocumentSyncKind::Incremental,
        "textDocumentSync kind union must round-trip numeric sync kind");
    Expect(
        !kind_copy.textDocumentSync->second,
        "textDocumentSync kind union must not populate options side");

    using SyncUnion = std::pair<optional<lsTextDocumentSyncKind>, optional<lsTextDocumentSyncOptions>>;
    SyncUnion kind_union;
    kind_union.first = lsTextDocumentSyncKind::Incremental;
    Expect(SerializeJson(kind_union) == "2", "textDocumentSync kind union must serialize as sync kind number");
    Expect(
        SerializeJson(kind_caps).find("\"textDocumentSync\":2") != std::string::npos,
        "server capabilities must serialize textDocumentSync as sync kind number");

    lsServerCapabilities options_caps;
    lsTextDocumentSyncOptions options;
    options.openClose = true;
    options.change = lsTextDocumentSyncKind::Full;
    options_caps.textDocumentSync =
        std::make_pair(optional<lsTextDocumentSyncKind>(), optional<lsTextDocumentSyncOptions>(options));
    lsServerCapabilities const options_copy = RoundTrip(options_caps);
    Expect(
        options_copy.textDocumentSync && options_copy.textDocumentSync->second &&
            options_copy.textDocumentSync->second->openClose &&
            *options_copy.textDocumentSync->second->openClose == true &&
            options_copy.textDocumentSync->second->change &&
            *options_copy.textDocumentSync->second->change == lsTextDocumentSyncKind::Full,
        "textDocumentSync options union must round-trip sync options object");
    Expect(
        !options_copy.textDocumentSync->first,
        "textDocumentSync options union must not populate kind side");

    SyncUnion const parsed_kind = ParseJson<SyncUnion>("1");
    Expect(
        parsed_kind.first && *parsed_kind.first == lsTextDocumentSyncKind::Full,
        "textDocumentSync must parse numeric JSON as sync kind");

    SyncUnion const parsed_options = ParseJson<SyncUnion>(R"({"openClose":true,"change":2})");
    Expect(
        parsed_options.second && parsed_options.second->openClose &&
            *parsed_options.second->openClose == true && parsed_options.second->change &&
            *parsed_options.second->change == lsTextDocumentSyncKind::Incremental,
        "textDocumentSync must parse object JSON as sync options");
}

void TestDocumentUriEscapingRoundTrip()
{
    AbsolutePath path("/tmp/lspcpp space/with#symbols(a).cpp");
    lsDocumentUri uri(path);

    Expect(uri.raw_uri_.find("%20") != std::string::npos, "document URI must escape spaces");
    Expect(uri.raw_uri_.find("%23") != std::string::npos, "document URI must escape reserved characters");
    Expect(uri.GetRawPath() == path.path(), "document URI raw path must decode escaped characters");
    Expect(uri.GetAbsolutePath().path() == path.path(), "document URI absolute path must decode back to original path");

    lsDocumentUri const uri_copy = RoundTrip(uri);
    Expect(uri_copy == uri, "document URI must round-trip its raw URI");
}

void TestDocumentUriFromPathAndSimpleFileUri()
{
    AbsolutePath path("/tmp/simple.cpp");
    lsDocumentUri from_path = lsDocumentUri::FromPath(path);
    lsDocumentUri from_ctor(path);

    Expect(from_path == from_ctor, "FromPath must produce the same URI as the path constructor");
#if defined(_WIN32)
    Expect(from_path.raw_uri_ == "file:///tmp/simple.cpp", "simple file URI must use the file scheme prefix on Windows");
#else
    Expect(from_path.raw_uri_ == "file:///tmp/simple.cpp", "simple file URI must use the file scheme prefix on Unix");
#endif
    Expect(from_path.GetRawPath() == path.path(), "simple file URI must decode to the original path");
    Expect(from_path.GetAbsolutePath().path() == path.path(), "simple file URI absolute path must match the source path");
    Expect(
        make_file_scheme_uri(path.path()) == from_path.raw_uri_,
        "make_file_scheme_uri must use the same encoding as lsDocumentUri::SetPath");
}

void TestDocumentUriReservedCharacterEscaping()
{
    AbsolutePath path("/tmp/a b#$&()+,;?@x.cpp");
    lsDocumentUri uri(path);

    Expect(uri.raw_uri_.find("%20") != std::string::npos, "document URI must escape spaces");
    Expect(uri.raw_uri_.find("%23") != std::string::npos, "document URI must escape #");
    Expect(uri.raw_uri_.find("%24") != std::string::npos, "document URI must escape $");
    Expect(uri.raw_uri_.find("%26") != std::string::npos, "document URI must escape &");
    Expect(uri.raw_uri_.find("%28") != std::string::npos, "document URI must escape (");
    Expect(uri.raw_uri_.find("%29") != std::string::npos, "document URI must escape )");
    Expect(uri.raw_uri_.find("%2B") != std::string::npos, "document URI must escape +");
    Expect(uri.raw_uri_.find("%2C") != std::string::npos, "document URI must escape ,");
    Expect(uri.raw_uri_.find("%3B") != std::string::npos, "document URI must escape ;");
    Expect(uri.raw_uri_.find("%3F") != std::string::npos, "document URI must escape ?");
    Expect(uri.raw_uri_.find("%40") != std::string::npos, "document URI must escape @");
    Expect(uri.GetRawPath() == path.path(), "document URI must decode every escaped reserved character");
}

void TestDocumentUriPercentAndUtf8Escaping()
{
    AbsolutePath percent_path("/tmp/100%done.cpp");
    lsDocumentUri percent_uri(percent_path);
    Expect(
        percent_uri.raw_uri_.find("%25") != std::string::npos,
        "document URI must escape literal percent signs");
    Expect(
        percent_uri.GetRawPath() == percent_path.path(),
        "document URI must decode escaped percent signs back to the original path");

    AbsolutePath utf8_path("/tmp/\xE4\xB8\xAD.cpp");
    lsDocumentUri utf8_uri(utf8_path);
    Expect(
        utf8_uri.raw_uri_.find("%E4%B8%AD") != std::string::npos,
        "document URI must percent-encode UTF-8 path bytes");
    Expect(utf8_uri.GetRawPath() == utf8_path.path(), "document URI must decode UTF-8 bytes back to the path");
}

void TestDocumentUriFileLocalhostAndInvalidPercentDecoding()
{
    lsDocumentUri localhost_uri;
    localhost_uri.raw_uri_ = "file://localhost/tmp/from-client.cpp";
    Expect(
        localhost_uri.GetRawPath() == "/tmp/from-client.cpp",
        "file://localhost URIs must decode to local absolute paths");
    Expect(
        localhost_uri.GetAbsolutePath().path() == "/tmp/from-client.cpp",
        "file://localhost URIs must normalize as local file paths");

    lsDocumentUri uppercase_localhost_uri;
    uppercase_localhost_uri.raw_uri_ = "FILE://LOCALHOST/tmp/upper-client.cpp";
    Expect(
        uppercase_localhost_uri.GetRawPath() == "/tmp/upper-client.cpp",
        "file scheme and localhost authority matching must be case-insensitive");

    lsDocumentUri invalid_percent_uri;
    invalid_percent_uri.raw_uri_ = "file:///tmp/bad%ZZ.cpp";
    Expect(
        invalid_percent_uri.GetRawPath() == "/tmp/bad%ZZ.cpp",
        "invalid percent escapes must be preserved instead of corrupting the path");
}

void TestDocumentUriRelativePathAndQueryFragmentGuards()
{
    std::string const relative_uri = make_file_scheme_uri("relative/path.cpp");
    Expect(relative_uri == "file:///relative/path.cpp", "relative paths must not become URI authorities");

    lsDocumentUri relative_document_uri(AbsolutePath("relative/path.cpp"));
    Expect(
        relative_document_uri.raw_uri_.empty(),
        "lsDocumentUri must not encode invalid relative AbsolutePath values");
    Expect(
        relative_document_uri.GetAbsolutePath().empty(),
        "invalid relative document URIs must not produce an AbsolutePath");

    lsDocumentUri query_uri;
    query_uri.raw_uri_ = "file:///tmp/name%3Finside.cpp?query=value#fragment";
    Expect(
        query_uri.GetRawPath() == "/tmp/name?inside.cpp",
        "file URI query and fragment must not become part of the decoded file path");

    lsDocumentUri fragment_uri;
    fragment_uri.raw_uri_ = "file:///tmp/hash%23inside.cpp#fragment";
    Expect(
        fragment_uri.GetRawPath() == "/tmp/hash#inside.cpp",
        "file URI fragment must be stripped while encoded path hashes still decode");
}

void TestAbsolutePathFromNormalized()
{
    AbsolutePath valid = AbsolutePath::FromNormalized("/tmp/normalized.cpp");
    Expect(valid.valid(), "FromNormalized must accept absolute paths");
    Expect(valid.path() == "/tmp/normalized.cpp", "FromNormalized must preserve the normalized string");

    AbsolutePath preserved = AbsolutePath::FromNormalized("/tmp/a/../b");
    Expect(preserved.valid(), "FromNormalized must accept absolute paths with dot components");
    Expect(preserved.path() == "/tmp/a/../b", "FromNormalized must not re-normalize the input string");

    AbsolutePath relative = AbsolutePath::FromNormalized("relative/path.cpp");
    Expect(relative.empty(), "FromNormalized must reject relative paths");
    Expect(!relative.valid(), "FromNormalized relative results must be invalid");

    AbsolutePath empty = AbsolutePath::FromNormalized("");
    Expect(empty.empty(), "FromNormalized must reject empty strings");
    Expect(!empty.valid(), "FromNormalized empty results must be invalid");
}

void TestDirectoryEnsuresTrailingSlash()
{
    Directory without_slash(AbsolutePath("/tmp/dir"));
    Expect(without_slash.path == "/tmp/dir/", "Directory must append trailing slash to path");

    Directory with_slash(AbsolutePath("/tmp/dir/"));
    Expect(with_slash.path == "/tmp/dir/", "Directory must keep a single trailing slash");

    Expect(without_slash == with_slash, "Directory equality must treat trailing slash variants as equal");
}

void TestAbsolutePathJsonRoundTrip()
{
    AbsolutePath path("/tmp/json-roundtrip.cpp");
    AbsolutePath const copy = RoundTrip(path);
    Expect(copy.path() == path.path(), "AbsolutePath JSON must round-trip valid paths");
    Expect(copy.valid(), "AbsolutePath JSON round-trip must stay valid");

    std::string const json = SerializeJson(path);
    Expect(json == "\"/tmp/json-roundtrip.cpp\"", "AbsolutePath JSON writer must serialize path string");
}

void TestAbsolutePathValidationNormalizesRelativePaths()
{
    AbsolutePath raw("relative/../guarded/missing.cpp");
    Expect(raw.empty(), "AbsolutePath construction must reject relative path strings");
    Expect(!raw.valid(), "AbsolutePath construction must keep relative paths invalid");

    AbsolutePath normalized("relative/../guarded/missing.cpp");
    Expect(!normalized.valid(), "default validation must not qualify relative paths implicitly");

    AbsolutePath normalized_explicitly = lsp::NormalizePath("relative/../guarded/missing.cpp", false);
    Expect(normalized_explicitly.valid(), "NormalizePath must qualify relative paths explicitly");
    Expect(lsp::IsAbsolutePath(normalized_explicitly.path()), "NormalizePath must return absolute paths");
    Expect(
        lsp::EndsWith(normalized_explicitly.path(), "/guarded/missing.cpp"),
        "NormalizePath must collapse dot-dot components even when the file is missing");

    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"("relative/from-json.cpp")");
    AbsolutePath from_json;
    Reflect(reader, from_json);
    Expect(!from_json.valid(), "AbsolutePath JSON reads must reject relative path strings");
}

void TestAbsolutePathQualifyConsistency()
{
    AbsolutePath empty;
    Expect(empty.empty(), "default AbsolutePath must start with an empty path");
    Expect(!empty.valid(), "default AbsolutePath must not mark the empty path as valid");

    AbsolutePath relative("relative/path.cpp");
    Expect(!relative.valid(), "relative paths must remain invalid");

    AbsolutePath absolute("/tmp/absolute-path.cpp");
    Expect(absolute.valid(), "absolute paths must remain valid");

    AbsolutePath failed = lsp::NormalizePath("", false);
    Expect(failed.empty(), "failed NormalizePath results must keep an empty path");
    Expect(!failed.valid(), "failed NormalizePath results must not be valid");
}

void TestAbsolutePathOperatorsAndConversion()
{
    AbsolutePath a("/tmp/a.cpp");
    AbsolutePath same("/tmp/a.cpp");
    AbsolutePath b("/tmp/b.cpp");

    Expect(a == same, "AbsolutePath equality must compare path strings");
    Expect(a != b, "AbsolutePath inequality must compare path strings");
    Expect(a < b, "AbsolutePath ordering must compare path strings");
    Expect(b > a, "AbsolutePath greater-than must compare path strings");
    Expect(a.path() == "/tmp/a.cpp", "AbsolutePath path accessor must return path");

    std::ostringstream out;
    out << a;
    Expect(out.str() == "/tmp/a.cpp", "AbsolutePath stream output must write path");
}

void TestNormalizePathAllowsMissingPathsWithoutStatState()
{
    AbsolutePath normalized = lsp::NormalizePath("missing-root/child/../leaf.cpp", false);
    Expect(lsp::IsAbsolutePath(normalized.path()), "NormalizePath must qualify missing relative paths");
    Expect(
        lsp::EndsWith(normalized.path(), "/missing-root/leaf.cpp"),
        "NormalizePath must normalize missing path components without requiring stat data");

    AbsolutePath empty = lsp::NormalizePath("", false);
    Expect(empty.empty(), "NormalizePath must reject empty paths instead of manufacturing a path");
    Expect(!empty.valid(), "NormalizePath empty results must not be valid");
}

void TestAbsolutePathValidatePreservesAbsoluteInput()
{
    AbsolutePath preserved("/tmp/a/../b.cpp");
    Expect(preserved.path() == "/tmp/b.cpp", "default validation must normalize already absolute inputs");
    Expect(preserved.valid(), "default validation must keep already absolute inputs valid");
}

#ifndef _WIN32
void TestNormalizePathCollapsesAbsoluteComponents()
{
    AbsolutePath root = lsp::NormalizePath("/", false);
    Expect(root.path() == "/", "NormalizePath must preserve the filesystem root");
    Expect(root.valid(), "normalized filesystem root must be valid");

    AbsolutePath collapsed = lsp::NormalizePath("/tmp//a/./b/../c/", false);
    Expect(collapsed.path() == "/tmp/a/c", "NormalizePath must collapse slashes, dots, dot-dots, and trailing slash");

    AbsolutePath unc = lsp::NormalizePath("//server/share/../docs/file.cpp", false);
    Expect(unc.path() == "//server/docs/file.cpp", "NormalizePath must preserve leading double-slash UNC roots");

    AbsolutePath above_root = lsp::NormalizePath("/../../", false);
    Expect(above_root.path() == "/", "NormalizePath dot-dots above root must remain at root");
}

void TestNormalizePathEnsureExists()
{
    ScopedTempDir temp;
    Expect(!temp.path.empty(), "test must create a temporary directory");
    if (temp.path.empty())
    {
        return;
    }

    std::string const dir = temp.path + "/dir";
    std::string const real = temp.path + "/real";
    Expect(mkdir(dir.c_str(), 0700) == 0, "test must create a directory");
    Expect(mkdir(real.c_str(), 0700) == 0, "test must create a real symlink target directory");

    std::string const file = dir + "/file.cpp";
    std::ofstream(file.c_str()) << "int main() {}\n";
    std::string const real_child = real + "/child.cpp";
    std::ofstream(real_child.c_str()) << "int child() {}\n";

    AbsolutePath existing = lsp::NormalizePath(dir + "/../dir/file.cpp", true);
    Expect(existing.path() == file, "NormalizePath ensure_exists must accept existing paths and collapse components");

    AbsolutePath missing = lsp::NormalizePath(dir + "/missing.cpp", true);
    Expect(missing.empty(), "NormalizePath ensure_exists must reject missing paths");
    Expect(!missing.valid(), "NormalizePath missing results must not be valid");

    AbsolutePath file_as_dir = lsp::NormalizePath(file + "/child.cpp", true);
    Expect(file_as_dir.empty(), "NormalizePath ensure_exists must reject file components used as directories");

    std::string const link = temp.path + "/link";
    if (symlink("real", link.c_str()) == 0)
    {
        AbsolutePath through_link = lsp::NormalizePath(link + "/child.cpp", true);
        Expect(
            through_link.path() == link + "/child.cpp",
            "NormalizePath must validate through symlinks without expanding the symlink path");
    }
}

void TestDocumentUriDotDotNormalization()
{
    lsDocumentUri uri;
    uri.raw_uri_ = "file:///tmp/a/../b.cpp";
    Expect(uri.GetAbsolutePath().path() == "/tmp/b.cpp", "file URI GetAbsolutePath must normalize dot-dot components");
}
#endif

void TestAbsolutePathRecognitionGuards()
{
    Expect(lsp::IsUnixAbsolutePath("/"), "Unix root must be recognized as absolute");
    Expect(lsp::IsWindowsAbsolutePath("C:/"), "Windows drive roots with forward slashes must be absolute");
    Expect(lsp::IsWindowsAbsolutePath("C:\\"), "Windows drive roots with backslashes must be absolute");
    Expect(lsp::IsWindowsAbsolutePath("\\\\server\\share"), "Windows UNC paths must be absolute");
    Expect(!lsp::IsWindowsAbsolutePath("C:relative.cpp"), "drive-relative Windows paths must not be absolute");
}

#if defined(_WIN32)
void TestDocumentUriWindowsDriveLetterEscaping()
{
    AbsolutePath path("C:/Users/test/main.cpp");
    lsDocumentUri uri(path);

    Expect(uri.raw_uri_.find("C%3A") != std::string::npos, "Windows drive letters must escape the colon");
    Expect(uri.GetRawPath() == path.path(), "Windows drive letter URI must decode back to the original path");
}
#endif

#if defined(_WIN32)
void TestDocumentUriDotDotNormalizationWindows()
{
    lsDocumentUri uri;
    uri.raw_uri_ = "file:///C:/tmp/a/../b.cpp";
    Expect(
        uri.GetAbsolutePath().path() == "C:/tmp/b.cpp",
        "Windows file URI GetAbsolutePath must normalize dot-dot components");
}
#endif

void TestDocumentUriGetAbsolutePathInvalidPaths()
{
    lsDocumentUri localhost_without_path;
    localhost_without_path.raw_uri_ = "file://localhost";
    Expect(
        !localhost_without_path.GetAbsolutePath().valid(),
        "file://localhost without path must not produce valid AbsolutePath");
    Expect(
        localhost_without_path.GetAbsolutePath().empty(),
        "file://localhost without path must return empty AbsolutePath");

    lsDocumentUri empty_uri;
    Expect(!empty_uri.GetAbsolutePath().valid(), "empty document URI GetAbsolutePath must be invalid");
    Expect(empty_uri.GetAbsolutePath().empty(), "empty document URI GetAbsolutePath must return empty AbsolutePath");

    lsDocumentUri non_file;
    non_file.raw_uri_ = "untitled:Untitled-1";
    Expect(!non_file.GetAbsolutePath().valid(), "non-file URI GetAbsolutePath must be invalid");
    Expect(non_file.GetAbsolutePath().empty(), "non-file URI GetAbsolutePath must return empty AbsolutePath");

#if defined(_WIN32)
    lsDocumentUri invalid_utf8;
    invalid_utf8.raw_uri_ = "file:///tmp/%FF.cpp";
    Expect(
        !invalid_utf8.GetAbsolutePath().valid(),
        "GetAbsolutePath must return invalid paths when Windows UTF-8 conversion rejects URI bytes");
#endif
}

void TestDocumentUriUncAuthorityRoundTrip()
{
    lsDocumentUri unc;
    unc.raw_uri_ = "file://server/share/docs/file.cpp";
    Expect(
        unc.GetRawPath() == "//server/share/docs/file.cpp",
        "UNC authority file URI must decode to UNC absolute paths");
    Expect(unc.GetAbsolutePath().valid(), "UNC file URI GetAbsolutePath must produce valid AbsolutePath");
    Expect(
        unc.GetAbsolutePath().path() == "//server/share/docs/file.cpp",
        "UNC file URI GetAbsolutePath must preserve the UNC authority path");

    lsDocumentUri const unc_copy = RoundTrip(unc);
    Expect(unc_copy == unc, "UNC document URI must round-trip through JSON");

    lsDocumentUri localhost;
    localhost.raw_uri_ = "file://localhost/tmp/roundtrip.cpp";
    lsDocumentUri const localhost_copy = RoundTrip(localhost);
    Expect(localhost_copy == localhost, "localhost authority document URI must round-trip through JSON");
    Expect(
        localhost_copy.GetAbsolutePath().path() == "/tmp/roundtrip.cpp",
        "localhost authority round-trip must still normalize to local absolute paths");
}

#ifndef _WIN32
void TestNormalizePathForceLowerIgnoredOnUnix()
{
    AbsolutePath default_case = lsp::NormalizePath("/tmp/MixedCase/File.cpp", false);
    AbsolutePath explicit_lower = lsp::NormalizePath("/tmp/MixedCase/File.cpp", false, true);
    AbsolutePath preserve_case = lsp::NormalizePath("/tmp/MixedCase/File.cpp", false, false);
    Expect(explicit_lower.path() == default_case.path(), "default force_lower must match explicit true on Unix");
    Expect(preserve_case.path() == default_case.path(), "force_lower_on_windows must be ignored on Unix");
}
#endif

#if defined(_WIN32)
void TestNormalizePathWindowsDriveLetterAndForceLower()
{
    AbsolutePath lowered = lsp::NormalizePath("C:/Users/MIXED/File.cpp", false, true);
    Expect(lowered.valid(), "Windows NormalizePath must accept drive-letter paths");
    Expect(lowered.path() == "c:/users/mixed/file.cpp", "Windows NormalizePath must lower-case paths by default");

    AbsolutePath preserved = lsp::NormalizePath("C:/Users/MIXED/File.cpp", false, false);
    Expect(preserved.valid(), "Windows NormalizePath must accept drive-letter paths without force_lower");
    Expect(preserved.path().find("MIXED") != std::string::npos, "force_lower_on_windows=false must preserve path casing");
    Expect(lowered.path() != preserved.path(), "force_lower_on_windows must change normalized path casing");
}
#endif

#if defined(_WIN32)
void TestNormalizePathWindowsEnsureExistsAndMaxPathGuard()
{
    std::string const file = "lspcpp-normalize-existing.tmp";
    std::remove(file.c_str());
    std::ofstream(file.c_str()) << "int main() {}\n";

    AbsolutePath existing = lsp::NormalizePath(file, true);
    Expect(existing.valid(), "Windows NormalizePath ensure_exists must accept existing files");
    Expect(lsp::EndsWith(existing.path(), "/lspcpp-normalize-existing.tmp"), "Windows NormalizePath must normalize existing files");

    AbsolutePath missing = lsp::NormalizePath("lspcpp-normalize-missing.tmp", true);
    Expect(missing.empty(), "Windows NormalizePath ensure_exists must reject missing paths");
    Expect(!missing.valid(), "Windows NormalizePath missing results must not be valid");

    AbsolutePath too_long = lsp::NormalizePath(std::string(300, 'a'), false);
    Expect(too_long.empty(), "Windows NormalizePath must reject paths whose normalized length reaches MAX_PATH");
    Expect(!too_long.valid(), "Windows NormalizePath long-path results must not be valid");

    std::remove(file.c_str());
}
#endif

void TestDocumentUriDefaultCopyEqualityAndSwap()
{
    lsDocumentUri empty;
    Expect(empty.raw_uri_.empty(), "default document URI must start empty");
    Expect(empty == lsDocumentUri(), "default document URIs must compare equal");

    AbsolutePath path("/tmp/equality.cpp");
    lsDocumentUri uri(path);
    lsDocumentUri copied(uri);

    Expect(copied == uri, "copy constructor must preserve document URI");
    Expect(uri == uri.raw_uri_, "document URI must compare equal to its raw string form");
    Expect(!(uri == "file:///other.cpp"), "document URI string comparison must reject different URIs");

    lsDocumentUri other;
    other.swap(uri);
    Expect(other == copied, "swap must move the encoded URI to the other object");
    Expect(uri.raw_uri_.empty(), "swap must leave the previous owner empty");
}

void TestDocumentUriNonFileSchemePassthrough()
{
    lsDocumentUri untitled;
    untitled.raw_uri_ = "untitled:Untitled-1";

    Expect(
        untitled.GetRawPath() == "untitled:Untitled-1",
        "non-file URI GetRawPath must return the raw URI unchanged");
    Expect(
        !untitled.GetAbsolutePath().valid(),
        "non-file URI GetAbsolutePath must not manufacture an AbsolutePath");

    lsDocumentUri const untitled_copy = RoundTrip(untitled);
    Expect(untitled_copy == untitled, "non-file document URI must round-trip through JSON");
}

void TestDocumentUriJsonDeserialization()
{
    rapidjson::Document document;
    JsonReader reader = MakeReader(document, R"("file:///client/sent.cpp")");
    lsDocumentUri from_client;
    Reflect(reader, from_client);

    Expect(
        from_client.raw_uri_ == "file:///client/sent.cpp",
        "JSON deserialization must populate raw_uri_ from client-provided strings");
    Expect(
        from_client == std::string("file:///client/sent.cpp"),
        "deserialized document URI must compare equal to the source JSON string");
}

void TestOptionalFieldsAreOmittedWhenUnsetAndReadFromNull()
{
    lsCompletionItem item;
    item.label = "plain";
    std::string const json = SerializeJson(item);
    Expect(json.find("detail") == std::string::npos, "unset optional fields must be omitted from JSON objects");
    Expect(json.find("insertText") == std::string::npos, "unset insertText must be omitted from JSON objects");

    rapidjson::Document document;
    JsonReader reader = MakeReader(document, "null");
    optional<std::string> value = std::string("present");
    Reflect(reader, value);
    Expect(value && *value == "present", "reading null into an optional must preserve current nullopt compatibility");
}

void TestLocationListEitherRoundTrip()
{
    lsRange const range(lsPosition(1, 0), lsPosition(1, 4));

    LocationListEither::Either locations;
    locations.first = std::vector<lsLocation> {lsLocation(MakeDocumentUri("file:///tmp/a.cpp"), range)};
    LocationListEither::Either const locations_copy = RoundTrip(locations);
    Expect(locations_copy.first && locations_copy.first->size() == 1, "Location[] union must round-trip array size");
    Expect(
        (*locations_copy.first)[0].uri == MakeDocumentUri("file:///tmp/a.cpp"),
        "Location[] union must round-trip target URI");
    Expect((*locations_copy.first)[0].range == range, "Location[] union must round-trip range");

    LocationListEither::Either links;
    LocationLink link;
    link.originSelectionRange = lsRange(lsPosition(0, 0), lsPosition(0, 1));
    link.targetUri = MakeDocumentUri("file:///tmp/b.cpp");
    link.targetRange = range;
    link.targetSelectionRange = range;
    links.second = std::vector<LocationLink> {link};
    LocationListEither::Either const links_copy = RoundTrip(links);
    Expect(links_copy.second && links_copy.second->size() == 1, "LocationLink[] union must round-trip array size");
    Expect(
        (*links_copy.second)[0].targetUri == MakeDocumentUri("file:///tmp/b.cpp"),
        "LocationLink[] union must round-trip target URI");

    LocationListEither::Either const parsed_locations = ParseJson<LocationListEither::Either>(
        R"([{"uri":"file:///tmp/a.cpp","range":{"start":{"line":1,"character":0},"end":{"line":1,"character":4}}}])");
    Expect(parsed_locations.first && parsed_locations.first->size() == 1, "Location[] JSON must parse as location vector");
    Expect(!parsed_locations.second, "Location[] JSON must not populate LocationLink side");

    LocationListEither::Either const parsed_links = ParseJson<LocationListEither::Either>(
        R"([{"originSelectionRange":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"targetUri":"file:///tmp/b.cpp","targetRange":{"start":{"line":1,"character":0},"end":{"line":1,"character":4}},"targetSelectionRange":{"start":{"line":1,"character":0},"end":{"line":1,"character":4}}}])");
    Expect(parsed_links.second && parsed_links.second->size() == 1, "LocationLink[] JSON must parse as link vector");
    Expect(!parsed_links.first, "LocationLink[] JSON must not populate Location side");

    LocationListEither::Either const parsed_single_location = ParseJson<LocationListEither::Either>(
        R"({"uri":"file:///tmp/single.cpp","range":{"start":{"line":2,"character":0},"end":{"line":2,"character":8}}})");
    Expect(
        parsed_single_location.first && parsed_single_location.first->size() == 1,
        "single Location JSON must parse as a one-element Location[] union");
    Expect(!parsed_single_location.second, "single Location JSON must not populate LocationLink side");
    Expect(
        parsed_single_location.first && (*parsed_single_location.first)[0].uri.raw_uri_ == "file:///tmp/single.cpp",
        "single Location JSON must preserve URI");
}

void TestCompletionResponseArrayVsList()
{
    lsCompletionItem item;
    item.label = "foo";

    TextDocumentComplete::Either array_result;
    array_result.first = std::vector<lsCompletionItem> {item};
    TextDocumentComplete::Either const array_copy = RoundTrip(array_result);
    Expect(
        array_copy.first && array_copy.first->size() == 1 && (*array_copy.first)[0].label == "foo",
        "CompletionItem[] union must round-trip item label");
    Expect(!array_copy.second, "CompletionItem[] union must not populate CompletionList side");

    TextDocumentComplete::Either list_result;
    CompletionList list;
    list.isIncomplete = true;
    list.items.push_back(item);
    list_result.second = list;
    TextDocumentComplete::Either const list_copy = RoundTrip(list_result);
    Expect(
        list_copy.second && list_copy.second->items.size() == 1 && list_copy.second->items[0].label == "foo",
        "CompletionList union must round-trip nested items");
    Expect(
        list_copy.second && list_copy.second->isIncomplete,
        "CompletionList union must round-trip isIncomplete flag");
    Expect(!list_copy.first, "CompletionList union must not populate array side");

    TextDocumentComplete::Either const parsed_array = ParseJson<TextDocumentComplete::Either>(R"([{"label":"bar"}])");
    Expect(
        parsed_array.first && (*parsed_array.first)[0].label == "bar",
        "CompletionItem[] JSON must parse as completion item array");

    TextDocumentComplete::Either const parsed_list =
        ParseJson<TextDocumentComplete::Either>(R"({"isIncomplete":true,"items":[{"label":"baz"}]})");
    Expect(
        parsed_list.second && parsed_list.second->items[0].label == "baz",
        "CompletionList JSON must parse as completion list object");
}

void TestCodeActionEitherVariants()
{
    TextDocumentCodeAction::Either command_either;
    lsCommandWithAny command;
    command.title = "Run Fix";
    command.command = "editor.action.fixAll";
    command_either.first = command;
    TextDocumentCodeAction::Either const command_copy = RoundTrip(command_either);
    Expect(
        command_copy.first && command_copy.first->command == "editor.action.fixAll",
        "CodeAction command variant must round-trip command id");
    Expect(!command_copy.second, "CodeAction command variant must not populate CodeAction side");

    TextDocumentCodeAction::Either action_either;
    CodeAction action;
    action.title = "Organize Imports";
    action.kind = std::string("source.organizeImports");
    action.diagnostics = std::vector<lsDiagnostic>();
    action_either.second = action;
    TextDocumentCodeAction::Either const action_copy = RoundTrip(action_either);
    Expect(
        action_copy.second && action_copy.second->title == "Organize Imports",
        "CodeAction object variant must round-trip title");
    Expect(!action_copy.first, "CodeAction object variant must not populate command side");

    TextDocumentCodeAction::Either const parsed_command =
        ParseJson<TextDocumentCodeAction::Either>(R"({"title":"Quick Fix","command":"cmd.quickFix"})");
    Expect(
        parsed_command.first && parsed_command.first->command == "cmd.quickFix",
        "CodeAction JSON with string command must parse as command variant");

    TextDocumentCodeAction::Either const parsed_action = ParseJson<TextDocumentCodeAction::Either>(
        R"({"title":"Rename","kind":"quickfix","diagnostics":[]})");
    Expect(
        parsed_action.second && parsed_action.second->title == "Rename",
        "CodeAction JSON with diagnostics must parse as CodeAction variant");

    TextDocumentCodeAction::Either const parsed_title_only =
        ParseJson<TextDocumentCodeAction::Either>(R"({"title":"Refactor this"})");
    Expect(
        parsed_title_only.second && parsed_title_only.second->title == "Refactor this",
        "title-only CodeAction JSON must parse as CodeAction variant");
    Expect(!parsed_title_only.first, "title-only CodeAction JSON must not populate Command side");
}

void TestWorkspaceEditDocumentChanges()
{
    lsWorkspaceEdit workspace_edit;
    workspace_edit.documentChanges = std::vector<lsWorkspaceEdit::Either>();

    lsWorkspaceEdit::Either document_edit_entry;
    lsTextDocumentEdit document_edit;
    document_edit.textDocument.uri = MakeDocumentUri("file:///tmp/a.cpp");
    document_edit.textDocument.version = 2;
    lsTextEdit text_edit;
    text_edit.range = lsRange(lsPosition(0, 0), lsPosition(0, 3));
    text_edit.newText = "inserted";
    document_edit.edits.push_back(text_edit);
    document_edit_entry.first = document_edit;
    workspace_edit.documentChanges->push_back(document_edit_entry);

    lsWorkspaceEdit::Either create_entry;
    create_entry.second = lsp::Any();
    create_entry.second->SetJsonString(
        R"({"kind":"create","uri":"file:///tmp/new.cpp"})",
        lsp::Any::kObjectType);
    workspace_edit.documentChanges->push_back(create_entry);

    lsWorkspaceEdit const workspace_copy = RoundTrip(workspace_edit);
    Expect(
        workspace_copy.documentChanges && workspace_copy.documentChanges->size() == 2,
        "workspace edit documentChanges must round-trip entry count");
    Expect(
        workspace_copy.documentChanges->at(0).first &&
            workspace_copy.documentChanges->at(0).first->edits[0].newText == "inserted",
        "workspace edit documentChanges must round-trip text document edits");
    Expect(
        workspace_copy.documentChanges->at(1).second &&
            workspace_copy.documentChanges->at(1).second->Data().find("\"kind\":\"create\"") != std::string::npos,
        "workspace edit documentChanges must round-trip resource operations");

    lsWorkspaceEdit const parsed = ParseJson<lsWorkspaceEdit>(R"({
        "documentChanges": [
            {
                "textDocument": {"uri": "file:///tmp/a.cpp", "version": 1},
                "edits": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}, "newText": "bar"}]
            },
            {"kind": "create", "uri": "file:///tmp/new.cpp", "options": {"overwrite": true}}
        ]
    })");
    Expect(parsed.documentChanges && parsed.documentChanges->size() == 2, "workspace edit JSON must parse documentChanges");
    Expect(
        parsed.documentChanges->at(0).first &&
            parsed.documentChanges->at(0).first->textDocument.uri.raw_uri_ == "file:///tmp/a.cpp",
        "workspace edit JSON must parse text document edit URI");
    Expect(
        parsed.documentChanges->at(1).second &&
            parsed.documentChanges->at(1).second->Data().find("file:///tmp/new.cpp") != std::string::npos,
        "workspace edit JSON must parse create resource operation");
}

void TestHoverContentsAllRepresentations()
{
    TextDocumentHover::Result const markup_hover = ParseJson<TextDocumentHover::Result>(
        R"({"contents":{"kind":"markdown","value":"**bold**"}})");
    Expect(
        markup_hover.contents.second && markup_hover.contents.second->value == "**bold**",
        "hover MarkupContent JSON must parse contents.kind/value");

    TextDocumentHover::Result const marked_hover = ParseJson<TextDocumentHover::Result>(
        R"({"contents":[{"language":"cpp","value":"int main();"}]})");
    Expect(
        marked_hover.contents.first && marked_hover.contents.first->size() == 1,
        "hover MarkedString array JSON must parse as marked content array");
    Expect(
        marked_hover.contents.first && (*marked_hover.contents.first)[0].second &&
            (*marked_hover.contents.first)[0].second->value == "int main();",
        "hover MarkedString object JSON must parse language/value pair");

    TextDocumentHover::Result const string_hover =
        ParseJson<TextDocumentHover::Result>(R"({"contents":["plain hover text"]})");
    Expect(
        string_hover.contents.first && string_hover.contents.first->size() == 1,
        "hover string array JSON must parse as marked string array");
    Expect(
        string_hover.contents.first && (*string_hover.contents.first)[0].first &&
            *(*string_hover.contents.first)[0].first == "plain hover text",
        "hover plain string JSON must parse as string marked content");

    TextDocumentHover::Result markup_result;
    MarkupContent markdown;
    markdown.kind = "markdown";
    markdown.value = "**symbol**";
    markup_result.contents.second = markdown;
    TextDocumentHover::Result const markup_copy = RoundTrip(markup_result);
    Expect(
        markup_copy.contents.second && markup_copy.contents.second->value == "**symbol**",
        "hover MarkupContent must round-trip through JSON");
}

void TestTypeGuardInspiredMalformedInputs()
{
    lsPosition missing_character;
    Expect(
        TryParseJson(R"({"line":2})", missing_character),
        "position with missing character should parse with default character");
    Expect(
        missing_character.line == 2 && missing_character.character == 0,
        "position missing character must default to zero");

    lsPosition wrong_position;
    bool const parsed_wrong_position = TryParseJson(R"({"line":"bad","character":0})", wrong_position);
    Expect(
        !parsed_wrong_position || (wrong_position.line == 0 && wrong_position.character == 0),
        "position with wrong field type must be rejected or leave default value");

    lsRange missing_end;
    Expect(TryParseJson(R"({"start":{"line":1,"character":2}})", missing_end), "range with missing end must not crash");
    Expect(missing_end.start.line == 1 && missing_end.start.character == 2, "range start must parse when end is absent");
    Expect(missing_end.end == lsPosition(), "range missing end must keep default end");

    lsRange wrong_range;
    bool const parsed_wrong_range = TryParseJson(
        R"({"start":{"line":0,"character":0},"end":{"line":"bad","character":1}})", wrong_range);
    Expect(
        !parsed_wrong_range || wrong_range.end == lsPosition(),
        "range with wrong nested position must be rejected or keep default end");

    TextDocumentHover::Result direct_string_hover;
    Expect(
        TryParseJson(R"({"contents":"plain hover text"})", direct_string_hover),
        "hover direct string contents must parse");
    Expect(
        direct_string_hover.contents.first && direct_string_hover.contents.first->size() == 1,
        "hover direct string contents must become a single MarkedString entry");
    Expect(
        direct_string_hover.contents.first && (*direct_string_hover.contents.first)[0].first &&
            *(*direct_string_hover.contents.first)[0].first == "plain hover text",
        "hover direct string contents must preserve text");

    TextDocumentHover::Result null_range_hover;
    Expect(
        TryParseJson(R"({"contents":{"kind":"markdown","value":"docs"},"range":null})", null_range_hover),
        "hover null range must parse");
    Expect(!null_range_hover.range, "hover null range must remain unset");

    TextDocumentHover::Result null_contents_hover;
    Expect(TryParseJson(R"({"contents":null})", null_contents_hover), "hover null contents must not crash");
    Expect(
        !null_contents_hover.contents.first && !null_contents_hover.contents.second,
        "hover null contents must not populate either representation");

    TextDocumentHover::Result array_with_null_hover;
    Expect(
        TryParseJson(R"({"contents":[null]})", array_with_null_hover),
        "hover array containing null must not crash the parser");

    lsTextEdit missing_new_text;
    Expect(
        TryParseJson(
            R"({"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}})",
            missing_new_text),
        "text edit missing newText must not crash");
    Expect(missing_new_text.newText.empty(), "text edit missing newText must default to empty string");

    lsTextEdit missing_range;
    Expect(TryParseJson(R"({"newText":"replacement"})", missing_range), "text edit missing range must not crash");
    Expect(missing_range.range == lsRange(), "text edit missing range must keep default range");
    Expect(missing_range.newText == "replacement", "text edit missing range must still parse newText");

    std::pair<optional<std::string>, optional<lsMarkedString>> marked_string;
    Expect(TryParseJson(R"("plain marked string")", marked_string), "MarkedString string form must parse");
    Expect(marked_string.first && *marked_string.first == "plain marked string", "MarkedString string form mismatch");

    std::pair<optional<std::string>, optional<lsMarkedString>> marked_object;
    Expect(TryParseJson(R"({"language":"cpp","value":"int main();"})", marked_object), "MarkedString object form must parse");
    Expect(marked_object.second && marked_object.second->language, "MarkedString object form must parse language");
    Expect(
        marked_object.second && marked_object.second->value == "int main();",
        "MarkedString object form must parse value");
}

void TestProgressTokenParamsRoundTrip()
{
    lsInitializeParams const initialize_params = ParseJson<lsInitializeParams>(
        R"({"processId":1,"capabilities":{},"workDoneToken":"token"})");
    Expect(
        initialize_params.workDoneToken && initialize_params.workDoneToken->first &&
            *initialize_params.workDoneToken->first == "token",
        "initialize params must parse a string workDoneToken");

    lsInitializeParams no_token_params;
    no_token_params.processId = 1;
    Expect(
        SerializeJson(no_token_params).find("workDoneToken") == std::string::npos,
        "unset workDoneToken must be omitted from serialized params");

    lsDocumentSymbolParams const symbol_params = ParseJson<lsDocumentSymbolParams>(
        R"({"textDocument":{"uri":"file:///tmp/a.cpp"},"workDoneToken":"work","partialResultToken":42})");
    Expect(
        symbol_params.workDoneToken && symbol_params.workDoneToken->first &&
            *symbol_params.workDoneToken->first == "work",
        "documentSymbol params must parse a string workDoneToken");
    Expect(
        symbol_params.partialResultToken && symbol_params.partialResultToken->second &&
            *symbol_params.partialResultToken->second == 42,
        "documentSymbol params must parse a numeric partialResultToken");

    lsDocumentSymbolParams const round_tripped = RoundTrip(symbol_params);
    Expect(
        round_tripped.workDoneToken && round_tripped.workDoneToken->first &&
            *round_tripped.workDoneToken->first == "work",
        "documentSymbol workDoneToken must survive a JSON round-trip");
    Expect(
        round_tripped.partialResultToken && round_tripped.partialResultToken->second &&
            *round_tripped.partialResultToken->second == 42,
        "documentSymbol partialResultToken must survive a JSON round-trip");
}

void TestDocumentSymbolEitherVariants()
{
    TextDocumentDocumentSymbol::Either const symbol_information = ParseJson<TextDocumentDocumentSymbol::Either>(
        R"({"name":"legacy","kind":12,"location":{"uri":"file:///tmp/a.cpp","range":{"start":{"line":1,"character":0},"end":{"line":1,"character":6}}}})");
    Expect(
        symbol_information.first && symbol_information.first->name == "legacy",
        "DocumentSymbol Either must parse SymbolInformation objects with location");
    Expect(!symbol_information.second, "SymbolInformation JSON must not populate hierarchical DocumentSymbol side");

    TextDocumentDocumentSymbol::Either const document_symbol = ParseJson<TextDocumentDocumentSymbol::Either>(
        R"({"name":"modern","kind":12,"range":{"start":{"line":2,"character":0},"end":{"line":4,"character":1}},"selectionRange":{"start":{"line":2,"character":5},"end":{"line":2,"character":11}},"children":[]})");
    Expect(
        document_symbol.second && document_symbol.second->name == "modern",
        "DocumentSymbol Either must parse hierarchical DocumentSymbol objects");
    Expect(!document_symbol.first, "hierarchical DocumentSymbol JSON must not populate SymbolInformation side");
    Expect(
        document_symbol.second && document_symbol.second->children && document_symbol.second->children->empty(),
        "DocumentSymbol children array must parse");
}

void TestProtocolJsonHandlerParsesPolymorphicResponseEnvelopes()
{
    lsp::ProtocolJsonHandler handler;

    ExpectParsesResponse(
        handler,
        td_hover::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":40,"result":{"contents":[{"language":"cpp","value":"int main();"}]}})",
        "ProtocolJsonHandler must parse hover MarkedString array response envelopes");
    ExpectParsesResponse(
        handler,
        td_codeAction::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":41,"result":[{"title":"Title-only action"}]})",
        "ProtocolJsonHandler must parse title-only codeAction response envelopes");
}

void TestProtocolJsonHandlerParsesErrorResponses()
{
    lsp::ProtocolJsonHandler handler;
    char const* error_json =
        R"({"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"Request failed"}})";

    ExpectParsesErrorResponse(
        handler, td_initialize::request::kMethodInfo, error_json, "ProtocolJsonHandler must parse initialize error responses");
    ExpectParsesErrorResponse(
        handler, td_completion::request::kMethodInfo, error_json, "ProtocolJsonHandler must parse completion error responses");
    ExpectParsesErrorResponse(
        handler, td_hover::request::kMethodInfo, error_json, "ProtocolJsonHandler must parse hover error responses");
    ExpectParsesErrorResponse(
        handler,
        td_definition::request::kMethodInfo,
        error_json,
        "ProtocolJsonHandler must parse definition error responses");
    ExpectParsesErrorResponse(
        handler,
        td_codeAction::request::kMethodInfo,
        error_json,
        "ProtocolJsonHandler must parse codeAction error responses");
}

void TestProtocolJsonHandlerRegistersCoreRequestsAndNotifications()
{
    lsp::ProtocolJsonHandler handler;

    ExpectParsesRequest(
        handler,
        td_completion::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":1,"method":"textDocument/completion","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2},"context":{"triggerKind":1}}})",
        "ProtocolJsonHandler must parse textDocument/completion requests");
    ExpectParsesRequest(
        handler,
        td_hover::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2}}})",
        "ProtocolJsonHandler must parse textDocument/hover requests");
    ExpectParsesRequest(
        handler,
        td_signatureHelp::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":15,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2}}})",
        "ProtocolJsonHandler must parse textDocument/signatureHelp requests");
    ExpectParsesRequest(
        handler,
        td_definition::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":3,"method":"textDocument/definition","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2}}})",
        "ProtocolJsonHandler must parse textDocument/definition requests");
    ExpectParsesRequest(
        handler,
        td_references::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":4,"method":"textDocument/references","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2},"context":{"includeDeclaration":true}}})",
        "ProtocolJsonHandler must parse textDocument/references requests");
    ExpectParsesRequest(
        handler,
        td_rename::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":5,"method":"textDocument/rename","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2},"newName":"renamed"}})",
        "ProtocolJsonHandler must parse textDocument/rename requests");
    ExpectParsesRequest(
        handler,
        td_formatting::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":6,"method":"textDocument/formatting","params":{"textDocument":{"uri":"file:///a.cpp"},"options":{"tabSize":4,"insertSpaces":true}}})",
        "ProtocolJsonHandler must parse textDocument/formatting requests");
    ExpectParsesRequest(
        handler,
        td_codeAction::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":7,"method":"textDocument/codeAction","params":{"textDocument":{"uri":"file:///a.cpp"},"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"context":{"diagnostics":[]}}})",
        "ProtocolJsonHandler must parse textDocument/codeAction requests");
    ExpectParsesRequest(
        handler,
        td_symbol::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":8,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":"file:///a.cpp"}}})",
        "ProtocolJsonHandler must parse textDocument/documentSymbol requests");
    ExpectParsesRequest(
        handler,
        wp_symbol::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":9,"method":"workspace/symbol","params":{"query":"foo"}})",
        "ProtocolJsonHandler must parse workspace/symbol requests");
    ExpectParsesRequest(
        handler,
        wp_executeCommand::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":16,"method":"workspace/executeCommand","params":{"command":"build.project","arguments":[]}})",
        "ProtocolJsonHandler must parse workspace/executeCommand requests");

    ExpectParsesNotification(
        handler,
        Notify_TextDocumentPublishDiagnostics::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":"file:///a.cpp","diagnostics":[]}})",
        "ProtocolJsonHandler must parse publishDiagnostics notifications");
    ExpectParsesNotification(
        handler,
        Notify_TextDocumentDidOpen::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///a.cpp","languageId":"cpp","version":1,"text":"int main(){}"}}})",
        "ProtocolJsonHandler must parse didOpen notifications");
    ExpectParsesNotification(
        handler,
        Notify_TextDocumentDidChange::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///a.cpp","version":2},"contentChanges":[{"text":"int main(){return 0;}"}]}})",
        "ProtocolJsonHandler must parse didChange notifications");
    ExpectParsesNotification(
        handler,
        Notify_TextDocumentDidClose::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":"file:///a.cpp"}}})",
        "ProtocolJsonHandler must parse didClose notifications");

    ExpectParsesRequest(
        handler,
        td_initialize::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":10,"method":"initialize","params":{"processId":1234,"capabilities":{}}})",
        "ProtocolJsonHandler must parse initialize requests");
    ExpectParsesRequest(
        handler,
        td_shutdown::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":11,"method":"shutdown","params":null})",
        "ProtocolJsonHandler must parse shutdown requests");
    ExpectParsesRequest(
        handler,
        td_declaration::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":12,"method":"textDocument/declaration","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2}}})",
        "ProtocolJsonHandler must parse textDocument/declaration requests");
    ExpectParsesRequest(
        handler,
        td_codeLens::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":13,"method":"textDocument/codeLens","params":{"textDocument":{"uri":"file:///a.cpp"}}})",
        "ProtocolJsonHandler must parse textDocument/codeLens requests");
    ExpectParsesRequest(
        handler,
        td_highlight::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":14,"method":"textDocument/documentHighlight","params":{"textDocument":{"uri":"file:///a.cpp"},"position":{"line":1,"character":2}}})",
        "ProtocolJsonHandler must parse textDocument/documentHighlight requests");

    ExpectParsesNotification(
        handler,
        Notify_InitializedNotification::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"initialized","params":{}})",
        "ProtocolJsonHandler must parse initialized notifications");
    ExpectParsesNotification(
        handler,
        Notify_TextDocumentDidSave::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"textDocument/didSave","params":{"textDocument":{"uri":"file:///a.cpp"},"text":"saved"}})",
        "ProtocolJsonHandler must parse didSave notifications");
    ExpectParsesNotification(
        handler,
        td_willSave::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"textDocument/willSave","params":{"textDocument":{"uri":"file:///a.cpp"},"reason":1}})",
        "ProtocolJsonHandler must parse willSave notifications");
    ExpectParsesNotification(
        handler,
        Notify_LogMessage::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"window/logMessage","params":{"type":3,"message":"server log"}})",
        "ProtocolJsonHandler must parse window/logMessage notifications");

    ExpectParsesResponse(
        handler,
        td_initialize::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":20,"result":{"capabilities":{"hoverProvider":true}}})",
        "ProtocolJsonHandler must parse initialize responses");
    ExpectParsesResponse(
        handler,
        td_shutdown::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":21,"result":null})",
        "ProtocolJsonHandler must parse shutdown responses");
    ExpectParsesResponse(
        handler,
        td_completion::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":22,"result":{"isIncomplete":false,"items":[{"label":"item"}]}})",
        "ProtocolJsonHandler must parse completion responses");
    ExpectParsesResponse(
        handler,
        td_hover::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":23,"result":{"contents":{"kind":"markdown","value":"doc"}}})",
        "ProtocolJsonHandler must parse hover responses");
    ExpectParsesResponse(
        handler,
        td_definition::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":24,"result":[{"uri":"file:///a.cpp","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}]})",
        "ProtocolJsonHandler must parse definition responses");
    ExpectParsesResponse(
        handler,
        td_codeAction::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":25,"result":[{"title":"Fix","command":"cmd.fix"}]})",
        "ProtocolJsonHandler must parse codeAction responses");
    ExpectParsesResponse(
        handler,
        td_codeLens::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":26,"result":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}]})",
        "ProtocolJsonHandler must parse codeLens responses");
    ExpectParsesResponse(
        handler,
        td_highlight::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":27,"result":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"kind":1}]})",
        "ProtocolJsonHandler must parse documentHighlight responses");
    ExpectParsesResponse(
        handler,
        td_declaration::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":28,"result":[{"uri":"file:///a.cpp","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}]})",
        "ProtocolJsonHandler must parse declaration responses");

    ExpectParsesResponse(
        handler,
        td_definition::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":29,"result":{"uri":"file:///single.cpp","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}})",
        "ProtocolJsonHandler must parse single Location definition responses");
}

void TestProtocolJsonHandlerParsesNoParamsMessages()
{
    lsp::ProtocolJsonHandler handler;

    ExpectParsesRequest(
        handler,
        td_shutdown::request::kMethodInfo,
        R"({"jsonrpc":"2.0","id":30,"method":"shutdown"})",
        "ProtocolJsonHandler must parse shutdown requests without params");
    ExpectParsesNotification(
        handler,
        Notify_Exit::notify::kMethodInfo,
        R"({"jsonrpc":"2.0","method":"exit"})",
        "ProtocolJsonHandler must parse exit notifications without params");
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
    RUN_TEST(TestPositionRangeLocationAndTextEditRoundTrip);
    RUN_TEST(TestWorkspaceEditAndDiagnosticRoundTrip);
    RUN_TEST(TestCompletionHoverAndInitializeRoundTrip);
    RUN_TEST(TestTextDocumentSyncUnionRoundTrip);
    RUN_TEST(TestDocumentUriEscapingRoundTrip);
    RUN_TEST(TestDocumentUriFromPathAndSimpleFileUri);
    RUN_TEST(TestDocumentUriReservedCharacterEscaping);
    RUN_TEST(TestDocumentUriPercentAndUtf8Escaping);
    RUN_TEST(TestDocumentUriFileLocalhostAndInvalidPercentDecoding);
    RUN_TEST(TestDocumentUriRelativePathAndQueryFragmentGuards);
    RUN_TEST(TestAbsolutePathFromNormalized);
    RUN_TEST(TestDirectoryEnsuresTrailingSlash);
    RUN_TEST(TestAbsolutePathJsonRoundTrip);
    RUN_TEST(TestAbsolutePathValidationNormalizesRelativePaths);
    RUN_TEST(TestAbsolutePathQualifyConsistency);
    RUN_TEST(TestAbsolutePathOperatorsAndConversion);
    RUN_TEST(TestNormalizePathAllowsMissingPathsWithoutStatState);
    RUN_TEST(TestAbsolutePathValidatePreservesAbsoluteInput);
#ifndef _WIN32
    RUN_TEST(TestNormalizePathCollapsesAbsoluteComponents);
    RUN_TEST(TestNormalizePathEnsureExists);
    RUN_TEST(TestDocumentUriDotDotNormalization);
#endif
    RUN_TEST(TestAbsolutePathRecognitionGuards);
#if defined(_WIN32)
    RUN_TEST(TestDocumentUriWindowsDriveLetterEscaping);
    RUN_TEST(TestNormalizePathWindowsDriveLetterAndForceLower);
    RUN_TEST(TestNormalizePathWindowsEnsureExistsAndMaxPathGuard);
    RUN_TEST(TestDocumentUriDotDotNormalizationWindows);
#endif
    RUN_TEST(TestDocumentUriGetAbsolutePathInvalidPaths);
    RUN_TEST(TestDocumentUriUncAuthorityRoundTrip);
#ifndef _WIN32
    RUN_TEST(TestNormalizePathForceLowerIgnoredOnUnix);
#endif
    RUN_TEST(TestDocumentUriDefaultCopyEqualityAndSwap);
    RUN_TEST(TestDocumentUriNonFileSchemePassthrough);
    RUN_TEST(TestDocumentUriJsonDeserialization);
    RUN_TEST(TestOptionalFieldsAreOmittedWhenUnsetAndReadFromNull);
    RUN_TEST(TestLocationListEitherRoundTrip);
    RUN_TEST(TestCompletionResponseArrayVsList);
    RUN_TEST(TestCodeActionEitherVariants);
    RUN_TEST(TestWorkspaceEditDocumentChanges);
    RUN_TEST(TestHoverContentsAllRepresentations);
    RUN_TEST(TestTypeGuardInspiredMalformedInputs);
    RUN_TEST(TestProgressTokenParamsRoundTrip);
    RUN_TEST(TestDocumentSymbolEitherVariants);
    RUN_TEST(TestProtocolJsonHandlerParsesPolymorphicResponseEnvelopes);
    RUN_TEST(TestProtocolJsonHandlerParsesErrorResponses);
    RUN_TEST(TestProtocolJsonHandlerRegistersCoreRequestsAndNotifications);
    RUN_TEST(TestProtocolJsonHandlerParsesNoParamsMessages);
    return test::Failures() == 0 ? 0 : 1;
}
