#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_close.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include "LibLsp/lsp/working_files.h"
#include "test_helpers.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{
using test::DummyLog;
using test::Expect;
using test::FeedableIStream;
using test::StringOStream;

AbsolutePath MakePath(std::string const& name)
{
    return AbsolutePath("/tmp/lspcpp-text-document-sync-suite/" + name);
}

std::shared_ptr<WorkingFile> OpenFile(WorkingFiles& files, AbsolutePath const& path, std::string text, int version = 1)
{
    lsTextDocumentItem open;
    open.uri = lsDocumentUri(path);
    open.languageId = "cpp";
    open.version = version;
    open.text = std::move(text);
    return files.OnOpen(open);
}

std::string ContentOf(WorkingFiles& files, std::shared_ptr<WorkingFile>& file)
{
    std::string content;
    Expect(files.GetFileBufferContent(file, content), "expected file content to be available");
    return content;
}

std::vector<size_t> LineStarts(std::string const& text)
{
    std::vector<size_t> starts;
    starts.push_back(0);
    for (size_t i = 0; i < text.size(); ++i)
    {
        if (text[i] == '\n')
        {
            starts.push_back(i + 1);
        }
    }
    return starts;
}

lsPosition PositionAt(std::string const& text, size_t offset)
{
    offset = std::min(offset, text.size());
    unsigned line = 0;
    size_t line_start = 0;
    for (size_t i = 0; i < offset; ++i)
    {
        if (text[i] == '\n')
        {
            ++line;
            line_start = i + 1;
        }
    }
    return lsPosition(static_cast<int>(line), static_cast<int>(offset - line_start));
}

size_t LineLength(std::string const& text, std::vector<size_t> const& starts, size_t line)
{
    size_t const start = starts[line];
    size_t const next = (line + 1 < starts.size()) ? starts[line + 1] : text.size();
    size_t end = next;
    if (end > start && text[end - 1] == '\n')
    {
        --end;
    }
    return end - start;
}

void AssertValidLineOffsets(WorkingFile& file, std::string const& expected)
{
    auto const starts = LineStarts(expected);
    Expect(file.LineOffsetCountForTest() == starts.size(), "line offset count must match document line count");

    for (size_t line = 0; line < starts.size(); ++line)
    {
        Expect(
            file.GetOffsetForPosition(lsPosition(static_cast<int>(line), 0)) == static_cast<int>(starts[line]),
            "line start offset mismatch");
        size_t const length = LineLength(expected, starts, line);
        Expect(
            file.GetOffsetForPosition(lsPosition(static_cast<int>(line), static_cast<int>(length))) ==
                static_cast<int>(starts[line] + length),
            "line end offset mismatch");
    }

    Expect(
        file.GetOffsetForPosition(lsPosition(static_cast<int>(starts.size() + 10), 0)) ==
            static_cast<int>(expected.size()),
        "out-of-range line must clamp to document end");
}

void AssertDocument(WorkingFiles& files, std::shared_ptr<WorkingFile>& file, std::string const& expected, int version)
{
    Expect(ContentOf(files, file) == expected, "document content mismatch after didChange");
    Expect(file->version == version, "document version mismatch after didChange");
    AssertValidLineOffsets(*file, expected);
}

lsTextDocumentContentChangeEvent FullEdit(std::string text)
{
    lsTextDocumentContentChangeEvent edit;
    edit.text = std::move(text);
    return edit;
}

lsTextDocumentContentChangeEvent RangedEdit(lsPosition start, lsPosition end, std::string text)
{
    lsTextDocumentContentChangeEvent edit;
    edit.range = lsRange(start, end);
    edit.text = std::move(text);
    return edit;
}

lsTextDocumentContentChangeEvent ReplaceFirst(std::string& model, std::string const& old_text, std::string replacement)
{
    size_t const start = model.find(old_text);
    Expect(start != std::string::npos, "test fixture target text must exist");
    size_t const end = start + old_text.size();
    auto edit = RangedEdit(PositionAt(model, start), PositionAt(model, end), replacement);
    model.replace(start, old_text.size(), replacement);
    return edit;
}

lsTextDocumentContentChangeEvent InsertAfter(std::string& model, std::string const& marker, std::string insertion)
{
    size_t const start = model.find(marker);
    Expect(start != std::string::npos, "test fixture marker text must exist");
    size_t const offset = start + marker.size();
    auto edit = RangedEdit(PositionAt(model, offset), PositionAt(model, offset), insertion);
    model.insert(offset, insertion);
    return edit;
}

lsTextDocumentContentChangeEvent InsertAtOffset(std::string& model, size_t offset, std::string insertion)
{
    offset = std::min(offset, model.size());
    auto edit = RangedEdit(PositionAt(model, offset), PositionAt(model, offset), insertion);
    model.insert(offset, insertion);
    return edit;
}

lsTextDocumentContentChangeEvent DeleteRange(std::string& model, size_t start, size_t end)
{
    start = std::min(start, model.size());
    end = std::min(end, model.size());
    auto edit = RangedEdit(PositionAt(model, start), PositionAt(model, end), "");
    model.erase(start, end - start);
    return edit;
}

std::shared_ptr<WorkingFile> ApplyChanges(
    WorkingFiles& files,
    AbsolutePath const& path,
    int version,
    std::vector<lsTextDocumentContentChangeEvent> changes)
{
    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.textDocument.version = optional<int>(version);
    change.contentChanges = std::move(changes);
    return files.OnChange(change);
}

void TestFullContentUpdates()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("full-updates.cpp");
    auto file = OpenFile(files, path, "abc123", 0);

    auto changed = ApplyChanges(files, path, 1, {FullEdit("efg456")});
    Expect(changed == file, "full update must return the tracked file");
    AssertDocument(files, file, "efg456", 1);

    ApplyChanges(files, path, 2, {FullEdit("hello")});
    ApplyChanges(files, path, 3, {FullEdit("world")});
    AssertDocument(files, file, "world", 3);
}

void TestIncrementalRemovals()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("incremental-removals.cpp");
    std::string model = "function abc() {\n  console.log(\"hello, world!\");\n}\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {ReplaceFirst(model, ", world!", "")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {ReplaceFirst(model, "  console.log(\"hello\");\n", "")});
    AssertDocument(files, file, model, 3);
}

void TestIncrementalMultiLineRemovals()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("multi-line-removals.cpp");
    std::string model = "function abc() {\n  foo();\n  bar();\n  baz();\n}\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {ReplaceFirst(model, "  foo();\n  bar();\n", "")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {ReplaceFirst(model, "  baz();\n}", "}")});
    AssertDocument(files, file, model, 3);
}

void TestIncrementalAdditions()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("incremental-additions.cpp");
    std::string model = "function abc() {\n  console.log(\"hello\");\n}\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {InsertAfter(model, "hello", ", world!")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {InsertAfter(model, "console.log(\"hello, world!\");", "\n  bar();")});
    AssertDocument(files, file, model, 3);
}

void TestSingleLineReplacements()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("single-line-replacements.cpp");
    std::string model = "function abc() {\n  console.log(\"hello, world!\");\n}\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {ReplaceFirst(model, "hello, world!", "hello, test case!!!")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {ReplaceFirst(model, "hello, test case!!!", "short")});
    AssertDocument(files, file, model, 3);

    ApplyChanges(files, path, 4, {ReplaceFirst(model, "short", "equal")});
    AssertDocument(files, file, model, 4);
}

void TestMultiLineReplacements()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("multi-line-replacements.cpp");
    std::string model = "a1\nb1\nc1\na2\nb2\nc2\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {ReplaceFirst(model, "b1\nc1", "b1\nx1\ny1\nc1")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {ReplaceFirst(model, "b1\nx1\ny1\nc1", "b1")});
    AssertDocument(files, file, model, 3);

    ApplyChanges(files, path, 4, {ReplaceFirst(model, "a2\nb2\nc2", "d2\ne2\nf2")});
    AssertDocument(files, file, model, 4);
}

void TestHugeMultiLineInsertion()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("huge-insertion.cpp");
    std::string model = "prefix\nsuffix\n";
    auto file = OpenFile(files, path, model, 1);

    std::string insertion;
    for (int i = 0; i < 19999; ++i)
    {
        insertion += "line ";
        insertion += std::to_string(i);
        insertion += "\n";
    }
    ApplyChanges(files, path, 2, {InsertAfter(model, "prefix\n", insertion)});
    AssertDocument(files, file, model, 2);
}

void TestSeveralIncrementalChangesInOneNotification()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("several-changes.cpp");
    std::string model = "function abc() {\n  console.log(\"hello\");\n}\n";
    auto file = OpenFile(files, path, model, 1);

    std::vector<lsTextDocumentContentChangeEvent> changes;
    changes.push_back(InsertAfter(model, "abc", "defg"));
    changes.push_back(ReplaceFirst(model, "\"hello\"", "\"hello, world!\""));
    changes.push_back(InsertAfter(model, "abcdefg", "hij"));

    ApplyChanges(files, path, 2, std::move(changes));
    AssertDocument(files, file, model, 2);
}

void TestAppendDeleteAndCharacterReplace()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("append-delete-replace.cpp");
    std::string model = "0123456789\nabcdefghij\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {InsertAtOffset(model, 10, "A")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {InsertAtOffset(model, model.size(), "tail\nmore")});
    AssertDocument(files, file, model, 3);

    ApplyChanges(files, path, 4, {DeleteRange(model, 1, 4)});
    AssertDocument(files, file, model, 4);

    size_t const multi_start = model.find("abcd");
    ApplyChanges(files, path, 5, {DeleteRange(model, multi_start, multi_start + 7)});
    AssertDocument(files, file, model, 5);

    ApplyChanges(files, path, 6, {ReplaceFirst(model, "0", "X")});
    AssertDocument(files, file, model, 6);

    ApplyChanges(files, path, 7, {ReplaceFirst(model, "tail", "TAIL-UPDATED")});
    AssertDocument(files, file, model, 7);
}

void TestEditStyleInsertReplaceAndMultiline()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("edits-style.cpp");
    std::string model = "012345678901234567890123456789";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {InsertAtOffset(model, 0, "a"), InsertAtOffset(model, 2, "b")});
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {ReplaceFirst(model, "345", "XYZ")});
    AssertDocument(files, file, model, 3);

    ApplyChanges(files, path, 4, {ReplaceFirst(model, "XYZ67", "one\ntwo\nthree")});
    AssertDocument(files, file, model, 4);
}

void TestInvalidRangesClamp()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("invalid-ranges.cpp");
    std::string model = "abc\ndef\n";
    auto file = OpenFile(files, path, model, 1);

    ApplyChanges(files, path, 2, {RangedEdit(lsPosition(-1, -1), lsPosition(0, 1), "A")});
    model.replace(0, 1, "A");
    AssertDocument(files, file, model, 2);

    ApplyChanges(files, path, 3, {RangedEdit(lsPosition(-10, 0), lsPosition(1, 1), "prefix")});
    model.replace(0, std::string("Abc\nd").size(), "prefix");
    AssertDocument(files, file, model, 3);

    size_t const from_middle = model.find("ef");
    ApplyChanges(files, path, 4, {RangedEdit(PositionAt(model, from_middle), lsPosition(99, 99), "")});
    model.erase(from_middle);
    AssertDocument(files, file, model, 4);

    ApplyChanges(files, path, 5, {RangedEdit(lsPosition(99, 99), lsPosition(99, 99), "\nend")});
    model += "\nend";
    AssertDocument(files, file, model, 5);

    ApplyChanges(files, path, 6, {RangedEdit(lsPosition(-1, 0), lsPosition(99, 99), "whole")});
    model = "whole";
    AssertDocument(files, file, model, 6);
}

bool WaitUntil(std::function<bool()> const& predicate)
{
    for (int i = 0; i < 100; ++i)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
}

std::string JsonEscape(std::string const& value)
{
    std::string out;
    for (char c : value)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(c);
        }
    }
    return out;
}

// Runs didOpen/didChange LSP frames through a RemoteEndPoint into WorkingFiles,
// mirroring vscode-languageserver-node server TextDocuments manager tests.
struct EndpointDocumentHarness
{
    DummyLog log;
    WorkingFiles files;
    std::shared_ptr<FeedableIStream> input_stream = std::make_shared<FeedableIStream>();
    std::shared_ptr<StringOStream> output_stream = std::make_shared<StringOStream>();
    std::unique_ptr<RemoteEndPoint> point;
    std::atomic<int> handled_notifications {0};

    EndpointDocumentHarness()
    {
        auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
        auto endpoint = std::make_shared<GenericEndpoint>(log);
        // Exercise document-sync ordering under multiple workers: didOpen and
        // incremental didChange notifications must still be applied in wire order.
        point.reset(new RemoteEndPoint(json_handler, endpoint, log, lsp::JSONStreamStyle::Standard, 4));
        point->registerHandler(
            [this](Notify_TextDocumentDidOpen::notify const& notification)
            {
                auto params = notification.params;
                files.OnOpen(params.textDocument);
                handled_notifications.fetch_add(1, std::memory_order_relaxed);
            });
        point->registerHandler(
            [this](Notify_TextDocumentDidChange::notify const& notification)
            {
                files.OnChange(notification.params);
                handled_notifications.fetch_add(1, std::memory_order_relaxed);
            });
        point->startProcessingMessages(input_stream, output_stream);
    }

    ~EndpointDocumentHarness()
    {
        point->stop();
    }

    void Open(std::string const& uri, std::string const& text, int version)
    {
        input_stream->append(test::MakeLspFrame(
            R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":")" + uri +
            R"(","languageId":"javascript","version":)" + std::to_string(version) + R"(,"text":")" +
            JsonEscape(text) + R"("}}})"));
    }

    void Change(std::string const& uri, int version, std::string const& content_changes_json)
    {
        input_stream->append(test::MakeLspFrame(
            R"({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":")" + uri +
            R"(","version":)" + std::to_string(version) + R"(},"contentChanges":)" + content_changes_json + "}}"));
    }

    bool WaitForNotifications(int expected)
    {
        return WaitUntil(
            [this, expected]()
            {
                return handled_notifications.load(std::memory_order_relaxed) >= expected;
            });
    }
};

void ExpectEndpointDocument(EndpointDocumentHarness& harness, AbsolutePath const& path, std::string const& expected, int version)
{
    auto file = harness.files.GetFileByFilename(path);
    Expect(file != nullptr, "endpoint-managed document must exist after didOpen");
    if (file)
    {
        AssertDocument(harness.files, file, expected, version);
    }
}

void TestSeveralFullContentUpdatesThroughEndpoint()
{
    EndpointDocumentHarness harness;
    AbsolutePath const path = MakePath("endpoint-full-updates.js");
    std::string const uri = lsDocumentUri(path).raw_uri_;

    harness.Open(uri, "abc123", 1);
    harness.Change(uri, 2, R"([{"text":"hello"},{"text":"world"}])");
    Expect(harness.WaitForNotifications(2), "full-update didChange frame must reach WorkingFiles");

    ExpectEndpointDocument(harness, path, "world", 2);
}

void TestIncrementalRemovalThroughEndpoint()
{
    EndpointDocumentHarness harness;
    AbsolutePath const path = MakePath("endpoint-incremental-remove.js");
    std::string const uri = lsDocumentUri(path).raw_uri_;

    harness.Open(uri, "function abc() {\n  console.log(\"hello, world!\");\n}", 1);
    harness.Change(
        uri, 2, R"([{"text":"","range":{"start":{"line":1,"character":20},"end":{"line":1,"character":28}}}])");
    Expect(harness.WaitForNotifications(2), "incremental removal frame must reach WorkingFiles");

    ExpectEndpointDocument(harness, path, "function abc() {\n  console.log(\"hello\");\n}", 2);
}

void TestIncrementalAdditionThroughEndpoint()
{
    EndpointDocumentHarness harness;
    AbsolutePath const path = MakePath("endpoint-incremental-add.js");
    std::string const uri = lsDocumentUri(path).raw_uri_;

    harness.Open(uri, "function abc() {\n  console.log(\"hello\");\n}", 1);
    harness.Change(
        uri, 2,
        R"([{"text":", world!","range":{"start":{"line":1,"character":20},"end":{"line":1,"character":20}}}])");
    Expect(harness.WaitForNotifications(2), "incremental addition frame must reach WorkingFiles");

    ExpectEndpointDocument(harness, path, "function abc() {\n  console.log(\"hello, world!\");\n}", 2);
}

void TestIncrementalReplacementThroughEndpoint()
{
    EndpointDocumentHarness harness;
    AbsolutePath const path = MakePath("endpoint-incremental-replace.js");
    std::string const uri = lsDocumentUri(path).raw_uri_;

    harness.Open(uri, "function abc() {\n  console.log(\"hello, world!\");\n}", 1);
    harness.Change(
        uri, 2,
        R"([{"text":"hello, test case!!!","range":{"start":{"line":1,"character":15},"end":{"line":1,"character":28}}}])");
    Expect(harness.WaitForNotifications(2), "incremental replacement frame must reach WorkingFiles");

    ExpectEndpointDocument(harness, path, "function abc() {\n  console.log(\"hello, test case!!!\");\n}", 2);
}

void TestSeveralIncrementalChangesThroughEndpoint()
{
    EndpointDocumentHarness harness;
    AbsolutePath const path = MakePath("endpoint-incremental-several.js");
    std::string const uri = lsDocumentUri(path).raw_uri_;

    harness.Open(uri, "function abc() {\n  console.log(\"hello, world!\");\n}", 1);
    // Positions of later edits are relative to the document state after
    // earlier edits within the same didChange, matching vscode semantics.
    harness.Change(
        uri, 2,
        R"([{"text":"defg","range":{"start":{"line":0,"character":12},"end":{"line":0,"character":12}}},)"
        R"({"text":"hello, test case!!!","range":{"start":{"line":1,"character":15},"end":{"line":1,"character":28}}},)"
        R"({"text":"hij","range":{"start":{"line":0,"character":16},"end":{"line":0,"character":16}}}])");
    Expect(harness.WaitForNotifications(2), "multi-edit didChange frame must reach WorkingFiles");

    ExpectEndpointDocument(
        harness, path, "function abcdefghij() {\n  console.log(\"hello, test case!!!\");\n}", 2);
}

void TestDidOpenChangeCloseThroughEndpoint()
{
    DummyLog log;
    WorkingFiles files;
    AbsolutePath const path = MakePath("endpoint.cpp");
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    std::atomic<bool> opened {false};
    std::atomic<bool> changed {false};
    std::atomic<bool> closed {false};

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](Notify_TextDocumentDidOpen::notify const& notification)
        {
            auto params = notification.params;
            opened.store(files.OnOpen(params.textDocument) != nullptr, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidChange::notify const& notification)
        {
            changed.store(files.OnChange(notification.params) != nullptr, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidClose::notify const& notification)
        {
            closed.store(files.OnClose(notification.params.textDocument), std::memory_order_relaxed);
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const uri = lsDocumentUri(path).raw_uri_;
    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":")" + uri +
        R"(","languageId":"cpp","version":1,"text":"abc123"}}})"));
    Expect(WaitUntil([&]() { return opened.load(std::memory_order_relaxed); }), "didOpen must reach WorkingFiles");

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":")" + uri +
        R"(","version":2},"contentChanges":[{"text":"efg456"}]}})"));
    Expect(WaitUntil([&]() { return changed.load(std::memory_order_relaxed); }), "didChange must reach WorkingFiles");

    auto file = files.GetFileByFilename(path);
    Expect(file != nullptr, "didOpen must make the endpoint document available by path");
    if (file)
    {
        AssertDocument(files, file, "efg456", 2);
    }

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":")" + uri +
        R"("}}})"));
    Expect(WaitUntil([&]() { return closed.load(std::memory_order_relaxed); }), "didClose must reach WorkingFiles");
    point.stop();

    std::string content;
    Expect(!files.GetFileBufferContent(path, content), "didClose must remove the endpoint document");
}

// Adapted from clangd/test/version.test
void TestDidChangeVersionPropagationThroughEndpoint()
{
    DummyLog log;
    WorkingFiles files;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);
    auto input_stream = std::make_shared<FeedableIStream>();
    auto output_stream = std::make_shared<StringOStream>();
    AbsolutePath const path = MakePath("version-endpoint.c");
    std::atomic<bool> opened {false};

    RemoteEndPoint point(json_handler, endpoint, log);
    point.registerHandler(
        [&](Notify_TextDocumentDidOpen::notify const& notification)
        {
            auto params = notification.params;
            opened.store(files.OnOpen(params.textDocument) != nullptr, std::memory_order_relaxed);
        });
    point.registerHandler(
        [&](Notify_TextDocumentDidChange::notify const& notification)
        {
            files.OnChange(notification.params);
        });
    point.startProcessingMessages(input_stream, output_stream);

    std::string const uri = lsDocumentUri(path).raw_uri_;
    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":")" + uri +
        R"(","languageId":"c","version":0,"text":""}}})"));
    Expect(WaitUntil([&]() { return opened.load(std::memory_order_relaxed); }), "version scenario didOpen must dispatch");

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":")" + uri +
        R"(","version":5},"contentChanges":[{"text":"a"}]}})"));
    Expect(
        WaitUntil(
            [&]()
            {
                auto current = files.GetFileByFilename(path);
                if (current == nullptr)
                {
                    return false;
                }
                std::string content;
                return files.GetFileBufferContent(current, content) && content == "a" && current->version == 5;
            }),
        "version scenario didChange must apply explicit version and content");
    auto file = files.GetFileByFilename(path);
    Expect(file != nullptr, "version scenario file must exist after didChange");
    if (file)
    {
        AssertDocument(files, file, "a", 5);
    }

    input_stream->append(test::MakeLspFrame(
        R"({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":")" + uri +
        R"("},"contentChanges":[{"text":"b"}]}})"));
    Expect(
        WaitUntil(
            [&]()
            {
                auto current = files.GetFileByFilename(path);
                if (current == nullptr)
                {
                    return false;
                }
                std::string content;
                return files.GetFileBufferContent(current, content) && content == "b" && current->version == 5;
            }),
        "version scenario didChange without version must preserve previous version");
    if (file)
    {
        AssertDocument(files, file, "b", 5);
    }

    point.stop();
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestFullContentUpdates);
    RUN_TEST(TestIncrementalRemovals);
    RUN_TEST(TestIncrementalMultiLineRemovals);
    RUN_TEST(TestIncrementalAdditions);
    RUN_TEST(TestSingleLineReplacements);
    RUN_TEST(TestMultiLineReplacements);
    RUN_TEST(TestHugeMultiLineInsertion);
    RUN_TEST(TestSeveralIncrementalChangesInOneNotification);
    RUN_TEST(TestAppendDeleteAndCharacterReplace);
    RUN_TEST(TestEditStyleInsertReplaceAndMultiline);
    RUN_TEST(TestInvalidRangesClamp);
    RUN_TEST(TestDidOpenChangeCloseThroughEndpoint);
    RUN_TEST(TestSeveralFullContentUpdatesThroughEndpoint);
    RUN_TEST(TestIncrementalRemovalThroughEndpoint);
    RUN_TEST(TestIncrementalAdditionThroughEndpoint);
    RUN_TEST(TestIncrementalReplacementThroughEndpoint);
    RUN_TEST(TestSeveralIncrementalChangesThroughEndpoint);
    RUN_TEST(TestDidChangeVersionPropagationThroughEndpoint);
    return test::Failures() == 0 ? 0 : 1;
}
