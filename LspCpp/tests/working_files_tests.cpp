#include "LibLsp/lsp/working_files.h"
#include "LibLsp/lsp/utils.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace
{
using test::Expect;

AbsolutePath MakePath(std::string const& name)
{
    return AbsolutePath("/tmp/lspcpp-working-files-suite/" + name);
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

lsTextDocumentContentChangeEvent RangedEdit(lsPosition start, lsPosition end, std::string text)
{
    lsTextDocumentContentChangeEvent edit;
    edit.range = lsRange(start, end);
    edit.text = std::move(text);
    return edit;
}

std::string ContentOf(WorkingFiles& files, std::shared_ptr<WorkingFile>& file)
{
    std::string content;
    Expect(files.GetFileBufferContent(file, content), "expected open file content to be available");
    return content;
}

void TestOffsetBoundariesAndLineEndings()
{
    WorkingFiles files;
    auto file = OpenFile(files, MakePath("offsets.cpp"), "a\r\nbb\nccc");

    Expect(file->GetOffsetForPosition(lsPosition(0, 0)) == 0, "offset for file start must be zero");
    Expect(file->GetOffsetForPosition(lsPosition(0, 2)) == 2, "CRLF line should count carriage return as a character");
    Expect(file->GetOffsetForPosition(lsPosition(1, 1)) == 4, "offset for second line must use cached line start");
    Expect(file->GetOffsetForPosition(lsPosition(2, 3)) == 9, "offset at end of last line mismatch");
    Expect(
        file->GetOffsetForPosition(lsPosition(20, 0)) == static_cast<int>(file->GetContentNoLock().size()),
        "out-of-range line must clamp to end of document");
    Expect(
        file->GetOffsetForPosition(lsPosition(2, 99)) == static_cast<int>(file->GetContentNoLock().size()),
        "out-of-range character must clamp to end of document");
}

void TestUtf8OffsetsUseCodePointCounting()
{
    WorkingFiles files;
    auto file = OpenFile(files, MakePath("utf8.cpp"), "a\xc3\xa9\nb\xf0\x9f\x98\x80z");

    Expect(file->GetOffsetForPosition(lsPosition(0, 1)) == 1, "ASCII prefix offset mismatch");
    Expect(file->GetOffsetForPosition(lsPosition(0, 2)) == 3, "UTF-8 two-byte code point must advance as one character");
    Expect(
        file->GetOffsetForPosition(lsPosition(1, 3)) == 9,
        "UTF-8 four-byte code point must advance as two UTF-16 code units");
    Expect(
        file->GetOffsetForPosition(lsPosition(1, 4)) == 10,
        "cached offsets must resume single-unit counting after a surrogate pair");
    Expect(
        file->GetOffsetForPosition(lsPosition(1, 2)) == 9,
        "character indexes inside a surrogate pair must snap past the whole code point");
}

void TestMultipleEditsRebuildLineOffsetsSequentially()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("multi-edit.cpp");
    auto file = OpenFile(files, path, "one\ntwo\nthree\n");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.textDocument.version = optional<int>(2);
    change.contentChanges.push_back(RangedEdit(lsPosition(0, 0), lsPosition(0, 0), "zero\n"));
    change.contentChanges.push_back(RangedEdit(lsPosition(2, 0), lsPosition(3, 0), "TWO-AND-"));
    change.contentChanges.push_back(RangedEdit(lsPosition(2, 8), lsPosition(2, 13), "THREE"));

    files.OnChange(change);

    Expect(ContentOf(files, file) == "zero\none\nTWO-AND-THREE\n", "multiple ranged edits must apply sequentially");
    Expect(file->version == 2, "WorkingFiles must update version from didChange");
    Expect(file->LineOffsetCountForTest() == 4, "line offsets must be rebuilt after multi-edit change");
}

void TestFullReplaceAndDocumentBoundaryEdits()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("full-replace.cpp");
    auto file = OpenFile(files, path, "middle");

    lsTextDocumentDidChangeParams full_change;
    full_change.textDocument.uri = lsDocumentUri(path);
    full_change.textDocument.version = optional<int>(2);
    lsTextDocumentContentChangeEvent full_edit;
    full_edit.text = "alpha\nomega";
    full_change.contentChanges.push_back(full_edit);
    files.OnChange(full_change);
    Expect(ContentOf(files, file) == "alpha\nomega", "full document replace must replace all content");

    lsTextDocumentDidChangeParams boundary_change;
    boundary_change.textDocument.uri = lsDocumentUri(path);
    boundary_change.textDocument.version = optional<int>(3);
    boundary_change.contentChanges.push_back(RangedEdit(lsPosition(0, 0), lsPosition(0, 0), "BEGIN\n"));
    boundary_change.contentChanges.push_back(RangedEdit(lsPosition(2, 5), lsPosition(2, 5), "\nEND"));
    files.OnChange(boundary_change);

    Expect(
        ContentOf(files, file) == "BEGIN\nalpha\nomega\nEND",
        "boundary inserts must work at document start and end");
    Expect(file->LineOffsetCountForTest() == 4, "line offsets must include every line start after boundary edits");
}

void TestLifecycleOpenCloseClearAndDirectories()
{
    WorkingFiles files;
    AbsolutePath const first = MakePath("lifecycle/a.cpp");
    AbsolutePath const second = MakePath("lifecycle/b.cpp");
    auto first_file = OpenFile(files, first, "old", 1);
    auto reopened = OpenFile(files, first, "new", 7);
    auto second_file = OpenFile(files, second, "other", 1);

    Expect(first_file == reopened, "reopening the same URI must reuse the existing WorkingFile");
    Expect(reopened->version == 7, "reopening must update file version");
    Expect(ContentOf(files, reopened) == "new", "reopening must replace file content");

    std::wstring wide_content;
    Expect(files.GetFileBufferContent(reopened, wide_content), "wstring content lookup must succeed");
    Expect(wide_content == L"new", "wstring content lookup must convert current buffer");

    lsTextDocumentIdentifier close;
    close.uri = lsDocumentUri(first);
    Expect(files.OnClose(close), "OnClose must remove an open file");
    std::string closed_content;
    Expect(!files.GetFileBufferContent(first, closed_content), "closed file lookup by path must fail");
    Expect(!files.OnClose(close), "OnClose must report false for already-closed file");

    files.CloseFilesInDirectory({Directory(AbsolutePath("/tmp/lspcpp-working-files-suite/lifecycle/"))});
    Expect(!files.GetFileBufferContent(second, closed_content), "CloseFilesInDirectory must remove matching files");

    AbsolutePath const clear_path = MakePath("clear.cpp");
    OpenFile(files, clear_path, "present");
    files.Clear();
    Expect(!files.GetFileBufferContent(clear_path, closed_content), "Clear must remove all tracked files");
}

void TestCloseFilesInDirectoryTrailingSlashMatching()
{
    WorkingFiles files;
    AbsolutePath const first = MakePath("dir-match/a.cpp");
    AbsolutePath const second = MakePath("dir-match/b.cpp");
    AbsolutePath const unrelated = MakePath("other-dir/file.cpp");
    OpenFile(files, first, "first");
    OpenFile(files, second, "second");
    OpenFile(files, unrelated, "stay");

    files.CloseFilesInDirectory({Directory(AbsolutePath("/tmp/lspcpp-working-files-suite/dir-match"))});

    std::string content;
    Expect(!files.GetFileBufferContent(first, content), "CloseFilesInDirectory must match directory paths without trailing slash");
    Expect(!files.GetFileBufferContent(second, content), "CloseFilesInDirectory must remove every file in the matched directory");
    Expect(files.GetFileBufferContent(unrelated, content), "CloseFilesInDirectory must not remove files outside the matched directory");
    Expect(content == "stay", "CloseFilesInDirectory must leave unrelated files untouched");

    OpenFile(files, first, "again");
    files.CloseFilesInDirectory({Directory(AbsolutePath("/tmp/lspcpp-working-files-suite/dir-match/"))});
    Expect(!files.GetFileBufferContent(first, content), "CloseFilesInDirectory must match directory paths with trailing slash");
}

void TestOnSaveWritesCurrentContent()
{
    WorkingFiles files;
    AbsolutePath const path("/tmp/lspcpp-working-files-save-test.txt");
    std::remove(path.path().c_str());
    auto file = OpenFile(files, path, "saved content");

    lsTextDocumentIdentifier save;
    save.uri = lsDocumentUri(path);
    auto saved = files.OnSave(save);

    Expect(saved == file, "OnSave must return the saved WorkingFile");
    auto disk_content = lsp::ReadContent(path);
    Expect(disk_content.has_value(), "OnSave must create the target file");
    if (disk_content)
    {
        Expect(*disk_content == "saved content", "OnSave must write the current buffer content");
    }
    std::remove(path.path().c_str());
}

void TestNonFileUrisDoNotEnterWorkingFiles()
{
    WorkingFiles files;

    lsTextDocumentItem open;
    open.uri.raw_uri_ = "untitled:Untitled-1";
    open.languageId = "cpp";
    open.version = 1;
    open.text = "scratch";
    Expect(!files.OnOpen(open), "non-file URIs must not be tracked as working files");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri.raw_uri_ = "untitled:Untitled-1";
    change.contentChanges.push_back(lsTextDocumentContentChangeEvent());
    change.contentChanges.back().text = "changed";
    Expect(!files.OnChange(change), "non-file URI changes must not create working files");

    lsTextDocumentIdentifier close;
    close.uri.raw_uri_ = "untitled:Untitled-1";
    Expect(!files.OnClose(close), "non-file URI closes must be ignored");
    Expect(!files.OnSave(close), "non-file URI saves must be ignored");
}

void TestWorkingFilesRangedEditUtf16SurrogatePairs()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("surrogate-edit.cpp");
    // "prefix" + U+1F600 (😀, UTF-16 surrogate pair) + "suffix"
    std::string const original = "prefix\xf0\x9f\x98\x80suffix";
    auto file = OpenFile(files, path, original);

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.textDocument.version = optional<int>(2);
    change.contentChanges.push_back(RangedEdit(lsPosition(0, 6), lsPosition(0, 8), "X"));
    files.OnChange(change);

    Expect(
        ContentOf(files, file) == "prefixXsuffix",
        "ranged edit spanning a UTF-16 surrogate pair must replace the whole code point");
    Expect(
        file->GetOffsetForPosition(lsPosition(0, 7)) == 7,
        "offsets must stay valid after surrogate-pair edit");
}

void TestWorkingFilesUriAliasFindsSameFile()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("uri-alias.cpp");

    lsTextDocumentItem open;
    open.uri.raw_uri_ = "file:///tmp/lspcpp-working-files-suite/alias/../uri-alias.cpp";
    open.languageId = "cpp";
    open.version = 1;
    open.text = "initial";
    auto file = files.OnOpen(open);
    Expect(file != nullptr, "OnOpen must accept aliased file URIs");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.textDocument.version = optional<int>(2);
    change.contentChanges.push_back(RangedEdit(lsPosition(0, 0), lsPosition(0, 7), "updated"));
    auto changed = files.OnChange(change);

    Expect(changed == file, "didChange with normalized URI must find the same WorkingFile");
    Expect(ContentOf(files, file) == "updated", "didChange with normalized URI must apply to the open file");
}

#if defined(_WIN32)
void TestWorkingFilesWindowsUriCasingUsesSameMapKey()
{
    WorkingFiles files;
    AbsolutePath const path("C:/tmp/lspcpp-working-files-suite/casing.cpp");

    lsTextDocumentItem open;
    open.uri.raw_uri_ = "file:///C:/tmp/lspcpp-working-files-suite/casing.cpp";
    open.languageId = "cpp";
    open.version = 1;
    open.text = "original";
    auto file = files.OnOpen(open);
    Expect(file != nullptr, "OnOpen must accept mixed-case Windows file URIs");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri.raw_uri_ = "file:///c:/tmp/lspcpp-working-files-suite/casing.cpp";
    change.textDocument.version = optional<int>(2);
    lsTextDocumentContentChangeEvent edit;
    edit.text = "changed";
    change.contentChanges.push_back(edit);
    auto changed = files.OnChange(change);

    Expect(changed == file, "didChange with different URI casing must find the same WorkingFile on Windows");
    Expect(ContentOf(files, file) == "changed", "didChange with different URI casing must apply to the open file");
}
#endif

void TestWorkingFilesChangeCloseSaveWithoutOpen()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("never-opened.cpp");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.contentChanges.push_back(lsTextDocumentContentChangeEvent());
    change.contentChanges.back().text = "ghost";
    Expect(!files.OnChange(change), "OnChange must fail when the file was never opened");

    lsTextDocumentIdentifier id;
    id.uri = lsDocumentUri(path);
    Expect(!files.OnClose(id), "OnClose must fail when the file was never opened");
    Expect(!files.OnSave(id), "OnSave must fail when the file was never opened");
}

void TestWorkingFilesRejectsInvalidFileUris()
{
    WorkingFiles files;

    lsTextDocumentItem open;
    open.uri.raw_uri_ = "file://localhost";
    open.languageId = "cpp";
    open.version = 1;
    open.text = "content";
    Expect(!files.OnOpen(open), "OnOpen must reject file://localhost without path");

    open.uri.raw_uri_ = "";
    Expect(!files.OnOpen(open), "OnOpen must reject empty file URI");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri.raw_uri_ = "file://localhost";
    change.contentChanges.push_back(lsTextDocumentContentChangeEvent());
    change.contentChanges.back().text = "x";
    Expect(!files.OnChange(change), "OnChange must reject file://localhost without path");

    change.textDocument.uri.raw_uri_ = "";
    Expect(!files.OnChange(change), "OnChange must reject empty file URI");

    lsTextDocumentIdentifier id;
    id.uri.raw_uri_ = "file://localhost";
    Expect(!files.OnClose(id), "OnClose must reject file://localhost without path");

    id.uri.raw_uri_ = "";
    Expect(!files.OnClose(id), "OnClose must reject empty file URI");

    id.uri.raw_uri_ = "file://localhost";
    Expect(!files.OnSave(id), "OnSave must reject file://localhost without path");
}

void TestOnSaveWritesContentAfterChange()
{
    WorkingFiles files;
    AbsolutePath const path("/tmp/lspcpp-save-after-change.txt");
    std::remove(path.path().c_str());
    auto file = OpenFile(files, path, "initial");

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    change.textDocument.version = optional<int>(2);
    lsTextDocumentContentChangeEvent edit;
    edit.text = "modified on disk";
    change.contentChanges.push_back(edit);
    files.OnChange(change);

    lsTextDocumentIdentifier save;
    save.uri = lsDocumentUri(path);
    auto saved = files.OnSave(save);

    Expect(saved == file, "OnSave after change must return the same WorkingFile");
    auto disk_content = lsp::ReadContent(path);
    Expect(disk_content.has_value(), "OnSave after change must create the target file");
    if (disk_content)
    {
        Expect(*disk_content == "modified on disk", "OnSave after change must write the updated buffer content");
    }
    std::remove(path.path().c_str());
}

void TestCloseFilesInDirectoryMultipleRoots()
{
    WorkingFiles files;
    AbsolutePath const in_dir_a = MakePath("multi-root/a/file.cpp");
    AbsolutePath const in_dir_b = MakePath("multi-root/b/file.cpp");
    AbsolutePath const in_dir_c = MakePath("multi-root-c/file.cpp");
    AbsolutePath const unrelated = MakePath("multi-root-unrelated/stay.cpp");
    OpenFile(files, in_dir_a, "a");
    OpenFile(files, in_dir_b, "b");
    OpenFile(files, in_dir_c, "c");
    OpenFile(files, unrelated, "stay");

    files.CloseFilesInDirectory(
        {Directory(AbsolutePath("/tmp/lspcpp-working-files-suite/multi-root/a/")),
         Directory(AbsolutePath("/tmp/lspcpp-working-files-suite/multi-root/b/")),
         Directory(AbsolutePath("/tmp/lspcpp-working-files-suite/multi-root-c/"))});

    std::string content;
    Expect(
        !files.GetFileBufferContent(in_dir_a, content),
        "CloseFilesInDirectory must remove files under the first matched root");
    Expect(
        !files.GetFileBufferContent(in_dir_b, content),
        "CloseFilesInDirectory must remove files under the second matched root");
    Expect(
        !files.GetFileBufferContent(in_dir_c, content),
        "CloseFilesInDirectory must remove files under the third matched root");
    Expect(
        files.GetFileBufferContent(unrelated, content),
        "CloseFilesInDirectory must leave files outside all matched roots");
    Expect(content == "stay", "CloseFilesInDirectory must not disturb unrelated files");
}

void TestGetFileByFilenameRejectsInvalidPath()
{
    WorkingFiles files;
    OpenFile(files, MakePath("lookup.cpp"), "present");

    Expect(
        files.GetFileByFilename(AbsolutePath("relative/path.cpp")) == nullptr,
        "GetFileByFilename must reject relative paths");
    Expect(files.GetFileByFilename(AbsolutePath("")) == nullptr, "GetFileByFilename must reject empty paths");
}

void TestOnChangeWithoutVersionPreservesVersion()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("version-preserve.cpp");
    OpenFile(files, path, "content", 1);
    auto file = OpenFile(files, path, "content", 5);

    lsTextDocumentDidChangeParams change;
    change.textDocument.uri = lsDocumentUri(path);
    lsTextDocumentContentChangeEvent edit;
    edit.text = "updated";
    change.contentChanges.push_back(edit);
    files.OnChange(change);

    Expect(file->version == 5, "OnChange without version must preserve the existing file version");
    Expect(ContentOf(files, file) == "updated", "OnChange without version must still apply content changes");
}

void TestConcurrentReadSnapshotsDuringChanges()
{
    WorkingFiles files;
    AbsolutePath const path = MakePath("concurrent.cpp");
    auto file = OpenFile(files, path, "value 0\n");
    std::atomic<bool> stop {false};
    std::atomic<bool> saw_bad_snapshot {false};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i)
    {
        readers.emplace_back(
            [&]()
            {
                while (!stop.load(std::memory_order_relaxed))
                {
                    std::string content;
                    if (files.GetFileBufferContent(file, content))
                    {
                        bool const complete =
                            content.find("value ") == 0 && !content.empty() && content.back() == '\n';
                        if (!complete)
                        {
                            saw_bad_snapshot.store(true, std::memory_order_relaxed);
                        }
                    }
                }
            });
    }

    for (int i = 1; i <= 200; ++i)
    {
        lsTextDocumentDidChangeParams change;
        change.textDocument.uri = lsDocumentUri(path);
        change.textDocument.version = optional<int>(i + 1);
        lsTextDocumentContentChangeEvent edit;
        edit.text = "value " + std::to_string(i) + "\n";
        change.contentChanges.push_back(edit);
        files.OnChange(change);
    }

    stop.store(true, std::memory_order_relaxed);
    for (auto& reader : readers)
    {
        reader.join();
    }

    Expect(!saw_bad_snapshot.load(std::memory_order_relaxed), "concurrent readers must only observe complete snapshots");
    Expect(ContentOf(files, file) == "value 200\n", "last concurrent change must win");
}

// Adapted from clangd/test/positionencoding.test
void TestMultibyteCharacterColumnOffsets()
{
    WorkingFiles files;
    auto file = OpenFile(files, MakePath("position-encoding.cpp"), "/*\xc3\xb6*/int x;\nint y=x;");

    Expect(file->GetOffsetForPosition(lsPosition(1, 6)) == 19, "column offset must count multibyte characters as one column");
    Expect(file->GetOffsetForPosition(lsPosition(0, 9)) == 10, "first-line column offset must reach the x identifier");
}

// Adapted from clangd/test/version.test
void TestVersionPropagationFromClangdScenario()
{
    WorkingFiles files;
    AbsolutePath const foo_path = MakePath("version-foo.c");
    AbsolutePath const bar_path = MakePath("version-bar.c");

    lsTextDocumentItem open_foo;
    open_foo.uri = lsDocumentUri(foo_path);
    open_foo.languageId = "c";
    open_foo.version = 0;
    open_foo.text = "";
    auto foo = files.OnOpen(open_foo);
    Expect(foo->version == 0, "didOpen must preserve explicit version zero");

    lsTextDocumentDidChangeParams change_foo;
    change_foo.textDocument.uri = lsDocumentUri(foo_path);
    change_foo.textDocument.version = optional<int>(5);
    lsTextDocumentContentChangeEvent full_foo;
    full_foo.text = "a";
    change_foo.contentChanges.push_back(full_foo);
    files.OnChange(change_foo);
    Expect(foo->version == 5, "didChange must apply explicit version from clangd version.test");

    lsTextDocumentDidChangeParams infer_foo;
    infer_foo.textDocument.uri = lsDocumentUri(foo_path);
    lsTextDocumentContentChangeEvent full_infer;
    full_infer.text = "b";
    infer_foo.contentChanges.push_back(full_infer);
    files.OnChange(infer_foo);
    Expect(foo->version == 5, "didChange without version must preserve the previous version");

    lsTextDocumentItem open_bar;
    open_bar.uri = lsDocumentUri(bar_path);
    open_bar.languageId = "c";
    open_bar.version = 1;
    open_bar.text = "";
    files.OnOpen(open_bar);

    lsTextDocumentItem reopen_bar;
    reopen_bar.uri = lsDocumentUri(bar_path);
    reopen_bar.languageId = "c";
    reopen_bar.version = 42;
    reopen_bar.text = "";
    auto bar = files.OnOpen(reopen_bar);
    Expect(bar->version == 42, "reopening must preserve explicit non-zero version");

    lsTextDocumentDidChangeParams change_bar;
    change_bar.textDocument.uri = lsDocumentUri(bar_path);
    change_bar.textDocument.version = optional<int>(123);
    lsTextDocumentContentChangeEvent full_bar;
    full_bar.text = "d";
    change_bar.contentChanges.push_back(full_bar);
    files.OnChange(change_bar);
    Expect(bar->version == 123, "didChange must honor explicit version override");
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
RUN_TEST(TestOffsetBoundariesAndLineEndings);
    RUN_TEST(TestUtf8OffsetsUseCodePointCounting);
    RUN_TEST(TestMultipleEditsRebuildLineOffsetsSequentially);
    RUN_TEST(TestFullReplaceAndDocumentBoundaryEdits);
    RUN_TEST(TestLifecycleOpenCloseClearAndDirectories);
    RUN_TEST(TestCloseFilesInDirectoryTrailingSlashMatching);
    RUN_TEST(TestOnSaveWritesCurrentContent);
    RUN_TEST(TestNonFileUrisDoNotEnterWorkingFiles);
    RUN_TEST(TestWorkingFilesRangedEditUtf16SurrogatePairs);
    RUN_TEST(TestWorkingFilesUriAliasFindsSameFile);
#if defined(_WIN32)
    RUN_TEST(TestWorkingFilesWindowsUriCasingUsesSameMapKey);
#else
    // TestWorkingFilesWindowsUriCasingUsesSameMapKey requires Windows path casing semantics.
#endif
    RUN_TEST(TestWorkingFilesChangeCloseSaveWithoutOpen);
    RUN_TEST(TestWorkingFilesRejectsInvalidFileUris);
    RUN_TEST(TestOnSaveWritesContentAfterChange);
    RUN_TEST(TestCloseFilesInDirectoryMultipleRoots);
    RUN_TEST(TestGetFileByFilenameRejectsInvalidPath);
    RUN_TEST(TestOnChangeWithoutVersionPreservesVersion);
    RUN_TEST(TestMultibyteCharacterColumnOffsets);
    RUN_TEST(TestVersionPropagationFromClangdScenario);
    RUN_TEST(TestConcurrentReadSnapshotsDuringChanges);
    return test::Failures() == 0 ? 0 : 1;
}
