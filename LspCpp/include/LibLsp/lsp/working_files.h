#pragma once

#include "LibLsp/lsp/lsp_diagnostic.h"
#include "LibLsp/lsp/AbsolutePath.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_close.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include <mutex>
#include <string>
#include <memory>
#include <vector>
#include "Directory.h"

struct WorkingFiles;
struct WorkingFilesData;
struct WorkingFile
{

    int version = 0;
    AbsolutePath filename;
    Directory directory;
    WorkingFiles& parent;
    std::atomic<long long> counter;
    WorkingFile(WorkingFiles&, AbsolutePath const& filename, std::string const& buffer_content);
    WorkingFile(WorkingFiles&, AbsolutePath const& filename, std::string&& buffer_content);
    void RebuildLineOffsets();
    int GetOffsetForPosition(lsPosition position) const;
    size_t LineOffsetCountForTest() const
    {
        return line_offsets.size();
    }
    std::string const& GetContentNoLock() const
    {
        return buffer_content;
    }

protected:
    friend struct WorkingFiles;
    std::string buffer_content;
    std::vector<size_t> line_offsets;
};

struct WorkingFiles
{

    WorkingFiles();
    ~WorkingFiles();

    void CloseFilesInDirectory(std::vector<Directory> const&);
    std::shared_ptr<WorkingFile> OnOpen(lsTextDocumentItem& open);
    std::shared_ptr<WorkingFile> OnChange(lsTextDocumentDidChangeParams const& change);
    bool OnClose(lsTextDocumentIdentifier const& close);
    std::shared_ptr<WorkingFile> OnSave(lsTextDocumentIdentifier const& _save);

    bool GetFileBufferContent(AbsolutePath const& filename, std::wstring& out)
    {
        auto file = GetFileByFilename(filename);
        if (!file)
        {
            return false;
        }
        return GetFileBufferContent(file, out);
    }
    bool GetFileBufferContent(AbsolutePath const& filename, std::string& out)
    {
        auto file = GetFileByFilename(filename);
        if (!file)
        {
            return false;
        }
        return GetFileBufferContent(file, out);
    }
    bool GetFileBufferContent(std::shared_ptr<WorkingFile>&, std::string& out);
    bool GetFileBufferContent(std::shared_ptr<WorkingFile>&, std::wstring& out);

    // Find the file with the given filename.
    std::shared_ptr<WorkingFile> GetFileByFilename(AbsolutePath const& filename);

    void Clear();

private:
    std::shared_ptr<WorkingFile> GetFileByFilenameNoLock(AbsolutePath const& filename);

    WorkingFilesData* d_ptr;
};
