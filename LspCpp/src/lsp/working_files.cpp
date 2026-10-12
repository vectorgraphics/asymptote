#include "LibLsp/lsp/working_files.h"
#include <algorithm>
#include <climits>
#include <numeric>
#include "LibLsp/lsp/utils.h"
#include <memory>
#include "LibLsp/lsp/AbsolutePath.h"
using namespace lsp;
struct WorkingFilesData
{
    std::map<AbsolutePath, std::shared_ptr<WorkingFile>> files;
    std::mutex files_mutex; // Protects |d_ptr->files|.
};

namespace
{
std::vector<size_t> BuildLineOffsets(std::string const& content)
{
    std::vector<size_t> offsets;
    offsets.push_back(0);
    for (size_t i = 0; i < content.size(); ++i)
    {
        if (content[i] == '\n')
        {
            offsets.push_back(i + 1);
        }
    }
    return offsets;
}

void RebuildLineOffsets(WorkingFile& file)
{
    file.RebuildLineOffsets();
}

int GetCachedOffsetForPosition(WorkingFile const& file, lsPosition position)
{
    return file.GetOffsetForPosition(position);
}
} // namespace

WorkingFile::WorkingFile(WorkingFiles& _parent, AbsolutePath const& filename, std::string const& buffer_content)
    : filename(filename), directory(filename), parent(_parent), counter(0), buffer_content(buffer_content)
{
    directory = Directory(AbsolutePath::FromNormalized(GetDirName(filename.path())));
    this->RebuildLineOffsets();
}

WorkingFile::WorkingFile(WorkingFiles& _parent, AbsolutePath const& filename, std::string&& buffer_content)
    : filename(filename), directory(filename), parent(_parent), counter(0), buffer_content(buffer_content)
{
    directory = Directory(AbsolutePath::FromNormalized(GetDirName(filename.path())));
    this->RebuildLineOffsets();
}

void WorkingFile::RebuildLineOffsets()
{
    line_offsets = BuildLineOffsets(buffer_content);
}

int WorkingFile::GetOffsetForPosition(lsPosition position) const
{
    if (line_offsets.empty())
    {
        return ::GetOffsetForPosition(position, buffer_content);
    }

    size_t i = buffer_content.size();
    if (position.line < line_offsets.size())
    {
        i = line_offsets[position.line];
    }

    while (position.character > 0 && i < buffer_content.size())
    {
        uint8_t const lead = uint8_t(buffer_content[i++]);
        if (lead >= 128)
        {
            while (i < buffer_content.size() && uint8_t(buffer_content[i]) >= 128 && uint8_t(buffer_content[i]) < 192)
            {
                i++;
            }
        }
        // Match lsp::GetOffsetForPosition: code points above U+FFFF count as
        // two UTF-16 code units (a surrogate pair).
        // Saturating subtraction: |character| is unsigned.
        unsigned const units = (lead >= 0xF0) ? 2u : 1u;
        position.character -= std::min(position.character, units);
    }
    return static_cast<int>(i);
}

WorkingFiles::WorkingFiles() : d_ptr(new WorkingFilesData())
{
}

WorkingFiles::~WorkingFiles()
{
    delete d_ptr;
}

void WorkingFiles::CloseFilesInDirectory(std::vector<Directory> const& directories)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    std::vector<AbsolutePath> files_to_be_delete;

    for (auto& it : d_ptr->files)
    {
        for (auto& dir : directories)
        {
            if (it.second->directory == dir)
            {
                files_to_be_delete.emplace_back(it.first);
            }
        }
    }

    for (auto& it : files_to_be_delete)
    {
        d_ptr->files.erase(it);
    }
}

std::shared_ptr<WorkingFile> WorkingFiles::GetFileByFilename(AbsolutePath const& filename)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    return GetFileByFilenameNoLock(filename);
}

std::shared_ptr<WorkingFile> WorkingFiles::GetFileByFilenameNoLock(AbsolutePath const& filename)
{
    if (!filename.valid())
    {
        return nullptr;
    }

    auto const findIt = d_ptr->files.find(filename);
    if (findIt != d_ptr->files.end())
    {
        return findIt->second;
    }
    return nullptr;
}

std::shared_ptr<WorkingFile> WorkingFiles::OnOpen(lsTextDocumentItem& open)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

    AbsolutePath filename = open.uri.GetAbsolutePath();
    if (!filename.valid())
    {
        return {};
    }

    // The file may already be open.
    if (auto file = GetFileByFilenameNoLock(filename))
    {
        file->version = open.version;
        file->buffer_content.swap(open.text);
        RebuildLineOffsets(*file);

        return file;
    }

    auto const& it =
        d_ptr->files.insert({filename, std::make_shared<WorkingFile>(*this, filename, std::move(open.text))});
    return it.first->second;
}

std::shared_ptr<WorkingFile> WorkingFiles::OnChange(lsTextDocumentDidChangeParams const& change)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

    AbsolutePath filename = change.textDocument.uri.GetAbsolutePath();
    if (!filename.valid())
    {
        return {};
    }
    auto file = GetFileByFilenameNoLock(filename);
    if (!file)
    {
        return {};
    }

    if (change.textDocument.version)
    {
        file->version = *change.textDocument.version;
    }
    file->counter.fetch_add(1, std::memory_order_relaxed);
    for (lsTextDocumentContentChangeEvent const& diff : change.contentChanges)
    {
        // Per the spec replace everything if the rangeLength and range are not set.
        // See https://github.com/Microsoft/language-server-protocol/issues/9.
        if (!diff.range)
        {
            file->buffer_content = diff.text;
            RebuildLineOffsets(*file);
        }
        else
        {
            int start_offset = GetCachedOffsetForPosition(*file, diff.range->start);
            // Ignore TextDocumentContentChangeEvent.rangeLength which causes trouble
            // when UTF-16 surrogate pairs are used.
            int end_offset = GetCachedOffsetForPosition(*file, diff.range->end);
            file->buffer_content.replace(
                file->buffer_content.begin() + start_offset, file->buffer_content.begin() + end_offset, diff.text
            );
            RebuildLineOffsets(*file);
        }
    }
    return file;
}

bool WorkingFiles::OnClose(lsTextDocumentIdentifier const& close)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

    AbsolutePath filename = close.uri.GetAbsolutePath();
    if (!filename.valid())
    {
        return false;
    }
    auto const findIt = d_ptr->files.find(filename);
    if (findIt != d_ptr->files.end())
    {
        d_ptr->files.erase(findIt);
        return true;
    }
    return false;
}

std::shared_ptr<WorkingFile> WorkingFiles::OnSave(lsTextDocumentIdentifier const& _save)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

    AbsolutePath filename = _save.uri.GetAbsolutePath();
    if (!filename.valid())
    {
        return {};
    }
    auto const findIt = d_ptr->files.find(filename);
    if (findIt != d_ptr->files.end())
    {
        std::shared_ptr<WorkingFile>& file = findIt->second;
        lsp::WriteToFile(file->filename.path(), file->GetContentNoLock());
        return findIt->second;
    }
    return {};
}

bool WorkingFiles::GetFileBufferContent(std::shared_ptr<WorkingFile>& file, std::string& out)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    if (file)
    {
        out = file->buffer_content;
        return true;
    }
    return false;
}
bool WorkingFiles::GetFileBufferContent(std::shared_ptr<WorkingFile>& file, std::wstring& out)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    if (file)
    {
        out = lsp::s2ws(file->buffer_content);
        return true;
    }
    return false;
}
void WorkingFiles::Clear()
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    d_ptr->files.clear();
}
