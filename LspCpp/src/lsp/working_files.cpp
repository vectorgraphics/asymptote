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
    std::map<AbsolutePath, std::shared_ptr<WorkingFile> > files;
    std::mutex files_mutex;  // Protects |d_ptr->files|.
};

WorkingFile::WorkingFile(WorkingFiles& _parent, const AbsolutePath& filename,
                         const std::string& buffer_content)
  : filename(filename), directory(filename), parent(_parent), counter(0), buffer_content(buffer_content)
{
       directory = Directory(GetDirName(filename.path));
}

WorkingFile::WorkingFile(WorkingFiles& _parent, const AbsolutePath& filename,
                         std::string&& buffer_content)
  : filename(filename), directory(filename), parent(_parent), counter(0), buffer_content(buffer_content)
{
    directory = Directory(GetDirName(filename.path));
}

WorkingFiles::WorkingFiles():d_ptr(new WorkingFilesData())
{
}

WorkingFiles::~WorkingFiles()
{
    delete d_ptr;

}



void WorkingFiles::CloseFilesInDirectory(const std::vector<Directory>& directories)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    std::vector<AbsolutePath> files_to_be_delete;

    for(auto& it : d_ptr->files)
    {
        for (auto& dir : directories)
        {
            if (it.second->directory == dir) {
                files_to_be_delete.emplace_back(it.first);
            }
        }
    }

    for(auto& it : files_to_be_delete)
    {
        d_ptr->files.erase(it);
    }
}




std::shared_ptr<WorkingFile> WorkingFiles::GetFileByFilename(const AbsolutePath& filename) {
  std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
  return GetFileByFilenameNoLock(filename);
}

std::shared_ptr<WorkingFile> WorkingFiles::GetFileByFilenameNoLock(
    const AbsolutePath& filename) {
    const auto findIt = d_ptr->files.find(filename);
    if ( findIt != d_ptr->files.end())
    {
        return findIt->second;
    }
  return nullptr;
}



std::shared_ptr<WorkingFile>  WorkingFiles::OnOpen( lsTextDocumentItem& open) {
  std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

  AbsolutePath filename = open.uri.GetAbsolutePath();

  // The file may already be open.
  if (auto file = GetFileByFilenameNoLock(filename)) {
    file->version = open.version;
    file->buffer_content.swap(open.text);

    return file;
  }

  const auto& it =  d_ptr->files.insert({ filename,std::make_shared<WorkingFile>(*this,filename, std::move(open.text)) });
  return  it.first->second;
}


std::shared_ptr<WorkingFile>  WorkingFiles::OnChange(const lsTextDocumentDidChangeParams& change) {
  std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

  AbsolutePath filename = change.textDocument.uri.GetAbsolutePath();
  auto file = GetFileByFilenameNoLock(filename);
  if (!file) {
    return {};
  }

  if (change.textDocument.version)
    file->version = *change.textDocument.version;
  file->counter.fetch_add(1, std::memory_order_relaxed);
  for (const lsTextDocumentContentChangeEvent& diff : change.contentChanges) {
    // Per the spec replace everything if the rangeLength and range are not set.
    // See https://github.com/Microsoft/language-server-protocol/issues/9.
    if (!diff.range) {
      file->buffer_content = diff.text;

    } else {
      int start_offset =
          GetOffsetForPosition(diff.range->start, file->buffer_content);
      // Ignore TextDocumentContentChangeEvent.rangeLength which causes trouble
      // when UTF-16 surrogate pairs are used.
      int end_offset =
          GetOffsetForPosition(diff.range->end, file->buffer_content);
      file->buffer_content.replace(file->buffer_content.begin() + start_offset,
          file->buffer_content.begin() + end_offset,
                                   diff.text);

    }
  }
  return  file;
}

bool WorkingFiles::OnClose(const lsTextDocumentIdentifier& close) {
  std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

  AbsolutePath filename = close.uri.GetAbsolutePath();
  const auto findIt = d_ptr->files.find(filename);
  if( findIt != d_ptr->files.end())
  {
      d_ptr->files.erase(findIt);
	  return true;
  }
  return false;
}

std::shared_ptr<WorkingFile> WorkingFiles::OnSave(const lsTextDocumentIdentifier& _save)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);

    AbsolutePath filename = _save.uri.GetAbsolutePath();
    const auto findIt = d_ptr->files.find(filename);
    if (findIt != d_ptr->files.end())
    {
        std::shared_ptr<WorkingFile>& file = findIt->second;
        lsp::WriteToFile(file->filename, file->GetContentNoLock());
        return findIt->second;
    }
    return  {};

}

bool WorkingFiles::GetFileBufferContent(std::shared_ptr<WorkingFile>&file, std::string& out)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    if (file)
    {
        out = file->buffer_content;
        return  true;
    }
    return  false;
}
bool WorkingFiles::GetFileBufferContent(std::shared_ptr<WorkingFile>& file, std::wstring& out)
{
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    if (file)
    {
        out = lsp::s2ws(file->buffer_content);
        return  true;
    }
    return  false;
}
void  WorkingFiles::Clear() {
    std::lock_guard<std::mutex> lock(d_ptr->files_mutex);
    d_ptr->files.clear();
}