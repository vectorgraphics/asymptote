#include "LibLsp/lsp/utils.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>

#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <sys/stat.h>

#include "LibLsp/lsp/lsPosition.h"
#include "utf8.h"
#ifdef  _WIN32
#include <Windows.h>
#endif


// DEFAULT_RESOURCE_DIRECTORY is passed with quotes for non-MSVC compilers, ie,
// foo vs "foo".
#if defined(_MSC_VER)
#define _STRINGIFY(x) #x
#define ENSURE_STRING_MACRO_ARGUMENT(x) _STRINGIFY(x)
#else
#define ENSURE_STRING_MACRO_ARGUMENT(x) x
#endif
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string.hpp>
namespace lsp
{


// See http://stackoverflow.com/a/2072890
bool EndsWith(std::string value, std::string ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool StartsWith(std::string value, std::string start) {
  if (start.size() > value.size())
    return false;
  return std::equal(start.begin(), start.end(), value.begin());
}

bool AnyStartsWith(const std::vector<std::string>& values,
                   const std::string& start) {
  return std::any_of(
      std::begin(values), std::end(values),
      [&start](const std::string& value) { return StartsWith(value, start); });
}

bool StartsWithAny(const std::string& value,
                   const std::vector<std::string>& startings) {
  return std::any_of(std::begin(startings), std::end(startings),
                     [&value](const std::string& starting) {
                       return StartsWith(value, starting);
                     });
}

bool EndsWithAny(const std::string& value,
                 const std::vector<std::string>& endings) {
  return std::any_of(
      std::begin(endings), std::end(endings),
      [&value](const std::string& ending) { return EndsWith(value, ending); });
}

bool FindAnyPartial(const std::string& value,
                    const std::vector<std::string>& values) {
  return std::any_of(std::begin(values), std::end(values),
                     [&value](const std::string& v) {
                       return value.find(v) != std::string::npos;
                     });
}

std::string GetDirName(std::string path) {

  ReplaceAll(path, "\\", "/");
  if (path.size() && path.back() == '/')
    path.pop_back();
  size_t last_slash = path.find_last_of('/');
  if (last_slash == std::string::npos)
    return "./";
  return path.substr(0, last_slash + 1);
}

std::string GetBaseName(const std::string& path) {
  size_t last_slash = path.find_last_of('/');
  if (last_slash != std::string::npos && (last_slash + 1) < path.size())
    return path.substr(last_slash + 1);
  return path;
}

std::string StripFileType(const std::string& path) {
  size_t last_period = path.find_last_of('.');
  if (last_period != std::string::npos)
    return path.substr(0, last_period);
  return path;
}

// See http://stackoverflow.com/a/29752943
std::string ReplaceAll(const std::string& source,
                       const std::string& from,
                       const std::string& to) {
  std::string result;
  result.reserve(source.length());  // avoids a few memory allocations

  std::string::size_type last_pos = 0;
  std::string::size_type find_pos;

  while (std::string::npos != (find_pos = source.find(from, last_pos))) {
    result.append(source, last_pos, find_pos - last_pos);
    result += to;
    last_pos = find_pos + from.length();
  }

  // Care for the rest after last occurrence
  result += source.substr(last_pos);

  return result;
}

std::vector<std::string> SplitString(const std::string& str,
                                     const std::string& delimiter) {
  // http://stackoverflow.com/a/13172514
  std::vector<std::string> strings;

  std::string::size_type pos = 0;
  std::string::size_type prev = 0;
  while ((pos = str.find(delimiter, prev)) != std::string::npos) {
    strings.emplace_back(str.substr(prev, pos - prev));
    prev = pos + 1;
  }

  // To get the last substring (or only, if delimiter is not found)
  strings.emplace_back(str.substr(prev));

  return strings;
}

void EnsureEndsInSlash(std::string& path) {
  if (path.empty() || path[path.size() - 1] != '/')
    path += '/';
}

std::string EscapeFileName(std::string path) {
  if (path.size() && path.back() == '/')
    path.pop_back();
  std::replace(path.begin(), path.end(), '\\', '@');
  std::replace(path.begin(), path.end(), '/', '@');
  std::replace(path.begin(), path.end(), ':', '@');
  return path;
}

// http://stackoverflow.com/a/6089413
std::istream& SafeGetline(std::istream& is, std::string& t) {
  t.clear();

  // The characters in the stream are read one-by-one using a std::streambuf.
  // That is faster than reading them one-by-one using the std::istream. Code
  // that uses streambuf this way must be guarded by a sentry object. The sentry
  // object performs various tasks, such as thread synchronization and updating
  // the stream state.

  std::istream::sentry se(is, true);
  std::streambuf* sb = is.rdbuf();

  for (;;) {
    int c = sb->sbumpc();
    if (c == EOF) {
      // Also handle the case when the last line has no line ending
      if (t.empty())
        is.setstate(std::ios::eofbit);
      return is;
    }

    t += (char)c;

    if (c == '\n')
      return is;
  }
}

bool FileExists(const std::string& filename) {
  std::ifstream cache(filename);
  return cache.is_open();
}

boost::optional<std::string> ReadContent(const AbsolutePath& filename) {

  std::ifstream cache;
  cache.open(filename.path);

  try {
    return std::string(std::istreambuf_iterator<char>(cache),
                       std::istreambuf_iterator<char>());
  } catch (std::ios_base::failure&) {
    return {};
  }
}

std::vector<std::string> ReadLinesWithEnding(const AbsolutePath& filename) {
  std::vector<std::string> result;

  std::ifstream input(filename.path);
  for (std::string line; SafeGetline(input, line);)
    result.emplace_back(line);

  return result;
}

bool WriteToFile(const std::string& filename, const std::string& content) {
  std::ofstream file(filename,
                     std::ios::out | std::ios::trunc | std::ios::binary);
  if (!file.good()) {

    return false;
  }

  file << content;
  return true;
}


std::string FormatMicroseconds(long long microseconds) {
  long long milliseconds = microseconds / 1000;
  long long remaining = microseconds - milliseconds;

  // Only show two digits after the dot.
  while (remaining >= 100)
    remaining /= 10;

  return std::to_string(milliseconds) + "." + std::to_string(remaining) + "ms";
}



std::string UpdateToRnNewlines(std::string output) {
  size_t idx = 0;
  while (true) {
    idx = output.find('\n', idx);

    // No more matches.
    if (idx == std::string::npos)
      break;

    // Skip an existing "\r\n" match.
    if (idx > 0 && output[idx - 1] == '\r') {
      ++idx;
      continue;
    }

    // Replace "\n" with "\r|n".
    output.replace(output.begin() + idx, output.begin() + idx + 1, "\r\n");
  }

  return output;
}



bool IsAbsolutePath(const std::string& path) {
  return IsUnixAbsolutePath(path) || IsWindowsAbsolutePath(path);
}

bool IsUnixAbsolutePath(const std::string& path) {
  return !path.empty() && path[0] == '/';
}

bool IsWindowsAbsolutePath(const std::string& path) {
  auto is_drive_letter = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  };

  return path.size() > 3 && path[1] == ':' &&
         (path[2] == '/' || path[2] == '\\') && is_drive_letter(path[0]);
}

bool IsDirectory(const std::string& path) {
  struct stat path_stat;

  if (stat(path.c_str(), &path_stat) != 0) {
    perror("cannot access path");
    return false;
  }

  return path_stat.st_mode & S_IFDIR;
}

    std::string ws2s(std::wstring const& wstr) {
        if(sizeof(wchar_t) == 2){
            std::string narrow;
            utf8::utf16to8(wstr.begin(), wstr.end(), std::back_inserter(narrow));
            return narrow;
        }else{
            std::string narrow;
            utf8::utf32to8(wstr.begin(), wstr.end(), std::back_inserter(narrow));
            return narrow;
        }

    }
    std::wstring s2ws(const std::string& str) {
        std::wstring wide;
        if(sizeof(wchar_t) == 2){
            utf8::utf8to16(str.begin(), str.end(), std::back_inserter(wide));
            return wide;
        }else{
            utf8::utf8to32(str.begin(), str.end(), std::back_inserter(wide));
            return wide;
        }
    }

#ifdef _WIN32

#else
// Returns the canonicalized absolute pathname, without expanding symbolic
// links. This is a variant of realpath(2), C++ rewrite of
// https://github.com/freebsd/freebsd/blob/master/lib/libc/stdlib/realpath.c
AbsolutePath RealPathNotExpandSymlink(std::string path,
        bool ensure_exists) {
        if (path.empty()) {
                errno = EINVAL;
                return {};
        }
        if (path[0] == '\0') {
                errno = ENOENT;
                return {};
        }

        // Do not use PATH_MAX because it is tricky on Linux.
        // See https://eklitzke.org/path-max-is-tricky
        char tmp[1024];
        std::string resolved;
        size_t i = 0;
        struct stat sb;
        if (path[0] == '/') {
                resolved = "/";
                i = 1;
        }
        else {
                if (!getcwd(tmp, sizeof tmp) && ensure_exists)
                        return {};
                resolved = tmp;
        }

        while (i < path.size()) {
                auto j = path.find('/', i);
                if (j == std::string::npos)
                        j = path.size();
                auto next_token = path.substr(i, j - i);
                i = j + 1;
                if (resolved.back() != '/')
                        resolved += '/';
                if (next_token.empty() || next_token == ".") {
                        // Handle consequential slashes and "."
                        continue;
                }
                else if (next_token == "..") {
                        // Strip the last path component except when it is single "/"
                        if (resolved.size() > 1)
                                resolved.resize(resolved.rfind('/', resolved.size() - 2) + 1);
                        continue;
                }
                // Append the next path component.
                // Here we differ from realpath(3), we use stat(2) instead of
                // lstat(2) because we do not want to resolve symlinks.
                resolved += next_token;
                if (stat(resolved.c_str(), &sb) != 0 && ensure_exists)
                        return {};
                if (!S_ISDIR(sb.st_mode) && j < path.size() && ensure_exists) {
                        errno = ENOTDIR;
                        return {};
                }
        }

        // Remove trailing slash except when a single "/".
        if (resolved.size() > 1 && resolved.back() == '/')
                resolved.pop_back();
        return AbsolutePath(resolved, true /*validate*/);
}
#endif


AbsolutePath NormalizePath(const std::string& path0,
        bool ensure_exists ,
        bool force_lower_on_windows) {
#ifdef _WIN32

        std::wstring path = lsp::s2ws(path0);

        wchar_t buffer[MAX_PATH] = (L"");

        // Normalize the path name, ie, resolve `..`.
        unsigned long len = GetFullPathNameW(path.c_str(), MAX_PATH, buffer, nullptr);
        if (!len)
                return {};
        path = std::wstring(buffer, len);

        // Get the actual casing of the path, ie, if the file on disk is `C:\FooBar`
        // and this function is called with `c:\fooBar` this will return `c:\FooBar`.
        // (drive casing is lowercase).
        if (ensure_exists) {
                len = GetLongPathNameW(path.c_str(), buffer, MAX_PATH);
                if (!len)
                        return {};
                path = std::wstring(buffer, len);
        }

        // Empty paths have no meaning.
        if (path.empty())
                return {};

        // We may need to normalize the drive name to upper-case; at the moment
        // vscode sends lower-case path names.
        /*
        path[0] = toupper(path[0]);
        */
        // Make the path all lower-case, since windows is case-insensitive.
        if (force_lower_on_windows) {
                for (size_t i = 0; i < path.size(); ++i)
                        path[i] = (wchar_t)tolower(path[i]);
        }

        // cquery assumes forward-slashes.
        std::replace(path.begin(), path.end(), '\\', '/');


        return AbsolutePath(lsp::ws2s(path), false /*validate*/);
#else

        return RealPathNotExpandSymlink(path0, ensure_exists);

#endif


}

// VSCode (UTF-16) disagrees with Emacs lsp-mode (UTF-8) on how to represent
// text documents.
// We use a UTF-8 iterator to approximate UTF-16 in the specification (weird).
// This is good enough and fails only for UTF-16 surrogate pairs.
int GetOffsetForPosition(lsPosition position, const std::string& content) {
        size_t i = 0;
        // Iterate lines until we have found the correct line.
        while (position.line > 0 && i < content.size()) {
                if (content[i] == '\n')
                        position.line--;
                i++;
        }
        // Iterate characters on the target line.
        while (position.character > 0 && i < content.size()) {
                if (uint8_t(content[i++]) >= 128) {
                        // Skip 0b10xxxxxx
                        while (i < content.size() && uint8_t(content[i]) >= 128 &&
                                uint8_t(content[i]) < 192)
                                i++;
                }
                position.character--;
        }
        return int(i);
}


lsPosition GetPositionForOffset(size_t offset,const  std::string& content) {
        lsPosition result;
        for (size_t i = 0; i < offset && i < content.length(); ++i) {
                if (content[i] == '\n') {
                        result.line++;
                        result.character = 0;
                }
                else {
                        result.character++;
                }
        }
        return result;
}

lsPosition CharPos(const  std::string& search,
        char character,
        int character_offset) {
        lsPosition result;
        size_t index = 0;
        while (index < search.size()) {
                char c = search[index];
                if (c == character)
                        break;
                if (c == '\n') {
                        result.line += 1;
                        result.character = 0;
                }
                else {
                        result.character += 1;
                }
                ++index;
        }
        assert(index < search.size());
        result.character += character_offset;
        return result;
}

void scanDirsUseRecursive(const std::wstring& rootPath, std::vector<std::wstring>& ret)
{
        namespace fs = boost::filesystem;
        fs::path fullpath(rootPath);
        if (!fs::exists(fullpath)) { return; }
        fs::recursive_directory_iterator end_iter;
        for (fs::recursive_directory_iterator iter(fullpath); iter != end_iter; iter++) {
                try {
                        if (fs::is_directory(*iter)) {
                                ret.push_back(iter->path().wstring());
                        }
                }
                catch (const std::exception& ex) {
                        continue;
                }
        }
}

void scanDirsNoRecursive(const std::wstring& rootPath, std::vector<std::wstring>& ret)
{
        namespace fs = boost::filesystem;
        boost::filesystem::path myPath(rootPath);
        if (!fs::exists(rootPath)) { return; }
        boost::filesystem::directory_iterator endIter;
        for (boost::filesystem::directory_iterator iter(myPath); iter != endIter; iter++) {
                if (boost::filesystem::is_directory(*iter)) {
                        ret.push_back(iter->path().wstring());
                }
        }
}

void scanFilesUseRecursive(
        const std::wstring& rootPath,
        std::vector<std::wstring>& ret,
        std::wstring suf) {
        namespace fs = boost::filesystem;
        boost::to_lower(suf);

        fs::path fullpath(rootPath);
        if (!fs::exists(fullpath)) { return; }
        fs::recursive_directory_iterator end_iter;
        for (fs::recursive_directory_iterator iter(fullpath); iter != end_iter; iter++) {
                try {
                        if (!fs::is_directory(*iter) && fs::is_regular_file(*iter)) {
                                auto temp_path = iter->path().wstring();
                                auto size = suf.size();
                                if (!size)
                                {
                                        ret.push_back(std::move(temp_path));
                                }
                                else
                                {

                                        if (temp_path.size() < size) continue;
                                        auto suf_temp = temp_path.substr(temp_path.size() - size);
                                        boost::to_lower(suf_temp);
                                        if (suf_temp == suf)
                                        {
                                                ret.push_back(std::move(temp_path));
                                        }
                                }
                        }
                }
                catch (const std::exception&) {
                        continue;
                }
        }
}

void scanFileNamesUseRecursive(const std::wstring& rootPath, std::vector<std::wstring>& ret,
        std::wstring strSuf)
{
        scanFilesUseRecursive(rootPath, ret, strSuf);
        std::vector<std::wstring> names;
        for (auto& it : ret)
        {
                if (it.size() >= rootPath.size())
                {
                        names.push_back(it.substr(rootPath.size()));
                }
        }
        ret.swap(names);
}

void scanFileNamesUseRecursive(const std::string& rootPath, std::vector<std::string>& ret, std::string strSuf)
{
        std::vector<std::wstring> out;
        scanFileNamesUseRecursive(s2ws(rootPath), out, s2ws(strSuf));
        for (auto& it : out)
        {
                ret.push_back(ws2s(it));
        }
}

void scanFilesUseRecursive(const std::string& rootPath, std::vector<std::string>& ret, std::string strSuf)
{
        std::vector<std::wstring> out;
        scanFilesUseRecursive(s2ws(rootPath), out, s2ws(strSuf));
        for (auto& it : out)
        {
                ret.push_back(ws2s(it));
        }
}


}
