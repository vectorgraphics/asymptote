/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#if defined(_WIN32)
#  include <filesystem>
#  include <system_error>
#  include <Windows.h>
#else
#  include <unistd.h>
#  ifdef __APPLE__
#    include <limits.h>
#    include <stdlib.h>
#    include <mach-o/dyld.h>
#  endif
#endif

#include "locate.h"
#include "settings.h"
#include "util.h"


namespace settings
{

// True if systemDir was resolved relative to the running executable rather
// than taken from the compiled-in ASYMPTOTE_SYSDIR. Statically zero-initialized
// before any dynamic initialization, so it is safe for resolveSysdir() -- which
// runs as a static initializer in settings.cc -- to assign it.
bool relocatedSysdir= false;

#if defined(_WIN32)
// Narrow a path to the encoding the rest of the program uses for paths.
//
// asy strings are 8-bit, and the file APIs this program reaches for on Windows
// are the ...A variants (PathFileExistsA in fileExists(), the narrow CRT
// elsewhere), so a path handed to them must be in the process code page -- not
// UTF-8. The narrowing done by path::string() follows that same code page, so
// its result is what those APIs expect.
//
// A path containing characters outside that code page is therefore not
// representable here, and path::string() substitutes for them silently.

// Returns "" rather than letting an exception escape: resolveSysdir() runs as a
// static initializer, where an escaping exception calls terminate() before
// main() rather than being caught anywhere.
static string narrowPath(std::filesystem::path const& path)
{
  try {
    return string(path.string().c_str());
  } catch (...) {
    return "";
  }
}

// Resolve symlinks, junctions and 8.3 short names in a path, or return "" if it
// cannot be resolved. This is the Win32 counterpart of the realpath() call in
// the __APPLE__ branch below, and exists for the same reason: GetModuleFileNameW
// reports the path the process was launched from, unresolved, and only reopening
// the file and asking for its final name gets past a reparse point.
//
// The argument is a path rather than a string so that the resolution runs on the
// wide form throughout, with narrowPath() applied once to the result. Narrowing
// first would resolve a path that had already lost characters.
static string canonicalPath(std::filesystem::path const& path)
{
  // The error_code overload, not the throwing one, for the reason given above
  // narrowPath().
  std::error_code ec;
  std::filesystem::path const resolved= std::filesystem::canonical(path, ec);
  return ec ? "" : narrowPath(resolved);
}
#endif

// Absolute path of the running executable, or "" if it cannot be determined.
// Declared in locate.h: rendererloader.cc and vkrender.cc resolve co-installed
// files (renderer shared libraries, ICD manifests) against it too.
string executablePath()
{
#if defined(_WIN32)
  // GetModuleFileNameW truncates rather than failing when the path does not
  // fit, so allocate for the longest path Windows documents (32767 characters)
  // rather than the MAX_PATH (260) that long-path support can exceed. That
  // limit counts characters, so the wide form is what it bounds: the same
  // number of bytes can fall short of it under a double-byte code page.
  DWORD const size= 32768;
  mem::vector<wchar_t> buf(size);
  DWORD len= GetModuleFileNameW(nullptr, buf.data(), size);
  if (len == 0 || len == size)// 0: failed; size: truncated
    return "";
  // GetModuleFileNameW does not resolve symlinks: launched through a link on
  // PATH it reports the link, whose directory holds no base/. Resolve it so
  // that such a link yields the real install prefix, as on the other two
  // platforms. Falling back to the unresolved path on failure mirrors what the
  // macOS branch does when realpath() fails.
  std::filesystem::path const exe(buf.data(), buf.data() + len);
  string resolved= canonicalPath(exe);
  return resolved.empty() ? narrowPath(exe) : resolved;
#elif defined(__APPLE__)
  char buf[4096];
  uint32_t size= (uint32_t) sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) != 0)
    return "";
  // _NSGetExecutablePath may return a path containing symlinks or "..";
  // resolve it so that a symlinked bin directory (Homebrew, MacPorts) yields
  // the real install prefix.
  //
  char resolved[PATH_MAX];
  if (realpath(buf, resolved) != nullptr)
    return string(resolved);
  return string(buf);
#else
  char buf[4096];
  ssize_t len= readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0)
    return "";
  return string(buf, len);
#endif
}

// The directory part of path, or nullopt if it has none. An optional rather
// than a string because a path directly under a POSIX root has an empty
// directory part ("/bin" -> ""), which "" as a return value would conflate with
// having no directory part at all ("asy" -> nullopt).
//
// A backslash is an ordinary character in a POSIX filename, so scanning for the
// last separator is exactly right there. On MSWindows it is not, which is why
// that branch defers to std::filesystem::path:
//
//  - "C:\asy.exe" splits to "C:", which is drive-*relative*: appending to it
//    resolves against the current directory of drive C:, not its root.
//    parent_path() yields "C:\" instead.
//  - 0x5C is a legal trail byte in Shift-JIS, Big5 and GBK, so a scan of the
//    narrow (process code page) form can split in the middle of a character.
//    path parses the wide form, where it cannot.
//  - a UNC path keeps its root name: "\\server\share" rather than "\\server".
//
// A root parent is the one result that carries a trailing separator ("C:\"), so
// a caller appending "/base" to it gets "C:\/base"; Win32 collapses the doubled
// separator, and no non-root parent is affected.
static optional<string> parentDir(string const& path)
{
#if defined(_WIN32)
  std::filesystem::path const full(path.c_str());
  if (!full.has_parent_path())
    return nullopt;
  // A parent that exists is never empty on MSWindows -- it is at minimum a root
  // ("C:\", "\") or a drive ("C:") -- so an empty result here means narrowPath()
  // failed. Report no parent rather than "", which the caller would otherwise
  // append to and get a path relative to the current drive.
  string const dir= narrowPath(full.parent_path());
  if (dir.empty())
    return nullopt;
  return dir;
#else
  size_t slash= path.find_last_of('/');
  if (slash == string::npos)
    return nullopt;
  return path.substr(0, slash);
#endif
}

// The directory containing the running executable, or "" if it cannot be
// determined. An executablePath() that is empty, or that has no separator at
// all, both yield "" here.
string executableDir()
{
  return parentDir(executablePath()).value_or("");
}

// A directory is only accepted as a base directory if it contains this file.
// Mere existence of the directory proves nothing: the compiled-in path may
// belong to an unrelated (or half-removed) Asymptote installation.
static bool isBaseDir(string const& dir)
{
  return fileExists(dir + "/plain.asy");
}

// Determine the system base directory.
//
// ASYMPTOTE_SYSDIR is passed in rather than read here. Under CMake it differs
// between asy and asy-ctan (the CTAN/TeXLive build defines it empty), but only
// settings.cc is compiled separately per executable; locate.cc is compiled once
// into asycore and linked into both, so a value read here would be identical
// for the two binaries. The autotools build has a single executable and is
// unaffected either way.
//
// Candidates are tried relative to the running executable first, so that a
// binary run in place from its build tree uses its own base/ even when some
// other Asymptote is installed at the compiled-in sysdir. Falling back to the
// compiled-in path last costs nothing for an installed binary, whose
// <prefix>/bin/asy resolves to the same <prefix>/share/asymptote either way.
//
// The build-tree candidate is always tried: <exedir>/base/plain.asy exists
// only in a build tree or a flat install, never on a system where asy came
// from a package, so it needs no opt-in. The install-tree and flat candidates
// are gated behind IS_RELOCATABLE.
//
string resolveSysdir(string const& compiledInSysdir)
{
  // An empty compiled-in sysdir marks a TeXLive build, whose data directory is
  // defined only by kpathsea. It must always defer to kpsewhich, so never
  // relocate it relative to the executable -- even when a base/ sits beside the
  // binary (e.g. when run in place from its build tree).
  if (compiledInSysdir.empty())
    return compiledInSysdir;

  // parentDir() rather than executableDir(), so that an executable sitting
  // directly in the filesystem root still gets its candidates tried.
  optional<string> const exeDir= parentDir(executablePath());
  if (exeDir) {
    string const& bindir= *exeDir;
    // Build tree: base/ sits next to the executable.
    string buildBase= bindir + "/base";
    if (isBaseDir(buildBase)) {
      relocatedSysdir= true;
      return buildBase;
    }
#ifdef IS_RELOCATABLE
    // Install tree: <prefix>/bin/asy with data in <prefix>/share/asymptote.
    optional<string> const prefix= parentDir(bindir);
    if (prefix) {
      string shareBase= *prefix + "/share/asymptote";
      if (isBaseDir(shareBase)) {
        relocatedSysdir= true;
        return shareBase;
      }
    }
    // Flat layout (the MSWindows installer): base files beside asy.exe.
    if (isBaseDir(bindir)) {
      relocatedSysdir= true;
      return bindir;
    }
#endif
  }
  return compiledInSysdir;
}

namespace fs
{

string extension(string name)
{
  size_t n = name.rfind(".");
  if (n != string::npos)
    return name.substr(n);
  else
    return string();
}

bool exists(string filename)
{
  return fileExists(filename);
}

} // namespace fs


file_list_t searchPath;

// Returns list of possible filenames, accounting for extensions.
file_list_t mungeFileName(string id, string suffix)
{
  string ext = fs::extension(id);
  file_list_t files;
  if (ext == "."+suffix) {
    files.push_back(id);
    files.push_back(id+"."+suffix);
  } else {
    files.push_back(id+"."+suffix);
    files.push_back(id);
  }
  return files;
}

// Join a directory with the given filename, to give the path to the file,
// avoiding unsightly joins such as 'dir//file.asy' in favour of 'dir/file.asy'
string join(string dir, string file, bool full)
{
  return dir == "." ? (full ? string(getPath())+"/"+file : file) :
    *dir.rbegin() == '/' ? dir + file :
    dir + "/" + file;
}

// Find the appropriate file, first looking in the local directory, then the
// directory given in settings, and finally the global system directory.
string locateFile(string id, bool full, string suffix)
{
  if(id.empty()) return "";
  file_list_t filenames = mungeFileName(id,suffix);
  for (auto const& leaf : filenames) {
    if (leaf[0] == '/') { // FIXME: Add windows path check
      string file = leaf;
      if (fs::exists(file))
        return file;
    } else {
      for (auto const& dir : searchPath) {
        string file = join(dir,leaf,full);
        if (fs::exists(file))
          return file;
      }
    }
  }
  return string();
}

} // namespace settings
