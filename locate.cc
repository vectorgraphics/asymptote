/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#if defined(_WIN32)
#  include <filesystem>// canonical
#  include <system_error>// error_code
#  include <Windows.h>
#else
#  include <unistd.h>
#  ifdef __APPLE__
#    include <limits.h>// PATH_MAX
#    include <stdlib.h>// realpath
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
// representable here, and path::string() substitutes for them silently. That is
// not a loss introduced by resolving the path in wide form below: the ...A APIs
// it would be handed to cannot open such a path either. Lifting that limitation
// means moving the whole process to UTF-8 (a manifest declaring it as the active
// code page), which is not something this file can do on its own.
//
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
// The MSVC standard library does exactly that -- CreateFileW with
// FILE_READ_ATTRIBUTES and FILE_FLAG_BACKUP_SEMANTICS, then
// GetFinalPathNameByHandleW with VOLUME_NAME_DOS -- and then puts the result
// back in a form the rest of this file expects, stripping the \\?\ prefix and
// respelling \\?\UNC\server\share as \\server\share. It also grows its buffer
// rather than failing on a long path, and falls back to the NT namespace for a
// volume with no drive letter. Calling it is preferred to reproducing it here:
// this is delicate platform detail that few readers of this file can review.
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
static string getExecutablePath()
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
  // the real install prefix. Linux does not need this: /proc/self/exe is
  // already fully resolved.
  //
  // PATH_MAX (POSIX, 1024 here -- not to be confused with Win32's MAX_PATH)
  // is required: realpath() may write that many bytes to the buffer.
  // Nothing else here uses it, since POSIX leaves it optional and glibc/Hurd
  // does not define it.
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
// When nothing matches, the compiled-in path is returned unchanged -- including
// when it is empty, which is how a TeXLive build says "I have no fixed data
// directory". initDir() sees the empty string and asks kpsewhich for TEXMFMAIN.
// That lookup is therefore the next candidate after the ones below, not a
// separate mode: a TeXLive binary run from its build tree uses the adjacent
// base/, and only a deployed one (bin/<platform>/asy, where no candidate
// matches) consults kpsewhich.
string resolveSysdir(string const& compiledInSysdir)
{
  string exe= getExecutablePath();
  if (!exe.empty()) {
    size_t slash= exe.find_last_of("/\\");
    if (slash != string::npos) {
      string bindir= exe.substr(0, slash);
      // Build tree: base/ sits next to the executable.
      string buildBase= bindir + "/base";
      if (isBaseDir(buildBase)) {
        relocatedSysdir= true;
        return buildBase;
      }
#ifdef IS_RELOCATABLE
      // Install tree: <prefix>/bin/asy with data in <prefix>/share/asymptote.
      size_t slash2= bindir.find_last_of("/\\");
      if (slash2 != string::npos) {
        string shareBase= bindir.substr(0, slash2) + "/share/asymptote";
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
