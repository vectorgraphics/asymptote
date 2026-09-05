/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#if defined(_WIN32)
#  include <filesystem>// canonical
#  include <system_error>// error_code
#  include <vector>
#  include <Windows.h>
#else
#  include <unistd.h>
#  if defined(__APPLE__)
#    include <limits.h>// PATH_MAX
#    include <stdlib.h>// realpath
#    include <mach-o/dyld.h>
#  elif defined(__FreeBSD__)
#    include <limits.h>// PATH_MAX
#    include <string.h>// strnlen
#    include <sys/types.h>// sysctl (documented prerequisite of sys/sysctl.h)
#    include <sys/sysctl.h>// sysctl, CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME
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
// representable here, and path::string() substitutes for them silently. The
// ...A APIs it would be handed to could not open such a path either; lifting
// the limitation means moving the whole process to UTF-8.
//
// Returns "" rather than letting an exception escape. resolveSysdir() guards
// its whole body for the static-initializer case, but this is also reached at
// runtime through executableDir(), and "" is the not-representable answer both
// paths already handle.
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
// std::filesystem::canonical is preferred to hand-rolling that
// CreateFileW/GetFinalPathNameByHandleW dance, which also has to strip the \\?\
// prefix, grow its buffer and cope with a volume that has no drive letter.
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
  // fit, reporting the buffer size when it does. MAX_PATH (260) covers every
  // path not using long-path support, so try the stack first and fall back to
  // the heap only for the rare path that needs it. The fallback is sized for
  // the longest path Windows can represent (32767 characters), so one retry
  // either fits or the path is unobtainable -- there is nothing to loop over.
  // That limit counts characters, so the wide form is what it bounds: the same
  // number of bytes can fall short of it under a double-byte code page.
  //
  // std::vector rather than mem::vector: the buffer never escapes and holds no
  // collectable pointers, so there is no reason to place it under the garbage
  // collector -- which would scan it for pointers.
  DWORD const maxSize= 32768;
  wchar_t stackBuf[MAX_PATH];
  std::vector<wchar_t> heapBuf;// stays empty unless the stack buffer is short
  wchar_t const* data= stackBuf;
  DWORD len= GetModuleFileNameW(nullptr, stackBuf, MAX_PATH);
  if (len == 0)// failed
    return "";
  if (len == MAX_PATH) {// truncated; retry at the documented maximum
    heapBuf.resize(maxSize);
    len= GetModuleFileNameW(nullptr, heapBuf.data(), maxSize);
    if (len == 0 || len == maxSize)
      return "";
    data= heapBuf.data();
  }
  // GetModuleFileNameW does not resolve symlinks: launched through a link on
  // PATH it reports the link, whose directory holds no base/. Resolve it so
  // that such a link yields the real install prefix, as on the other
  // platforms. Falling back to the unresolved path on failure mirrors what the
  // macOS branch does when realpath() fails.
  std::filesystem::path const exe(data, data + len);
  string resolved= canonicalPath(exe);
  return resolved.empty() ? narrowPath(exe) : resolved;
#elif defined(__APPLE__)
  char buf[4096];
  uint32_t size= (uint32_t) sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) != 0)
    return "";
  // _NSGetExecutablePath may return a path containing symlinks or "..";
  // resolve it so that a symlinked bin directory (Homebrew, MacPorts) yields
  // the real install prefix. The other POSIX branches arrive resolved already.
  // PATH_MAX is required here -- realpath() may write that many bytes -- and
  // avoided elsewhere in this file, since POSIX leaves it optional.
  char resolved[PATH_MAX];
  if (realpath(buf, resolved) != nullptr)
    return string(resolved);
  return string(buf);
#elif defined(__FreeBSD__)
  // procfs(5) is not mounted on a stock FreeBSD, and where it is it spells this
  // /proc/curproc/file; kern.proc.pathname is the supported query, and a pid of
  // -1 asks about the calling process. The other BSDs need their own branches:
  // NetBSD uses a different mib, OpenBSD has no equivalent. mib is non-const so
  // that it binds to the historical BSD prototype as well as FreeBSD's.
  int mib[4]= {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
  // No growth loop: the kernel builds this path in a MAXPATHLEN buffer, and
  // MAXPATHLEN and PATH_MAX are both 1024 here, so a longer path is one it
  // could not have produced.
  char buf[PATH_MAX];
  size_t size= sizeof(buf);
  if (sysctl(mib, 4, buf, &size, nullptr, 0) != 0)
    return "";
  // A process with no text vnode succeeds while writing nothing, which would
  // leave buf uninitialized.
  if (size == 0)
    return "";
  // A size past the buffer would be a kernel bug; clamp it before strnlen()
  // reads that far. size counts the terminating NUL.
  if (size > sizeof(buf))
    size= sizeof(buf);
  // Already resolved, like /proc/self/exe: the kernel reconstructs the path
  // from p_textvp, the vnode execve() reached after following any symlinks.
  return string(buf, strnlen(buf, size));
#else
  // Linux, and anything else carrying a Linux-style /proc. The kernel resolves
  // this symlink itself, so there is nothing left to canonicalize. A system
  // with neither /proc nor a branch above needs one written for it.
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
// The one candidate is base/ beside the running executable, tried before the
// compiled-in path so that a binary run in place from its build tree uses its
// own base/ rather than that of a separately installed Asymptote. It needs no
// opt-in: <exedir>/base/plain.asy exists only in a build tree or in a
// distribution that deliberately ships base/ beside the binary. The macOS
// bundle is laid out that way -- install-asy with bindir=<dest>/Asymptote
// asydir=<dest>/Asymptote/base -- so it relocates with no compiled-in path
// involved.
//
// Otherwise the compiled-in path is returned unchanged, including when it is
// empty: that is how a TeXLive build says it has no fixed data directory, and
// initDir() then asks kpsewhich for TEXMFMAIN.
//
// noexcept because this runs as a static initializer (settings.cc), where an
// escaping exception calls terminate() before main() rather than being caught
// anywhere. Marking it costs nothing there -- terminate() is what an escaping
// exception would produce either way -- and states the contract in a form the
// compiler checks rather than one a comment can drift away from. The body is
// guarded as a whole rather than at each allocating step: it is built out of
// strings and std::filesystem paths, so the throwing operations are too many to
// enumerate reliably, and all of them mean the same thing here.
//
string resolveSysdir(string const& compiledInSysdir) noexcept
{
  try {
    // parentDir() rather than executableDir(), so that an executable sitting
    // directly in the filesystem root still gets the candidate tried.
    optional<string> const exeDir= parentDir(executablePath());
    if (exeDir) {
      string adjacentBase= *exeDir + "/base";
      if (isBaseDir(adjacentBase)) {
        relocatedSysdir= true;
        return adjacentBase;
      }
    }
    return compiledInSysdir;
  } catch (...) {
    return compiledInSysdir;
  }
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
