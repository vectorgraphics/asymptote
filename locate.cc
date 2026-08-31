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
// representable here, and path::string() substitutes for them silently. That is
// not a loss introduced by resolving the path in wide form below: the ...A APIs
// it would be handed to cannot open such a path either. Lifting that limitation
// means moving the whole process to UTF-8 (a manifest declaring it as the active
// code page), which is not something this file can do on its own.
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
  // the real install prefix. The other two POSIX branches do not need this:
  // /proc/self/exe and kern.proc.pathname both arrive resolved.
  //
  // PATH_MAX (POSIX, 1024 here -- not to be confused with Win32's MAX_PATH)
  // is required: realpath() may write that many bytes to the buffer.
  // Nothing else here uses it, since POSIX leaves it optional and glibc/Hurd
  // does not define it.
  char resolved[PATH_MAX];
  if (realpath(buf, resolved) != nullptr)
    return string(resolved);
  return string(buf);
#elif defined(__FreeBSD__)
  // FreeBSD cannot use the /proc/self/exe branch below: procfs(5) is not
  // mounted on a stock system, and even where it is, it is FreeBSD's own procfs
  // -- which spells this /proc/curproc/file -- rather than the Linux one. The
  // supported way to ask is the kern.proc.pathname sysctl, present since
  // FreeBSD 6.0 and the same thing "procstat -b" reports. A pid of -1 in the
  // last mib slot asks about the calling process; the kernel special-cases it,
  // so it skips both the process lookup and the permission check that asking
  // about a numbered pid goes through.
  //
  // The other BSDs are deliberately not folded in here: NetBSD spells the same
  // query with a different mib (KERN_PROC_ARGS, with KERN_PROC_PATHNAME as the
  // *fourth* element), and OpenBSD has no equivalent at all. Each needs its own
  // branch, written by someone who can test it.
  //
  // mib is deliberately not const: FreeBSD's sysctl() takes the name as
  // const int*, but the historical BSD prototype does not, and a non-const
  // array binds to either.
  int mib[4]= {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
  // One PATH_MAX buffer is enough, and no growth loop is needed: the kernel
  // reconstructs this path into a buffer of MAXPATHLEN bytes, and MAXPATHLEN
  // and PATH_MAX are both 1024 on FreeBSD, so a path too long to fit here is
  // one the kernel could not have produced. If that ever ceases to hold,
  // sysctl() fails with ENOMEM and we return "" -- resolveSysdir() then uses
  // the compiled-in sysdir, which is what a non-relocatable build does anyway.
  // (PATH_MAX is used for the same reason as in the __APPLE__ branch above:
  // required here, and avoided everywhere else in this file because POSIX
  // leaves it optional.)
  char buf[PATH_MAX];
  size_t size= sizeof(buf);
  if (sysctl(mib, 4, buf, &size, nullptr, 0) != 0)
    return "";
  // Succeeding while writing nothing is a real outcome, not a contradiction:
  // the kernel returns an empty result rather than an error for a process with
  // no text vnode. Without this, buf would be read uninitialized.
  if (size == 0)
    return "";
  // Bound the reported length by the buffer before using it as one: sysctl()
  // cannot have written more than it was given room for, so a larger value
  // would be a kernel bug -- and strnlen() would run off the end on it.
  if (size > sizeof(buf))
    size= sizeof(buf);
  // size counts the terminating NUL that the kernel writes, so the string is
  // one shorter. strnlen() rather than size - 1 so that a result that somehow
  // arrived unterminated is bounded by what was actually written instead of
  // running off the end of the buffer.
  //
  // No realpath() here, unlike the __APPLE__ branch: this path arrives already
  // resolved, as /proc/self/exe does. The kernel does not record the string
  // passed to execve(); it reconstructs a path from p_textvp, the vnode of the
  // file that was executed. execve()'s lookup followed any symlinks on the way
  // to that vnode, so it is the target's vnode and not the link's, and the name
  // cache can only spell it as its own name in its own parent directory.
  return string(buf, strnlen(buf, size));
#else
  // Linux, and anything else carrying a Linux-style /proc. The kernel resolves
  // this symlink itself, so there is nothing left to canonicalize -- and
  // nothing to fall back on either: a system with neither /proc nor a branch
  // above needs one written for it, as FreeBSD did.
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
// When nothing matches, the compiled-in path is returned unchanged -- including
// when it is empty, which is how a TeXLive build says "I have no fixed data
// directory". initDir() sees the empty string and asks kpsewhich for TEXMFMAIN.
// That lookup is therefore the next candidate after the ones below, not a
// separate mode: a TeXLive binary run from its build tree uses the adjacent
// base/, and only a deployed one (bin/<platform>/asy, where no candidate
// matches) consults kpsewhich.
//
// noexcept because this runs as a static initializer (settings.cc), where an
// escaping exception calls terminate() before main() rather than being caught
// anywhere. Marking it costs nothing there -- terminate() is what an escaping
// exception would produce either way -- and states the contract in a form the
// compiler checks rather than one a comment can drift away from. The body is
// guarded as a whole rather than at each allocating step: every candidate is
// built from strings and std::filesystem paths, so the throwing operations are
// too many to enumerate reliably, and all of them mean the same thing here.
//
string resolveSysdir(string const& compiledInSysdir) noexcept
{
  try {
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
