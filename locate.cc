/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#if defined(_WIN32)
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

// Absolute path of the running executable, or "" if it cannot be determined.
static string getExecutablePath()
{
#if defined(_WIN32)
  // GetModuleFileNameA truncates rather than failing when the path does not
  // fit, so allocate for the longest path Windows documents (32767 chars)
  // rather than the MAX_PATH (260) that long-path support can exceed.
  DWORD const size= 32768;
  mem::vector<char> buf(size);
  DWORD len= GetModuleFileNameA(nullptr, buf.data(), size);
  if (len == 0 || len == size)// 0: failed; size: truncated
    return "";
  return string(buf.data(), len);
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
