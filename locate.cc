/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#ifdef IS_RELOCATABLE
#  if defined(_WIN32)
#    include <Windows.h>
#  else
#    include <unistd.h>
#    ifdef __APPLE__
#      include <mach-o/dyld.h>
#    endif
#  endif
#endif

#include "locate.h"
#include "settings.h"
#include "util.h"


namespace settings
{

#ifdef IS_RELOCATABLE
// Absolute path of the running executable, or "" if it cannot be determined.
static string getExecutablePath()
{
#  if defined(_WIN32)
  char buf[MAX_PATH];
  DWORD len= GetModuleFileNameA(nullptr, buf, sizeof(buf));
  if (len == 0 || len == sizeof(buf))
    return "";
  return string(buf, len);
#  elif defined(__APPLE__)
  char buf[4096];
  uint32_t size= (uint32_t) sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) != 0)
    return "";
  return string(buf);
#  else
  char buf[4096];
  ssize_t len= readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0)
    return "";
  return string(buf, len);
#  endif
}
#endif

// Determine the system base directory. Normally this is the compiled-in
// ASYMPTOTE_SYSDIR. For a relocatable binary (IS_RELOCATABLE), when that path
// does not exist on disk -- e.g. the binary is run in place from its build
// tree, or from a staged/relocated install -- locate base/ relative to the
// running executable instead, so it can find its data directory without -dir.
string initSysdir()
{
#ifdef IS_RELOCATABLE
  string sysdir= ASYMPTOTE_SYSDIR;
  if (!sysdir.empty() && fileExists(sysdir))
    return sysdir;
  string exe= getExecutablePath();
  if (!exe.empty()) {
    size_t slash= exe.find_last_of("/\\");
    if (slash != string::npos) {
      string bindir= exe.substr(0, slash);
      // Build tree: base/ sits next to the executable.
      string buildBase= bindir + "/base";
      if (fileExists(buildBase))
        return buildBase;
      // Install tree: <prefix>/bin/asy with data in <prefix>/share/asymptote.
      size_t slash2= bindir.find_last_of("/\\");
      if (slash2 != string::npos)
        return bindir.substr(0, slash2) + "/share/asymptote";
    }
  }
#endif
  return ASYMPTOTE_SYSDIR;
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
