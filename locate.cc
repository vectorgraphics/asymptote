/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#include <unistd.h>

#include "settings.h"
#include "locate.h"

using std::string;

namespace settings {

namespace fs {

string extension(std::string name)
{
  size_t n = name.rfind(".");
  if (n != string::npos)
    return name.substr(n);
  else
    return string();
}

bool exists(string filename)
{
  return ::access(filename.c_str(), R_OK) == 0;  
}

} // namespace fs


file_list_t searchPath;

// Returns list of possible filenames, accounting for extensions.
file_list_t mungeFileName(string id)
{
  string ext = fs::extension(id);
  file_list_t files;
  if (ext == "."+settings::suffix ||
      ext == "."+settings::guisuffix) {
    files.push_back(id);
    files.push_back(id+"."+settings::suffix);
  } else {
    files.push_back(id+"."+settings::suffix);
    files.push_back(id);
  }
  return files;
}


// Find the appropriate file, first looking in the local directory, then the
// directory given in settings, and finally the global system directory.
string locateFile(string id)
{
  file_list_t filenames = mungeFileName(id);
  for (file_list_t::iterator leaf = filenames.begin();
       leaf != filenames.end();
       ++leaf) {
    for (file_list_t::iterator dir = searchPath.begin();
         dir != searchPath.end();
         ++dir) {
      string file = *dir + "/" + *leaf;
      if (fs::exists(file))
        return file;
    }
  }
  return std::string();
}

} // namespace settings

 
