/*****
 * locate.cc
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include "settings.h"
#include "locate.h"

using std::string;

namespace settings {

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
fs::path locateFile(string id)
{
  file_list_t filenames = mungeFileName(id);
  for (file_list_t::iterator leaf = filenames.begin();
       leaf != filenames.end();
       ++leaf) {
    for (file_list_t::iterator dir = searchPath.begin();
         dir != searchPath.end();
         ++dir) {
      fs::path file = *dir / *leaf;
      if (exists(file))
        return file;
    }
  }
  return fs::path();
}

} // namespace settings

 
