/*****
 * locate.h
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#ifndef LOCATE_H
#define LOCATE_H

#include <list>
#include <string>

#include <boost/filesystem/path.hpp>

namespace fs = boost::filesystem;

namespace settings {

typedef std::list<fs::path> file_list_t;
extern file_list_t searchPath;

// Find the appropriate file, first looking in the local directory, then the
// directory given in settings, and finally the global system directory.
fs::path locateFile(std::string id);

} // namespace settings

#endif // LOCATE_H
