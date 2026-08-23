/*****
 * locate.h
 * Tom Prince 2005/03/24
 *
 * Locate files in search path.
 *****/

#ifndef LOCATE_H
#define LOCATE_H

#include "common.h"
#include "settings.h"

namespace settings {

typedef mem::list<string> file_list_t;
extern file_list_t searchPath;

// Absolute path of the running executable, or "" if it cannot be determined.
// Symlinks are resolved, so a binary reached through a link on PATH (or a
// symlinked bin directory, as Homebrew and MacPorts create) still yields the
// real install prefix. On MSWindows the result is narrowed to the process code
// page, which is the encoding the ...A file APIs used elsewhere expect.
string executablePath();

// The directory containing the running executable, without a trailing
// separator (except for a root directory, where the separator is part of it:
// "C:\"), or "" if it cannot be determined. Callers that need to resolve a
// co-installed file (base/, a renderer shared library, an ICD manifest) should
// use this rather than rolling their own: the search path is CWD-dependent and
// can pick up a stale system-wide installation.
string executableDir();

// Determine the system base directory (used to initialize settings::systemDir).
// Candidates relative to the running executable are preferred over the
// compiled-in ASYMPTOTE_SYSDIR, so that a binary run in place uses its own
// base/ rather than that of a separately installed Asymptote. Callers must pass
// ASYMPTOTE_SYSDIR in: under CMake it differs between asy and asy-ctan, and
// settings.cc is the only source file compiled separately per executable.
// See locate.cc.
string resolveSysdir(string const& compiledInSysdir);

// True if initSysdir() resolved systemDir relative to the running executable.
// Used to keep an installed Asymptote's configuration (the MSWindows registry
// entry) from overriding a binary that found its own base/.
extern bool relocatedSysdir;

// Find the appropriate file, first looking in the local directory, then the
// directory given in settings, and finally the global system directory.
string locateFile(string id, bool full=false, string suffix=settings::suffix);

namespace fs {

// Check to see if a file of given name exists.
bool exists(string filename);

}

} // namespace settings

#endif // LOCATE_H
