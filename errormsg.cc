/*****
 * errormsg.cc
 * Andy Hammerlindl 2002/06/17
 *
 * Used in all phases of the compiler to give error messages.
 *****/

#include <cstdio>
#include <cstdlib>
#include <regex>

#include "errormsg.h"
#include "interact.h"
#include "fileio.h"
#include "hashing.h"

errorstream em;

position nullPos;
static nullPosInitializer nullPosInit;

bool errorstream::interrupt=false;

namespace {
struct memStringHash {
  size_t operator()(const string& s) const {
    return hashing::hashSpan(s);
  }
};
// Storage for the filename registry. Index 0 is reserved for the empty
// string and corresponds to "no file" (i.e. nullPos).
mem::vector<string>& registryFilenames() {
  static mem::vector<string> v = [] {
    mem::vector<string> tmp;
    tmp.push_back(string());
    return tmp;
  }();
  return v;
}
mem::unordered_map<string, uint16_t, memStringHash>& registryIndex() {
  static mem::unordered_map<string, uint16_t, memStringHash> m;
  return m;
}
} // namespace

uint16_t positionFileRegistry::intern(const string& filename)
{
  if (filename.empty()) return 0;
  auto& idx = registryIndex();
  auto it = idx.find(filename);
  if (it != idx.end()) return it->second;
  auto& names = registryFilenames();
  size_t next = names.size();
  // Cap at uint16_t max; if exceeded, reuse last index (graceful fallback).
  if (next > 0xFFFFu) return uint16_t(0xFFFFu);
  uint16_t newIndex = uint16_t(next);
  names.push_back(filename);
  idx.emplace(filename, newIndex);
  return newIndex;
}

const string& positionFileRegistry::getFilename(uint16_t index)
{
  auto& names = registryFilenames();
  if (index >= names.size()) return names[0];
  return names[index];
}

namespace errormsg {
string moduleNameFromPath(const string& filename) {
  size_t start = filename.rfind('/');
  if (start == filename.npos)
    start = 0;
  else
    // Step over slash.
    ++start;

  size_t end = filename.rfind(".asy");
  if (end != filename.size() - 4)
    end = filename.size();

  return filename.substr(start, end-start);
}
} // namespace errormsg

using camp::newl;

ostream& operator<< (ostream& out, const position& pos)
{
  if (!pos)
    return out;

  string filename=pos.filename();

  if(filename != "-" && !(settings::getSetting<bool>("quiet") ||
                          settings::getSetting<bool>("where"))) {
    std::ifstream fin(filename.c_str());
    string s;
    size_t count=pos.line_;
    while(count > 0 && getline(fin,s)) {
      count--;
    }
    s=std::regex_replace(s,std::regex("\t")," ");
    out << s << endl;
    for(size_t i=1; i < pos.column_; ++i)
      out << " ";
    out << "^" << endl;
  }

  out << filename << ": ";
  out << pos.line_ << "." << pos.column_;

  if(settings::xasy) {
    camp::openpipeout();
    fprintf(camp::pipeout,"Error\n");
    fflush(camp::pipeout);
  }

  return out;
}

void errorstream::clear()
{
  sync();
  anyErrors = anyWarnings = false;
}

void errorstream::message(position pos, const string& s)
{
  if (mode == ErrorMode::SUPPRESS)
    return;
  if (floating) out << endl;
  out << pos << ": " << s;
  floating = true;
}

void errorstream::compiler(position pos)
{
  mode = ErrorMode::FORCE;
  message(pos,"Compiler bug; report to https://github.com/vectorgraphics/asymptote/issues:\n");
  anyErrors = true;
}

void errorstream::compiler()
{
  compiler(nullPos);
}

void errorstream::runtime(position pos)
{
  if (mode == ErrorMode::SUPPRESS)
    return;
  message(pos,"runtime: ");
  anyErrors = true;
}

void errorstream::error(position pos)
{
  if (mode == ErrorMode::SUPPRESS)
    return;
  message(pos,"");
  anyErrors = true;
}

void errorstream::warning(position pos, string s)
{
  if (mode == ErrorMode::SUPPRESS)
    return;
  message(pos,"warning ["+s+"]: ");
  anyWarnings = true;
}

void errorstream::warning(position pos)
{
  if (mode == ErrorMode::SUPPRESS)
    return;
  message(pos,"warning: ");
  anyWarnings = true;
}

void errorstream::fatal(position pos)
{
  mode = ErrorMode::FORCE;
  message(pos,"abort: ");
  anyErrors = true;
}

void errorstream::trace(position pos)
{
  if (mode == ErrorMode::SUPPRESS)
    return;
  static position lastpos;
  if(!pos || (pos.match(lastpos.filename()) && pos.match(lastpos.Line())))
    return;
  lastpos=pos;
  message(pos,"");
  sync();
}

void errorstream::cont()
{
  floating = false;
}

void errorstream::sync(bool reportTraceback)
{
  if (floating) out << endl;

  if(reportTraceback && traceback.size()) {
    bool first=true;
    for(auto p=this->traceback.rbegin(); p != this->traceback.rend(); ++p) {
      if(p->filename() != "-") {
        if(first) {
          out << newl << "TRACEBACK:";
          first=false;
        }
        cout << newl << (*p) << endl;
      }
    }
    traceback.clear();
  }

  floating = false;
}

void outOfMemory()
{
  cerr << "error: out of memory" << endl;
  exit(1);
}
