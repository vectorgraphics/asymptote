/*****
 * errormsg.h
 * Andy Hammerlindl 2002/06/17
 *
 * Used in all phases of the compiler to give error messages.
 *****/

#ifndef ERRORMSG_H
#define ERRORMSG_H

#include <iostream>
#include <exception>
#include "common.h"
#include "settings.h"
#include "symbolmaps.h"

using std::ostream;

struct handled_error : std::exception {}; // Exception to process next file.
struct interrupted : std::exception {};   // Exception to interrupt execution.
struct quit : std::exception {};          // Exception to quit current operation.
struct EofException : std::exception {};           // Exception to exit interactive mode.

// Strips the directory prefix and the '.asy' suffix from a filename,
// returning just the module name. Assumes '/' as the directory separator.
namespace errormsg {
string moduleNameFromPath(const string& filename);
} // namespace errormsg

// Registry that interns filenames so that positions can reference them by
// a small (16-bit) index instead of carrying a fileinfo pointer. This keeps
// the position struct compact (one 64-bit word) which improves cache
// locality of bytecode instructions that carry a position.
class positionFileRegistry {
public:
  // Returns the index for the given filename, allocating a new one if
  // necessary. Index 0 is reserved to mean "no file".
  static uint16_t intern(const string& filename);
  // Returns the filename associated with the given index. Returns the
  // empty string for index 0.
  static const string& getFilename(uint16_t index);
};

class fileinfo : public gc {
  string filename;
  size_t lineNum;
  uint16_t fileIndex_;

public:
  fileinfo(string filename, size_t lineNum=1)
    : filename(filename), lineNum(lineNum),
      fileIndex_(positionFileRegistry::intern(filename)) {}

  size_t line() const
  {
    return lineNum;
  }

  string name() const {
    return filename;
  }

  uint16_t fileIndex() const {
    return fileIndex_;
  }

  // The filename without the directory and without the '.asy' suffix.
  // Note that this assumes name are separated by a forward slash.
  string moduleName() const {
    return errormsg::moduleNameFromPath(filename);
  }

  // Specifies a newline symbol at the character position given.
  void newline() {
    ++lineNum;
  }

};

inline bool operator == (const fileinfo& a, const fileinfo& b)
{
  return a.line() == b.line() && a.name() == b.name();
}

class position : public gc {
  // Packed representation: a position is a 16-bit file index (0 means
  // "no file"), a 16-bit line number, a 16-bit column number, and 16
  // bits of padding. Total: one 64-bit word. Equality is a single word
  // comparison.
  uint16_t fileIndex_;
  uint16_t line_;
  uint16_t column_;
  uint16_t reserved_;

  // Saturating cast from a (possibly large) integral value to uint16_t.
  static uint16_t toU16(size_t v) {
    return v > 0xFFFFu ? uint16_t(0xFFFFu) : uint16_t(v);
  }

public:
  void init(fileinfo *f, Int p) {
    if (f) {
      fileIndex_ = f->fileIndex();
      line_ = toU16(f->line());
      column_ = toU16((size_t) p);
    } else {
      fileIndex_ = 0;
      line_ = 0;
      column_ = 0;
    }
    reserved_ = 0;
  }

  string filename() const {
    return positionFileRegistry::getFilename(fileIndex_);
  }

  size_t Line() const {
    return line_;
  }

  size_t Column() const {
    return column_;
  }

  position shift(unsigned int offset) const {
    position P=*this;
    P.line_ = uint16_t(P.line_ - offset);
    return P;
  }

  std::pair<size_t,size_t>LineColumn() const {
    return std::pair<size_t,size_t>(line_,column_);
  }

  bool match(const string& s) {
    return fileIndex_ != 0 && filename() == s;
  }

  bool match(size_t l) {
    return line_ == l;
  }

  bool matchColumn(size_t c) {
    return column_ == c;
  }

  bool operator! () const
  {
    return fileIndex_ == 0;
  }

  friend ostream& operator << (ostream& out, const position& pos);
  friend inline bool operator == (const position& a, const position& b);

  typedef std::pair<size_t, size_t> posInFile;
  typedef std::pair<std::string, posInFile> filePos;

  explicit operator AsymptoteLsp::filePos()
  {
    return std::make_pair((std::string) filename().c_str(),LineColumn());
  }

  void print(ostream& out) const
  {
    if (fileIndex_) {
      out << filename() << ":" << line_ << "." << column_;
    }
  }

  // Write out just the module name and line number.
  void printTerse(ostream& out) const
  {
    if (fileIndex_) {
      const string& fname = positionFileRegistry::getFilename(fileIndex_);
      out << errormsg::moduleNameFromPath(fname) << ":" << line_;
    }
  }
};

extern position nullPos;

struct nullPosInitializer {
  nullPosInitializer() {nullPos.init(NULL,0);}
};

inline bool operator == (const position& a, const position& b)
{
  // Since filenames are interned uniquely into fileIndex, comparing all
  // fields suffices. This is a single 64-bit comparison.
  return a.fileIndex_ == b.fileIndex_ && a.line_ == b.line_ &&
         a.column_ == b.column_;
}

string warning(string s);

enum class ErrorMode
{
  SUPPRESS,// Suppress warnings and errors.
  NORMAL,
  FORCE,// Like normal mode, but ignores attempts to change the mode.
};

class errorstream {
  ostream& out;
  bool anyErrors;
  bool anyWarnings;
  bool floating;        // Was a message output without a terminating newline?

  // Is there an error that warrants the asy process to return 1 instead of 0?
  bool anyStatusErrors;

  ErrorMode mode;
  void setMode(ErrorMode newMode)
  {
    if (mode != ErrorMode::FORCE)
      mode= newMode;
  }

public:
  static bool interrupt; // Is there a pending interrupt?

  using traceback_t = mem::list<position> ;
  traceback_t traceback;

  errorstream(ostream& out = cerr)
    : out(out), anyErrors(false), anyWarnings(false), floating(false),
      anyStatusErrors(false), mode(ErrorMode::NORMAL) {}


  void clear();

  void message(position pos, const string& s);

  void Interrupt(bool b) {
    interrupt=b;
  }

  // An error is encountered, not in the user's code, but in the way the
  // compiler works!  This may be augmented in the future with a message
  // to contact the compiler writers.
  void compiler();
  void compiler(position pos);

  // An error encountered when running compiled code.  This method does
  // not stop the executable, but the executable should be stopped
  // shortly after calling this method.
  void runtime(position pos);

  // Errors encountered when compiling making it impossible to run the code.
  void error(position pos);

  // Indicate potential problems in the code, but the code is still usable.
  void warning(position pos);
  void warning(position pos, string s);

  // Single a fatal error and execute the main process.
  void fatal(position pos);

  // Print out position in code to aid debugging.
  void trace(position pos);

  // Sends stuff to out to print.
  // NOTE: May later make it do automatic line breaking for long messages.
  template<class T>
  errorstream& operator << (const T& x) {
    if (mode != ErrorMode::SUPPRESS) {
      flush(out);
      out << x;
    }
    return *this;
  }

  // Reporting errors to the stream may be incomplete.  This draws the
  // appropriate newlines or file excerpts that may be needed at the end.
  void sync(bool reportTraceback=false);

  void cont();

  bool errors() const {
    return anyErrors;
  }

  bool warnings() const {
    return anyWarnings || errors();
  }

  void statusError() {
    anyStatusErrors=true;
  }

  // Returns true if no errors have occurred that should be reported by the
  // return value of the process.
  bool processStatus() const {
    return !anyStatusErrors;
  }

  bool isSuppressed() const {
    return mode == ErrorMode::SUPPRESS;
  }

  class ModeGuard
  {
    errorstream& es;
    ErrorMode oldMode;

  public:
    ModeGuard(errorstream& es, ErrorMode newMode) : es(es), oldMode(es.mode)
    {
      es.setMode(newMode);
    }
    ~ModeGuard() { es.setMode(oldMode); }

    ModeGuard(const ModeGuard&) = delete;
    ModeGuard& operator=(const ModeGuard&) = delete;
    ModeGuard(ModeGuard&&) = delete;
    ModeGuard& operator=(ModeGuard&&) = delete;
  };

  ModeGuard modeGuard(ErrorMode newMode) { return ModeGuard(*this, newMode); }
};

extern errorstream em;
void outOfMemory();

GC_DECLARE_PTRFREE(nullPosInitializer);

#endif
