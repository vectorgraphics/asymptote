/*****
 * errormsg.h
 * Andy Hammerlindl 2002/06/17
 *
 * Used in all phases of the compiler to give error messages.
 *****/

#ifndef ERRORMSG_H
#define ERRORMSG_H

#include <list>
#include <iostream>
#include "camperror.h"
#include "pool.h"
#include "settings.h"

using std::ostream;
using std::endl;

class fileinfo : public mempool::pooled<fileinfo> {
  std::string filename;
  std::list<int> linePos;
  int lineNum;

public:
  fileinfo(std::string filename)
    : filename(filename), lineNum(1)
  {
    linePos.push_front(0);
  }
  
  // Specifies a newline symbol at the character position given.
  void newline(int tokPos) {
    linePos.push_front(tokPos);
    ++lineNum;
  }

  // Prints out a position for an error message, with filename, row and column.
  ostream& print(ostream& out, int p);
};


class position {
  fileinfo *file;
  int p; // The offset in characters in the file.

public:
  /*position()
    : file(0), p(0) {} 

  position(fileinfo *file, int p)
    : file(file), p(p) {}
  */

  void init(fileinfo *file, int p)
  {
    this->file = file;
    this->p = p;
  }

  friend ostream& operator<< (ostream& out, const position& pos)
  {
    if (pos.file)
      pos.file->print(out, pos.p);
    else
      out << "<unknown pos>: ";
    return out;
  }

  static position nullPos()
  {
    position p;
    p.init(0,0);
    return p;
  }
};


extern position lastpos;

class errorstream {
  ostream& out;
  bool anyErrors;

  // If there is an error printed without the closing newline.
  bool floating;

  // Prints errors occured with camp and exits.  Does not return.
  void printCamp(position pos);

public:
  errorstream(ostream& out = std::cerr)
    : out(out), anyErrors(false), floating(false) {}

  // Print out position in code to aid debugging.
  void debug(position pos);
  
  // Errors encountered when compiling making it impossible to run the code.
  void error(position pos);

  // More information regarding the previous error.
  //void suberror(position pos);
  
  // Indicate potential problems in the code, but the code is still
  // usable.
  void warning(position pos);

  // An error encountered when running compiled code.  This method does
  // not stop the executable, but the executable should be stopped
  // shortly after calling this method.
  void runtime();
  void runtime(position pos);

  // An error is encountered, not in the user's code, but in the way the
  // compiler works!  This may be augmented in the future with a message
  // to contact the compiler writers.
  void compiler();
  void compiler(position pos);

  // Sends stuff to out to print.
  // NOTE: May later make it do automatic line breaking for long messages.
  template<class T>
  errorstream& operator<< (const T& x)
  {
    out << x;
    return *this;
  }

  // Reporting errors to the stream may be incomplete.  This draws the
  // appropriate newlines or file excerpts that may be needed at the end.
  void sync();

  // Camp has its own methods for reporting errors.  After a method in
  // camp is run, the function calling camp should call this to
  // report any errors.  If checkCamp() finds errors, it will print them
  // out, then stop execution.
  void checkCamp(position pos)
  {
    lastpos=pos;
    if (camp::errors())
      printCamp(pos);
    if (settings::verbose > 4) 
      debug(pos);
  }
  
  bool errors() {
    return anyErrors;
  }
  
};

struct handled_error {}; // Exception to process next file.
  
extern errorstream *em;

#endif
