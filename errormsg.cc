/*****
 * errormsg.cc
 * Andy Hammerlindl 2002/06/17
 *
 * Used in all phases of the compiler to give error messages.
 *****/

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

#include "errormsg.h"
#include "interact.h"

ostream& fileinfo::print(ostream& out, int pos)
{
  int num = lineNum;
  std::list<int>::iterator lines = linePos.begin();
  std::list<int>::iterator last = --linePos.end();
  while (lines != last && *lines >= pos) {
    ++lines; num--;
  }
  if(filename == "-" && interact::interactive && num > 1) num--;

  out << filename.string() << ": " << num << "." << pos-*lines << ": ";
  return out;
}

position errorstream::lastpos=position::nullPos();
bool errorstream::interrupt=false;

void errorstream::message(position pos, const std::string& s)
{
  if (floating) out << endl;
  out << pos << s;
  floating = true;
}

void errorstream::compiler(position pos)
{
  message(pos,"compiler: ");
  anyErrors = true;
}

void errorstream::runtime(position pos)
{
  message(pos,"runtime: ");
  anyErrors = true;
}

void errorstream::error(position pos)
{
  message(pos,"");
  anyErrors = true;
}

void errorstream::warning(position pos)
{
  message(pos,"warning: ");
  anyWarnings = true;
}

void errorstream::trace(position pos)
{
  message(pos,"");
  sync();
}

void errorstream::sync()
{
  if (floating) out << endl;
  floating = false;
}

void errorstream::printCamp(position pos)
{
  while (camp::errors()) {
    runtime(pos);
    *this << "camp: " << camp::getError();
    sync();
  }

  throw handled_error();
}
