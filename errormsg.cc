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

ostream& fileinfo::print(ostream& out, int pos)
{
  int num = lineNum;
  std::list<int>::iterator lines = linePos.begin();
  std::list<int>::iterator last = --linePos.end();
  while (lines != last && *lines >= pos) {
    ++lines; num--;
  }

  out << filename << ": " << num << "." << pos-*lines << ": ";
  return out;
}

position lastpos;

void errorstream::error(position pos)
{
  if (floating) out << endl;

  out << pos;

  floating = true;
  anyErrors = true;
}

void errorstream::warning(position pos)
{
  if (floating) out << endl;

  out << pos << "warning: ";

  floating = true;
}

void errorstream::runtime()
{
  if (floating) out << endl;

  out << "<unknown pos>: runtime: ";

  floating = true;
  anyErrors = true;
}

void errorstream::runtime(position pos)
{
  if (floating) out << endl;

  out << pos << "runtime: ";

  floating = true;
  anyErrors = true;
}

void errorstream::debug(position pos)
{
  if (floating) out << endl;

  out << pos << "runtime: ";

  floating = true;
  sync();
}

void errorstream::compiler()
{
  if (floating) out << endl;

  out << "compiler: ";

  floating = true;
  anyErrors = true;
}

void errorstream::compiler(position pos)
{
  if (floating) out << endl;

  out << pos << "compiler: ";

  floating = true;
  anyErrors = true;
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
