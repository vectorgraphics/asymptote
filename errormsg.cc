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

ostream& operator<< (ostream& out, const position& pos)
{
  if (!pos)
    return out;

  bool interact = pos.file->name() == "-" && interact::interactive
    && pos.line > 1;
  
  out << pos.file->name() << ": " << (interact ? pos.line-1 : pos.line) << "."
      << pos.column << ": ";
  return out;
}

bool errorstream::interrupt=false;

void errorstream::clear()
{
  sync();
  anyErrors = anyWarnings = false;
}

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

void errorstream::compiler()
{
  message(position::nullPos(),"compiler: ");
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
  if(!pos) return;
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

