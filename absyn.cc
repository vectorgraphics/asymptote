/****
 * absyn.cc
 * Tom Prince 2004/05/12
 *
 * Utility functions for syntax trees.
 *****/

#include "absyn.h"
#include "coenv.h"

namespace absyntax {

void absyn::markPos(trans::coenv& e)
{
  e.c.markPos(getPos());
}

absyn::~absyn()
{}

void prettyindent(ostream &out, int indent)
{
  for (int i = 0; i < indent; i++) out << " ";
}
void prettyname(ostream &out, std::string name, int indent) {
  prettyindent(out,indent);
  out << name << "\n";
}

} // namespace absyntax
