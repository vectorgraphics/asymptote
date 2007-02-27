/*****
 * parser.h
 * Tom Prince 2004/01/10
 *
 *****/

#ifndef PARSER_H
#define PARSER_H

#include "memory.h"
#include "absyn.h"

namespace parser {

// Opens and parses the file returning the abstract syntax tree.  If
// there is an unrecoverable parse error, returns null.
absyntax::file *parseFile(const mem::string& filename, const char *text=NULL);

// Parses string and returns the abstract syntax tree.  Any error in lexing or
// parsing will be reported and a handled_error thrown.  If the string is
// "extendable", then a parse error simply due to running out of input will not
// throw an exception, but will return null.
absyntax::file *parseString(const mem::string& code,
			    const mem::string& filename,
                            bool extendable=false);
} // namespace parser

#endif // PARSER_H
