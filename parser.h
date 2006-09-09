/*****
 * parser.h
 * Tom Prince 2004/01/10
 *
 *****/

#ifndef PARSER_H
#define PARSER_H

#include <string>
#include "absyn.h"

using std::string;

namespace parser {

// Opens and parses the file returning the abstract syntax tree.  If
// there is an unrecoverable parse error, returns null.
absyntax::file *parseFile(string filename);

// Parses string and returns the abstract syntax tree.
absyntax::file *parseString(string code, string filename);

// Parses standard input and returns the abstract syntax tree.
//absyntax::file *parseInteractive();

} // namespace parser

#endif // PARSER_H
