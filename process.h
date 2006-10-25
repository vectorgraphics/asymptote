/*****
 * process.h
 * Andy Hammerlindl 2006/08/19 
 *
 * Handles processing blocks of code (including files, strings, and the
 * interactive prompt, for listing and parse-only modes as well as actually
 * running it.
 *****/

#ifndef PROCESS_H
#define PROCESS_H

#include "memory.h"
#include "stm.h"
#include "stack.h"

// Process the code respecting the parseonly and listvariables flags of
// settings.
void processCode(absyntax::block *code);
void processFile(const std::string& filename);
void processPrompt();

// Run the code in its own environment.
void runCode(absyntax::block *code);
void runString(const std::string& string);
void runFile(const std::string& filename);
void runPrompt();

// Run the code in a given run-time environment.
typedef vm::interactiveStack istack;
void runCodeEmbedded(absyntax::block *code, trans::coenv &e, istack &s);
void runStringEmbedded(const std::string& string, trans::coenv &e, istack &s);
void runPromptEmbedded(trans::coenv &e, istack &s);

// Basic listing.
void doUnrestrictedList();

#endif
