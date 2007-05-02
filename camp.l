%{
/*****
 * camp.l
 * Andy Hammerlindl 2002/06/14
 *
 * The lexical analyzer of the Asymptote language.
 *****/

#include <iostream>
#include <fstream>
#include <sstream>

#include "util.h"
#include "modifier.h"
#include "exp.h"
#include "stm.h"
#include "fundec.h"
#include "errormsg.h"
#include "interact.h"
#include "lexical.h"

using namespace absyntax;
using mem::string;

#include "camp.tab.h"

#define YY_NO_UNPUT

fileinfo* fi;
int tokPos;
int charPos;
//int commentDepth = 0;

bool eof;
string eofMessage;

extern errorstream *em;

extern "C" int yywrap(void)
{
  charPos=1;
  return 1;
}

typedef size_t (*input_f) (char* bif, size_t max_size);

input_f yy_input = NULL;

void setlexer(input_f input, string filename) 
{
  YY_FLUSH_BUFFER;
  yywrap();
  fi = new fileinfo(filename);
  yy_input = input;
  tokPos = charPos = 1;

  eof=false;
  eofMessage="<no eof>";
}

#define YY_INPUT(buf,result,max_size) {result=yy_input(buf,max_size);}

position lexerPos()
{
  position p;
  p.init(fi, tokPos);
  return p;
}
  
namespace {
position here()

{
  return lexerPos();
}

void adjust()
{
  tokPos = charPos;
  charPos += yyleng;
  yylval.pos = here();
} 

void makesymbol(bool op=true, string text = yytext)
{
  adjust();
  yylval.ps.pos=here();
  yylval.ps.sym=op ? symbol::opTrans(text) : symbol::literalTrans(text);
}

void makeselfsymbol(bool op=true, string text = yytext)
{
  /* Copy all but the last digit, which should be an =. */
  assert(text[text.length()-1] == '=');
  text.erase(text.length()-1);
  makesymbol(op,text);
}

void makemod(trans::modifier mod) {
  yylval.mod.pos=here();
  yylval.mod.val=mod;
}

void makeperm(trans::permission perm) {
  yylval.perm.pos=here();
  yylval.perm.val=perm;
}

void newline()
{
  fi->newline();
  charPos = tokPos = 1;
}

void error(void)
{
  em->error(here());
}

}


// Used by the lexer rules to flag an unexpected end of input.  The message is
// the error message that should be reported, and may differ if, say the input
// ends in the middle of a string or comment.
void setEOF(string message) {
  eof=true;
  eofMessage=message;
}
  
// Called by code outside of the lexer to see if a parse error was caused by
// running out of input.
bool lexerEOF()
{
  return eof;
}

// Called by code outside of the lexer when it wants to report the unexpected
// eof as an error (instead of looking for more input).
void reportEOF() {
  assert(eof);
  error();
  *em << eofMessage;
  em->sync();
}

string stringbuild;

namespace {
void append(char c)
{
  stringbuild.push_back(c);
  yylval.pos = here();
}

void getstring(void)
{
  // NOTE: Replace here() with a position at the start of the string.
  makesymbol(false,stringbuild);
  string().swap(stringbuild);
}
}

%}

%x lexcomment
%x texstring
%x cstring
%x lexformat
%x opname

LETTER [_A-Za-z]
ESC \\
ENDL \\?(\r\n|\n|\r)
EXTRAOPS <<|>>|$|$$|@|@@|~

%%

<lexcomment>{
\/\*               {adjust(); /*commentDepth++;*/}
\*\/               {adjust(); /*commentDepth--;*/
                    /*if (commentDepth == 0)*/ BEGIN INITIAL; }
\r\n|\n|\r         {adjust(); newline(); continue; }
<<EOF>>            {adjust();
                    setEOF("comment not terminated");
                    BEGIN INITIAL;
                    return GARBAGE;
                   }
.                  {adjust(); continue; }
}

<texstring>{
\"/([ \t]|{ENDL})*[\"\'] {adjust(); BEGIN INITIAL;}
\"                 {adjust(); 
                    BEGIN INITIAL;
                    getstring(); 
                    return STRING; }
<<EOF>>            {adjust();
                    setEOF("string not terminated");
                    BEGIN INITIAL;
                    getstring(); 
                    return GARBAGE;
                   }
{ENDL}             {adjust(); newline(); append('\n'); continue; }
{ESC}{ESC}         {adjust(); append('\\'); append('\\'); continue; }
{ESC}\"            {adjust(); append('\"'); continue; }
.                  {adjust(); append(*yytext); }
}

<cstring>{
\'/([ \t]|{ENDL})*[\"\'] {adjust(); BEGIN INITIAL;}
\'                 {adjust(); 
                    BEGIN INITIAL;
                    getstring(); 
                    return STRING; }
<<EOF>>            {adjust();
                    setEOF("string not terminated");
                    BEGIN INITIAL;
                    getstring(); 
                    return GARBAGE;
                   }
{ENDL}             {adjust(); newline(); append('\n'); continue; }
{ESC}(\'|\"|\?|\\)       {adjust(); append(yytext[1]); continue; }
{ESC}a        {adjust(); append('\a'); continue; }
{ESC}b        {adjust(); append('\b'); continue; }
{ESC}f        {adjust(); append('\f'); continue; }
{ESC}n        {adjust(); append('\n'); continue; }
{ESC}r        {adjust(); append('\r'); continue; }
{ESC}t        {adjust(); append('\t'); continue; }
{ESC}v        {adjust(); append('\v'); continue; }
{ESC}[0-7]         {adjust();
                    char x=(char)(yytext[1]-'0');
                    append(x);
                    continue;
                   } 
{ESC}[0-7][0-7]    {adjust();
                    char x=(char)((yytext[1]-'0')*8+yytext[2]-'0');
                    append(x);
                    continue;
                   } 
{ESC}[0-3][0-7][0-7] {adjust();
                    char x=(char)((yytext[1]-'0')*64+(yytext[2]-'0')*8
                            +yytext[3]-'0');
                    append(x);
                    continue;
                   } 
{ESC}x[0-9,A-F]    {adjust();
                    char x=(char) (yytext[2] <= '9' ? yytext[2]-'0' : 
                                                      10+yytext[2]-'A');
                    append(x);
                    continue;
                   } 
{ESC}x[0-9,A-F][0-9,A-F] {adjust();
                    char x=(char) ((yytext[2] <= '9' ? yytext[2]-'0' : 
                                                      10+yytext[2]-'A')*16
                                  +(yytext[3] <= '9' ? yytext[3]-'0' : 
                                                      10+yytext[3]-'A'));
                    append(x);
                    continue;
                   } 
.                  {adjust(); append(*yytext); }
}

[ \t]              {adjust(); continue;}
{ENDL}             {adjust(); newline(); continue;}
\/\/[^\n]*         {adjust(); continue;}
","                {adjust(); return ','; }
":"                {adjust(); return ':'; }
";"                {adjust(); return ';'; }
"("                {adjust(); return '('; }
")"                {adjust(); return ')'; }
"["                {adjust(); return '['; }
"]"                {adjust(); return ']'; }
"{"                {adjust(); return '{'; }
"}"                {adjust(); return '}'; }
"."                {adjust(); return '.'; }
"..."              {adjust(); return ELLIPSIS; }
"+"                {makesymbol(); return '+'; }
"-"                {makesymbol(); return '-'; }
"*"                {makesymbol(); return '*'; }
"/"                {makesymbol(); return '/'; }
"%"                {makesymbol(); return '%'; }
"^"                {makesymbol(); return '^'; }
"**"               {makesymbol(true,"^"); return '^'; }
"?"                {adjust(); return '?'; }
"="                {adjust(); return ASSIGN; }
"=="               {makesymbol(); return EQ; }
"!="               {makesymbol(); return NEQ; }
"<"                {makesymbol(); return LT; }
"<="               {makesymbol(); return LE; }
">"                {makesymbol(); return GT; }
">="               {makesymbol(); return GE; }
"&&"               {makesymbol(); return CAND; }
"||"               {makesymbol(); return COR; }
"!"                {makesymbol(); return LOGNOT; }
"^^"               {makesymbol(); return CARETS; }
"::"               {makesymbol(); return COLONS; }
"++"               {makesymbol(); return INCR; }
".."               {makesymbol(); return DOTS; }
"--"               {makesymbol(); return DASHES; }
"---"              {makesymbol(); return LONGDASH; }
"&"                {makesymbol(); return AMPERSAND; }
"|"                {makesymbol(); return BAR; }
{EXTRAOPS}         {makesymbol(); return OPERATOR; }

"+="               {makeselfsymbol(); return ADD; }
"-="               {makeselfsymbol(); return SUBTRACT; }
"*="               {makeselfsymbol(); return TIMES; }
"/="               {makeselfsymbol(); return DIVIDE; }
"%="               {makeselfsymbol(); return MOD; }
"^="               {makeselfsymbol(); return EXPONENT; }

and                {adjust(); return AND; }
controls           {makesymbol(); return CONTROLS; }
tension            {makesymbol(); return TENSION; }
atleast            {makesymbol(); return ATLEAST; }
curl               {makesymbol(); return CURL; }

if                 {adjust(); return IF; }
else               {adjust(); return ELSE; }
while              {adjust(); return WHILE; }
for                {adjust(); return FOR; }
do                 {adjust(); return DO; }
return             {adjust(); return RETURN_; }
break              {adjust(); return BREAK; }
continue           {adjust(); return CONTINUE; }
struct             {adjust(); return STRUCT; }
typedef            {adjust(); return TYPEDEF; }
new                {adjust(); return NEW; }
access             {adjust(); return ACCESS; }
import             {adjust(); return IMPORT; }
unravel            {adjust(); return UNRAVEL; }
from               {adjust(); return FROM; }
include            {adjust(); return INCLUDE; }
quote              {adjust(); return QUOTE; }
static             {adjust(); makemod(trans::EXPLICIT_STATIC);
                              return MODIFIER; }
public             {adjust(); makeperm(trans::PUBLIC); return PERM; }
private            {adjust(); makeperm(trans::PRIVATE); return PERM; }
restricted         {adjust(); makeperm(trans::RESTRICTED); return PERM; }
this               {adjust(); return THIS; }
explicit           {adjust(); return EXPLICIT; }


[0-9]+             try {
  adjust(); yylval.e= new intExp(here(), lexical::cast<int>(yytext)); 
  } catch (lexical::bad_cast&) {
    error();
    *em << "invalid integer";
    yylval.e= new intExp(here(), 0); 
  } return LIT;
([0-9]*\.[0-9]+)|([0-9]+\.[0-9]*)|([0-9]*\.*[0-9]+e[-+]*[0-9]+)|([0-9]+\.[0-9]*e[-+]*[0-9]+)  try {
  adjust(); yylval.e= new realExp(here(), lexical::cast<double>(yytext));
  } catch (lexical::bad_cast&) {
    error();
    *em << "invalid real";
    yylval.e= new realExp(here(), 0); 
  } return LIT;
true               {
  adjust(); yylval.e= new booleanExp(here(), true); return LIT; }
false              {
  adjust(); yylval.e= new booleanExp(here(), false); return LIT; }
null               {
  adjust(); yylval.e= new nullExp(here()); return LIT; }
cycle              {
  adjust(); yylval.e= new cycleExp(here()); return LIT; }
newframe        {
  adjust(); yylval.e= new newPictureExp(here()); return LIT; }

operator           {adjust(); BEGIN opname; }
<opname>{
[ \t\r]            {adjust(); continue;} 
{ENDL}             {adjust(); newline(); continue;}
<<EOF>>            {adjust();
                    setEOF("missing operator name");
                    BEGIN INITIAL;
                    return GARBAGE;
                   }
"**"               { makesymbol(true, "^");
                     BEGIN INITIAL;
                     return ID;
                   }
[-+*/%^!<>]|==|!=|<=|>=|&|\||\^\^|\.\.|::|--|---|\+\+|{EXTRAOPS} {
  makesymbol();
  BEGIN INITIAL;
  return ID;}
{LETTER}({LETTER}|[0-9])* {
  makesymbol(true);
  BEGIN INITIAL;
  return ID; }
.                   {}
}

{LETTER}({LETTER}|[0-9])* {
  makesymbol(false);
  return ID; }  

\/\*               {adjust(); /*commentDepth = 1;*/ BEGIN lexcomment; }
\"                 {adjust(); BEGIN texstring; }
\'                 {adjust(); BEGIN cstring; }

<<EOF>>            { setEOF("unexpected end of input"); yyterminate(); }

.                  {
  adjust(); error(); *em <<"invalid token '" << yytext << "'"; }

