%{
/*****
 * camp.l
 * Andy Hammerlindl 2002/06/14
 *
 * The lexical analyzer of the Asymptote language.
 *****/

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>

#include "stm.h"
#include "errormsg.h"
#include "interact.h"

using namespace absyntax;
using boost::lexical_cast;

#include "camp.tab.h"

#define YY_NO_UNPUT

fileinfo* fi;
int tokPos;
int charPos;
//int commentDepth = 0;

// Needed by the cxx compiler
#ifdef __DECCXX_LIBCXX_RH70
extern "C" int fileno(FILE *);
#endif

std::streambuf *yy_streambuf = 0;
bool my_streambuf = false;
bool interactive;
bool eof;

extern errorstream *em;

extern "C" int yywrap(void)
{
  if(my_streambuf) delete yy_streambuf;
  yy_streambuf=NULL;
  my_streambuf=false;
  charPos=1;
  return 1;
}

bool setlexer(std::string filename)
{
  yywrap();
  
  if(filename == "-") {
    yy_streambuf = std::cin.rdbuf();
    my_streambuf=false;
    interactive = interact::interactive;
  } else {
    std::filebuf *filebuf = new std::filebuf();
    if (!filebuf->open(filename.c_str(),std::ios::in)) {
      delete filebuf;
      return false;
    }
    yy_streambuf = filebuf;
    my_streambuf = true;
    interactive = false;
  }

  eof=false;
  fi = new fileinfo(filename);

  tokPos = charPos = 1;

  return true;
}

size_t yy_input(char *buf, size_t max_size)
{
  size_t count;

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  if(interactive) count=interact::interactive_input(buf,max_size);
  else
#endif
  {
    if(eof) return YY_NULL;
    count=yy_streambuf ? yy_streambuf->sgetn(buf,max_size) : 0;
    if(count < max_size) eof=true;
  }

  return count ? count : YY_NULL;
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

void makesymbol(std::string text = yytext)
{
  adjust();
  yylval.ps.pos=here();
  yylval.ps.sym=symbol::trans(text);
}

void makeselfsymbol(std::string text = yytext)
{
  /* Copy all but the last digit, which should be an =. */
  assert(text[text.length()-1] == '=');
  text.erase(text.length()-1);
  makesymbol(text);
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

std::string stringbuild;

namespace {
void append(char c)
{
  stringbuild.push_back(c);
  yylval.pos = here();
}

void getstring(void)
{
  // NOTE: Replace here() with a position at the start of the string.
  makesymbol(stringbuild);
  std::string().swap(stringbuild);
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

%%

<lexcomment>{
\/\*               {adjust(); /*commentDepth++;*/}
\*\/               {adjust(); /*commentDepth--;*/
                    /*if (commentDepth == 0)*/ BEGIN INITIAL; }
\r\n|\n|\r         {adjust(); newline(); continue; }
<<EOF>>            {adjust();
                    error();
                    *em << "comment not terminated";
                    BEGIN INITIAL;
                    yyterminate();
                   }
.                  {adjust(); continue; }
}

<texstring>{
\"/[ \t\r\n]*[\"\'] {adjust(); BEGIN INITIAL;}
\"                 {adjust(); 
                    BEGIN INITIAL;
                    getstring(); 
                    return STRING; }
<<EOF>>            {adjust();
                    error();
                    *em << "string not terminated";
                    BEGIN INITIAL;
                    yyterminate();
                   }
\r\n|\n|\r         {adjust(); newline(); append('\n'); continue; }
{ESC}\"            {adjust(); append('\"'); continue; }
.                  {adjust(); append(*yytext); }
}

<cstring>{
\'/[ \t\r\n]*[\"\'] {adjust(); BEGIN INITIAL;}
\'                 {adjust(); 
                    BEGIN INITIAL;
                    getstring(); 
                    return STRING; }
<<EOF>>            {adjust();
                    error();
                    *em << "string not terminated";
                    BEGIN INITIAL;
                    yyterminate();
                   }
\r\n|\n|\r         {adjust(); newline(); append('\n'); continue; }
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
\r\n|\n|\r         {adjust(); newline(); continue; }
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
"+"                {makesymbol(); return '+'; }
"-"                {makesymbol(); return '-'; }
"*"                {makesymbol(); return '*'; }
"/"                {makesymbol(); return '/'; }
"%"                {makesymbol(); return '%'; }
"^"                {makesymbol(); return '^'; }
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
"++"               {makesymbol(); return INCR; }
".."               {makesymbol(); return DOTS; }
"--"               {makesymbol(); return DASHES; }

"+="               {makeselfsymbol(); return ADD; }
"-="               {makeselfsymbol(); return SUBTRACT; }
"*="               {makeselfsymbol(); return TIMES; }
"/="               {makeselfsymbol(); return DIVIDE; }
"%="               {makeselfsymbol(); return MOD; }
"^="               {makeselfsymbol(); return EXPONENT; }

and                {adjust(); return AND; }
cycle              {makesymbol(); return CYCLE; }
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
import             {adjust(); return IMPORT; }
static             {adjust(); return STATIC; }
public             {adjust(); return PUBLIC_TOK; }
private            {adjust(); return PRIVATE_TOK; }
this               {adjust(); return THIS; }
explicit           {adjust(); return EXPLICIT; }


void|bool|int|real|string|pair|transform|guide|path|pen|frame {
  makesymbol();
  return PRIM; }

[0-9]+             try {
  adjust(); yylval.e= new intExp(here(), lexical_cast<int>(yytext)); 
  } catch (boost::bad_lexical_cast&) {
    error();
    *em << "integer not valid";
    yylval.e= new intExp(here(), 0); 
  } return LIT;
([0-9]*\.[0-9]+)|([0-9]+\.[0-9]*)|([0-9]*\.*[0-9]+e[-+]*[0-9]+)|([0-9]+\.[0-9]*e[-+]*[0-9]+)  try {
  adjust(); yylval.e= new realExp(here(), lexical_cast<double>(yytext));
  } catch (boost::bad_lexical_cast&) {
    error();
    *em << "real not valid";
    yylval.e= new realExp(here(), 0); 
  } return LIT;
true               {
  adjust(); yylval.e= new booleanExp(here(), true); return LIT; }
false              {
  adjust(); yylval.e= new booleanExp(here(), false); return LIT; }
null               {
  adjust(); yylval.e= new nullExp(here()); return LIT; }
nullframe        {
  adjust(); yylval.e= new nullPictureExp(here()); return LIT; }
nullpath        {
  adjust(); yylval.e= new nullPathExp(here()); return LIT; }

operator           {adjust(); BEGIN opname; }
<opname>{
[ \t\r]            {adjust(); continue;} 
\r\n|\n|\r         {adjust(); newline(); continue;}
<<EOF>>            {adjust();
                    error();
                    *em << "missing operator name";
                    BEGIN INITIAL;
                    yyterminate();
                   }
spec               { makesymbol("<spec>");
                     BEGIN INITIAL;
                     return ID;
                   }
[-+*/%^!<>]|==|!=|<=|>=|&&|\|\||\^\^|\.\.|--|cycle|curl|tension|atleast|controls {
  makesymbol();
  BEGIN INITIAL;
  return ID;}
.                   {}
}

{LETTER}({LETTER}|[0-9])* {
  makesymbol();
  return ID; }  

\/\*               {adjust(); /*commentDepth = 1;*/ BEGIN lexcomment; }
\"                 {adjust(); BEGIN texstring; }
\'                 {adjust(); BEGIN cstring; }

.                  {
  adjust(); error(); *em <<"invalid token '" << yytext << "'"; }

