%{
/*****
 * camp.y
 * Andy Hammerlindl 08/12/2002
 *
 * The grammar of the camp language.
 *****/

#include "errormsg.h"
#include "exp.h"
#include "newexp.h"
#include "dec.h"
#include "fundec.h"
#include "stm.h"
#include "modifier.h"
#include "opsymbols.h"

// Avoid error messages with unpatched bison-1.875:
#ifndef __attribute__
#define __attribute__(x)
#endif

// Used when a position needs to be determined and no token is
// available.  Defined in camp.l.
position lexerPos();

bool lexerEOF();

int yylex(void); /* function prototype */

void yyerror(const char *s)
{
  if (!lexerEOF()) {
    em.error(lexerPos());
    em << s;
    em.cont();
  }
}

// Check if the symbol given is "keyword".  Returns true in this case and
// returns false and reports an error otherwise.
bool checkKeyword(position pos, symbol sym)
{
  if (sym != symbol::trans("keyword")) {
    em.error(pos);
    em << "expected 'keyword' here";

    return false;
  }
  return true;
}

namespace absyntax { file *root; }

using namespace absyntax;
using sym::symbol;
using mem::string;
%}

%union {
  position pos;
  bool boo;
  struct {
    position pos;
    sym::symbol sym;
  } ps;
  absyntax::name *n;
  absyntax::varinit *vi;
  absyntax::arrayinit *ai;
  absyntax::exp *e;
  absyntax::stringExp *stre;
  absyntax::specExp *se;
  absyntax::joinExp *j;
  absyntax::explist *elist;
  absyntax::argument arg;
  absyntax::arglist *alist;
  absyntax::slice *slice;
  absyntax::dimensions *dim;
  absyntax::ty  *t;
  absyntax::decid *di;
  absyntax::decidlist *dil;
  absyntax::decidstart *dis;
  absyntax::runnable *run;
  struct {
    position pos;
    trans::permission val;
  } perm;
  struct {
    position pos;
    trans::modifier val;
  } mod;
  absyntax::modifierList *ml;
  //absyntax::program *prog;
  absyntax::vardec *vd;
  //absyntax::vardecs *vds;
  absyntax::dec *d;
  absyntax::idpair *ip;
  absyntax::idpairlist *ipl;
  absyntax::stm *s;
  absyntax::block *b;
  absyntax::stmExpList *sel;
  //absyntax::funheader *fh;
  absyntax::formal *fl;
  absyntax::formals *fls;
}  

%token <ps> ID SELFOP
            DOTS COLONS DASHES INCR LONGDASH
            CONTROLS TENSION ATLEAST CURL
            COR CAND BAR AMPERSAND EQ NEQ LT LE GT GE CARETS
            '+' '-' '*' '/' '%' '^' LOGNOT OPERATOR
%token <pos> LOOSE ASSIGN '?' ':'
             DIRTAG JOIN_PREC AND
             '{' '}' '(' ')' '.' ','  '[' ']' ';' ELLIPSIS
             ACCESS UNRAVEL IMPORT INCLUDE FROM QUOTE STRUCT TYPEDEF NEW
             IF ELSE WHILE DO FOR BREAK CONTINUE RETURN_
             THIS EXPLICIT
             GARBAGE
%token <e>   LIT
%token <stre> STRING
%token <perm> PERM
%token <mod> MODIFIER

%right ASSIGN SELFOP
%right '?' ':'
%left  COR
%left  CAND
%left  BAR
%left  AMPERSAND
%left  EQ NEQ
%left  LT LE GT GE
%left  OPERATOR

%left  CARETS
%left  JOIN_PREC DOTS COLONS DASHES INCR LONGDASH
%left  DIRTAG CONTROLS TENSION ATLEAST AND
%left  CURL '{' '}'

%left  '+' '-' 
%left  '*' '/' '%' LIT
%left  UNARY
%right '^'
%left  LOGNOT
%left  EXP_IN_PARENS_RULE
%left  '(' ')'

%type  <b>   fileblock bareblock block
%type  <n>   name
%type  <run> runnable
%type  <ml>  modifiers
%type  <d>   dec fundec typedec
%type  <ps>  strid
%type  <ip>  idpair stridpair
%type  <ipl> idpairlist stridpairlist
%type  <vd>  vardec barevardec 
%type  <t>   type celltype
%type  <dim> dims
%type  <dil> decidlist
%type  <di>  decid
%type  <dis> decidstart
%type  <vi>  varinit
%type  <ai>  arrayinit basearrayinit varinits
%type  <fl>  formal
%type  <fls> formals
%type  <e>   value exp fortest
%type  <arg> argument
%type  <slice> slice
%type  <j>   join basicjoin
%type  <e>   tension controls
%type  <se>  dir
%type  <elist> dimexps
%type  <alist> arglist tuple
%type  <s>   stm stmexp blockstm
%type  <run> forinit
%type  <sel> forupdate stmexplist
%type  <boo> explicitornot

/* There are four shift/reduce conflicts:
 *   the dangling ELSE in IF (exp) IF (exp) stm ELSE stm
 *   new ID
 *   the argument id=exp is taken as an argument instead of an assignExp
 *   explicit cast
 */
%expect 4

/* Enable grammar debugging. */
/*%debug*/

%%

file:
  fileblock        { absyntax::root = $1; }
;

fileblock:
  /* empty */      { $$ = new file(lexerPos(), false); }
| fileblock runnable
                   { $$ = $1; $$->add($2); }
;

bareblock:
  /* empty */      { $$ = new block(lexerPos(), true); }
| bareblock runnable
                   { $$ = $1; $$->add($2); }
;

name:
  ID               { $$ = new simpleName($1.pos, $1.sym); }
| name '.' ID      { $$ = new qualifiedName($2, $1, $3.sym); }
| '%'              { $$ = new simpleName($1.pos,
                                  symbol::trans("operator answer")); }
;

runnable:
  dec              { $$ = $1; }
| stm              { $$ = $1; }
| modifiers dec
                   { $$ = new modifiedRunnable($1->getPos(), $1, $2); }
| modifiers stm
                   { $$ = new modifiedRunnable($1->getPos(), $1, $2); }
;

modifiers:
  MODIFIER         { $$ = new modifierList($1.pos); $$->add($1.val); }
| PERM             { $$ = new modifierList($1.pos); $$->add($1.val); }
| modifiers MODIFIER
                   { $$ = $1; $$->add($2.val); }
| modifiers PERM
                   { $$ = $1; $$->add($2.val); }
;

dec:
  vardec           { $$ = $1; }
| fundec           { $$ = $1; }
| typedec          { $$ = $1; }
| ACCESS stridpairlist ';'
                   { $$ = new accessdec($1, $2); }
| FROM name UNRAVEL idpairlist ';'
                   { $$ = new unraveldec($1, $2, $4); }
| FROM name UNRAVEL '*' ';'
                   { $$ = new unraveldec($1, $2, WILDCARD); }
| UNRAVEL name ';' { $$ = new unraveldec($1, $2, WILDCARD); }
| FROM strid ACCESS idpairlist ';'
                   { $$ = new fromaccessdec($1, $2.sym, $4); }
| FROM strid ACCESS '*' ';'
                   { $$ = new fromaccessdec($1, $2.sym, WILDCARD); }
| IMPORT stridpair ';'
                   { $$ = new importdec($1, $2); }
| INCLUDE ID ';'   { $$ = new includedec($1, $2.sym); }                   
| INCLUDE STRING ';'
                   { $$ = new includedec($1, $2->getString()); }
;

idpair:
  ID               { $$ = new idpair($1.pos, $1.sym); }
/* ID 'as' ID */
| ID ID ID         { $$ = new idpair($1.pos, $1.sym, $2.sym , $3.sym); }
;

idpairlist:
  idpair           { $$ = new idpairlist(); $$->add($1); }
| idpairlist ',' idpair
                   { $$ = $1; $$->add($3); }
;

strid:
  ID               { $$ = $1; }
| STRING           { $$.pos = $1->getPos();
                     $$.sym = symbol::literalTrans($1->getString()); }
;

stridpair:
  ID               { $$ = new idpair($1.pos, $1.sym); }
/* strid 'as' ID */
| strid ID ID      { $$ = new idpair($1.pos, $1.sym, $2.sym , $3.sym); }
;

stridpairlist:
  stridpair        { $$ = new idpairlist(); $$->add($1); }
| stridpairlist ',' stridpair
                   { $$ = $1; $$->add($3); }
;

vardec:
  barevardec ';'   { $$ = $1; }
;

barevardec:
  type decidlist   { $$ = new vardec($1->getPos(), $1, $2); }
;

type:
  celltype         { $$ = $1; }
| name dims        { $$ = new arrayTy($1, $2); }
;

celltype:
  name             { $$ = new nameTy($1); }
;

dims:
 '[' ']'           { $$ = new dimensions($1); }
| dims '[' ']'     { $$ = $1; $$->increase(); }
;

dimexps:
  '[' exp ']'      { $$ = new explist($1); $$->add($2); }
| dimexps '[' exp ']'
                   { $$ = $1; $$->add($3); }
;

decidlist:
  decid            { $$ = new decidlist($1->getPos()); $$->add($1); }
| decidlist ',' decid
                   { $$ = $1; $$->add($3); }
;

decid:
  decidstart       { $$ = new decid($1->getPos(), $1); }
| decidstart ASSIGN varinit
                   { $$ = new decid($1->getPos(), $1, $3); }
;

decidstart:
  ID               { $$ = new decidstart($1.pos, $1.sym); }
| ID dims          { $$ = new decidstart($1.pos, $1.sym, $2); }
| ID '(' ')'       { $$ = new fundecidstart($1.pos, $1.sym, 0,
                                            new formals($2)); }
| ID '(' formals ')'
                   { $$ = new fundecidstart($1.pos, $1.sym, 0, $3); }
;

varinit:
  exp              { $$ = $1; }
| arrayinit        { $$ = $1; }
;

block:
  '{' bareblock '}'
                   { $$ = $2; }
;

arrayinit:
  '{' '}'          { $$ = new arrayinit($1); }
| '{' ELLIPSIS varinit '}'
                   { $$ = new arrayinit($1); $$->addRest($3); }
| '{' basearrayinit '}'
                   { $$ = $2; }
| '{' basearrayinit ELLIPSIS varinit '}'
                   { $$ = $2; $$->addRest($4); }
;

basearrayinit:
  ','              { $$ = new arrayinit($1); }
| varinits         { $$ = $1; }
| varinits ','     { $$ = $1; }
;

varinits:
  varinit          { $$ = new arrayinit($1->getPos());
		     $$->add($1);}
| varinits ',' varinit
                   { $$ = $1; $$->add($3); }
;

formals:
  formal           { $$ = new formals($1->getPos()); $$->add($1); }
| ELLIPSIS formal  { $$ = new formals($1); $$->addRest($2); }
| formals ',' formal
                   { $$ = $1; $$->add($3); }
| formals ELLIPSIS formal
                   { $$ = $1; $$->addRest($3); }
;

explicitornot:
  EXPLICIT         { $$ = true; }
|                  { $$ = false; }
;

formal:
  explicitornot type
                   { $$ = new formal($2->getPos(), $2, 0, 0, $1, 0); }
| explicitornot type decidstart
                   { $$ = new formal($2->getPos(), $2, $3, 0, $1, 0); }
| explicitornot type decidstart ASSIGN varinit
                   { $$ = new formal($2->getPos(), $2, $3, $5, $1, 0); }
/* The uses of ID below are 'keyword' qualifiers before the parameter name. */
| explicitornot type ID decidstart
                   { bool k = checkKeyword($3.pos, $3.sym);
                     $$ = new formal($2->getPos(), $2, $4, 0, $1, k); }
| explicitornot type ID decidstart ASSIGN varinit
                   { bool k = checkKeyword($3.pos, $3.sym);
                     $$ = new formal($2->getPos(), $2, $4, $6, $1, k); }
;

fundec:
  type ID '(' ')' blockstm
                   { $$ = new fundec($3, $1, $2.sym, new formals($3), $5); }
| type ID '(' formals ')' blockstm
                   { $$ = new fundec($3, $1, $2.sym, $4, $6); }
;

typedec:
  STRUCT ID block  { $$ = new recorddec($1, $2.sym, $3); }
| TYPEDEF vardec   { $$ = new typedec($1, $2); }
;

slice:
  ':'              { $$ = new slice($1, 0, 0); }
| exp ':'          { $$ = new slice($2, $1, 0); }
| ':' exp          { $$ = new slice($1, 0, $2); }
| exp ':' exp      { $$ = new slice($2, $1, $3); }
;

value:
  value '.' ID     { $$ = new fieldExp($2, $1, $3.sym); } 
| name '[' exp ']' { $$ = new subscriptExp($2,
                              new nameExp($1->getPos(), $1), $3); }
| value '[' exp ']'{ $$ = new subscriptExp($2, $1, $3); }
| name '[' slice ']' { $$ = new sliceExp($2,
                              new nameExp($1->getPos(), $1), $3); }
| value '[' slice ']'{ $$ = new sliceExp($2, $1, $3); }
| name '(' ')'     { $$ = new callExp($2,
                                      new nameExp($1->getPos(), $1),
                                      new arglist()); } 
| name '(' arglist ')'
                   { $$ = new callExp($2, 
                                      new nameExp($1->getPos(), $1),
                                      $3); }
| value '(' ')'    { $$ = new callExp($2, $1, new arglist()); }
| value '(' arglist ')'
                   { $$ = new callExp($2, $1, $3); }
| '(' exp ')' %prec EXP_IN_PARENS_RULE
                   { $$ = $2; }
| '(' name ')' %prec EXP_IN_PARENS_RULE
                   { $$ = new nameExp($2->getPos(), $2); }
| THIS             { $$ = new thisExp($1); }
;

argument:
  exp              { $$.name = symbol::nullsym; $$.val=$1; }
| ID ASSIGN exp    { $$.name = $1.sym; $$.val=$3; }
;

arglist:
  argument         { $$ = new arglist(); $$->add($1); }
| ELLIPSIS argument
                   { $$ = new arglist(); $$->addRest($2); }
| arglist ',' argument
                   { $$ = $1; $$->add($3); }
| arglist ELLIPSIS argument
                   { $$ = $1; $$->addRest($3); }
;

/* A list of two or more expressions, separated by commas. */
tuple:
  exp ',' exp      { $$ = new arglist(); $$->add($1); $$->add($3); }
| tuple ',' exp    { $$ = $1; $$->add($3); }
;

exp:
  name             { $$ = new nameExp($1->getPos(), $1); }
| value            { $$ = $1; }
| LIT              { $$ = $1; }
| STRING           { $$ = $1; }
/* This is for scaling expressions such as 105cm */
| LIT exp          { $$ = new scaleExp($1->getPos(), $1, $2); }
| '(' name ')' exp
                   { $$ = new castExp($2->getPos(), new nameTy($2), $4); }
| '(' name dims ')' exp
                   { $$ = new castExp($2->getPos(), new arrayTy($2, $3), $5); }
| '+' exp %prec UNARY
                   { $$ = new unaryExp($1.pos, $2, $1.sym); }
| '-' exp %prec UNARY
                   { $$ = new unaryExp($1.pos, $2, $1.sym); }
| LOGNOT exp       { $$ = new unaryExp($1.pos, $2, $1.sym); }
| exp '+' exp      { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp '-' exp      { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp '*' exp      { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp '/' exp      { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp '%' exp      { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp '^' exp      { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp LT exp       { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp LE exp       { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp GT exp       { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp GE exp       { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp EQ exp       { $$ = new equalityExp($2.pos, $1, $2.sym, $3); }
| exp NEQ exp      { $$ = new equalityExp($2.pos, $1, $2.sym, $3); }
| exp CAND exp     { $$ = new andExp($2.pos, $1, $2.sym, $3); }
| exp COR exp      { $$ = new orExp($2.pos, $1, $2.sym, $3); }
| exp CARETS exp   { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp AMPERSAND exp{ $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp BAR       exp{ $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp OPERATOR exp { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| exp INCR exp     { $$ = new binaryExp($2.pos, $1, $2.sym, $3); }
| NEW celltype
                   { $$ = new newRecordExp($1, $2); }
| NEW celltype dimexps
                   { $$ = new newArrayExp($1, $2, $3, 0, 0); }
| NEW celltype dimexps dims
                   { $$ = new newArrayExp($1, $2, $3, $4, 0); }
| NEW celltype dims
                   { $$ = new newArrayExp($1, $2, 0, $3, 0); }
| NEW celltype dims arrayinit
                   { $$ = new newArrayExp($1, $2, 0, $3, $4); }
| NEW celltype '(' ')' blockstm
                   { $$ = new newFunctionExp($1, $2, new formals($3), $5); }
| NEW celltype dims '(' ')' blockstm
                   { $$ = new newFunctionExp($1,
                                             new arrayTy($2->getPos(), $2, $3),
                                             new formals($4),
                                             $6); }
| NEW celltype '(' formals ')' blockstm
                   { $$ = new newFunctionExp($1, $2, $4, $6); }
| NEW celltype dims '(' formals ')' blockstm
                   { $$ = new newFunctionExp($1,
                                             new arrayTy($2->getPos(), $2, $3),
                                             $5,
                                             $7); }
| exp '?' exp ':' exp
                   { $$ = new conditionalExp($2, $1, $3, $5); }
| exp ASSIGN exp   { $$ = new assignExp($2, $1, $3); }
| '(' tuple ')'    { $$ = new callExp($1, new nameExp($1, SYM_TUPLE), $2); }
| exp join exp %prec JOIN_PREC 
                   { $2->pushFront($1); $2->pushBack($3); $$ = $2; }
| exp dir %prec DIRTAG
                   { $2->setSide(camp::OUT);
                     joinExp *jexp =
                         new joinExp($2->getPos(), SYM_DOTS);
                     $$=jexp;
                     jexp->pushBack($1); jexp->pushBack($2); }
| INCR exp %prec UNARY
                   { $$ = new prefixExp($1.pos, $2, SYM_PLUS); }
| DASHES exp %prec UNARY
                   { $$ = new prefixExp($1.pos, $2, SYM_MINUS); }
/* Illegal - will be caught during translation. */
| exp INCR %prec UNARY 
                   { $$ = new postfixExp($2.pos, $1, SYM_PLUS); }
| exp SELFOP exp   { $$ = new selfExp($2.pos, $1, $2.sym, $3); }
| QUOTE '{' fileblock '}'
                   { $$ = new quoteExp($1, $3); }
;

// This verbose definition is because leaving empty as an expansion for dir
// made a whack of reduce/reduce errors.
join:
  DASHES           { $$ = new joinExp($1.pos,$1.sym); }
| basicjoin %prec JOIN_PREC 
                   { $$ = $1; }
| dir basicjoin %prec JOIN_PREC
                   { $1->setSide(camp::OUT);
                     $$ = $2; $$->pushFront($1); }
| basicjoin dir %prec JOIN_PREC 
                   { $2->setSide(camp::IN);
                     $$ = $1; $$->pushBack($2); }
| dir basicjoin dir %prec JOIN_PREC
                   { $1->setSide(camp::OUT); $3->setSide(camp::IN);
                     $$ = $2; $$->pushFront($1); $$->pushBack($3); }
;

dir:
  '{' CURL exp '}' { $$ = new specExp($2.pos, $2.sym, $3); }
| '{' exp '}'      { $$ = new specExp($1, symbol::opTrans("spec"), $2); }
| '{' exp ',' exp '}'
                   { $$ = new specExp($1, symbol::opTrans("spec"),
				      new pairExp($3, $2, $4)); }
| '{' exp ',' exp ',' exp '}'
                   { $$ = new specExp($1, symbol::opTrans("spec"),
				      new tripleExp($3, $2, $4, $6)); }
;

basicjoin:
  DOTS             { $$ = new joinExp($1.pos, $1.sym); }
| DOTS tension DOTS
                   { $$ = new joinExp($1.pos, $1.sym); $$->pushBack($2); }
| DOTS controls DOTS
                   { $$ = new joinExp($1.pos, $1.sym); $$->pushBack($2); }
| COLONS           { $$ = new joinExp($1.pos, $1.sym); }
| LONGDASH         { $$ = new joinExp($1.pos, $1.sym); }
;

tension:
  TENSION exp      { $$ = new binaryExp($1.pos, $2, $1.sym,
                              new booleanExp($1.pos, false)); }
| TENSION exp AND exp
                   { $$ = new ternaryExp($1.pos, $2, $1.sym, $4,
                              new booleanExp($1.pos, false)); }
| TENSION ATLEAST exp 
                   { $$ = new binaryExp($1.pos, $3, $1.sym,
                              new booleanExp($2.pos, true)); }
| TENSION ATLEAST exp AND exp
                   { $$ = new ternaryExp($1.pos, $3, $1.sym, $5,
                              new booleanExp($2.pos, true)); }
;

controls:
  CONTROLS exp     { $$ = new unaryExp($1.pos, $2, $1.sym); }
| CONTROLS exp AND exp
                   { $$ = new binaryExp($1.pos, $2, $1.sym, $4); }
;

stm:
  ';'              { $$ = new emptyStm($1); }
| blockstm         { $$ = $1; }
| stmexp ';'       { $$ = $1; }
| IF '(' exp ')' stm
                   { $$ = new ifStm($1, $3, $5); }
| IF '(' exp ')' stm ELSE stm
                   { $$ = new ifStm($1, $3, $5, $7); }
| WHILE '(' exp ')' stm
                   { $$ = new whileStm($1, $3, $5); }
| DO stm WHILE '(' exp ')' ';'
                   { $$ = new doStm($1, $2, $5); }
| FOR '(' forinit ';' fortest ';' forupdate ')' stm
                   { $$ = new forStm($1, $3, $5, $7, $9); }
| FOR '(' type ID ':' exp ')' stm
                   { $$ = new extendedForStm($1, $3, $4.sym, $6, $8); }
| BREAK ';'        { $$ = new breakStm($1); }
| CONTINUE ';'     { $$ = new continueStm($1); }
| RETURN_ ';'       { $$ = new returnStm($1); }
| RETURN_ exp ';'   { $$ = new returnStm($1, $2); }
;

stmexp:
  exp              { $$ = new expStm($1->getPos(), $1); }
;

blockstm:
  block            { $$ = new blockStm($1->getPos(), $1); }
;

forinit:
  /* empty */      { $$ = 0; }
| stmexplist       { $$ = $1; }
| barevardec       { $$ = $1; }
;

fortest:
  /* empty */      { $$ = 0; }
| exp              { $$ = $1; }
;

forupdate:
  /* empty */      { $$ = 0; }
| stmexplist       { $$ = $1; }
;

stmexplist:
  stmexp           { $$ = new stmExpList($1->getPos()); $$->add($1); }
| stmexplist ',' stmexp
                   { $$ = $1; $$->add($3); }
;
