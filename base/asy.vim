" Vim syntax file
" Language:	Asymptote
" Maintainer:	Andy Hammerlindl
" Last Change:	2005 Aug 23

" Hacked together from Bram Moolenaar's C syntax file, and Claudio Fleiner's
" Java syntax file.

" For version 5.x: Clear all syntax items
" For version 6.x: Quit when a syntax file was already loaded
if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

" A bunch of useful C keywords
syn keyword	asyStatement	break return continue unravel
syn keyword	asyConditional	if else
syn keyword	asyRepeat	while for do
syn keyword     asyExternal     access from import include
syn keyword     asyOperator     new operator

syn keyword	asyTodo		contained TODO FIXME XXX

" asyCommentGroup allows adding matches for special things in comments
syn cluster	asyCommentGroup	contains=asyTodo

" String and Character constants
" Highlight special characters (those proceding a double backslash) differently
syn match	asySpecial	display contained "\\\\."
" Highlight line continuation slashes
syn match	asySpecial	display contained "\\$"
syn region	asyString	start=+"+ skip=+\\\\\|\\"+ end=+"+ contains=asySpecial
  " asyCppString: same as asyString, but ends at end of line
if 0
syn region	asyCppString	start=+"+ skip=+\\\\\|\\"\|\\$+ excludenl end=+"+ end='$' contains=asySpecial
endif

"when wanted, highlight trailing white space
if exists("asy_space_errors")
  if !exists("asy_no_trail_space_error")
    syn match	asySpaceError	display excludenl "\s\+$"
  endif
  if !exists("asy_no_tab_space_error")
    syn match	asySpaceError	display " \+\t"me=e-1
  endif
endif

"catch errors caused by wrong parenthesis and brackets
syn cluster	asyParenGroup	contains=asyParenError,asyIncluded,asySpecial,asyCommentSkip,asyCommentString,asyComment2String,@asyCommentGroup,asyCommentStartError,asyUserCont,asyUserLabel,asyBitField,asyCommentSkip,asyOctalZero,asyCppOut,asyCppOut2,asyCppSkip,asyFormat,asyNumber,asyFloat,asyOctal,asyOctalError,asyNumbersCom
if exists("asy_no_bracket_error")
  syn region	asyParen		transparent start='(' end=')' contains=ALLBUT,@asyParenGroup,asyCppParen,asyCppString
  " asyCppParen: same as asyParen but ends at end-of-line; used in asyDefine
  syn region	asyCppParen	transparent start='(' skip='\\$' excludenl end=')' end='$' contained contains=ALLBUT,@asyParenGroup,asyParen,asyString
  syn match	asyParenError	display ")"
  syn match	asyErrInParen	display contained "[{}]"
else
  syn region	asyParen	transparent start='(' end=')' contains=ALLBUT,@asyParenGroup,asyCppParen,asyErrInBracket,asyCppBracket,asyCppString
  " asyCppParen: same as asyParen but ends at end-of-line; used in asyDefine
  syn region	asyCppParen	transparent start='(' skip='\\$' excludenl end=')' end='$' contained contains=ALLBUT,@asyParenGroup,asyErrInBracket,asyParen,asyBracket,asyString
if 0
  syn match	asyParenError	display "[\])]"
  syn match	asyErrInParen	display contained "[\]]"
endif
  syn region	asyBracket	transparent start='\[' end=']' contains=ALLBUT,@asyParenGroup,asyErrInParen,asyCppParen,asyCppBracket,asyCppString
  " asyCppBracket: same as asyParen but ends at end-of-line; used in asyDefine
  syn region	asyCppBracket	transparent start='\[' skip='\\$' excludenl end=']' end='$' contained contains=ALLBUT,@asyParenGroup,asyErrInParen,asyParen,asyBracket,asyString
  syn match	asyErrInBracket	display contained "[);]"
endif

"integer number, or floating point number without a dot and with "f".
syn case ignore
syn match	asyNumbers	display transparent "\<\d\|\.\d" contains=asyNumber,asyFloat
syn match       asyNumber       display contained "\d\+"
"floating point number, with dot, optional exponent
syn match	asyFloat	display contained "\d\+\.\d*\(e[-+]\=\d\+\)\="
"floating point number, starting with a dot, optional exponent
syn match	asyFloat	display contained "\.\d\+\(e[-+]\=\d\+\)\="
"floating point number, without dot, with exponent
syn match	asyFloat	display contained "\d\+e[-+]\=\d\+"
syn case match

if exists("asy_comment_strings")
  " A comment can contain asyString, asyCharacter and asyNumber.
  " But a "*/" inside a asyString in a asyComment DOES end the comment!  So we
  " need to use a special type of asyString: asyCommentString, which also ends on
  " "*/", and sees a "*" at the start of the line as comment again.
  " Unfortunately this doesn't very well work for // type of comments :-(
  syntax match	asyCommentSkip	contained "^\s*\*\($\|\s\+\)"
  syntax region asyCommentString	contained start=+L\="+ skip=+\\\\\|\\"+ end=+"+ end=+\*/+me=s-1 contains=asySpecial,asyCommentSkip
  syntax region asyComment2String	contained start=+L\="+ skip=+\\\\\|\\"+ end=+"+ end="$" contains=asySpecial
  syntax region  asyCommentL	start="//" skip="\\$" end="$" keepend contains=@asyCommentGroup,asyComment2String,asyCharacter,asyNumbersCom,asySpaceError
  syntax region asyComment	matchgroup=asyCommentStart start="/\*" matchgroup=NONE end="\*/" contains=@asyCommentGroup,asyCommentStartError,asyCommentString,asyCharacter,asyNumbersCom,asySpaceError
else
  syn region	asyCommentL	start="//" skip="\\$" end="$" keepend contains=@asyCommentGroup,asySpaceError
  syn region	asyComment	matchgroup=asyCommentStart start="/\*" matchgroup=NONE end="\*/" contains=@asyCommentGroup,asyCommentStartError,asySpaceError
endif
" keep a // comment separately, it terminates a preproc. conditional
syntax match	asyCommentError	display "\*/"
syntax match	asyCommentStartError display "/\*"me=e-1 contained

syn keyword	asyType		void bool int real string
syn keyword	asyType		pair triple transform guide path pen frame
syn keyword     asyType         picture

syn keyword	asyStructure	struct typedef
syn keyword     asyStorageClass static public readable private explicit

syn keyword     asyPathSpec     and cycle controls tension atleast curl

syn keyword     asyConstant     true false
syn keyword     asyConstant     null nullframe nullpath

if exists("asy_syn_plain")
  syn keyword	asyConstant	currentpicture currentpen currentprojection
  syn keyword	asyConstant	inch inches cm mm bp pt up down right left 
  syn keyword	asyConstant	E NE N NW W SW S SE
  syn keyword	asyConstant	ENE NNE NNW WNW WSW SSW SSE ESE
  syn keyword	asyConstant	I pi twopi
  syn keyword	asyConstant	solid dotted dashed dashdotted
  syn keyword	asyConstant	longdashed longdashdotted
  syn keyword	asyConstant	squarecap roundcap extendcap
  syn keyword	asyConstant	miterjoin roundjoin beveljoin
  syn keyword	asyConstant	zerowinding evenodd
  syn keyword	asyConstant	invisible black gray grey white
  syn keyword	asyConstant	lightgray lightgrey
  syn keyword	asyConstant	red green blue
  syn keyword	asyConstant	cmyk Cyan Magenta Yellow Black
  syn keyword	asyConstant	yellow magenta cyan
  syn keyword	asyConstant	brown darkgreen darkblue
  syn keyword	asyConstant	orange purple royalblue olive
  syn keyword	asyConstant	chartreuse fuchsia salmon lightblue springgreen
  syn keyword	asyConstant	pink
endif

syn sync ccomment asyComment minlines=15

" Define the default highlighting.
" For version 5.7 and earlier: only when not done already
" For version 5.8 and later: only when an item doesn't have highlighting yet
if version >= 508 || !exists("did_asy_syn_inits")
  if version < 508
    let did_asy_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink asyFormat		asySpecial
  HiLink asyCppString		asyString
  HiLink asyCommentL		asyComment
  HiLink asyCommentStart		asyComment
  HiLink asyLabel			Label
  HiLink asyUserLabel		Label
  HiLink asyConditional		Conditional
  HiLink asyRepeat		Repeat
  HiLink asyCharacter		Character
  HiLink asySpecialCharacter	asySpecial
  HiLink asyNumber		Number
  HiLink asyOctal			Number
  HiLink asyOctalZero		PreProc	 " link this to Error if you want
  HiLink asyFloat			Float
  HiLink asyOctalError		asyError
  HiLink asyParenError		asyError
  HiLink asyErrInParen		asyError
  HiLink asyErrInBracket		asyError
  HiLink asyCommentError		asyError
  HiLink asyCommentStartError	asyError
  HiLink asySpaceError		asyError
  HiLink asySpecialError		asyError
  HiLink asyOperator		Operator
  HiLink asyStructure		Structure
  HiLink asyStorageClass		StorageClass
  HiLink asyExternal		Include
  HiLink asyPreProc		PreProc
  HiLink asyDefine		Macro
  HiLink asyIncluded		asyString
  HiLink asyError			Error
  HiLink asyStatement		Statement
  HiLink asyPreCondit		PreCondit
  HiLink asyType			Type
  HiLink asyConstant		Constant
  HiLink asyCommentString		asyString
  HiLink asyComment2String	asyString
  HiLink asyCommentSkip		asyComment
  HiLink asyString		String
  HiLink asyComment		Comment
  HiLink asySpecial		SpecialChar
  HiLink asyTodo			Todo
  HiLink asyCppSkip		asyCppOut
  HiLink asyCppOut2		asyCppOut
  HiLink asyCppOut		Comment
  HiLink asyPathSpec		Statement
		

  delcommand HiLink
endif

let b:current_syntax = "c"

" vim: ts=8
