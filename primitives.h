/*****
 * primitives.h
 * Andy Hammerlindl 2007/04/27
 *
 * A list of the primitive types in Asymptote, defined using the
 * PRIMITIVE(name,Name,asyName) macro.  This macro should be defined in by the
 * code including this file for the context at hand.
 *
 * name - the name of the type in C++ code ex: boolean
 * Name - the same name capitalized        ex: Boolean
 * asyName - the name in Asymptote code    ex: bool
 *
 *****/

// No ifndef because this file may be included multiple times in different
// contexts.

// How to use this header:
// 1. define a macro PRIMITIVE(name,Name,asyName) in the file one wants to
//    include this header to
// 2. If one wants to include error type, define also PRIMERROR
// 3. If one is working on cases where there may be conflicting names, define
//    EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE to exclude them
// 4. If one wants to manually define the primitives, define
//    PRIMITIVES_MACRO_ONLY. A DEFINE_PRIMTIVES macro is required for
//    definition in this case.
// 5. Include this file
// 6. If PRIMITIVES_MACRO_ONLY has been define, the macro DEFINE_PRIMTIVES
//    will expands to PRIMITIVE(...) of all primitive types
// 7. Optionally undefine PRIMITIVE and all the relating macros

/* null is not a primitive type. */

#undef DEFINE_PRIMERROR
#undef DEFINE_PRIMITIVES_POTENTIALLY_CONFLICTING
#undef DEFINE_PRIMTIVES

#ifdef PRIMERROR
#define DEFINE_PRIMERROR PRIMITIVE(error,Error,<error>)
#else
#define DEFINE_PRIMERROR
#endif

// When used as an enum symbol, these types may conflict with existing
// declarations, hence we provide the option to exclude them. These cases
// may have to be dealt with separately
#ifndef EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#define DEFINE_PRIMITIVES_POTENTIALLY_CONFLICTING \
  PRIMITIVE(Int,Int,int) \
  PRIMITIVE(string,String,string)
#else
#define DEFINE_PRIMITIVES_POTENTIALLY_CONFLICTING
#endif

#define DEFINE_PRIMTIVES \
  PRIMITIVE(void,Void,void) \
  PRIMITIVE(inferred,Inferred,var) \
  PRIMITIVE(boolean,Boolean,bool) \
  PRIMITIVE(real,Real,real) \
  PRIMITIVE(pair,Pair,pair) \
  PRIMITIVE(triple,Triple,triple) \
  PRIMITIVE(transform,Transform,transform) \
  PRIMITIVE(guide,Guide,guide) \
  PRIMITIVE(path,Path,path) \
  PRIMITIVE(path3,Path3,path3) \
  PRIMITIVE(cycleToken,CycleToken,cycleToken) \
  PRIMITIVE(tensionSpecifier,TensionSpecifier,tensionSpecifier) \
  PRIMITIVE(curlSpecifier,CurlSpecifier,curlSpecifier) \
  PRIMITIVE(pen,Pen,pen) \
  PRIMITIVE(picture,Picture,frame) \
  PRIMITIVE(file,File,file) \
  PRIMITIVE(code,Code,code) \
  DEFINE_PRIMERROR DEFINE_PRIMITIVES_POTENTIALLY_CONFLICTING

// this allows the macros to be expanded in certain editors with macro
// expansion feature, allowing for better readability of the code

#ifndef PRIMITIVES_MACRO_ONLY
DEFINE_PRIMTIVES
#endif
