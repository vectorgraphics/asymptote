/*****
 * builtin.cc
 * Tom Prince 2004/08/25
 *
 * Initialize builtins.
 *****/

#include <cmath>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "builtin.h"
#include "entry.h"
#include "import.h"
#include "runtime.h"
#include "cast.h"
#include "types.h"
#include "pair.h"
#include "arrayop.h"
#include "pow.h"

using namespace types;
using namespace camp;

namespace trans {
using camp::pair;
using camp::transform;
using vm::stack;

// The base environments for built-in types and functions
void base_tenv(tenv &ret)
{
  ret.enter(symbol::trans("void"), primVoid());
  ret.enter(symbol::trans("bool"), primBoolean());
  ret.enter(symbol::trans("int"), primInt());
  ret.enter(symbol::trans("real"), primReal());
  ret.enter(symbol::trans("string"), primString());
  
  ret.enter(symbol::trans("pair"), primPair());
  ret.enter(symbol::trans("transform"), primTransform());
  ret.enter(symbol::trans("guide"), primGuide());
  ret.enter(symbol::trans("path"), primPath());
  ret.enter(symbol::trans("pen"), primPen());
  ret.enter(symbol::trans("frame"), primPicture());

  ret.enter(symbol::trans("file"), primFile());
}

// Macro to make a function.
inline void addFunc(venv &ve, access *a, ty *result, const char *name, 
		    ty *t1 = 0, ty *t2 = 0, ty *t3 = 0, ty* t4 = 0,
		    ty *t5 = 0, ty *t6 = 0, ty *t7 = 0, ty *t8 = 0)
{
  function *fun = new function(result);

  if (t1) fun->add(t1);
  if (t2) fun->add(t2);
  if (t3) fun->add(t3);
  if (t4) fun->add(t4);
  if (t5) fun->add(t5);
  if (t6) fun->add(t6);
  if (t7) fun->add(t7);
  if (t8) fun->add(t8);

  varEntry *ent = new varEntry(fun, a);

  ve.enter(symbol::trans(name), ent);
}

inline void addFunc(venv &ve, bltin f, ty *result, const char *name, 
		    ty *t1 = 0, ty *t2 = 0, ty *t3 = 0, ty* t4 = 0,
		    ty *t5 = 0, ty *t6 = 0, ty *t7 = 0, ty *t8 = 0)
{
  access *a = new bltinAccess(f);
  addFunc(ve, a, result, name, t1, t2, t3, t4, t5, t6, t7, t8);
}
  
inline void addFunc(venv &ve, inst::opcode o, ty *result, const char *name, 
		    ty *t1 = 0, ty *t2 = 0, ty *t3 = 0, ty* t4 = 0,
		    ty *t5 = 0, ty *t6 = 0, ty *t7 = 0, ty *t8 = 0)
{
  access *a = new instAccess(o);
  addFunc(ve, a, result, name, t1, t2, t3, t4, t5, t6, t7, t8);
}
  
  
void addRealFunc0(venv &ve, bltin fcn, const char *name)
{
  addFunc(ve, fcn, primReal(), name);
}

template<double (*fcn)(double)>
void addRealFunc(venv &ve, const char* name)
{
  addFunc(ve, run::realReal<fcn>, primReal(), name, primReal());
  addFunc(ve, run::realArrayFunc<fcn>, realArray(), name, realArray());
}

#define addRealFunc(fcn) addRealFunc<fcn>(ve, #fcn);

extern "C" double asinh(double x);
extern "C" double acosh(double x);
extern "C" double atanh(double x);
extern "C" double cbrt(double x);

void addRealFunc2(venv &ve, bltin fcn, const char *name)
{
  addFunc(ve, fcn, primReal(), name, primReal(), primReal());
}

// The identity access, i.e. no instructions are encoded for a cast or
// operation, and no fuctions are called.
identAccess id;

void addUnaryOperators(venv &ve)
{
  addFunc(ve, &id, primInt(), "+", primInt());
  addFunc(ve, inst::i_negate, primInt(), "-", primInt());

  addFunc(ve, &id, primReal(), "+", primReal());
  addFunc(ve, inst::f_negate, primReal(), "-", primReal());

  addFunc(ve, inst::log_not, primBoolean(), "!", primBoolean());

  addFunc(ve, &id, primPair(), "+", primPair());
  addFunc(ve, run::pairNegate, primPair(), "-", primPair());
}

/* To avoid typing the same type three times. */
void addSimpleOperator(venv &ve, access *a, ty *t, const char *name)
{
  addFunc(ve, a, t, name, t, t);
}
void addSimpleOperator(venv &ve, inst::opcode o, ty *t, const char *name)
{
  addFunc(ve, o, t, name, t, t);
}
void addSimpleOperator(venv &ve, bltin f, ty *t, const char *name)
{
  addFunc(ve, f, t, name, t, t);
}
void addBooleanOperator(venv &ve, access *a, ty *t, const char *name)
{
  addFunc(ve, a, primBoolean(), name, t, t);
}
void addBooleanOperator(venv &ve, inst::opcode o, ty *t, const char *name)
{
  addFunc(ve, o, primBoolean(), name, t, t);
}
void addBooleanOperator(venv &ve, bltin f, ty *t, const char *name)
{
  addFunc(ve, f, primBoolean(), name, t, t);
}

template<class T, template <class S> class op>
inline void addArrayOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addFunc(ve,run::arrayOp<T,op>,t1,name,t1,t2);
  addFunc(ve,run::opArray<T,op>,t1,name,t2,t1);
  addSimpleOperator(ve,run::arrayArrayOp<T,op>,t1,name);
}

template<class T, template <class S> class op>
inline void addArrayBooleanOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addFunc(ve,run::arrayOp<T,op>,boolArray(),name,t1,t2);
  addFunc(ve,run::opArray<T,op>,boolArray(),name,t2,t1);
  addFunc(ve,run::arrayArrayOp<T,op>,boolArray(),name,t1,t1);
}

template<class T>
inline void addUnorderedArrayOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addArrayBooleanOps<T,equals>(ve,t1,"==",t2);
  addArrayBooleanOps<T,notequals>(ve,t1,"!=",t2);
  
  addFunc(ve,run::writen<T>,primVoid(),"write",t2);
  addFunc(ve,run::write2<T>,primVoid(),"write",t2,t2);
  addFunc(ve,run::write3<T>,primVoid(),"write",t2,t2,t2);
  addFunc(ve,run::showArray<T>,primVoid(),"write",t1);
  addFunc(ve,run::showArray2<T>,primVoid(),"write",t3);
  
  addFunc(ve,run::write<T>,primVoid(),"write",primFile(),t2);
  addFunc(ve,run::writeArray<T>,primVoid(),"write",primFile(),t1);
  addFunc(ve,run::writeArray2<T>,primVoid(),"write",primFile(),t3);
  addFunc(ve,run::writeArray3<T>,primVoid(),"write",primFile(),t4);
}

template<class T>
inline void addOrderedArrayOps(venv &ve, ty *t1, ty *t2, ty *t3)
{
  addArrayBooleanOps<T,less>(ve,t1,"<",t2);
  addArrayBooleanOps<T,lessequals>(ve,t1,"<=",t2);
  addArrayBooleanOps<T,greaterequals>(ve,t1,">=",t2);
  addArrayBooleanOps<T,greater>(ve,t1,">",t2);
  
  addFunc(ve,run::minArray<T>,t2,"min",t1);
  addFunc(ve,run::maxArray<T>,t2,"max",t1);
  addFunc(ve,run::sortArray<T>,t1,"sort",t1);
  addFunc(ve,run::sortArray2<T>,t3,"sort",t3);
  
  addArrayOps<T,min>(ve,t1,"min",t2);
  addArrayOps<T,max>(ve,t1,"max",t2);
}

template<class T>
inline void addArrayOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addArrayOps<T,plus>(ve,t1,"+",t2);
  addArrayOps<T,minus>(ve,t1,"-",t2);
  addArrayOps<T,times>(ve,t1,"*",t2);
  addArrayOps<T,divide>(ve,t1,"/",t2);
  addArrayOps<T,power>(ve,t1,"^",t2);
  
  addFunc(ve,&id,t1,"+",t1);
  addFunc(ve,run::arrayNegate<T>,t1,"-",t1);
  
  addFunc(ve,run::sumArray<T>,t2,"sum",t1);
  
  addUnorderedArrayOps<T>(ve,t1,t2,t3,t4);
}

function *intRealFunction()
{
  function *ft = new function(primInt());
  ft->add(primReal());

  return ft;
}

function *realPairFunction()
{
  function *ft = new function(primReal());
  ft->add(primPair());

  return ft;
}

void addOperators(venv &ve) 
{
  addSimpleOperator(ve,inst::i_plus,primInt(),"+");
  addSimpleOperator(ve,inst::i_minus,primInt(),"-");
  addSimpleOperator(ve,inst::i_times,primInt(),"*");
  addSimpleOperator(ve,inst::i_divide,primInt(),"/");
  addSimpleOperator(ve,run::intIntMod,primInt(),"%");
  addSimpleOperator(ve,run::intIntPow,primInt(),"^");
  addBooleanOperator(ve,inst::i_lt,primInt(),"<");
  addBooleanOperator(ve,inst::i_le,primInt(),"<=");
  addBooleanOperator(ve,inst::i_eq,primInt(),"==");
  addBooleanOperator(ve,inst::i_ge,primInt(),">=");
  addBooleanOperator(ve,inst::i_gt,primInt(),">");
  addBooleanOperator(ve,inst::i_neq,primInt(),"!=");
  addSimpleOperator(ve,run::binaryOp<int,min>,primInt(),"min");
  addSimpleOperator(ve,run::binaryOp<int,max>,primInt(),"max");

  addSimpleOperator(ve,inst::f_plus,primReal(),"+");
  addSimpleOperator(ve,inst::f_minus,primReal(),"-");
  addSimpleOperator(ve,inst::f_times,primReal(),"*");
  addSimpleOperator(ve,inst::f_divide,primReal(),"/");
  addSimpleOperator(ve,run::realRealMod,primReal(),"%");
  addSimpleOperator(ve,run::realRealPow,primReal(),"^");
  addBooleanOperator(ve,inst::f_lt,primReal(),"<");
  addBooleanOperator(ve,inst::f_le,primReal(),"<=");
  addBooleanOperator(ve,inst::f_eq,primReal(),"==");
  addBooleanOperator(ve,inst::f_ge,primReal(),">=");
  addBooleanOperator(ve,inst::f_gt,primReal(),">");
  addBooleanOperator(ve,inst::f_neq,primReal(),"!=");
  addSimpleOperator(ve,run::binaryOp<double,min>,primReal(),"min");
  addSimpleOperator(ve,run::binaryOp<double,max>,primReal(),"max");
  
  addFunc(ve,run::realIntPow,primReal(),"^",primReal(),primInt());

  // This works as we force all booleans to be 1 or 0 as an integer.
  addBooleanOperator(ve,inst::log_eq,primBoolean(),"==");
  addBooleanOperator(ve,inst::log_neq,primBoolean(),"!=");
  addBooleanOperator(ve,run::boolXor,primBoolean(),"^");

  addBooleanOperator(ve,run::boolTrue,primNull(),"==");
  addBooleanOperator(ve,run::intZero,primNull(),"!=");

  addSimpleOperator(ve,run::binaryOp<string,plus>,primString(),"+");
  addBooleanOperator(ve,run::binaryOp<string,less>,primString(),"<");
  addBooleanOperator(ve,run::binaryOp<string,lessequals>,primString(),"<=");
  addBooleanOperator(ve,run::binaryOp<string,equals>,primString(),"==");
  addBooleanOperator(ve,run::binaryOp<string,greater>,primString(),">");
  addBooleanOperator(ve,run::binaryOp<string,greaterequals>,primString(),">=");
  addBooleanOperator(ve,run::binaryOp<string,notequals>,primString(),"!=");
  addSimpleOperator(ve,run::binaryOp<string,min>,primString(),"min");
  addSimpleOperator(ve,run::binaryOp<string,max>,primString(),"max");

  addSimpleOperator(ve,run::binaryOp<pair,plus>,primPair(),"+");
  addSimpleOperator(ve,run::binaryOp<pair,minus>,primPair(),"-");
  addSimpleOperator(ve,run::binaryOp<pair,times>,primPair(),"*");
  addSimpleOperator(ve,run::binaryOp<pair,divide>,primPair(),"/");
  addSimpleOperator(ve,run::binaryOp<pair,power>,primPair(),"^");
  
  addBooleanOperator(ve,run::binaryOp<pair,equals>,primPair(),"==");
  addBooleanOperator(ve,run::binaryOp<pair,notequals>,primPair(),"!=");

  addSimpleOperator(ve,run::transformTransformMult,primTransform(),"*");
  addFunc(ve,run::transformPairMult,primPair(),"*",primTransform(),
	  primPair());
  addFunc(ve,run::transformPathMult,primPath(),"*",primTransform(),
	  primPath());
  addFunc(ve,run::transformPenMult,primPen(),"*",primTransform(),
	  primPen());
  addFunc(ve,run::transformFrameMult,primPicture(),"*",primTransform(),
	  primPicture());
  addFunc(ve,run::transformPow,primTransform(),"^",primTransform(),
	  primInt());

  addSimpleOperator(ve,run::penPenPlus,primPen(),"+");
  addFunc(ve,run::realPenTimes,primPen(),"*",primReal(),primPen());
  addFunc(ve,run::penRealTimes,primPen(),"*",primPen(),primReal());
  addBooleanOperator(ve,run::boolPenEq,primPen(),"==");

  addFunc(ve,run::arrayBoolNegate,boolArray(),"!",boolArray());
  addArrayBooleanOps<bool,And>(ve,boolArray(),"&&",primBoolean());
  addArrayBooleanOps<bool,Or>(ve,boolArray(),"||",primBoolean());
  addArrayBooleanOps<bool,Xor>(ve,boolArray(),"^",primBoolean());
  
  addUnorderedArrayOps<bool>(ve,boolArray(),primBoolean(),
			     boolArray2(),boolArray3());
  addArrayOps<int>(ve,intArray(),primInt(),intArray2(),intArray3());
  addArrayOps<double>(ve,realArray(),primReal(),realArray2(),realArray3());
  addArrayOps<pair>(ve,pairArray(),primPair(),pairArray2(),pairArray3());
  addUnorderedArrayOps<string>(ve,stringArray(),primString(),
			       stringArray2(),stringArray3());
  
  addOrderedArrayOps<int>(ve,intArray(),primInt(),intArray2());
  addOrderedArrayOps<double>(ve,realArray(),primReal(),realArray2());
  addOrderedArrayOps<string>(ve,stringArray(),primString(),stringArray2());
  
  addArrayOps<int,mod>(ve,intArray(),"%",primInt());
  addArrayOps<double,mod>(ve,realArray(),"%",primReal());
}

double identity(double x) {return x;}
double pow10(double x) {return pow(10.0,x);}

// NOTE: We should move all of these into a "builtin" module.
void base_venv(venv &ve)
{
  addUnaryOperators(ve);
  addOperators(ve);

  addFunc(ve,run::draw,primVoid(),"draw",primPicture(),primPath(), primPen());
  addFunc(ve,run::fill,primVoid(),"fill",primPicture(),primPath(),
	  primPen(),primPair(),primReal(),primPen(),primPair(),primReal());
  addFunc(ve,run::clip,primVoid(),"clip",primPicture(),primPath());
  addFunc(ve,run::add,primVoid(),"add",primPicture(),primPicture());
  addFunc(ve,run::label,primVoid(),"_label",primPicture(),
	  primString(),primReal(),primPair(),primPair(),primPen());
  
  addFunc(ve,run::shipout,primVoid(),"shipout",primString(),
	  primPicture(),primPicture(),primString(),primBoolean());
  
  addFunc(ve,run::reset,primVoid(),"reset");
  
  addFunc(ve,run::overwrite,primVoid(),"overwrite",primInt());
  
  addFunc(ve,run::abort,primVoid(),"abort",primString());
  addFunc(ve,run::exit,primVoid(),"exit",primInt());
  
  addFunc(ve,run::execute,primVoid(),"execute",primString());

  addFunc(ve,run::stringFilePrefix,primString(),"fileprefix");
  
  addFunc(ve,run::postscript,primVoid(),"postscript",primPicture(),
	  primString());
  addFunc(ve,run::tex,primVoid(),"tex",primPicture(),primString());
  addFunc(ve,run::texPreamble,primVoid(),"texpreamble",primString());
  addFunc(ve,run::layer,primVoid(),"layer",primPicture());
  
  addFunc(ve,run::intIntMax,primInt(),"intMax");
  addRealFunc0(ve,run::realPi,"pi");
  addRealFunc0(ve,run::realInfinity,"Infinity");
  addRealFunc0(ve,run::realRealMax,"realMax");
  addRealFunc0(ve,run::realRealMin,"realMin");
  addRealFunc0(ve,run::realRealEpsilon,"realEpsilon");

  addRealFunc(sin);
  addRealFunc(cos);
  addRealFunc(tan);
  addRealFunc(asin);
  addRealFunc(acos);
  addRealFunc(atan);
  addRealFunc(exp);
  addRealFunc(log);
  addRealFunc(log10);
  addRealFunc(sinh);
  addRealFunc(cosh);
  addRealFunc(tanh);
  addRealFunc(asinh);
  addRealFunc(acosh);
  addRealFunc(atanh);
  addRealFunc(sqrt);
  addRealFunc(cbrt);
  addRealFunc(fabs);
  addRealFunc<fabs>(ve,"abs");

  addRealFunc(pow10);
  addRealFunc(identity);
  
  addRealFunc2(ve,run::realAtan2,"atan2");
  addRealFunc2(ve,run::realHypot,"hypot");
  addRealFunc2(ve,run::realFmod,"fmod");
  addRealFunc2(ve,run::realDrem,"drem");
  
  addFunc(ve,run::intAbs,primInt(),"abs",primInt());
  addFunc(ve,run::intCeil,primInt(),"ceil",primReal());
  addFunc(ve,run::intFloor,primInt(),"floor",primReal());
  addFunc(ve,run::intSgn,primInt(),"sgn",primReal());
  addFunc(ve,run::intRound,primInt(),"round",primReal());
  
  addFunc(ve,run::intRand,primInt(),"rand");
  addFunc(ve,run::intSrand,primVoid(),"srand",primInt());
  addFunc(ve,run::intRandMax,primInt(),"randMax");
  
  addFunc(ve,run::pairXPart,primReal(),"xpart",primPair());
  addFunc(ve,run::pairYPart,primReal(),"ypart",primPair());
  addFunc(ve,run::pairLength,primReal(),"length",primPair());
  addFunc(ve,run::pairLength,primReal(),"abs",primPair());
  addFunc(ve,run::pairAngle,primReal(),"angle",primPair());
  addFunc(ve,run::pairDegrees,primReal(),"Angle",primPair());
  addFunc(ve,run::pairUnit,primPair(),"unit",primPair());
  addFunc(ve,run::realDir,primPair(),"dir",primReal());
  addFunc(ve,run::pairExpi,primPair(),"expi",primReal());
  addFunc(ve,run::pairConj,primPair(),"conj",primPair());
  addFunc(ve,run::pairDot,primReal(),"Dot",primPair(),primPair());

  addFunc(ve,run::suppressOutput,primVoid(),"suppressoutput",primBoolean());
  addFunc(ve,run::upToDate,primVoid(),"uptodate",primBoolean());
  
  addFunc(ve,run::stringLength,primInt(),"length",primString());
  addFunc(ve,run::stringFind,primInt(),"find",primString(),primString(),
	  primInt());
  addFunc(ve,run::stringRfind,primInt(),"rfind",primString(),primString(),
	  primInt());
  addFunc(ve,run::stringSubstr,primString(),"substr",primString(),primInt(),
	  primInt());
  addFunc(ve,run::stringReverse,primString(),"reverse",primString());
  addFunc(ve,run::stringInsert,primString(),"insert",primString(),primInt(),
	  primString());
  addFunc(ve,run::stringErase,primString(),"erase",primString(),primInt(),
	  primInt());
  addFunc(ve,run::stringReplace,primString(),"replace",primString(),
	  stringArray2());
  addFunc(ve,run::stringFormatReal,primString(),"format",primString(),
	  primReal());
  addFunc(ve,run::stringFormatInt,primString(),"format",primString(),
	  primInt());
  addFunc(ve,run::stringTime,primString(),"time",primString());
  
  addFunc(ve,run::system,primInt(),"system",primString());
  addFunc(ve,run::merge,primInt(),"merge",primString(),primString(),
	  primBoolean());
  addFunc(ve,run::boolDeconstruct,primBoolean(),"deconstruct");
  
  addFunc(ve,run::pathIntPoint,primPair(),"point",primPath(),primInt());
  addFunc(ve,run::pathRealPoint,primPair(),"point",primPath(),primReal());
  
  addFunc(ve,run::pathIntPrecontrol,primPair(),"precontrol",primPath(),
	  primInt());
  addFunc(ve,run::pathRealPrecontrol,primPair(),"precontrol",primPath(),
	  primReal());
  
  addFunc(ve,run::pathIntPostcontrol,primPair(),"postcontrol",primPath(),
	  primInt());
  addFunc(ve,run::pathRealPostcontrol,primPair(),"postcontrol",primPath(),
	  primReal());
  
  addFunc(ve,run::pathIntDirection,primPair(),"direction",primPath(),
	  primInt());
  addFunc(ve,run::pathRealDirection,primPair(),"direction",primPath(),
	  primReal());
  addFunc(ve,run::pathIntDirection,primPair(),"dir",primPath(),primInt());
  addFunc(ve,run::pathRealDirection,primPair(),"dir",primPath(),primReal());

  addFunc(ve,run::pathCyclic,primBoolean(),"cyclic",primPath());
  addFunc(ve,run::pathStraight,primBoolean(),"straight",primPath(),primInt());
  addFunc(ve,run::pathReverse,primPath(),"reverse",primPath());
  addFunc(ve,run::pathSubPath,primPath(),"subpath",primPath(),
	  primInt(),primInt());
  addFunc(ve,run::pathSubPathReal,primPath(),"subpath",primPath(),
	  primReal(),primReal());
  
  addFunc(ve,run::pathLength,primInt(),"length",primPath());
  addFunc(ve,run::pathArcLength,primReal(),"arclength",primPath());
  addFunc(ve,run::pathArcTimeOfLength,primReal(),"arctime",primPath(),
	  primReal());
  addFunc(ve,run::pathDirectionTime,primReal(),"dirtime",primPath(),
	  primPair());
  addFunc(ve,run::pathIntersectionTime,primPair(),"intersect",primPath(),
	  primPath());
  
  addFunc(ve,run::pathSize,primInt(),"size",primPath());
  addFunc(ve,run::pathMax,primPair(),"max",primPath());
  addFunc(ve,run::pathMin,primPair(),"min",primPath());
  
  addFunc(ve,run::frameMax,primPair(),"max",primPicture());
  addFunc(ve,run::frameMin,primPair(),"min",primPicture());
  
  addFunc(ve,run::lineType,primPen(),"linetype",primString());
  addFunc(ve,run::rgb,primPen(),"rgb",primReal(),primReal(),primReal());
  addFunc(ve,run::cmyk,primPen(),"cmyk",primReal(),primReal(),primReal(),
	  primReal());
  addFunc(ve,run::gray,primPen(),"gray",primReal());
  addFunc(ve,run::colors,realArray(),"colors",primPen());
  addFunc(ve,run::pattern,primPen(),"pattern",primString());
  addFunc(ve,run::defaultpen,primPen(),"defaultpen");
  addFunc(ve,run::invisiblepen,primPen(),"invisible");
  addFunc(ve,run::lineCap,primPen(),"linecap",primInt());
  addFunc(ve,run::lineJoin,primPen(),"linejoin",primInt());
  addFunc(ve,run::lineWidth,primPen(),"linewidth",primReal());
  addFunc(ve,run::penLineWidth,primReal(),"linewidth",primPen());
  addFunc(ve,run::defaultLineWidth,primVoid(),"defaultlinewidth",primReal());
  addFunc(ve,run::fontSize,primPen(),"fontsize",primReal());
  addFunc(ve,run::penFontSize,primReal(),"fontsize",primPen());
  addFunc(ve,run::defaultFontSize,primVoid(),"defaultfontsize",primReal());
  addFunc(ve,run::penMax,primPair(),"max",primPen());
  addFunc(ve,run::penMin,primPair(),"min",primPen());
  
  // Transform creation
  
  addFunc(ve,run::transformIdentity,primTransform(),"identity");
  addFunc(ve,run::transformInverse,primTransform(),"inverse",primTransform());
  addFunc(ve,run::transformShift,primTransform(),"shift",primPair());
  addFunc(ve,run::transformXscale,primTransform(),"xscale",primReal());
  addFunc(ve,run::transformYscale,primTransform(),"yscale",primReal());
  addFunc(ve,run::transformScale,primTransform(),"scale",primReal());
  addFunc(ve,run::transformScaleInt,primTransform(),"scale",primInt());
  addFunc(ve,run::transformScalePair,primTransform(),"scale",primPair());
  addFunc(ve,run::transformSlant,primTransform(),"slant",primReal());
  addFunc(ve,run::transformRotate,primTransform(),"rotate",primReal(),
	  primPair());
  addFunc(ve,run::transformReflect,primTransform(),"reflect",primPair(),
	  primPair());
  
  // I/O functions

  addFunc(ve,run::fileOpenOut,primFile(),"output",primString());
  addFunc(ve,run::fileOpenIn,primFile(),"input",primString());
  addFunc(ve,run::fileOpenXOut,primFile(),"xoutput",primString());
  addFunc(ve,run::fileOpenXIn,primFile(),"xinput",primString());

  addFunc(ve,run::fileEol,primBoolean(),"eol",primFile());
  addFunc(ve,run::fileEof,primBoolean(),"eof",primFile());
  addFunc(ve,run::fileError,primBoolean(),"error",primFile());
  addFunc(ve,run::fileClear,primVoid(),"clear",primFile());
  addFunc(ve,run::fileClose,primVoid(),"close",primFile());
  addFunc(ve,run::filePrecision,primVoid(),"precision",primFile(),primInt());
  addFunc(ve,run::fileFlush,primVoid(),"flush",primFile());

  addFunc(ve,run::fileDimension1,primFile(),"dimension",primFile(),primInt());
  addFunc(ve,run::fileDimension2,primFile(),"dimension",primFile(),primInt(),
	  primInt());
  addFunc(ve,run::fileDimension3,primFile(),"dimension",primFile(),primInt(),
	  primInt(),primInt());
  addFunc(ve,run::fileCSVMode,primFile(),"csv",primFile());
  addFunc(ve,run::fileLineMode,primFile(),"line",primFile());
  addFunc(ve,run::fileArray1,primFile(),"read1",primFile());
  addFunc(ve,run::fileArray2,primFile(),"read2",primFile());
  addFunc(ve,run::fileArray3,primFile(),"read3",primFile());
  addFunc(ve,run::readChar,primString(),"getc",primFile());

  addFunc(ve,run::writeP<pen>,primVoid(),"write",primFile(),primPen());
  addFunc(ve,run::writeP<guide>,primVoid(),"write",primFile(),primGuide());
  addFunc(ve,run::writeP<transform>,primVoid(),"write",primFile(),
	  primTransform());
  addFunc(ve,run::showArray<string>,primVoid(),"write",stringArray());
  
  // Array functions
  
  addFunc(ve,run::arrayFunction,realArray(),"eval",realPairFunction(),
	  pairArray());
  addFunc(ve,run::arrayFunction,intArray(),"eval",intRealFunction(),
	  realArray());
  
  addFunc(ve,run::arrayAll,primBoolean(),"all",boolArray());
  addFunc(ve,run::arrayBoolSum,primInt(),"sum",boolArray());
  
  addFunc(ve,run::intSequence,intArray(),"sequence",primInt());
  addFunc(ve,run::arrayFind,primInt(),"find",boolArray(),primInt());
#ifdef HAVE_LIBFFTW3
  addFunc(ve,run::pairArrayFFT,pairArray(),"fft",pairArray(),primInt());
#endif
}

void base_menv(menv&)
{
}

} //namespace trans
