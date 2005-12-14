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
#include "runtime.h"
#include "types.h"

#include "castop.h"
#include "mathop.h"
#include "arrayop.h"
#include "vm.h"

#include "coder.h"
#include "exp.h"

using namespace types;
using namespace camp;

namespace trans {
using camp::transform;
using camp::pair;
using vm::bltin;
using run::divide;
using mem::string;

void gen_base_venv(venv &ve);

void addType(tenv &te, const char *name, ty *t)
{
  te.enter(symbol::trans(name), new tyEntry(t,0));
}

// The base environments for built-in types and functions
void base_tenv(tenv &te)
{
  addType(te, "void", primVoid());
  addType(te, "bool", primBoolean());
  addType(te, "int", primInt());
  addType(te, "real", primReal());
  addType(te, "string", primString());
  
  addType(te, "pair", primPair());
  addType(te, "triple", primTriple());
  addType(te, "transform", primTransform());
  addType(te, "guide", primGuide());
  addType(te, "path", primPath());
  addType(te, "pen", primPen());
  addType(te, "frame", primPicture());

  addType(te, "file", primFile());
  addType(te, "code", primCode());
}

// Add a function with one or more default arguments.
void addFunc(venv &ve, bltin f, ty *result, const char *name, 
	     ty *t1, const char *s1, bool d1,
	     ty *t2, const char *s2, bool d2,
	     ty *t3, const char *s3, bool d3,
	     ty *t4, const char *s4, bool d4,
	     ty *t5, const char *s5, bool d5,
	     ty *t6, const char *s6, bool d6,
	     ty *t7, const char *s7, bool d7,
	     ty *t8, const char *s8, bool d8)
{
  access *a = new bltinAccess(f);
  function *fun = new function(result);

  if (t1) fun->add(t1,false,s1,d1);
  if (t2) fun->add(t2,false,s2,d2);
  if (t3) fun->add(t3,false,s3,d3);
  if (t4) fun->add(t4,false,s4,d4);
  if (t5) fun->add(t5,false,s5,d5);
  if (t6) fun->add(t6,false,s6,d6);
  if (t7) fun->add(t7,false,s7,d7);
  if (t8) fun->add(t8,false,s8,d8);

  varEntry *ent = new varEntry(fun, a);
  
  ve.enter(symbol::trans(name), ent);
}
  
void addFunc(venv &ve, access *a, ty *result, symbol *id,
	     ty *t1=0, const char *s1="", bool d1=false) 
{
  function *fun = new function(result);
  if (t1) fun->add(t1,false,s1,d1);
  varEntry *ent = new varEntry(fun, a);
  ve.enter(id, ent);
}

void addFunc(venv &ve, access *a, ty *result, const char *name,
	     ty *t1=0, const char *s1="", bool d1=false) 
{
  addFunc(ve,a,result,symbol::trans(name),t1,s1,d1);
}

// Add a rest function with zero or more default/explicit arguments.
void addRestFunc(venv &ve, bltin f, ty *result, const char *name,
		 ty *trest, bool erest=false,
		 ty *t1=0, const char *s1="", bool d1=false, bool e1=false,
		 ty *t2=0, const char *s2="", bool d2=false, bool e2=false,
		 ty *t3=0, const char *s3="", bool d3=false, bool e3=false,
		 ty *t4=0, const char *s4="", bool d4=false, bool e4=false)
//                   type              name        default        explicit
{
  access *a = new bltinAccess(f);
  function *fun = new function(result);

  if (t1) fun->add(t1,e1,s1,d1);
  if (t2) fun->add(t2,e2,s2,d2);
  if (t3) fun->add(t3,e3,s3,d3);
  if (t4) fun->add(t4,e4,s4,d4);
  
  if (trest) fun->addRest(trest,erest);

  varEntry *ent = new varEntry(fun, a);

  ve.enter(symbol::trans(name), ent);
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

void addRealFunc2(venv &ve, bltin fcn, const char *name)
{
  addFunc(ve,fcn,primReal(),name,primReal(),"a=",false,primReal(),"b=",false);
}

void addInitializer(venv &ve, ty *t, access *a)
{
  addFunc(ve, a, t, symbol::initsym);
}

void addInitializer(venv &ve, ty *t, bltin f)
{
  access *a = new bltinAccess(f);
  addInitializer(ve, t, a);
}

// Specifies that source may be cast to target, but only if an explicit
// cast expression is used.
void addExplicitCast(venv &ve, ty *target, ty *source, access *a) {
  addFunc(ve, a, target, symbol::ecastsym, source);
}

// Specifies that source may be implicitly cast to target by the
// function or instruction stores at a.
void addCast(venv &ve, ty *target, ty *source, access *a) {
  //addExplicitCast(target,source,a);
  addFunc(ve, a, target, symbol::castsym, source);
}

void addExplicitCast(venv &ve, ty *target, ty *source, bltin f) {
  addExplicitCast(ve, target, source, new bltinAccess(f));
}

void addCast(venv &ve, ty *target, ty *source, bltin f) {
  addCast(ve, target, source, new bltinAccess(f));
}

// The identity access, i.e. no instructions are encoded for a cast or
// operation, and no functions are called.
identAccess id;

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

function *voidFileFunction()
{
  function *ft = new function(primVoid());
  ft->add(primFile());

  return ft;
}

void addInitializers(venv &ve)
{
  addInitializer(ve, primBoolean(), run::boolFalse);
  addInitializer(ve, primInt(), run::intZero);
  addInitializer(ve, primReal(), run::realZero);

  addInitializer(ve, primString(), run::emptyString);
  addInitializer(ve, primPair(), run::pairZero);
  addInitializer(ve, primTriple(), run::tripleZero);
  addInitializer(ve, primTransform(), run::transformIdentity);
  addInitializer(ve, primGuide(), run::nullGuide);
  addInitializer(ve, primPath(), run::nullPath);
  addInitializer(ve, primPen(), run::newPen);
  addInitializer(ve, primPicture(), run::newPicture);
  addInitializer(ve, primFile(), run::standardOut);
}

void addCasts(venv &ve)
{
  addExplicitCast(ve, primString(), primInt(), run::stringCast<int>);
  addExplicitCast(ve, primString(), primReal(), run::stringCast<double>);
  addExplicitCast(ve, primString(), primPair(), run::stringCast<pair>);
  addExplicitCast(ve, primString(), primTriple(), run::stringCast<triple>);
  addExplicitCast(ve, primInt(), primString(), run::castString<int>);
  addExplicitCast(ve, primReal(), primString(), run::castString<double>);
  addExplicitCast(ve, primPair(), primString(), run::castString<pair>);
  addExplicitCast(ve, primTriple(), primString(), run::castString<triple>);

  addExplicitCast(ve, primInt(), primReal(), run::castDoubleInt);

  addCast(ve, primReal(), primInt(), run::cast<int,double>);
  addCast(ve, primPair(), primInt(), run::cast<int,pair>);
  addCast(ve, primPair(), primReal(), run::cast<double,pair>);
  
  addCast(ve, primPath(), primPair(), run::cast<pair,path>);
  addCast(ve, primGuide(), primPair(), run::pairToGuide);
  addCast(ve, primGuide(), primPath(), run::pathToGuide);
  addCast(ve, primPath(), primGuide(), run::guideToPath);

  addCast(ve, primFile(), primNull(), run::nullFile);
  addCast(ve, primString(), primFile(), run::read<string>);

  // Vectorized casts.
  addExplicitCast(ve, intArray(), realArray(), run::arrayToArray<double,int>);
  
  addCast(ve, stringArray(), primFile(), run::readArray<string>);
  addCast(ve, stringArray2(), primFile(), run::readArray<string>);
  addCast(ve, stringArray3(), primFile(), run::readArray<string>);

  addCast(ve, realArray(), intArray(), run::arrayToArray<int,double>);
  addCast(ve, pairArray(), intArray(), run::arrayToArray<int,pair>);
  addCast(ve, pairArray(), realArray(), run::arrayToArray<double,pair>);
}

void addGuideOperators(venv &ve)
{
  // The guide operators .. and -- take an array of guides, and turn them
  // into a single guide.
  addRestFunc(ve, run::dotsGuide, primGuide(), "..", guideArray());
  addRestFunc(ve, run::dashesGuide, primGuide(), "--", guideArray());
}

/* Avoid typing the same type three times. */
void addSimpleOperator(venv &ve, bltin f, ty *t, const char *name)
{
  addFunc(ve, f, t, name, t, "a=", false, t, "b=", false);
}
void addBooleanOperator(venv &ve, bltin f, ty *t, const char *name)
{
  addFunc(ve, f, primBoolean(), name, t, "a=", false, t, "b=", false);
}

template<class T, template <class S> class op>
void addOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addSimpleOperator(ve,run::binaryOp<T,op>,t2,name);
  addFunc(ve,run::arrayOp<T,op>,t1,name,t1,"a=",false,t2,"b=",false);
  addFunc(ve,run::opArray<T,op>,t1,name,t2,"a=",false,t1,"b=",false);
  addSimpleOperator(ve,run::arrayArrayOp<T,op>,t1,name);
}

template<class T, template <class S> class op>
void addBooleanOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addBooleanOperator(ve,run::binaryOp<T,op>,t2,name);
  addFunc(ve,run::arrayOp<T,op>,boolArray(),name,t1,"a=",false,t2,"b=",false);
  addFunc(ve,run::opArray<T,op>,boolArray(),name,t2,"a=",false,t1,"b=",false);
  addFunc(ve,run::arrayArrayOp<T,op>,boolArray(),name,t1,"a=",false,
	  t1,"b=",false);
}

void addWrite(venv &ve, bltin f, ty *t1, ty *t2)
{
  addRestFunc(ve,f,primVoid(),"write",t1,false,primFile(),"file",true,false,
	      primString(),"s",true,false,t2,"x",false,false,
	      voidFileFunction(),"suffix",true,false);
}

template<class T>
void addUnorderedOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addBooleanOps<T,run::equals>(ve,t1,"==",t2);
  addBooleanOps<T,run::notequals>(ve,t1,"!=",t2);
  
  addCast(ve,t2,primFile(),run::read<T>);
  addCast(ve,t1,primFile(),run::readArray<T>);
  addCast(ve,t3,primFile(),run::readArray<T>);
  addCast(ve,t4,primFile(),run::readArray<T>);
  
  addWrite(ve,run::write<T>,t1,t2);
  addRestFunc(ve,run::writeArray<T>,primVoid(),"write",t3,false,
	      primFile(),"file",true,false,primString(),"s",true,false,
	      t1,"a",false,true);
  addFunc(ve,run::writeArray2<T>,primVoid(),"write",primFile(),"file",true,t3);
  addFunc(ve,run::writeArray3<T>,primVoid(),"write",primFile(),"file",true,t4);
}

template<class T>
void addOrderedOps(venv &ve, ty *t1, ty *t2, ty *t3)
{
  addBooleanOps<T,run::less>(ve,t1,"<",t2);
  addBooleanOps<T,run::lessequals>(ve,t1,"<=",t2);
  addBooleanOps<T,run::greaterequals>(ve,t1,">=",t2);
  addBooleanOps<T,run::greater>(ve,t1,">",t2);
  
  addFunc(ve,run::minArray<T>,t2,"min",t1,"a=",false);
  addFunc(ve,run::maxArray<T>,t2,"max",t1,"a=",false);
  addFunc(ve,run::sortArray<T>,t1,"sort",t1,"a=",false);
  addFunc(ve,run::sortArray2<T>,t3,"sort",t3,"a=",false);
  addFunc(ve,run::searchArray<T>,primInt(),"search",t1,"a=",false,t2,
	  "b=",false);
  
  addOps<T,run::min>(ve,t1,"min",t2);
  addOps<T,run::max>(ve,t1,"max",t2);
}

template<class T>
void addBasicOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addOps<T,run::plus>(ve,t1,"+",t2);
  addOps<T,run::minus>(ve,t1,"-",t2);
  
  addFunc(ve,&id,t1,"+",t1,"a=",false);
  addFunc(ve,&id,t2,"+",t2,"a=",false);
  addFunc(ve,run::arrayNegate<T>,t1,"-",t1,"a=",false);
  addFunc(ve,run::Negate<T>,t2,"-",t2,"a=",false);
  
  addFunc(ve,run::sumArray<T>,t2,"sum",t1,"a=",false);
  addUnorderedOps<T>(ve,t1,t2,t3,t4);
}

template<class T>
void addOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, bool divide=true)
{
  addBasicOps<T>(ve,t1,t2,t3,t4);
  addOps<T,run::times>(ve,t1,"*",t2);
  if(divide) addOps<T,run::divide>(ve,t1,"/",t2);
  addOps<T,run::power>(ve,t1,"^",t2);
}

void addOperators(venv &ve) 
{
  addSimpleOperator(ve,run::binaryOp<string,run::plus>,primString(),"+");
  
  addBooleanOps<bool,run::And>(ve,boolArray(),"&&",primBoolean());
  addBooleanOps<bool,run::Or>(ve,boolArray(),"||",primBoolean());
  addBooleanOps<bool,run::Xor>(ve,boolArray(),"^",primBoolean());
  
  addUnorderedOps<bool>(ve,boolArray(),primBoolean(),boolArray2(),
			boolArray3());
  addOps<int>(ve,intArray(),primInt(),intArray2(),intArray3(),false);
  addOps<double>(ve,realArray(),primReal(),realArray2(),realArray3());
  addOps<pair>(ve,pairArray(),primPair(),pairArray2(),pairArray3());
  addBasicOps<triple>(ve,tripleArray(),primTriple(),tripleArray2(),
		      tripleArray3());
  addUnorderedOps<string>(ve,stringArray(),primString(),stringArray2(),
			  stringArray3());
  
  addFunc(ve,run::binaryOp<int,divide>,primReal(),"/",
	  primInt(),"a=",false,primInt(),"b=",false);
  addFunc(ve,run::arrayOp<int,divide>,realArray(),"/",
	  intArray(),"a=",false,primInt(),"b=",false);
  addFunc(ve,run::opArray<int,divide>,realArray(),"/",
	  primInt(),"a=",false,intArray(),"b=",false);
  addFunc(ve,run::arrayArrayOp<int,divide>,realArray(),"/",
	  intArray(),"a=",false,intArray(),"b=",false);
  
  addOrderedOps<int>(ve,intArray(),primInt(),intArray2());
  addOrderedOps<double>(ve,realArray(),primReal(),realArray2());
  addOrderedOps<string>(ve,stringArray(),primString(),stringArray2());
  
  addOps<int,run::mod>(ve,intArray(),"%",primInt());
  addOps<double,run::mod>(ve,realArray(),"%",primReal());
}

double identity(double x) {return x;}
double pow10(double x) {return pow(10.0,x);}

// NOTE: We should move all of these into a "builtin" module.
void base_venv(venv &ve)
{
  addInitializers(ve);
  addCasts(ve);
  addOperators(ve);
  addGuideOperators(ve);
  
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
  
  addFunc(ve,run::writestring,primVoid(),"write",primFile(),"file",true,
	  primString(),"s",false,voidFileFunction(),"suffix",true);
  
  addWrite(ve,run::write<pen>,penArray(),primPen());
  addWrite(ve,run::write<transform>,transformArray(),primTransform());
  addWrite(ve,run::writeP<guide>,guideArray(),primGuide());

  addFunc(ve,run::arrayFunction,realArray(),"map",
	  realPairFunction(),"f=",false,
	  pairArray(),"a=",false);
  addFunc(ve,run::arrayFunction,intArray(),"map",
	  intRealFunction(),"f=",false,
	  realArray(),"a=",false);
  
#ifdef HAVE_LIBFFTW3
  addFunc(ve,run::pairArrayFFT,pairArray(),"fft",pairArray(),"",false,
	  primInt(),"sign",true);
#endif

  gen_base_venv(ve);
}

void base_menv(menv&)
{
}

} //namespace trans
