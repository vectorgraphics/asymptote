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
#include "refaccess.h"
#include "settings.h"

using namespace types;
using namespace camp;

namespace trans {
using camp::transform;
using camp::pair;
using vm::bltin;
using run::divide;
using run::less;
using run::greater;
using run::plus;
using run::minus;
using mem::string;

using namespace run;  
  
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

const formal noformal(0);  

void addFunc(venv &ve, access *a, ty *result, symbol *id,
	     formal f1=noformal, formal f2=noformal,
	     formal f3=noformal, formal f4=noformal,
	     formal f5=noformal, formal f6=noformal,
	     formal f7=noformal, formal f8=noformal)
{
  function *fun = new function(result);

  if (f1.t) fun->add(f1);
  if (f2.t) fun->add(f2);
  if (f3.t) fun->add(f3);
  if (f4.t) fun->add(f4);
  if (f5.t) fun->add(f5);
  if (f6.t) fun->add(f6);
  if (f7.t) fun->add(f7);
  if (f8.t) fun->add(f8);

  varEntry *ent = new varEntry(fun, a);
  
  ve.enter(id, ent);
}

// Add a function with one or more default arguments.
void addFunc(venv &ve, bltin f, ty *result, const char *name, 
	     formal f1, formal f2, formal f3, formal f4,
	     formal f5, formal f6, formal f7, formal f8)
{
  access *a = new bltinAccess(f);
  addFunc(ve,a,result,symbol::trans(name),f1,f2,f3,f4,f5,f6,f7,f8);
}
  
void addFunc(venv &ve, access *a, ty *result, const char *name, formal f1)
{
  addFunc(ve,a,result,symbol::trans(name),f1);
}

// Add a rest function with zero or more default/explicit arguments.
void addRestFunc(venv &ve, bltin f, ty *result, const char *name, formal frest,
		 formal f1=noformal, formal f2=noformal,
		 formal f3=noformal, formal f4=noformal,
		 formal f5=noformal, formal f6=noformal,
		 formal f7=noformal, formal f8=noformal)
{
  access *a = new bltinAccess(f);
  function *fun = new function(result);

  if (f1.t) fun->add(f1);
  if (f2.t) fun->add(f2);
  if (f3.t) fun->add(f3);
  if (f4.t) fun->add(f4);
  if (f5.t) fun->add(f5);
  if (f6.t) fun->add(f6);
  if (f7.t) fun->add(f7);
  if (f8.t) fun->add(f8);

  if (frest.t) fun->addRest(frest);

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
  addFunc(ve, realReal<fcn>, primReal(), name, formal(primReal(),"x"));
  addFunc(ve, realArrayFunc<fcn>, realArray(), name, 
	  formal(realArray(),"a"));
}

#define addRealFunc(fcn) addRealFunc<fcn>(ve, #fcn);

void addRealFunc2(venv &ve, bltin fcn, const char *name)
{
  addFunc(ve,fcn,primReal(),name,formal(primReal(),"a"),
	  formal(primReal(),"b"));
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

template<class T>
void addConstant(venv &ve, T value, ty *t, const char *name) {
  item* ref=new item;
  *ref=value;
  access *a = new itemRefAccess(ref);
  varEntry *ent = new varEntry(t, a, READONLY, settings::getSettingsModule());
  ve.enter(symbol::trans(name), ent);
}

// The identity access, i.e. no instructions are encoded for a cast or
// operation, and no functions are called.
identAccess id;

function *intRealFunction()
{
  return new function(primInt(),primReal());
}

function *realPairFunction()
{
  return new function(primReal(),primPair());
}

function *voidFileFunction()
{
  return new function(primVoid(),primFile());
}

void addInitializers(venv &ve)
{
  addInitializer(ve, primBoolean(), boolFalse);
  addInitializer(ve, primInt(), intZero);
  addInitializer(ve, primReal(), realZero);

  addInitializer(ve, primString(), emptyString);
  addInitializer(ve, primPair(), pairZero);
  addInitializer(ve, primTriple(), tripleZero);
  addInitializer(ve, primTransform(), transformIdentity);
  addInitializer(ve, primGuide(), nullGuide);
  addInitializer(ve, primPath(), nullPath);
  addInitializer(ve, primPen(), newPen);
  addInitializer(ve, primPicture(), newPicture);
  addInitializer(ve, primFile(), standardOut);
}

void addCasts(venv &ve)
{
  addExplicitCast(ve, primString(), primInt(), stringCast<int>);
  addExplicitCast(ve, primString(), primReal(), stringCast<double>);
  addExplicitCast(ve, primString(), primPair(), stringCast<pair>);
  addExplicitCast(ve, primString(), primTriple(), stringCast<triple>);
  addExplicitCast(ve, primInt(), primString(), castString<int>);
  addExplicitCast(ve, primReal(), primString(), castString<double>);
  addExplicitCast(ve, primPair(), primString(), castString<pair>);
  addExplicitCast(ve, primTriple(), primString(), castString<triple>);

  addExplicitCast(ve, primInt(), primReal(), castDoubleInt);

  addCast(ve, primReal(), primInt(), cast<int,double>);
  addCast(ve, primPair(), primInt(), cast<int,pair>);
  addCast(ve, primPair(), primReal(), cast<double,pair>);
  
  addCast(ve, primPath(), primPair(), cast<pair,path>);
  addCast(ve, primGuide(), primPair(), pairToGuide);
  addCast(ve, primGuide(), primPath(), pathToGuide);
  addCast(ve, primPath(), primGuide(), guideToPath);

  addCast(ve, primFile(), primNull(), nullFile);

  // Vectorized casts.
  addExplicitCast(ve, intArray(), realArray(), arrayToArray<double,int>);
  
  addCast(ve, realArray(), intArray(), arrayToArray<int,double>);
  addCast(ve, pairArray(), intArray(), arrayToArray<int,pair>);
  addCast(ve, pairArray(), realArray(), arrayToArray<double,pair>);
}

void addGuideOperators(venv &ve)
{
  // The guide operators .. and -- take an array of guides, and turn them
  // into a single guide.
  addRestFunc(ve, dotsGuide, primGuide(), "..", guideArray());
  addRestFunc(ve, dashesGuide, primGuide(), "--", guideArray());
}

/* Avoid typing the same type three times. */
void addSimpleOperator(venv &ve, bltin f, ty *t, const char *name)
{
  addFunc(ve,f,t,name,formal(t,"a"),formal(t,"b"));
}
void addBooleanOperator(venv &ve, bltin f, ty *t, const char *name)
{
  addFunc(ve,f,primBoolean(),name,formal(t,"a"),formal(t,"b"));
}

template<class T, template <class S> class op>
void addOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addSimpleOperator(ve,binaryOp<T,op>,t1,name);
  addFunc(ve,opArray<T,op>,t2,name,formal(t1,"a"),formal(t2,"b"));
  addFunc(ve,arrayOp<T,op>,t2,name,formal(t2,"a"),formal(t1,"b"));
  addSimpleOperator(ve,arrayArrayOp<T,op>,t2,name);
}

template<class T, template <class S> class op>
void addBooleanOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addBooleanOperator(ve,binaryOp<T,op>,t1,name);
  addFunc(ve,opArray<T,op>,boolArray(),name,formal(t1,"a"),formal(t2,"b"));
  addFunc(ve,arrayOp<T,op>,boolArray(),name,formal(t2,"a"),formal(t1,"b"));
  addFunc(ve,arrayArrayOp<T,op>,boolArray(),name,formal(t2,"a"),
	  formal(t2,"b"));
}

void addWrite(venv &ve, bltin f, ty *t1, ty *t2)
{
  addRestFunc(ve,f,primVoid(),"write",t2,
	      formal(primFile(),"file",true),formal(primString(),"s",true),
	      formal(t1,"x"),formal(voidFileFunction(),"suffix",true));
}

template<class T>
void addUnorderedOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addBooleanOps<T,equals>(ve,t1,"==",t2);
  addBooleanOps<T,notequals>(ve,t1,"!=",t2);
  
  addCast(ve,t1,primFile(),read<T>);
  addCast(ve,t2,primFile(),readArray<T>);
  addCast(ve,t3,primFile(),readArray<T>);
  addCast(ve,t4,primFile(),readArray<T>);
  
  addWrite(ve,write<T>,t1,t2);
  addRestFunc(ve,writeArray<T>,primVoid(),"write",t3,
	      formal(primFile(),"file",true),formal(primString(),"s",true),
	      formal(t2,"a",false,true));
  addFunc(ve,writeArray2<T>,primVoid(),"write",
	  formal(primFile(),"file",true),t3);
  addFunc(ve,writeArray3<T>,primVoid(),"write",
	  formal(primFile(),"file",true),t4);
}

template<class T>
void addOrderedOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addBooleanOps<T,less>(ve,t1,"<",t2);
  addBooleanOps<T,lessequals>(ve,t1,"<=",t2);
  addBooleanOps<T,greaterequals>(ve,t1,">=",t2);
  addBooleanOps<T,greater>(ve,t1,">",t2);
  
  addFunc(ve,binopArray<T,less>,t1,"min",formal(t2,"a"));
  addFunc(ve,binopArray<T,greater>,t1,"max",formal(t2,"a"));
  addFunc(ve,binopArray2<T,less>,t1,"min",formal(t3,"a"));
  addFunc(ve,binopArray2<T,greater>,t1,"max",formal(t3,"a"));
  addFunc(ve,binopArray3<T,less>,t1,"min",formal(t4,"a"));
  addFunc(ve,binopArray3<T,greater>,t1,"max",formal(t4,"a"));
  
  addFunc(ve,sortArray<T>,t2,"sort",formal(t2,"a"));
  addFunc(ve,sortArray2<T>,t3,"sort",formal(t3,"a"));
  
  addFunc(ve,searchArray<T>,primInt(),"search",formal(t2,"a"),
	  formal(t1,"key"));
  
  addOps<T,run::min>(ve,t1,"min",t2);
  addOps<T,run::max>(ve,t1,"max",t2);
}

template<class T>
void addBasicOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addOps<T,plus>(ve,t1,"+",t2);
  addOps<T,minus>(ve,t1,"-",t2);
  
  addFunc(ve,&id,t1,"+",formal(t1,"a"));
  addFunc(ve,&id,t2,"+",formal(t2,"a"));
  addFunc(ve,Negate<T>,t1,"-",formal(t1,"a"));
  addFunc(ve,arrayNegate<T>,t2,"-",formal(t2,"a"));
  addFunc(ve,interp<T>,t1,"interp",formal(t1,"a"),formal(t1,"b"),
	  formal(primReal(),"t"));
  
  addFunc(ve,sumArray<T>,t1,"sum",formal(t2,"a"));
  addUnorderedOps<T>(ve,t1,t2,t3,t4);
}

template<class T>
void addOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, bool divide=true)
{
  addBasicOps<T>(ve,t1,t2,t3,t4);
  addOps<T,times>(ve,t1,"*",t2);
  if(divide) addOps<T,run::divide>(ve,t1,"/",t2);
  addOps<T,power>(ve,t1,"^",t2);
}

void addOperators(venv &ve) 
{
  addSimpleOperator(ve,binaryOp<string,plus>,primString(),"+");
  
  addBooleanOps<bool,And>(ve,primBoolean(),"&&",boolArray());
  addBooleanOps<bool,Or>(ve,primBoolean(),"||",boolArray());
  addBooleanOps<bool,Xor>(ve,primBoolean(),"^",boolArray());
  
  addUnorderedOps<bool>(ve,primBoolean(),boolArray(),boolArray2(),
			boolArray3());
  addOps<int>(ve,primInt(),intArray(),intArray2(),intArray3(),false);
  addOps<double>(ve,primReal(),realArray(),realArray2(),realArray3());
  addOps<pair>(ve,primPair(),pairArray(),pairArray2(),pairArray3());
  addBasicOps<triple>(ve,primTriple(),tripleArray(),tripleArray2(),
		      tripleArray3());
  addUnorderedOps<string>(ve,primString(),stringArray(),stringArray2(),
			  stringArray3());
  
  addFunc(ve,binaryOp<int,divide>,primReal(),"/",
	  formal(primInt(),"a"),formal(primInt(),"b"));
  addFunc(ve,arrayOp<int,divide>,realArray(),"/",
	  formal(intArray(),"a"),formal(primInt(),"b"));
  addFunc(ve,opArray<int,divide>,realArray(),"/",
	  formal(primInt(),"a"),formal(intArray(),"b"));
  addFunc(ve,arrayArrayOp<int,divide>,realArray(),"/",
	  formal(intArray(),"a"),formal(intArray(),"b"));
  
  addOrderedOps<int>(ve,primInt(),intArray(),intArray2(),intArray3());
  addOrderedOps<double>(ve,primReal(),realArray(),realArray2(),realArray3());
  addOrderedOps<string>(ve,primString(),stringArray(),stringArray2(),
			stringArray3());
  
  addOps<int,mod>(ve,primInt(),"%",intArray());
  addOps<double,mod>(ve,primReal(),"%",realArray());
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
  
  addFunc(ve,writestring,primVoid(),"write",
	  formal(primFile(),"file",true),
	  formal(primString(),"s"),
	  formal(voidFileFunction(),"suffix",true));
  
  addWrite(ve,write<pen>,primPen(),penArray());
  addWrite(ve,write<transform>,primTransform(),transformArray());
  addWrite(ve,write<guide *>,primGuide(),guideArray());

  addFunc(ve,arrayFunction,realArray(),"map",
	  formal(realPairFunction(),"f"),
	  formal(pairArray(),"a"));
  addFunc(ve,arrayFunction,intArray(),"map",
	  formal(intRealFunction(),"f"),
	  formal(realArray(),"a"));
  
#ifdef HAVE_LIBFFTW3
  addFunc(ve,pairArrayFFT,pairArray(),"fft",formal(pairArray(),"a"),
	  formal(primInt(),"sign",true));
#endif

  addConstant<int>(ve, INT_MAX, primInt(), "intMax");
  addConstant<double>(ve, HUGE_VAL, primReal(), "inf");
  addConstant<double>(ve, DBL_MAX, primReal(), "realMax");
  addConstant<double>(ve, DBL_MIN, primReal(), "realMin");
  addConstant<double>(ve, DBL_EPSILON, primReal(), "realEpsilon");
  addConstant<double>(ve, RAND_MAX, primReal(), "randMax");
  addConstant<double>(ve, PI, primReal(), "pi");

  gen_base_venv(ve);
}

void base_menv(menv&)
{
}

} //namespace trans
