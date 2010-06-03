/*****
 * builtin.cc
 * Tom Prince 2004/08/25
 *
 * Initialize builtins.
 *****/

#include <cmath>

#include "builtin.h"
#include "entry.h"

#include "runtime.h"
#include "runpicture.h"
#include "runlabel.h"
#include "runhistory.h"
#include "runarray.h"
#include "runfile.h"
#include "runsystem.h"
#include "runstring.h"
#include "runpair.h"
#include "runtriple.h"
#include "runpath.h"
#include "runpath3d.h"
#include "runmath.h"

#include "types.h"

#include "castop.h"
#include "mathop.h"
#include "arrayop.h"
#include "vm.h"

#include "coder.h"
#include "exp.h"
#include "refaccess.h"
#include "settings.h"

#include "opsymbols.h"
#include "builtin.symbols.h"

#ifdef HAVE_LIBGSL  
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#endif
  
using namespace types;
using namespace camp;
using namespace vm;  

namespace trans {
using camp::transform;
using camp::pair;
using vm::bltin;
using run::divide;
using run::less;
using run::greater;
using run::plus;
using run::minus;

using namespace run;  
  
void gen_runtime_venv(venv &ve);
void gen_runbacktrace_venv(venv &ve);
void gen_runpicture_venv(venv &ve);
void gen_runlabel_venv(venv &ve);
void gen_runhistory_venv(venv &ve);
void gen_runarray_venv(venv &ve);
void gen_runfile_venv(venv &ve);
void gen_runsystem_venv(venv &ve);
void gen_runstring_venv(venv &ve);
void gen_runpair_venv(venv &ve);
void gen_runtriple_venv(venv &ve);
void gen_runpath_venv(venv &ve);
void gen_runpath3d_venv(venv &ve);
void gen_runmath_venv(venv &ve);


void addType(tenv &te, symbol *name, ty *t)
{
  te.enter(name, new tyEntry(t,0,0,position()));
}

// The base environments for built-in types and functions
void base_tenv(tenv &te)
{
#define PRIMITIVE(name,Name,asyName) \
        addType(te, symbol::trans(#asyName), prim##Name());
#include "primitives.h"
#undef PRIMITIVE
}

const formal noformal(0);  

void addFunc(venv &ve, access *a, ty *result, symbol *id,
             formal f1=noformal, formal f2=noformal, formal f3=noformal,
             formal f4=noformal, formal f5=noformal, formal f6=noformal,
             formal f7=noformal, formal f8=noformal, formal f9=noformal,
             formal fA=noformal, formal fB=noformal, formal fC=noformal,
             formal fD=noformal, formal fE=noformal, formal fF=noformal,
             formal fG=noformal, formal fH=noformal, formal fI=noformal)
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
  if (f9.t) fun->add(f9);
  if (fA.t) fun->add(fA);
  if (fB.t) fun->add(fB);
  if (fC.t) fun->add(fC);
  if (fD.t) fun->add(fD);
  if (fE.t) fun->add(fE);
  if (fF.t) fun->add(fF);
  if (fG.t) fun->add(fG);
  if (fH.t) fun->add(fH);
  if (fI.t) fun->add(fI);

  // NOTE: If the function is a field, we should encode the defining record in
  // the entry
  varEntry *ent = new varEntry(fun, a, 0, position());
  
  ve.enter(id, ent);
}

// Add a function with one or more default arguments.
void addFunc(venv &ve, bltin f, ty *result, symbol *name, 
             formal f1, formal f2, formal f3, formal f4, formal f5, formal f6,
             formal f7, formal f8, formal f9, formal fA, formal fB, formal fC,
             formal fD, formal fE, formal fF, formal fG, formal fH, formal fI)
{
  REGISTER_BLTIN(f, name);
  access *a = new bltinAccess(f);
  addFunc(ve,a,result,name,f1,f2,f3,f4,f5,f6,f7,f8,f9,
      fA,fB,fC,fD,fE,fF,fG,fH,fI);
}

#if 0
// Add a function with one or more default arguments.
void addFunc(venv &ve, bltin f, ty *result, const char *name, 
             formal f1, formal f2, formal f3, formal f4, formal f5, formal f6,
             formal f7, formal f8, formal f9, formal fA, formal fB, formal fC,
             formal fD, formal fE, formal fF, formal fG, formal fH, formal fI)
{
  REGISTER_BLTIN(f, name);
  access *a = new bltinAccess(f);
  addFunc(ve,a,result,symbol::trans(name),f1,f2,f3,f4,f5,f6,f7,f8,f9,
          fA,fB,fC,fD,fE,fF,fG,fH,fI);
}
  
void addFunc(venv &ve, access *a, ty *result, const char *name, formal f1)
{
  addFunc(ve,a,result,symbol::trans(name),f1);
}
#endif

void addOpenFunc(venv &ve, bltin f, ty *result, symbol *name)
{
  function *fun = new function(result, signature::OPEN);

  REGISTER_BLTIN(f, name);
  access *a= new bltinAccess(f);

  varEntry *ent = new varEntry(fun, a, 0, position());
  
  ve.enter(name, ent);
}


// Add a rest function with zero or more default/explicit arguments.
void addRestFunc(venv &ve, bltin f, ty *result, symbol *name, formal frest,
                 formal f1=noformal, formal f2=noformal, formal f3=noformal,
                 formal f4=noformal, formal f5=noformal, formal f6=noformal,
                 formal f7=noformal, formal f8=noformal, formal f9=noformal)
{
  REGISTER_BLTIN(f, name);
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
  if (f9.t) fun->add(f9);

  if (frest.t) fun->addRest(frest);

  varEntry *ent = new varEntry(fun, a, 0, position());

  ve.enter(name, ent);
}

void addRealFunc0(venv &ve, bltin fcn, symbol *name)
{
  addFunc(ve, fcn, primReal(), name);
}

template<double (*fcn)(double)>
void addRealFunc(venv &ve, symbol *name)
{
  addFunc(ve, realReal<fcn>, primReal(), name, formal(primReal(),SYM(x)));
  addFunc(ve, arrayFunc<double,double,fcn>, realArray(), name,
          formal(realArray(),SYM(a)));
}

#define addRealFunc(fcn, sym) addRealFunc<fcn>(ve, sym);
  
void addRealFunc2(venv &ve, bltin fcn, symbol *name)
{
  addFunc(ve,fcn,primReal(),name,formal(primReal(),SYM(a)),
          formal(primReal(),SYM(b)));
}

template <double (*func)(double, int)>
void realRealInt(vm::stack *s) {
  Int n = pop<Int>(s);
  double x = pop<double>(s);
  s->push(func(x, intcast(n)));
}

template<double (*fcn)(double, int)>
void addRealIntFunc(venv& ve, symbol *name, symbol *arg1,
                    symbol *arg2) {
  addFunc(ve, realRealInt<fcn>, primReal(), name, formal(primReal(), arg1),
          formal(primInt(), arg2));
}

#ifdef HAVE_LIBGSL  
bool GSLerror=false;
  
types::dummyRecord *GSLModule;

types::record *getGSLModule()
{
  return GSLModule;
}

inline void checkGSLerror()
{
  if(GSLerror) {
    GSLerror=false;
    throw handled_error();
  }
}
  
template <double (*func)(double)>
void realRealGSL(vm::stack *s) 
{
  double x=pop<double>(s);
  s->push(func(x));
  checkGSLerror();
}

template <double (*func)(double, gsl_mode_t)>
void realRealDOUBLE(vm::stack *s) 
{
  double x=pop<double>(s);
  s->push(func(x,GSL_PREC_DOUBLE));
  checkGSLerror();
}

template <double (*func)(double, double, gsl_mode_t)>
void realRealRealDOUBLE(vm::stack *s) 
{
  double y=pop<double>(s);
  double x=pop<double>(s);
  s->push(func(x,y,GSL_PREC_DOUBLE));
  checkGSLerror();
}

template <double (*func)(unsigned)>
void realIntGSL(vm::stack *s) 
{
  s->push(func(unsignedcast(pop<Int>(s))));
  checkGSLerror();
}

template <double (*func)(int, double)>
void realIntRealGSL(vm::stack *s) 
{
  double x=pop<double>(s);
  
  s->push(func(intcast(pop<Int>(s)),x));
  checkGSLerror();
}

template <double (*func)(double, double)>
void realRealRealGSL(vm::stack *s) 
{
  double x=pop<double>(s);
  double n=pop<double>(s);
  s->push(func(n,x));
  checkGSLerror();
}

template <int (*func)(double, double, double)>
void intRealRealRealGSL(vm::stack *s) 
{
  double x=pop<double>(s);
  double n=pop<double>(s);
  double a=pop<double>(s);
  s->push(func(a,n,x));
  checkGSLerror();
}

template <double (*func)(double, double, double)>
void realRealRealRealGSL(vm::stack *s) 
{
  double x=pop<double>(s);
  double n=pop<double>(s);
  double a=pop<double>(s);
  s->push(func(a,n,x));
  checkGSLerror();
}

template <double (*func)(double, unsigned)>
void realRealIntGSL(vm::stack *s) 
{
  Int n=pop<Int>(s);
  double x=pop<double>(s);
  s->push(func(x,unsignedcast(n)));
  checkGSLerror();
}

// Add a GSL special function from the GNU GSL library
template<double (*fcn)(double)>
void addGSLRealFunc(symbol *name, symbol *arg1=SYM(x))
{
  addFunc(GSLModule->e.ve, realRealGSL<fcn>, primReal(), name,
          formal(primReal(),arg1));
}

// Add a GSL_PREC_DOUBLE GSL special function.
template<double (*fcn)(double, gsl_mode_t)>
void addGSLDOUBLEFunc(symbol *name, symbol *arg1=SYM(x))
{
  addFunc(GSLModule->e.ve, realRealDOUBLE<fcn>, primReal(), name,
          formal(primReal(),arg1));
}

template<double (*fcn)(double, double, gsl_mode_t)>
void addGSLDOUBLE2Func(symbol *name, symbol *arg1=SYM(phi),
                       symbol *arg2=SYM(k))
{
  addFunc(GSLModule->e.ve, realRealRealDOUBLE<fcn>, primReal(), name, 
          formal(primReal(),arg1), formal(primReal(),arg2));
}

template <double (*func)(double, double, double, gsl_mode_t)>
void realRealRealRealDOUBLE(vm::stack *s) 
{
  double z=pop<double>(s);
  double y=pop<double>(s);
  double x=pop<double>(s);
  s->push(func(x,y,z,GSL_PREC_DOUBLE));
  checkGSLerror();
}

template<double (*fcn)(double, double, double, gsl_mode_t)>
void addGSLDOUBLE3Func(symbol *name, symbol *arg1, symbol *arg2,
                       symbol *arg3)
{
  addFunc(GSLModule->e.ve, realRealRealRealDOUBLE<fcn>, primReal(), name, 
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3));
}

template <double (*func)(double, double, double, double, gsl_mode_t)>
void realRealRealRealRealDOUBLE(vm::stack *s) 
{
  double z=pop<double>(s);
  double y=pop<double>(s);
  double x=pop<double>(s);
  double w=pop<double>(s);
  s->push(func(w,x,y,z,GSL_PREC_DOUBLE));
  checkGSLerror();
}

template<double (*fcn)(double, double, double, double, gsl_mode_t)>
void addGSLDOUBLE4Func(symbol *name, symbol *arg1, symbol *arg2,
                       symbol *arg3, symbol *arg4)
{
  addFunc(GSLModule->e.ve, realRealRealRealRealDOUBLE<fcn>, primReal(), name, 
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3), formal(primReal(), arg4));
}

template<double (*fcn)(unsigned)>
void addGSLIntFunc(symbol *name)
{
  addFunc(GSLModule->e.ve, realIntGSL<fcn>, primReal(), name,
          formal(primInt(),SYM(s)));
}

template <double (*func)(int)>
void realSignedGSL(vm::stack *s) 
{
  Int a = pop<Int>(s);
  s->push(func(intcast(a)));
  checkGSLerror();
}

template<double (*fcn)(int)>
void addGSLSignedFunc(symbol *name, symbol *arg1)
{
  addFunc(GSLModule->e.ve, realSignedGSL<fcn>, primReal(), name,
          formal(primInt(),arg1));
}

template<double (*fcn)(int, double)>
void addGSLIntRealFunc(symbol *name, symbol *arg1=SYM(n),
                       symbol *arg2=SYM(x))
{
  addFunc(GSLModule->e.ve, realIntRealGSL<fcn>, primReal(), name,
          formal(primInt(),arg1), formal(primReal(),arg2));
}

template<double (*fcn)(double, double)>
void addGSLRealRealFunc(symbol *name, symbol *arg1=SYM(nu),
                        symbol *arg2=SYM(x))
{
  addFunc(GSLModule->e.ve, realRealRealGSL<fcn>, primReal(), name,
          formal(primReal(),arg1), formal(primReal(),arg2));
}

template<double (*fcn)(double, double, double)>
void addGSLRealRealRealFunc(symbol *name, symbol *arg1,
                            symbol *arg2, symbol *arg3)
{
  addFunc(GSLModule->e.ve, realRealRealRealGSL<fcn>, primReal(), name,
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3));
}

template<int (*fcn)(double, double, double)>
void addGSLRealRealRealFuncInt(symbol *name, symbol *arg1,
                               symbol *arg2, symbol *arg3)
{
  addFunc(GSLModule->e.ve, intRealRealRealGSL<fcn>, primInt(), name,
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3));
}

template<double (*fcn)(double, unsigned)>
void addGSLRealIntFunc(symbol *name, symbol *arg1=SYM(nu),
                       symbol *arg2=SYM(s))
{
  addFunc(GSLModule->e.ve, realRealIntGSL<fcn>, primReal(), name, 
          formal(primReal(),arg1), formal(primInt(),arg2));
}

template<double (*func)(double, int)>
void realRealSignedGSL(vm::stack *s) 
{
  Int b = pop<Int>(s);
  double a = pop<double>(s);
  s->push(func(a, intcast(b)));
  checkGSLerror();
}

template<double (*fcn)(double, int)>
void addGSLRealSignedFunc(symbol *name, symbol *arg1, symbol *arg2)
{
  addFunc(GSLModule->e.ve, realRealSignedGSL<fcn>, primReal(), name, 
          formal(primReal(),arg1), formal(primInt(),arg2));
}

template<double (*func)(unsigned int, unsigned int)>
void realUnsignedUnsignedGSL(vm::stack *s) 
{
  Int b = pop<Int>(s);
  Int a = pop<Int>(s);
  s->push(func(unsignedcast(a), unsignedcast(b)));
  checkGSLerror();
}

template<double (*fcn)(unsigned int, unsigned int)>
void addGSLUnsignedUnsignedFunc(symbol *name, symbol *arg1,
                                symbol *arg2)
{
  addFunc(GSLModule->e.ve, realUnsignedUnsignedGSL<fcn>, primReal(), name, 
          formal(primInt(), arg1), formal(primInt(), arg2));
}

template<double (*func)(int, double, double)>
void realIntRealRealGSL(vm::stack *s) 
{
  double c = pop<double>(s);
  double b = pop<double>(s);
  Int a = pop<Int>(s);
  s->push(func(intcast(a), b, c));
  checkGSLerror();
}

template<double (*fcn)(int, double, double)>
void addGSLIntRealRealFunc(symbol *name, symbol *arg1,
                           symbol *arg2, symbol *arg3)
{
  addFunc(GSLModule->e.ve, realIntRealRealGSL<fcn>, primReal(), name, 
          formal(primInt(), arg1), formal(primReal(), arg2),
          formal(primReal(), arg3));
}

template<double (*func)(int, int, double)>
void realIntIntRealGSL(vm::stack *s) 
{
  double c = pop<double>(s);
  Int b = pop<Int>(s);
  Int a = pop<Int>(s);
  s->push(func(intcast(a), intcast(b), c));
  checkGSLerror();
}

template<double (*fcn)(int, int, double)>
void addGSLIntIntRealFunc(symbol *name, symbol *arg1, symbol *arg2,
                          symbol *arg3)
{
  addFunc(GSLModule->e.ve, realIntIntRealGSL<fcn>, primReal(), name, 
          formal(primInt(), arg1), formal(primInt(), arg2),
          formal(primReal(), arg3));
}

template<double (*func)(int, int, double, double)>
void realIntIntRealRealGSL(vm::stack *s) 
{
  double d = pop<double>(s);
  double c = pop<double>(s);
  Int b = pop<Int>(s);
  Int a = pop<Int>(s);
  s->push(func(intcast(a), intcast(b), c, d));
  checkGSLerror();
}

template<double (*fcn)(int, int, double, double)>
void addGSLIntIntRealRealFunc(symbol *name, symbol *arg1,
                              symbol *arg2, symbol *arg3,
                              symbol *arg4)
{
  addFunc(GSLModule->e.ve, realIntIntRealRealGSL<fcn>, primReal(), name, 
          formal(primInt(), arg1), formal(primInt(), arg2),
          formal(primReal(), arg3), formal(primReal(), arg4));
}

template<double (*func)(double, double, double, double)>
void realRealRealRealRealGSL(vm::stack *s) 
{
  double d = pop<double>(s);
  double c = pop<double>(s);
  double b = pop<double>(s);
  double a = pop<double>(s);
  s->push(func(a, b, c, d));
  checkGSLerror();
}

template<double (*fcn)(double, double, double, double)>
void addGSLRealRealRealRealFunc(symbol *name, symbol *arg1,
                                symbol *arg2, symbol *arg3,
                                symbol *arg4)
{
  addFunc(GSLModule->e.ve, realRealRealRealRealGSL<fcn>, primReal(), name, 
          formal(primReal(), arg1), formal(primReal(), arg2),
          formal(primReal(), arg3), formal(primReal(), arg4));
}

template<double (*func)(int, int, int, int, int, int)>
void realIntIntIntIntIntIntGSL(vm::stack *s) 
{
  Int f = pop<Int>(s);
  Int e = pop<Int>(s);
  Int d = pop<Int>(s);
  Int c = pop<Int>(s);
  Int b = pop<Int>(s);
  Int a = pop<Int>(s);
  s->push(func(intcast(a), intcast(b), intcast(c), intcast(d), intcast(e),
               intcast(f)));
  checkGSLerror();
}

template<double (*fcn)(int, int, int, int, int, int)>
void addGSLIntIntIntIntIntIntFunc(symbol *name, symbol *arg1,
                                  symbol *arg2, symbol *arg3,
                                  symbol *arg4, symbol *arg5,
                                  symbol *arg6)
{
  addFunc(GSLModule->e.ve, realIntIntIntIntIntIntGSL<fcn>, primReal(), name, 
          formal(primInt(), arg1), formal(primInt(), arg2),
          formal(primInt(), arg3), formal(primInt(), arg4),
          formal(primInt(), arg5), formal(primInt(), arg6));
}

template<double (*func)(int, int, int, int, int, int, int, int, int)>
void realIntIntIntIntIntIntIntIntIntGSL(vm::stack *s) 
{
  Int i = pop<Int>(s);
  Int h = pop<Int>(s);
  Int g = pop<Int>(s);
  Int f = pop<Int>(s);
  Int e = pop<Int>(s);
  Int d = pop<Int>(s);
  Int c = pop<Int>(s);
  Int b = pop<Int>(s);
  Int a = pop<Int>(s);
  s->push(func(intcast(a), intcast(b), intcast(c), intcast(d), intcast(e),
               intcast(f), intcast(g), intcast(h), intcast(i)));
  checkGSLerror();
}

template<double (*fcn)(int, int, int, int, int, int, int, int, int)>
void addGSLIntIntIntIntIntIntIntIntIntFunc(symbol *name, symbol *arg1,
                                           symbol *arg2, symbol *arg3,
                                           symbol *arg4, symbol *arg5,
                                           symbol *arg6, symbol *arg7,
                                           symbol *arg8, symbol *arg9)
{
  addFunc(GSLModule->e.ve, realIntIntIntIntIntIntIntIntIntGSL<fcn>, primReal(),
          name, formal(primInt(), arg1), formal(primInt(), arg2),
          formal(primInt(), arg3), formal(primInt(), arg4),
          formal(primInt(), arg5), formal(primInt(), arg6),
          formal(primInt(), arg7), formal(primInt(), arg8),
          formal(primInt(), arg9));
}

// Handle GSL errors gracefully.
void GSLerrorhandler(const char *reason, const char *, int, int) 
{
  if(!GSLerror) {
    vm::errornothrow(reason);
    GSLerror=true;
  }
}
#endif
  
void addInitializer(venv &ve, ty *t, access *a)
{
  addFunc(ve, a, t, symbol::initsym);
}

void addInitializer(venv &ve, ty *t, bltin f)
{
#ifdef DEBUG_BLTIN
  ostringstream s;
  s << "initializer for " << *t;
  REGISTER_BLTIN(f, s.str());
#endif
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
#ifdef DEBUG_BLTIN
  ostringstream s;
  s << "explicit cast from " << *source << " to " << *target;
  REGISTER_BLTIN(f, s.str());
#endif
  addExplicitCast(ve, target, source, new bltinAccess(f));
}

void addCast(venv &ve, ty *target, ty *source, bltin f) {
#ifdef DEBUG_BLTIN
  ostringstream s;
  s << "cast from " << *source << " to " << *target;
  REGISTER_BLTIN(f, s.str());
#endif
  addCast(ve, target, source, new bltinAccess(f));
}

template<class T>
void addVariable(venv &ve, T *ref, ty *t, symbol *name,
                 record *module=settings::getSettingsModule()) {
  access *a = new refAccess<T>(ref);
  varEntry *ent = new varEntry(t, a, PUBLIC, module, 0, position());
  ve.enter(name, ent);
}

template<class T>
void addVariable(venv &ve, T value, ty *t, symbol *name,
                 record *module=settings::getSettingsModule(),
                 permission perm=PUBLIC) {
  item* ref=new item;
  *ref=value;
  access *a = new itemRefAccess(ref);
  varEntry *ent = new varEntry(t, a, perm, module, 0, position());
  ve.enter(name, ent);
}

template<class T>
void addConstant(venv &ve, T value, ty *t, symbol *name,
                 record *module=settings::getSettingsModule()) {
  addVariable(ve,value,t,name,module,RESTRICTED);
}

// The identity access, i.e. no instructions are encoded for a cast or
// operation, and no functions are called.
identAccess id;

function *IntRealFunction()
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
  addInitializer(ve, primInt(), IntZero);
  addInitializer(ve, primReal(), realZero);

  addInitializer(ve, primString(), emptyString);
  addInitializer(ve, primPair(), pairZero);
  addInitializer(ve, primTriple(), tripleZero);
  addInitializer(ve, primTransform(), transformIdentity);
  addInitializer(ve, primGuide(), nullGuide);
  addInitializer(ve, primPath(), nullPath);
  addInitializer(ve, primPath3(), nullPath3);
  addInitializer(ve, primPen(), newPen);
  addInitializer(ve, primPicture(), newPicture);
  addInitializer(ve, primFile(), nullFile);
}

void addCasts(venv &ve)
{
  addExplicitCast(ve, primString(), primInt(), stringCast<Int>);
  addExplicitCast(ve, primString(), primReal(), stringCast<double>);
  addExplicitCast(ve, primString(), primPair(), stringCast<pair>);
  addExplicitCast(ve, primString(), primTriple(), stringCast<triple>);
  addExplicitCast(ve, primInt(), primString(), castString<Int>);
  addExplicitCast(ve, primReal(), primString(), castString<double>);
  addExplicitCast(ve, primPair(), primString(), castString<pair>);
  addExplicitCast(ve, primTriple(), primString(), castString<triple>);

  addExplicitCast(ve, primInt(), primReal(), castDoubleInt);

  addCast(ve, primReal(), primInt(), cast<Int,double>);
  addCast(ve, primPair(), primInt(), cast<Int,pair>);
  addCast(ve, primPair(), primReal(), cast<double,pair>);
  
  addCast(ve, primPath(), primPair(), cast<pair,path>);
  addCast(ve, primGuide(), primPair(), pairToGuide);
  addCast(ve, primGuide(), primPath(), pathToGuide);
  addCast(ve, primPath(), primGuide(), guideToPath);

  addCast(ve, primFile(), primNull(), nullFile);
  
  // Vectorized casts.
  addExplicitCast(ve, IntArray(), realArray(), arrayToArray<double,Int>);
  
  addCast(ve, realArray(), IntArray(), arrayToArray<Int,double>);
  addCast(ve, pairArray(), IntArray(), arrayToArray<Int,pair>);
  addCast(ve, pairArray(), realArray(), arrayToArray<double,pair>);
  
  addCast(ve, realArray2(), IntArray2(), array2ToArray2<Int,double>);
  addCast(ve, pairArray2(), IntArray2(), array2ToArray2<Int,pair>);
  addCast(ve, pairArray2(), realArray2(), array2ToArray2<double,pair>);
}

void addGuideOperators(venv &ve)
{
  // The guide operators .. and -- take an array of guides, and turn them
  // into a single guide.
  addRestFunc(ve, dotsGuide, primGuide(), SYM_DOTS, guideArray());
  addRestFunc(ve, dashesGuide, primGuide(), SYM_DASHES, guideArray());
}

/* Avoid typing the same type three times. */
void addSimpleOperator(venv &ve, bltin f, ty *t, symbol *name)
{
  addFunc(ve,f,t,name,formal(t,SYM(a)),formal(t,SYM(b)));
}
void addBooleanOperator(venv &ve, bltin f, ty *t, symbol *name)
{
  addFunc(ve,f,primBoolean(),name,formal(t,SYM(a)),formal(t,SYM(b)));
}

template<class T, template <class S> class op>
void addArray2Array2Op(venv &ve, ty *t3, symbol *name)
{
  addFunc(ve,array2Array2Op<T,op>,t3,name,formal(t3,SYM(a)),formal(t3,SYM(b)));
}

template<class T, template <class S> class op>
void addOpArray2(venv &ve, ty *t1, symbol *name, ty *t3)
{
  addFunc(ve,opArray2<T,T,op>,t3,name,formal(t1,SYM(a)),formal(t3,SYM(b)));
}

template<class T, template <class S> class op>
void addArray2Op(venv &ve, ty *t1, symbol *name, ty *t3)
{
  addFunc(ve,array2Op<T,T,op>,t3,name,formal(t3,SYM(a)),formal(t1,SYM(b)));
}

template<class T, template <class S> class op>
void addOps(venv &ve, ty *t1, symbol *name, ty *t2)
{
  addSimpleOperator(ve,binaryOp<T,op>,t1,name);
  addFunc(ve,opArray<T,T,op>,t2,name,formal(t1,SYM(a)),formal(t2,SYM(b)));
  addFunc(ve,arrayOp<T,T,op>,t2,name,formal(t2,SYM(a)),formal(t1,SYM(b)));
  addSimpleOperator(ve,arrayArrayOp<T,op>,t2,name);
}

template<class T, template <class S> class op>
void addBooleanOps(venv &ve, ty *t1, symbol *name, ty *t2)
{
  addBooleanOperator(ve,binaryOp<T,op>,t1,name);
  addFunc(ve,opArray<T,T,op>,
      booleanArray(),name,formal(t1,SYM(a)),formal(t2,SYM(b)));
  addFunc(ve,arrayOp<T,T,op>,
      booleanArray(),name,formal(t2,SYM(a)),formal(t1,SYM(b)));
  addFunc(ve,arrayArrayOp<T,op>,booleanArray(),name,formal(t2,SYM(a)),
          formal(t2,SYM(b)));
}

void addWrite(venv &ve, bltin f, ty *t1, ty *t2)
{
  addRestFunc(ve,f,primVoid(),SYM(write),t2,
              formal(primFile(),SYM(file),true),formal(primString(),SYM(s),true),
              formal(t1,SYM(x)),formal(voidFileFunction(),SYM(suffix),true));
}

template<class T>
void addUnorderedOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addBooleanOps<T,equals>(ve,t1,SYM_EQ,t2);
  addBooleanOps<T,notequals>(ve,t1,SYM_NEQ,t2);
   
  addFunc(ve, run::array2Equals<T>, primBoolean(), SYM_EQ, formal(t3, SYM(a)),
          formal(t3, SYM(b)));
  addFunc(ve, run::array2NotEquals<T>, primBoolean(), SYM_NEQ, formal(t3, SYM(a)),
          formal(t3, SYM(b)));
  
  addCast(ve,t1,primFile(),read<T>);
  addCast(ve,t2,primFile(),readArray<T>);
  addCast(ve,t3,primFile(),readArray<T>);
  addCast(ve,t4,primFile(),readArray<T>);
  
  addWrite(ve,write<T>,t1,t2);
  addRestFunc(ve,writeArray<T>,primVoid(),SYM(write),t3,
              formal(primFile(),SYM(file),true),formal(primString(),SYM(s),true),
              formal(t2,SYM(a),false,true));
  addFunc(ve,writeArray2<T>,primVoid(),SYM(write),
          formal(primFile(),SYM(file),true),t3);
  addFunc(ve,writeArray3<T>,primVoid(),SYM(write),
          formal(primFile(),SYM(file),true),t4);
}

inline double abs(pair z) {
  return z.length();
}

inline double abs(triple v) {
  return v.length();
}

inline pair conjugate(pair z) {
  return conj(z);
}

template<class T>
inline T negate(T x) {
  return -x;
}

template<class T, template <class S> class op>
void addBinOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, symbol *name)
{
  addFunc(ve,binopArray<T,op>,t1,name,formal(t2,SYM(a)));
  addFunc(ve,binopArray2<T,op>,t1,name,formal(t3,SYM(a)));
  addFunc(ve,binopArray3<T,op>,t1,name,formal(t4,SYM(a)));
}

template<class T>
void addOrderedOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addBooleanOps<T,less>(ve,t1,SYM_LT,t2);
  addBooleanOps<T,lessequals>(ve,t1,SYM_LE,t2);
  addBooleanOps<T,greaterequals>(ve,t1,SYM_GE,t2);
  addBooleanOps<T,greater>(ve,t1,SYM_GT,t2);
  
  addOps<T,run::min>(ve,t1,SYM(min),t2);
  addOps<T,run::max>(ve,t1,SYM(max),t2);
  addBinOps<T,run::min>(ve,t1,t2,t3,t4,SYM(min));
  addBinOps<T,run::max>(ve,t1,t2,t3,t4,SYM(max));
    
  addFunc(ve,sortArray<T>,t2,SYM(sort),formal(t2,SYM(a)));
  addFunc(ve,sortArray2<T>,t3,SYM(sort),formal(t3,SYM(a)));
  
  addFunc(ve,searchArray<T>,primInt(),SYM(search),formal(t2,SYM(a)),
          formal(t1,SYM(key)));
}

template<class T>
void addBasicOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, bool integer=false,
                 bool Explicit=false)
{
  addOps<T,plus>(ve,t1,SYM_PLUS,t2);
  addOps<T,minus>(ve,t1,SYM_MINUS,t2);
  
  addArray2Array2Op<T,plus>(ve,t3,SYM_PLUS);
  addArray2Array2Op<T,minus>(ve,t3,SYM_MINUS);

  addFunc(ve,&id,t1,SYM_PLUS,formal(t1,SYM(a)));
  addFunc(ve,&id,t2,SYM_PLUS,formal(t2,SYM(a)));
  addFunc(ve,Negate<T>,t1,SYM_MINUS,formal(t1,SYM(a)));
  addFunc(ve,arrayFunc<T,T,negate>,t2,SYM_MINUS,formal(t2,SYM(a)));
  addFunc(ve,arrayFunc2<T,T,negate>,t3,SYM_MINUS,formal(t3,SYM(a)));
  if(!integer) addFunc(ve,interp<T>,t1,SYM(interp),formal(t1,SYM(a),false,Explicit),
                       formal(t1,SYM(b),false,Explicit),
                       formal(primReal(),SYM(t)));
  
  addFunc(ve,sumArray<T>,t1,SYM(sum),formal(t2,SYM(a)));
  addUnorderedOps<T>(ve,t1,t2,t3,t4);
}

template<class T>
void addOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, bool integer=false,
            bool Explicit=false)
{
  addBasicOps<T>(ve,t1,t2,t3,t4,integer,Explicit);
  
  addOps<T,times>(ve,t1,SYM_TIMES,t2);
  addOpArray2<T,times>(ve,t1,SYM_TIMES,t3);
  addArray2Op<T,times>(ve,t1,SYM_TIMES,t3);
  
  if(!integer) {
    addOps<T,run::divide>(ve,t1,SYM_DIVIDE,t2);
    addArray2Op<T,run::divide>(ve,t1,SYM_DIVIDE,t3);
  }
      
  addOps<T,power>(ve,t1,SYM_CARET,t2);
}


// Adds standard functions for a newly added array type.
void addArrayOps(venv &ve, types::array *t)
{
  ty *ct = t->celltype;
  
  addFunc(ve, run::arrayAlias,
          primBoolean(), SYM(alias), formal(t, SYM(a)), formal(t, SYM(b)));

  addFunc(ve, run::newDuplicateArray,
          t, SYM(array), formal(primInt(), SYM(n)),
          formal(ct, SYM(value)),
          formal(primInt(), SYM(depth), /*optional=*/ true));

  switch (t->depth()) {
    case 1:
      addFunc(ve, run::arrayCopy, t, SYM(copy), formal(t, SYM(a)));
      addRestFunc(ve, run::arrayConcat, t, SYM(concat), new types::array(t));
      addFunc(ve, run::arraySequence,
              t, SYM(sequence), formal(new function(ct, primInt()), SYM(f)),
              formal(primInt(), SYM(n)));
      addFunc(ve, run::arrayFunction,
              t, SYM(map), formal(new function(ct, ct), SYM(f)), formal(t, SYM(a)));
      addFunc(ve, run::arraySort,
              t, SYM(sort), formal(t, SYM(a)),
              formal(new function(primBoolean(), ct, ct), SYM(f)));
      break;
    case 2:
      addFunc(ve, run::array2Copy, t, SYM(copy), formal(t, SYM(a)));
      addFunc(ve, run::array2Transpose, t, SYM(transpose), formal(t, SYM(a)));
      break;
    case 3:
      addFunc(ve, run::array3Copy, t, SYM(copy), formal(t, SYM(a)));
      addFunc(ve, run::array3Transpose, t, SYM(transpose), formal(t, SYM(a)),
              formal(IntArray(),SYM(perm)));
      break;
    default:
      break;
  }
}

void addRecordOps(venv &ve, record *r)
{
  addFunc(ve, run::boolMemEq, primBoolean(), SYM(alias), formal(r, SYM(a)),
          formal(r, SYM(b)));
  addFunc(ve, run::boolMemEq, primBoolean(), SYM_EQ, formal(r, SYM(a)),
          formal(r, SYM(b)));
  addFunc(ve, run::boolMemNeq, primBoolean(), SYM_NEQ, formal(r, SYM(a)),
          formal(r, SYM(b)));
}

void addFunctionOps(venv &ve, function *f)
{
  addFunc(ve, run::boolFuncEq, primBoolean(), SYM_EQ, formal(f, SYM(a)),
          formal(f, SYM(b)));
  addFunc(ve, run::boolFuncNeq, primBoolean(), SYM_NEQ, formal(f, SYM(a)),
          formal(f, SYM(b)));
}

void addOperators(venv &ve) 
{
  addSimpleOperator(ve,binaryOp<string,plus>,primString(),SYM_PLUS);
  
  addBooleanOps<bool,And>(ve,primBoolean(),SYM_AMPERSAND,booleanArray());
  addBooleanOps<bool,Or>(ve,primBoolean(),SYM_BAR,booleanArray());
  addBooleanOps<bool,Xor>(ve,primBoolean(),SYM_CARET,booleanArray());
  
  addUnorderedOps<bool>(ve,primBoolean(),booleanArray(),booleanArray2(),
                        booleanArray3());
  addOps<Int>(ve,primInt(),IntArray(),IntArray2(),IntArray3(),true);
  addOps<double>(ve,primReal(),realArray(),realArray2(),realArray3());
  addOps<pair>(ve,primPair(),pairArray(),pairArray2(),pairArray3(),false,true);
  addBasicOps<triple>(ve,primTriple(),tripleArray(),tripleArray2(),
                      tripleArray3());
  addFunc(ve,opArray<double,triple,times>,tripleArray(),SYM_TIMES,
          formal(primReal(),SYM(a)),formal(tripleArray(),SYM(b)));
  addFunc(ve,opArray2<double,triple,timesR>,tripleArray2(),SYM_TIMES,
          formal(primReal(),SYM(a)),formal(tripleArray2(),SYM(b)));
  addFunc(ve,arrayOp<triple,double,timesR>,tripleArray(),SYM_TIMES,
          formal(tripleArray(),SYM(a)),formal(primReal(),SYM(b)));
  addFunc(ve,array2Op<triple,double,timesR>,tripleArray2(),SYM_TIMES,
          formal(tripleArray2(),SYM(a)),formal(primReal(),SYM(b)));
  addFunc(ve,arrayOp<triple,double,divide>,tripleArray(),SYM_DIVIDE,
          formal(tripleArray(),SYM(a)),formal(primReal(),SYM(b)));

  addUnorderedOps<string>(ve,primString(),stringArray(),stringArray2(),
                          stringArray3());
  
  addSimpleOperator(ve,binaryOp<pair,minbound>,primPair(),SYM(minbound));
  addSimpleOperator(ve,binaryOp<pair,maxbound>,primPair(),SYM(maxbound));
  addSimpleOperator(ve,binaryOp<triple,minbound>,primTriple(),SYM(minbound));
  addSimpleOperator(ve,binaryOp<triple,maxbound>,primTriple(),SYM(maxbound));
  addBinOps<pair,minbound>(ve,primPair(),pairArray(),pairArray2(),pairArray3(),
                           SYM(minbound));
  addBinOps<pair,maxbound>(ve,primPair(),pairArray(),pairArray2(),pairArray3(),
                           SYM(maxbound));
  addBinOps<triple,minbound>(ve,primTriple(),tripleArray(),tripleArray2(),
                             tripleArray3(),SYM(minbound));
  addBinOps<triple,maxbound>(ve,primTriple(),tripleArray(),tripleArray2(),
                             tripleArray3(),SYM(maxbound));
  
  addFunc(ve,arrayFunc<double,pair,abs>,realArray(),SYM(abs),
          formal(pairArray(),SYM(a)));
  addFunc(ve,arrayFunc<double,triple,abs>,realArray(),SYM(abs),
          formal(tripleArray(),SYM(a)));
  
  addFunc(ve,arrayFunc<pair,pair,conjugate>,pairArray(),SYM(conj),
          formal(pairArray(),SYM(a)));
  addFunc(ve,arrayFunc2<pair,pair,conjugate>,pairArray2(),SYM(conj),
          formal(pairArray2(),SYM(a)));
  
  addFunc(ve,binaryOp<Int,divide>,primReal(),SYM_DIVIDE,
          formal(primInt(),SYM(a)),formal(primInt(),SYM(b)));
  addFunc(ve,arrayOp<Int,Int,divide>,realArray(),SYM_DIVIDE,
          formal(IntArray(),SYM(a)),formal(primInt(),SYM(b)));
  addFunc(ve,opArray<Int,Int,divide>,realArray(),SYM_DIVIDE,
          formal(primInt(),SYM(a)),formal(IntArray(),SYM(b)));
  addFunc(ve,arrayArrayOp<Int,divide>,realArray(),SYM_DIVIDE,
          formal(IntArray(),SYM(a)),formal(IntArray(),SYM(b)));
  
  addOrderedOps<Int>(ve,primInt(),IntArray(),IntArray2(),IntArray3());
  addOrderedOps<double>(ve,primReal(),realArray(),realArray2(),realArray3());
  addOrderedOps<string>(ve,primString(),stringArray(),stringArray2(),
                        stringArray3());
  
  addOps<Int,mod>(ve,primInt(),SYM_MOD,IntArray());
  addOps<double,mod>(ve,primReal(),SYM_MOD,realArray());
  
  addRestFunc(ve,diagonal<Int>,IntArray2(),SYM(diagonal),IntArray());
  addRestFunc(ve,diagonal<double>,realArray2(),SYM(diagonal),realArray());
  addRestFunc(ve,diagonal<pair>,pairArray2(),SYM(diagonal),pairArray());
}

dummyRecord *createDummyRecord(venv &ve, symbol *name)
{
  dummyRecord *r=new dummyRecord(name);
#ifdef DEBUG_FRAME
  vm::frame *f = new vm::frame("dummy record " + string(name), 0);
#else
  vm::frame *f = new vm::frame(0);
#endif
  addConstant(ve, f, r, name);
  addRecordOps(ve, r);
  return r;
}

double identity(double x) {return x;}
double pow10(double x) {return run::pow(10.0,x);}

// An example of an open function.
#ifdef OPENFUNCEXAMPLE
void openFunc(stack *Stack)
{
  vm::array *a=vm::pop<vm::array *>(Stack);
  size_t numArgs=checkArray(a);
  for (size_t k=0; k<numArgs; ++k)
    cout << k << ": " << (*a)[k];
  
  Stack->push<Int>((Int)numArgs);
}
#endif

// NOTE: We should move all of these into a "builtin" module.
void base_venv(venv &ve)
{
  addInitializers(ve);
  addCasts(ve);
  addOperators(ve);
  addGuideOperators(ve);
  
  addRealFunc(sin,SYM(sin));
  addRealFunc(cos,SYM(cos));
  addRealFunc(tan,SYM(tan));
  addRealFunc(asin,SYM(asin));
  addRealFunc(acos,SYM(acos));
  addRealFunc(atan,SYM(atan));
  addRealFunc(exp,SYM(exp));
  addRealFunc(log,SYM(log));
  addRealFunc(log10,SYM(log10));
  addRealFunc(sinh,SYM(sinh));
  addRealFunc(cosh,SYM(cosh));
  addRealFunc(tanh,SYM(tanh));
  addRealFunc(asinh,SYM(asinh));
  addRealFunc(acosh,SYM(acosh));
  addRealFunc(atanh,SYM(atanh));
  addRealFunc(sqrt,SYM(sqrt));
  addRealFunc(cbrt,SYM(cbrt));
  addRealFunc(fabs,SYM(fabs));
  addRealFunc<fabs>(ve,SYM(abs));
  addRealFunc(expm1,SYM(expm1));
  addRealFunc(log1p,SYM(log1p));
  addRealIntFunc<ldexp>(ve, SYM(ldexp), SYM(x), SYM(e));

  addRealFunc(pow10,SYM(pow10));
  addRealFunc(identity,SYM(identity));
  
#ifdef HAVE_LIBGSL  
  GSLModule=new dummyRecord(SYM(gsl));
  gsl_set_error_handler(GSLerrorhandler);
  
  // Common functions
  addGSLRealRealFunc<gsl_hypot>(SYM(hypot),SYM(x),SYM(y));
//  addGSLRealRealRealFunc<gsl_hypot3>(SYM(hypot),SYM(x),SYM(y),SYM(z));
  addGSLRealRealRealFuncInt<gsl_fcmp>(SYM(fcmp),SYM(x),SYM(y),SYM(epsilon));
  
  // Airy functions
  addGSLDOUBLEFunc<gsl_sf_airy_Ai>(SYM(Ai));
  addGSLDOUBLEFunc<gsl_sf_airy_Bi>(SYM(Bi));
  addGSLDOUBLEFunc<gsl_sf_airy_Ai_scaled>(SYM(Ai_scaled));
  addGSLDOUBLEFunc<gsl_sf_airy_Bi_scaled>(SYM(Bi_scaled));
  addGSLDOUBLEFunc<gsl_sf_airy_Ai_deriv>(SYM(Ai_deriv));
  addGSLDOUBLEFunc<gsl_sf_airy_Bi_deriv>(SYM(Bi_deriv));
  addGSLDOUBLEFunc<gsl_sf_airy_Ai_deriv_scaled>(SYM(Ai_deriv_scaled));
  addGSLDOUBLEFunc<gsl_sf_airy_Bi_deriv_scaled>(SYM(Bi_deriv_scaled));
  addGSLIntFunc<gsl_sf_airy_zero_Ai>(SYM(zero_Ai));
  addGSLIntFunc<gsl_sf_airy_zero_Bi>(SYM(zero_Bi));
  addGSLIntFunc<gsl_sf_airy_zero_Ai_deriv>(SYM(zero_Ai_deriv));
  addGSLIntFunc<gsl_sf_airy_zero_Bi_deriv>(SYM(zero_Bi_deriv));
  
  // Bessel functions
  addGSLRealFunc<gsl_sf_bessel_J0>(SYM(J0));
  addGSLRealFunc<gsl_sf_bessel_J1>(SYM(J1));
  addGSLIntRealFunc<gsl_sf_bessel_Jn>(SYM(Jn));
  addGSLRealFunc<gsl_sf_bessel_Y0>(SYM(Y0));
  addGSLRealFunc<gsl_sf_bessel_Y1>(SYM(Y1));
  addGSLIntRealFunc<gsl_sf_bessel_Yn>(SYM(Yn));
  addGSLRealFunc<gsl_sf_bessel_I0>(SYM(I0));
  addGSLRealFunc<gsl_sf_bessel_I1>(SYM(I1));
  addGSLIntRealFunc<gsl_sf_bessel_In>(SYM(I));
  addGSLRealFunc<gsl_sf_bessel_I0_scaled>(SYM(I0_scaled));
  addGSLRealFunc<gsl_sf_bessel_I1_scaled>(SYM(I1_scaled));
  addGSLIntRealFunc<gsl_sf_bessel_In_scaled>(SYM(I_scaled));
  addGSLRealFunc<gsl_sf_bessel_K0>(SYM(K0));
  addGSLRealFunc<gsl_sf_bessel_K1>(SYM(K1));
  addGSLIntRealFunc<gsl_sf_bessel_Kn>(SYM(K));
  addGSLRealFunc<gsl_sf_bessel_K0_scaled>(SYM(K0_scaled));
  addGSLRealFunc<gsl_sf_bessel_K1_scaled>(SYM(K1_scaled));
  addGSLIntRealFunc<gsl_sf_bessel_Kn_scaled>(SYM(K_scaled));
  addGSLRealFunc<gsl_sf_bessel_j0>(SYM(j0));
  addGSLRealFunc<gsl_sf_bessel_j1>(SYM(j1));
  addGSLRealFunc<gsl_sf_bessel_j2>(SYM(j2));
  addGSLIntRealFunc<gsl_sf_bessel_jl>(SYM(j),SYM(l));
  addGSLRealFunc<gsl_sf_bessel_y0>(SYM(y0));
  addGSLRealFunc<gsl_sf_bessel_y1>(SYM(y1));
  addGSLRealFunc<gsl_sf_bessel_y2>(SYM(y2));
  addGSLIntRealFunc<gsl_sf_bessel_yl>(SYM(y),SYM(l));
  addGSLRealFunc<gsl_sf_bessel_i0_scaled>(SYM(i0_scaled));
  addGSLRealFunc<gsl_sf_bessel_i1_scaled>(SYM(i1_scaled));
  addGSLRealFunc<gsl_sf_bessel_i2_scaled>(SYM(i2_scaled));
  addGSLIntRealFunc<gsl_sf_bessel_il_scaled>(SYM(i_scaled),SYM(l));
  addGSLRealFunc<gsl_sf_bessel_k0_scaled>(SYM(k0_scaled));
  addGSLRealFunc<gsl_sf_bessel_k1_scaled>(SYM(k1_scaled));
  addGSLRealFunc<gsl_sf_bessel_k2_scaled>(SYM(k2_scaled));
  addGSLIntRealFunc<gsl_sf_bessel_kl_scaled>(SYM(k_scaled),SYM(l));
  addGSLRealRealFunc<gsl_sf_bessel_Jnu>(SYM(J));
  addGSLRealRealFunc<gsl_sf_bessel_Ynu>(SYM(Y));
  addGSLRealRealFunc<gsl_sf_bessel_Inu>(SYM(I));
  addGSLRealRealFunc<gsl_sf_bessel_Inu_scaled>(SYM(I_scaled));
  addGSLRealRealFunc<gsl_sf_bessel_Knu>(SYM(K));
  addGSLRealRealFunc<gsl_sf_bessel_lnKnu>(SYM(lnK));
  addGSLRealRealFunc<gsl_sf_bessel_Knu_scaled>(SYM(K_scaled));
  addGSLIntFunc<gsl_sf_bessel_zero_J0>(SYM(zero_J0));
  addGSLIntFunc<gsl_sf_bessel_zero_J1>(SYM(zero_J1));
  addGSLRealIntFunc<gsl_sf_bessel_zero_Jnu>(SYM(zero_J));
  
  // Clausen functions
  addGSLRealFunc<gsl_sf_clausen>(SYM(clausen));
  
  // Coulomb functions
  addGSLRealRealFunc<gsl_sf_hydrogenicR_1>(SYM(hydrogenicR_1),SYM(Z),SYM(r));
  addGSLIntIntRealRealFunc<gsl_sf_hydrogenicR>(SYM(hydrogenicR),SYM(n),SYM(l),SYM(Z),
                                               SYM(r));
  // Missing: F_L(eta,x), G_L(eta,x), C_L(eta)
  
  // Coupling coefficients
  addGSLIntIntIntIntIntIntFunc<gsl_sf_coupling_3j>(SYM(coupling_3j),SYM(two_ja),
                                                   SYM(two_jb),SYM(two_jc),SYM(two_ma),
                                                   SYM(two_mb),SYM(two_mc));
  addGSLIntIntIntIntIntIntFunc<gsl_sf_coupling_6j>(SYM(coupling_6j),SYM(two_ja),
                                                   SYM(two_jb),SYM(two_jc),SYM(two_jd),
                                                   SYM(two_je),SYM(two_jf));
  addGSLIntIntIntIntIntIntIntIntIntFunc<gsl_sf_coupling_9j>(SYM(coupling_9j),
                                                            SYM(two_ja),SYM(two_jb),
                                                            SYM(two_jc),SYM(two_jd),
                                                            SYM(two_je),SYM(two_jf),
                                                            SYM(two_jg),SYM(two_jh),
                                                            SYM(two_ji));
  // Dawson function
  addGSLRealFunc<gsl_sf_dawson>(SYM(dawson));
  
  // Debye functions
  addGSLRealFunc<gsl_sf_debye_1>(SYM(debye_1));
  addGSLRealFunc<gsl_sf_debye_2>(SYM(debye_2));
  addGSLRealFunc<gsl_sf_debye_3>(SYM(debye_3));
  addGSLRealFunc<gsl_sf_debye_4>(SYM(debye_4));
  addGSLRealFunc<gsl_sf_debye_5>(SYM(debye_5));
  addGSLRealFunc<gsl_sf_debye_6>(SYM(debye_6));
  
  // Dilogarithm
  addGSLRealFunc<gsl_sf_dilog>(SYM(dilog));
  // Missing: complex dilogarithm
  
  // Elementary operations
  // we don't support errors at the moment
  
  // Elliptic integrals
  addGSLDOUBLEFunc<gsl_sf_ellint_Kcomp>(SYM(K),SYM(k));
  addGSLDOUBLEFunc<gsl_sf_ellint_Ecomp>(SYM(E),SYM(k));
  addGSLDOUBLE2Func<gsl_sf_ellint_Pcomp>(SYM(P),SYM(k),SYM(n));
  addGSLDOUBLE2Func<gsl_sf_ellint_F>(SYM(F));
  addGSLDOUBLE2Func<gsl_sf_ellint_E>(SYM(E));
  addGSLDOUBLE3Func<gsl_sf_ellint_P>(SYM(P),SYM(phi),SYM(k),SYM(n));
  addGSLDOUBLE3Func<gsl_sf_ellint_D>(SYM(D),SYM(phi),SYM(k),SYM(n));
  addGSLDOUBLE2Func<gsl_sf_ellint_RC>(SYM(RC),SYM(x),SYM(y));
  addGSLDOUBLE3Func<gsl_sf_ellint_RD>(SYM(RD),SYM(x),SYM(y),SYM(z));
  addGSLDOUBLE3Func<gsl_sf_ellint_RF>(SYM(RF),SYM(x),SYM(y),SYM(z));
  addGSLDOUBLE4Func<gsl_sf_ellint_RJ>(SYM(RJ),SYM(x),SYM(y),SYM(z),SYM(p));
  
  // Elliptic functions (Jacobi)
  // to be implemented
  
  // Error functions
  addGSLRealFunc<gsl_sf_erf>(SYM(erf));
  addGSLRealFunc<gsl_sf_erfc>(SYM(erfc));
  addGSLRealFunc<gsl_sf_log_erfc>(SYM(log_erfc));
  addGSLRealFunc<gsl_sf_erf_Z>(SYM(erf_Z));
  addGSLRealFunc<gsl_sf_erf_Q>(SYM(erf_Q));
  addGSLRealFunc<gsl_sf_hazard>(SYM(hazard));
  
  // Exponential functions
  addGSLRealRealFunc<gsl_sf_exp_mult>(SYM(exp_mult),SYM(x),SYM(y));
//  addGSLRealFunc<gsl_sf_expm1>(SYM(expm1));
  addGSLRealFunc<gsl_sf_exprel>(SYM(exprel));
  addGSLRealFunc<gsl_sf_exprel_2>(SYM(exprel_2));
  addGSLIntRealFunc<gsl_sf_exprel_n>(SYM(exprel),SYM(n),SYM(x));
  
  // Exponential integrals
  addGSLRealFunc<gsl_sf_expint_E1>(SYM(E1));
  addGSLRealFunc<gsl_sf_expint_E2>(SYM(E2));
//  addGSLIntRealFunc<gsl_sf_expint_En>(SYM(En),SYM(n),SYM(x));
  addGSLRealFunc<gsl_sf_expint_Ei>(SYM(Ei));
  addGSLRealFunc<gsl_sf_Shi>(SYM(Shi));
  addGSLRealFunc<gsl_sf_Chi>(SYM(Chi));
  addGSLRealFunc<gsl_sf_expint_3>(SYM(Ei3));
  addGSLRealFunc<gsl_sf_Si>(SYM(Si));
  addGSLRealFunc<gsl_sf_Ci>(SYM(Ci));
  addGSLRealFunc<gsl_sf_atanint>(SYM(atanint));
  
  // Fermi--Dirac function
  addGSLRealFunc<gsl_sf_fermi_dirac_m1>(SYM(FermiDiracM1));
  addGSLRealFunc<gsl_sf_fermi_dirac_0>(SYM(FermiDirac0));
  addGSLRealFunc<gsl_sf_fermi_dirac_1>(SYM(FermiDirac1));
  addGSLRealFunc<gsl_sf_fermi_dirac_2>(SYM(FermiDirac2));
  addGSLIntRealFunc<gsl_sf_fermi_dirac_int>(SYM(FermiDirac),SYM(j),SYM(x));
  addGSLRealFunc<gsl_sf_fermi_dirac_mhalf>(SYM(FermiDiracMHalf));
  addGSLRealFunc<gsl_sf_fermi_dirac_half>(SYM(FermiDiracHalf));
  addGSLRealFunc<gsl_sf_fermi_dirac_3half>(SYM(FermiDirac3Half));
  addGSLRealRealFunc<gsl_sf_fermi_dirac_inc_0>(SYM(FermiDiracInc0),SYM(x),SYM(b));
  
  // Gamma and beta functions
  addGSLRealFunc<gsl_sf_gamma>(SYM(gamma));
  addGSLRealFunc<gsl_sf_lngamma>(SYM(lngamma));
  addGSLRealFunc<gsl_sf_gammastar>(SYM(gammastar));
  addGSLRealFunc<gsl_sf_gammainv>(SYM(gammainv));
  addGSLIntFunc<gsl_sf_fact>(SYM(fact));
  addGSLIntFunc<gsl_sf_doublefact>(SYM(doublefact));
  addGSLIntFunc<gsl_sf_lnfact>(SYM(lnfact));
  addGSLIntFunc<gsl_sf_lndoublefact>(SYM(lndoublefact));
  addGSLUnsignedUnsignedFunc<gsl_sf_choose>(SYM(choose),SYM(n),SYM(m));
  addGSLUnsignedUnsignedFunc<gsl_sf_lnchoose>(SYM(lnchoose),SYM(n),SYM(m));
  addGSLIntRealFunc<gsl_sf_taylorcoeff>(SYM(taylorcoeff),SYM(n),SYM(x));
  addGSLRealRealFunc<gsl_sf_poch>(SYM(poch),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_lnpoch>(SYM(lnpoch),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_pochrel>(SYM(pochrel),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_gamma_inc>(SYM(gamma),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_gamma_inc_Q>(SYM(gamma_Q),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_gamma_inc_P>(SYM(gamma_P),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_beta>(SYM(beta),SYM(a),SYM(b));
  addGSLRealRealFunc<gsl_sf_lnbeta>(SYM(lnbeta),SYM(a),SYM(b));
  addGSLRealRealRealFunc<gsl_sf_beta_inc>(SYM(beta),SYM(a),SYM(b),SYM(x));
  
  // Gegenbauer functions
  addGSLRealRealFunc<gsl_sf_gegenpoly_1>(SYM(gegenpoly_1),SYM(lambda),SYM(x));
  addGSLRealRealFunc<gsl_sf_gegenpoly_2>(SYM(gegenpoly_2),SYM(lambda),SYM(x));
  addGSLRealRealFunc<gsl_sf_gegenpoly_3>(SYM(gegenpoly_3),SYM(lambda),SYM(x));
  addGSLIntRealRealFunc<gsl_sf_gegenpoly_n>(SYM(gegenpoly),SYM(n),SYM(lambda),SYM(x));
  
  // Hypergeometric functions
  addGSLRealRealFunc<gsl_sf_hyperg_0F1>(SYM(hy0F1),SYM(c),SYM(x));
  addGSLIntIntRealFunc<gsl_sf_hyperg_1F1_int>(SYM(hy1F1),SYM(m),SYM(n),SYM(x));
  addGSLRealRealRealFunc<gsl_sf_hyperg_1F1>(SYM(hy1F1),SYM(a),SYM(b),SYM(x));
  addGSLIntIntRealFunc<gsl_sf_hyperg_U_int>(SYM(U),SYM(m),SYM(n),SYM(x));
  addGSLRealRealRealFunc<gsl_sf_hyperg_U>(SYM(U),SYM(a),SYM(b),SYM(x));
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1>(SYM(hy2F1),SYM(a),SYM(b),SYM(c),SYM(x));
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1_conj>(SYM(hy2F1_conj),SYM(aR),SYM(aI),SYM(c),
                                                     SYM(x));
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1_renorm>(SYM(hy2F1_renorm),SYM(a),SYM(b),
                                                       SYM(c),SYM(x));
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1_conj_renorm>(SYM(hy2F1_conj_renorm),
                                                            SYM(aR),SYM(aI),SYM(c),SYM(x));
  addGSLRealRealRealFunc<gsl_sf_hyperg_2F0>(SYM(hy2F0),SYM(a),SYM(b),SYM(x));
  
  // Laguerre functions
  addGSLRealRealFunc<gsl_sf_laguerre_1>(SYM(L1),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_laguerre_2>(SYM(L2),SYM(a),SYM(x));
  addGSLRealRealFunc<gsl_sf_laguerre_3>(SYM(L3),SYM(a),SYM(x));
  addGSLIntRealRealFunc<gsl_sf_laguerre_n>(SYM(L),SYM(n),SYM(a),SYM(x));
  
  // Lambert W functions
  addGSLRealFunc<gsl_sf_lambert_W0>(SYM(W0));
  addGSLRealFunc<gsl_sf_lambert_Wm1>(SYM(Wm1));
  
  // Legendre functions and spherical harmonics
  addGSLRealFunc<gsl_sf_legendre_P1>(SYM(P1));
  addGSLRealFunc<gsl_sf_legendre_P2>(SYM(P2));
  addGSLRealFunc<gsl_sf_legendre_P3>(SYM(P3));
  addGSLIntRealFunc<gsl_sf_legendre_Pl>(SYM(Pl),SYM(l));
  addGSLRealFunc<gsl_sf_legendre_Q0>(SYM(Q0));
  addGSLRealFunc<gsl_sf_legendre_Q1>(SYM(Q1));
  addGSLIntRealFunc<gsl_sf_legendre_Ql>(SYM(Ql),SYM(l));
  addGSLIntIntRealFunc<gsl_sf_legendre_Plm>(SYM(Plm),SYM(l),SYM(m),SYM(x));
  addGSLIntIntRealFunc<gsl_sf_legendre_sphPlm>(SYM(sphPlm),SYM(l),SYM(m),SYM(x));
  addGSLRealRealFunc<gsl_sf_conicalP_half>(SYM(conicalP_half),SYM(lambda),SYM(x));
  addGSLRealRealFunc<gsl_sf_conicalP_mhalf>(SYM(conicalP_mhalf),SYM(lambda),SYM(x));
  addGSLRealRealFunc<gsl_sf_conicalP_0>(SYM(conicalP_0),SYM(lambda),SYM(x));
  addGSLRealRealFunc<gsl_sf_conicalP_1>(SYM(conicalP_1),SYM(lambda),SYM(x));
  addGSLIntRealRealFunc<gsl_sf_conicalP_sph_reg>(SYM(conicalP_sph_reg),SYM(l),
                                                 SYM(lambda),SYM(x));
  addGSLIntRealRealFunc<gsl_sf_conicalP_cyl_reg>(SYM(conicalP_cyl_reg),SYM(m),
                                                 SYM(lambda),SYM(x));
  addGSLRealRealFunc<gsl_sf_legendre_H3d_0>(SYM(H3d0),SYM(lambda),SYM(eta));
  addGSLRealRealFunc<gsl_sf_legendre_H3d_1>(SYM(H3d1),SYM(lambda),SYM(eta));
  addGSLIntRealRealFunc<gsl_sf_legendre_H3d>(SYM(H3d),SYM(l),SYM(lambda),SYM(eta));
  
  // Logarithm and related functions
  addGSLRealFunc<gsl_sf_log_abs>(SYM(logabs));
//  addGSLRealFunc<gsl_sf_log_1plusx>(SYM(log1p));
  addGSLRealFunc<gsl_sf_log_1plusx_mx>(SYM(log1pm));
  
  // Matthieu functions
  // to be implemented
  
  // Power function
  addGSLRealSignedFunc<gsl_sf_pow_int>(SYM(pow),SYM(x),SYM(n));
  
  // Psi (digamma) function
  addGSLSignedFunc<gsl_sf_psi_int>(SYM(psi),SYM(n));
  addGSLRealFunc<gsl_sf_psi>(SYM(psi));
  addGSLRealFunc<gsl_sf_psi_1piy>(SYM(psi_1piy),SYM(y));
  addGSLSignedFunc<gsl_sf_psi_1_int>(SYM(psi1),SYM(n));
  addGSLRealFunc<gsl_sf_psi_1>(SYM(psi1),SYM(x));
  addGSLIntRealFunc<gsl_sf_psi_n>(SYM(psi),SYM(n),SYM(x));
  
  // Synchrotron functions
  addGSLRealFunc<gsl_sf_synchrotron_1>(SYM(synchrotron_1));
  addGSLRealFunc<gsl_sf_synchrotron_2>(SYM(synchrotron_2));
  
  // Transport functions
  addGSLRealFunc<gsl_sf_transport_2>(SYM(transport_2));
  addGSLRealFunc<gsl_sf_transport_3>(SYM(transport_3));
  addGSLRealFunc<gsl_sf_transport_4>(SYM(transport_4));
  addGSLRealFunc<gsl_sf_transport_5>(SYM(transport_5));
  
  // Trigonometric functions
  addGSLRealFunc<gsl_sf_sinc>(SYM(sinc));
  addGSLRealFunc<gsl_sf_lnsinh>(SYM(lnsinh));
  addGSLRealFunc<gsl_sf_lncosh>(SYM(lncosh));
  
  // Zeta functions
  addGSLSignedFunc<gsl_sf_zeta_int>(SYM(zeta),SYM(n));
  addGSLRealFunc<gsl_sf_zeta>(SYM(zeta),SYM(s));
  addGSLSignedFunc<gsl_sf_zetam1_int>(SYM(zetam1),SYM(n));
  addGSLRealFunc<gsl_sf_zetam1>(SYM(zetam1),SYM(s));
  addGSLRealRealFunc<gsl_sf_hzeta>(SYM(hzeta),SYM(s),SYM(q));
  addGSLSignedFunc<gsl_sf_eta_int>(SYM(eta),SYM(n));
  addGSLRealFunc<gsl_sf_eta>(SYM(eta),SYM(s));
#endif
  
#ifdef STRUCTEXAMPLE
  dummyRecord *fun=createDummyRecord(ve, SYM(test));
  addFunc(fun->e.ve,realReal<sin>,primReal(),SYM(f),formal(primReal(),SYM(x)));
  addVariable<Int>(fun->e.ve,1,primInt(),SYM(x));
#endif
  
  addFunc(ve,writestring,primVoid(),SYM(write),
          formal(primFile(),SYM(file),true),
          formal(primString(),SYM(s)),
          formal(voidFileFunction(),SYM(suffix),true));
  
  addWrite(ve,write<transform>,primTransform(),transformArray());
  addWrite(ve,write<guide *>,primGuide(),guideArray());
  addWrite(ve,write<pen>,primPen(),penArray());
  addFunc(ve,arrayArrayOp<pen,equals>,booleanArray(),SYM_EQ,
          formal(penArray(),SYM(a)),formal(penArray(),SYM(b)));
  addFunc(ve,arrayArrayOp<pen,notequals>,booleanArray(),SYM_NEQ,
          formal(penArray(),SYM(a)),formal(penArray(),SYM(b)));

  addFunc(ve,arrayFunction,realArray(),SYM(map),
          formal(realPairFunction(),SYM(f)),
          formal(pairArray(),SYM(a)));
  addFunc(ve,arrayFunction,IntArray(),SYM(map),
          formal(IntRealFunction(),SYM(f)),
          formal(realArray(),SYM(a)));
  
  addConstant<Int>(ve, Int_MAX, primInt(), SYM(intMax));
  addConstant<Int>(ve, Int_MIN, primInt(), SYM(intMin));
  addConstant<double>(ve, HUGE_VAL, primReal(), SYM(inf));
  addConstant<double>(ve, run::infinity, primReal(), SYM(infinity));
  addConstant<double>(ve, DBL_MAX, primReal(), SYM(realMax));
  addConstant<double>(ve, DBL_MIN, primReal(), SYM(realMin));
  addConstant<double>(ve, DBL_EPSILON, primReal(), SYM(realEpsilon));
  addConstant<Int>(ve, DBL_DIG, primInt(), SYM(realDigits));
  addConstant<Int>(ve, RAND_MAX, primInt(), SYM(randMax));
  addConstant<double>(ve, PI, primReal(), SYM(pi));
  addConstant<string>(ve, string(settings::VERSION)+string(SVN_REVISION),
                      primString(),SYM(VERSION));
  
  addVariable<pen>(ve, &processData().currentpen, primPen(), SYM(currentpen));

#ifdef OPENFUNCEXAMPLE
  addOpenFunc(ve, openFunc, primInt(), SYM(openFunc));
#endif

  gen_runtime_venv(ve);
  gen_runbacktrace_venv(ve);
  gen_runpicture_venv(ve);
  gen_runlabel_venv(ve);
  gen_runhistory_venv(ve);
  gen_runarray_venv(ve);
  gen_runfile_venv(ve);
  gen_runsystem_venv(ve);
  gen_runstring_venv(ve);
  gen_runpair_venv(ve);
  gen_runtriple_venv(ve);
  gen_runpath_venv(ve);
  gen_runpath3d_venv(ve);
  gen_runmath_venv(ve);
}

} //namespace trans

namespace run {

double infinity=cbrt(DBL_MAX); // Reduced for tension atleast infinity

void arrayDeleteHelper(stack *Stack)
{
  array *a=pop<array *>(Stack);
  item itj=pop(Stack);
  bool jdefault=isdefault(itj);
  item iti=pop(Stack);
  Int i,j;
  if(isdefault(iti)) {
    if(jdefault) {
    (*a).clear();
    return;
    } else i=j=get<Int>(itj);
  } else {
    i=get<Int>(iti);
    j=jdefault ? i : get<Int>(itj);
  }

  size_t asize=checkArray(a);
  if(a->cyclic() && asize > 0) {
    if(j-i+1 >= (Int) asize) {
      (*a).clear();
      return;
    }
    i=imod(i,asize);
    j=imod(j,asize);
    if(j >= i) 
      (*a).erase((*a).begin()+i,(*a).begin()+j+1);
    else {
      (*a).erase((*a).begin()+i,(*a).end());
      (*a).erase((*a).begin(),(*a).begin()+j+1);
    }
    return;
  }
  
  if(i < 0 || i >= (Int) asize || i > j || j >= (Int) asize) {
    ostringstream buf;
    buf << "delete called on array of length " << (Int) asize 
        << " with out-of-bounds index range [" << i << "," << j << "]";
    error(buf);
  }

  (*a).erase((*a).begin()+i,(*a).begin()+j+1);
}

}
