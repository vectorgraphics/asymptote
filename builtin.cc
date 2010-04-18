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

void addType(tenv &te, const char *name, ty *t)
{
  te.enter(symbol::trans(name), new tyEntry(t,0,0,position()));
}

// The base environments for built-in types and functions
void base_tenv(tenv &te)
{
#define PRIMITIVE(name,Name,asyName)  addType(te, #asyName, prim##Name());
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

void addOpenFunc(venv &ve, bltin f, ty *result, const char *name)
{
  function *fun = new function(result, signature::OPEN);

  REGISTER_BLTIN(f, name);
  access *a= new bltinAccess(f);

  varEntry *ent = new varEntry(fun, a, 0, position());
  
  ve.enter(symbol::trans(name), ent);
}


// Add a rest function with zero or more default/explicit arguments.
void addRestFunc(venv &ve, bltin f, ty *result, const char *name, formal frest,
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
  addFunc(ve, arrayFunc<double,double,fcn>, realArray(), name,
          formal(realArray(),"a"));
}

#define addRealFunc(fcn) addRealFunc<fcn>(ve, #fcn);
  
void addRealFunc2(venv &ve, bltin fcn, const char *name)
{
  addFunc(ve,fcn,primReal(),name,formal(primReal(),"a"),
          formal(primReal(),"b"));
}

template <double (*func)(double, int)>
void realRealInt(vm::stack *s) {
  Int n = pop<Int>(s);
  double x = pop<double>(s);
  s->push(func(x, intcast(n)));
}

template<double (*fcn)(double, int)>
void addRealIntFunc(venv& ve, const char* name, const char* arg1,
                    const char* arg2) {
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
void addGSLRealFunc(const char* name, const char* arg1="x")
{
  addFunc(GSLModule->e.ve, realRealGSL<fcn>, primReal(), name,
          formal(primReal(),arg1));
}

// Add a GSL_PREC_DOUBLE GSL special function.
template<double (*fcn)(double, gsl_mode_t)>
void addGSLDOUBLEFunc(const char* name, const char* arg1="x")
{
  addFunc(GSLModule->e.ve, realRealDOUBLE<fcn>, primReal(), name,
          formal(primReal(),arg1));
}

template<double (*fcn)(double, double, gsl_mode_t)>
void addGSLDOUBLE2Func(const char* name, const char* arg1="phi",
                       const char* arg2="k")
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
void addGSLDOUBLE3Func(const char* name, const char* arg1, const char* arg2,
                       const char* arg3)
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
void addGSLDOUBLE4Func(const char* name, const char* arg1, const char* arg2,
                       const char* arg3, const char* arg4)
{
  addFunc(GSLModule->e.ve, realRealRealRealRealDOUBLE<fcn>, primReal(), name, 
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3), formal(primReal(), arg4));
}

template<double (*fcn)(unsigned)>
void addGSLIntFunc(const char* name)
{
  addFunc(GSLModule->e.ve, realIntGSL<fcn>, primReal(), name,
          formal(primInt(),"s"));
}

template <double (*func)(int)>
void realSignedGSL(vm::stack *s) 
{
  Int a = pop<Int>(s);
  s->push(func(intcast(a)));
  checkGSLerror();
}

template<double (*fcn)(int)>
void addGSLSignedFunc(const char* name, const char* arg1)
{
  addFunc(GSLModule->e.ve, realSignedGSL<fcn>, primReal(), name,
          formal(primInt(),arg1));
}

template<double (*fcn)(int, double)>
void addGSLIntRealFunc(const char* name, const char *arg1="n",
                       const char* arg2="x")
{
  addFunc(GSLModule->e.ve, realIntRealGSL<fcn>, primReal(), name,
          formal(primInt(),arg1), formal(primReal(),arg2));
}

template<double (*fcn)(double, double)>
void addGSLRealRealFunc(const char* name, const char* arg1="nu",
                        const char* arg2="x")
{
  addFunc(GSLModule->e.ve, realRealRealGSL<fcn>, primReal(), name,
          formal(primReal(),arg1), formal(primReal(),arg2));
}

template<double (*fcn)(double, double, double)>
void addGSLRealRealRealFunc(const char* name, const char* arg1,
                            const char* arg2, const char* arg3)
{
  addFunc(GSLModule->e.ve, realRealRealRealGSL<fcn>, primReal(), name,
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3));
}

template<int (*fcn)(double, double, double)>
void addGSLRealRealRealFuncInt(const char* name, const char* arg1,
                               const char* arg2, const char* arg3)
{
  addFunc(GSLModule->e.ve, intRealRealRealGSL<fcn>, primInt(), name,
          formal(primReal(),arg1), formal(primReal(),arg2),
          formal(primReal(), arg3));
}

template<double (*fcn)(double, unsigned)>
void addGSLRealIntFunc(const char* name, const char* arg1="nu",
                       const char* arg2="s")
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
void addGSLRealSignedFunc(const char* name, const char* arg1, const char* arg2)
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
void addGSLUnsignedUnsignedFunc(const char* name, const char* arg1,
                                const char* arg2)
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
void addGSLIntRealRealFunc(const char* name, const char* arg1,
                           const char* arg2, const char* arg3)
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
void addGSLIntIntRealFunc(const char* name, const char* arg1, const char* arg2,
                          const char* arg3)
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
void addGSLIntIntRealRealFunc(const char* name, const char* arg1,
                              const char* arg2, const char* arg3,
                              const char* arg4)
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
void addGSLRealRealRealRealFunc(const char* name, const char* arg1,
                                const char* arg2, const char* arg3,
                                const char* arg4)
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
void addGSLIntIntIntIntIntIntFunc(const char* name, const char* arg1,
                                  const char* arg2, const char* arg3,
                                  const char* arg4, const char* arg5,
                                  const char* arg6)
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
void addGSLIntIntIntIntIntIntIntIntIntFunc(const char* name, const char* arg1,
                                           const char* arg2, const char* arg3,
                                           const char* arg4, const char* arg5,
                                           const char* arg6, const char* arg7,
                                           const char* arg8, const char* arg9)
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
void addVariable(venv &ve, T *ref, ty *t, const char *name,
                 record *module=settings::getSettingsModule()) {
  access *a = new refAccess<T>(ref);
  varEntry *ent = new varEntry(t, a, PUBLIC, module, 0, position());
  ve.enter(symbol::trans(name), ent);
}

template<class T>
void addVariable(venv &ve, T value, ty *t, const char *name,
                 record *module=settings::getSettingsModule(),
                 permission perm=PUBLIC) {
  item* ref=new item;
  *ref=value;
  access *a = new itemRefAccess(ref);
  varEntry *ent = new varEntry(t, a, perm, module, 0, position());
  ve.enter(symbol::trans(name), ent);
}

template<class T>
void addConstant(venv &ve, T value, ty *t, const char *name,
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
void addArray2Array2Op(venv &ve, ty *t3, const char *name)
{
  addFunc(ve,array2Array2Op<T,op>,t3,name,formal(t3,"a"),formal(t3,"b"));
}

template<class T, template <class S> class op>
void addOpArray2(venv &ve, ty *t1, const char *name, ty *t3)
{
  addFunc(ve,opArray2<T,T,op>,t3,name,formal(t1,"a"),formal(t3,"b"));
}

template<class T, template <class S> class op>
void addArray2Op(venv &ve, ty *t1, const char *name, ty *t3)
{
  addFunc(ve,array2Op<T,T,op>,t3,name,formal(t3,"a"),formal(t1,"b"));
}

template<class T, template <class S> class op>
void addOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addSimpleOperator(ve,binaryOp<T,op>,t1,name);
  addFunc(ve,opArray<T,T,op>,t2,name,formal(t1,"a"),formal(t2,"b"));
  addFunc(ve,arrayOp<T,T,op>,t2,name,formal(t2,"a"),formal(t1,"b"));
  addSimpleOperator(ve,arrayArrayOp<T,op>,t2,name);
}

template<class T, template <class S> class op>
void addBooleanOps(venv &ve, ty *t1, const char *name, ty *t2)
{
  addBooleanOperator(ve,binaryOp<T,op>,t1,name);
  addFunc(ve,opArray<T,T,op>,
      booleanArray(),name,formal(t1,"a"),formal(t2,"b"));
  addFunc(ve,arrayOp<T,T,op>,
      booleanArray(),name,formal(t2,"a"),formal(t1,"b"));
  addFunc(ve,arrayArrayOp<T,op>,booleanArray(),name,formal(t2,"a"),
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
   
  addFunc(ve, run::array2Equals<T>, primBoolean(), "==", formal(t3, "a"),
          formal(t3, "b"));
  addFunc(ve, run::array2NotEquals<T>, primBoolean(), "!=", formal(t3, "a"),
          formal(t3, "b"));
  
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

inline double abs(pair z) {
  return z.length();
}

inline double abs(triple v) {
  return v.length();
}

inline pair conjugate(pair z) {
  return conj(z);
}

template<class T, template <class S> class op>
void addBinOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, const char *name)
{
  addFunc(ve,binopArray<T,op>,t1,name,formal(t2,"a"));
  addFunc(ve,binopArray2<T,op>,t1,name,formal(t3,"a"));
  addFunc(ve,binopArray3<T,op>,t1,name,formal(t4,"a"));
}

template<class T>
void addOrderedOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4)
{
  addBooleanOps<T,less>(ve,t1,"<",t2);
  addBooleanOps<T,lessequals>(ve,t1,"<=",t2);
  addBooleanOps<T,greaterequals>(ve,t1,">=",t2);
  addBooleanOps<T,greater>(ve,t1,">",t2);
  
  addOps<T,run::min>(ve,t1,"min",t2);
  addOps<T,run::max>(ve,t1,"max",t2);
  addBinOps<T,run::min>(ve,t1,t2,t3,t4,"min");
  addBinOps<T,run::max>(ve,t1,t2,t3,t4,"max");
    
  addFunc(ve,sortArray<T>,t2,"sort",formal(t2,"a"));
  addFunc(ve,sortArray2<T>,t3,"sort",formal(t3,"a"));
  
  addFunc(ve,searchArray<T>,primInt(),"search",formal(t2,"a"),
          formal(t1,"key"));
}

template<class T>
void addBasicOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, bool integer=false,
                 bool Explicit=false)
{
  addOps<T,plus>(ve,t1,"+",t2);
  addOps<T,minus>(ve,t1,"-",t2);
  
  addArray2Array2Op<T,plus>(ve,t3,"+");
  addArray2Array2Op<T,minus>(ve,t3,"-");

  addFunc(ve,&id,t1,"+",formal(t1,"a"));
  addFunc(ve,&id,t2,"+",formal(t2,"a"));
  addFunc(ve,Negate<T>,t1,"-",formal(t1,"a"));
  addFunc(ve,arrayNegate<T>,t2,"-",formal(t2,"a"));
  if(!integer) addFunc(ve,interp<T>,t1,"interp",formal(t1,"a",false,Explicit),
                       formal(t1,"b",false,Explicit),
                       formal(primReal(),"t"));
  
  addFunc(ve,sumArray<T>,t1,"sum",formal(t2,"a"));
  addUnorderedOps<T>(ve,t1,t2,t3,t4);
}

template<class T>
void addOps(venv &ve, ty *t1, ty *t2, ty *t3, ty *t4, bool integer=false,
            bool Explicit=false)
{
  addBasicOps<T>(ve,t1,t2,t3,t4,integer,Explicit);
  
  addOps<T,times>(ve,t1,"*",t2);
  addOpArray2<T,times>(ve,t1,"*",t3);
  addArray2Op<T,times>(ve,t1,"*",t3);
  
  if(!integer) {
    addOps<T,run::divide>(ve,t1,"/",t2);
    addArray2Op<T,run::divide>(ve,t1,"/",t3);
  }
      
  addOps<T,power>(ve,t1,"^",t2);
}


// Adds standard functions for a newly added array type.
void addArrayOps(venv &ve, types::array *t)
{
  ty *ct = t->celltype;
  
  addFunc(ve, run::arrayAlias,
          primBoolean(), "alias", formal(t, "a"), formal(t, "b"));

  addFunc(ve, run::newDuplicateArray,
          t, "array", formal(primInt(), "n"),
          formal(ct, "value"),
          formal(primInt(), "depth", /*optional=*/ true));

  switch (t->depth()) {
    case 1:
      addFunc(ve, run::arrayCopy, t, "copy", formal(t, "a"));
      addRestFunc(ve, run::arrayConcat, t, "concat", new types::array(t));
      addFunc(ve, run::arraySequence,
              t, "sequence", formal(new function(ct, primInt()), "f"),
              formal(primInt(), "n"));
      addFunc(ve, run::arrayFunction,
              t, "map", formal(new function(ct, ct), "f"), formal(t, "a"));
      addFunc(ve, run::arraySort,
              t, "sort", formal(t, "a"),
              formal(new function(primBoolean(), ct, ct), "f"));
      break;
    case 2:
      addFunc(ve, run::array2Copy, t, "copy", formal(t, "a"));
      addFunc(ve, run::array2Transpose, t, "transpose", formal(t, "a"));
      break;
    case 3:
      addFunc(ve, run::array3Copy, t, "copy", formal(t, "a"));
      addFunc(ve, run::array3Transpose, t, "transpose", formal(t, "a"),
              formal(IntArray(),"perm"));
      break;
    default:
      break;
  }
}

void addRecordOps(venv &ve, record *r)
{
  addFunc(ve, run::boolMemEq, primBoolean(), "alias", formal(r, "a"),
          formal(r, "b"));
  addFunc(ve, run::boolMemEq, primBoolean(), "==", formal(r, "a"),
          formal(r, "b"));
  addFunc(ve, run::boolMemNeq, primBoolean(), "!=", formal(r, "a"),
          formal(r, "b"));
}

void addFunctionOps(venv &ve, function *f)
{
  addFunc(ve, run::boolFuncEq, primBoolean(), "==", formal(f, "a"),
          formal(f, "b"));
  addFunc(ve, run::boolFuncNeq, primBoolean(), "!=", formal(f, "a"),
          formal(f, "b"));
}

void addOperators(venv &ve) 
{
  addSimpleOperator(ve,binaryOp<string,plus>,primString(),"+");
  
  addBooleanOps<bool,And>(ve,primBoolean(),"&",booleanArray());
  addBooleanOps<bool,Or>(ve,primBoolean(),"|",booleanArray());
  addBooleanOps<bool,Xor>(ve,primBoolean(),"^",booleanArray());
  
  addUnorderedOps<bool>(ve,primBoolean(),booleanArray(),booleanArray2(),
                        booleanArray3());
  addOps<Int>(ve,primInt(),IntArray(),IntArray2(),IntArray3(),true);
  addOps<double>(ve,primReal(),realArray(),realArray2(),realArray3());
  addOps<pair>(ve,primPair(),pairArray(),pairArray2(),pairArray3(),false,true);
  addBasicOps<triple>(ve,primTriple(),tripleArray(),tripleArray2(),
                      tripleArray3());
  addFunc(ve,opArray<double,triple,times>,tripleArray(),"*",
          formal(primReal(),"a"),formal(tripleArray(),"b"));
  addFunc(ve,arrayOp<triple,double,timesR>,tripleArray(),"*",
          formal(tripleArray(),"a"),formal(primReal(),"b"));
  addFunc(ve,arrayOp<triple,double,divide>,tripleArray(),"/",
          formal(tripleArray(),"a"),formal(primReal(),"b"));

  addUnorderedOps<string>(ve,primString(),stringArray(),stringArray2(),
                          stringArray3());
  
  addSimpleOperator(ve,binaryOp<pair,minbound>,primPair(),"minbound");
  addSimpleOperator(ve,binaryOp<pair,maxbound>,primPair(),"maxbound");
  addSimpleOperator(ve,binaryOp<triple,minbound>,primTriple(),"minbound");
  addSimpleOperator(ve,binaryOp<triple,maxbound>,primTriple(),"maxbound");
  addBinOps<pair,minbound>(ve,primPair(),pairArray(),pairArray2(),pairArray3(),
                           "minbound");
  addBinOps<pair,maxbound>(ve,primPair(),pairArray(),pairArray2(),pairArray3(),
                           "maxbound");
  addBinOps<triple,minbound>(ve,primTriple(),tripleArray(),tripleArray2(),
                             tripleArray3(),"minbound");
  addBinOps<triple,maxbound>(ve,primTriple(),tripleArray(),tripleArray2(),
                             tripleArray3(),"maxbound");
  
  addFunc(ve,arrayFunc<double,pair,abs>,realArray(),"abs",
          formal(pairArray(),"a"));
  addFunc(ve,arrayFunc<double,triple,abs>,realArray(),"abs",
          formal(tripleArray(),"a"));
  
  addFunc(ve,arrayFunc<pair,pair,conjugate>,pairArray(),"conj",
          formal(pairArray(),"a"));
  addFunc(ve,arrayFunc2<pair,pair,conjugate>,pairArray2(),"conj",
          formal(pairArray2(),"a"));
  
  addFunc(ve,binaryOp<Int,divide>,primReal(),"/",
          formal(primInt(),"a"),formal(primInt(),"b"));
  addFunc(ve,arrayOp<Int,Int,divide>,realArray(),"/",
          formal(IntArray(),"a"),formal(primInt(),"b"));
  addFunc(ve,opArray<Int,Int,divide>,realArray(),"/",
          formal(primInt(),"a"),formal(IntArray(),"b"));
  addFunc(ve,arrayArrayOp<Int,divide>,realArray(),"/",
          formal(IntArray(),"a"),formal(IntArray(),"b"));
  
  addOrderedOps<Int>(ve,primInt(),IntArray(),IntArray2(),IntArray3());
  addOrderedOps<double>(ve,primReal(),realArray(),realArray2(),realArray3());
  addOrderedOps<string>(ve,primString(),stringArray(),stringArray2(),
                        stringArray3());
  
  addOps<Int,mod>(ve,primInt(),"%",IntArray());
  addOps<double,mod>(ve,primReal(),"%",realArray());
  
  addRestFunc(ve,diagonal<Int>,IntArray2(),"diagonal",IntArray());
  addRestFunc(ve,diagonal<double>,realArray2(),"diagonal",realArray());
  addRestFunc(ve,diagonal<pair>,pairArray2(),"diagonal",pairArray());
}

dummyRecord *createDummyRecord(venv &ve, const char *name)
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
  addRealFunc(expm1);
  addRealFunc(log1p);
  addRealIntFunc<ldexp>(ve, "ldexp", "x", "e");

  addRealFunc(pow10);
  addRealFunc(identity);
  
#ifdef HAVE_LIBGSL  
  GSLModule=new dummyRecord(symbol::trans("gsl"));
  gsl_set_error_handler(GSLerrorhandler);
  
  // Common functions
  addGSLRealRealFunc<gsl_hypot>("hypot","x","y");
//  addGSLRealRealRealFunc<gsl_hypot3>("hypot","x","y","z");
  addGSLRealRealRealFuncInt<gsl_fcmp>("fcmp","x","y","epsilon");
  
  // Airy functions
  addGSLDOUBLEFunc<gsl_sf_airy_Ai>("Ai");
  addGSLDOUBLEFunc<gsl_sf_airy_Bi>("Bi");
  addGSLDOUBLEFunc<gsl_sf_airy_Ai_scaled>("Ai_scaled");
  addGSLDOUBLEFunc<gsl_sf_airy_Bi_scaled>("Bi_scaled");
  addGSLDOUBLEFunc<gsl_sf_airy_Ai_deriv>("Ai_deriv");
  addGSLDOUBLEFunc<gsl_sf_airy_Bi_deriv>("Bi_deriv");
  addGSLDOUBLEFunc<gsl_sf_airy_Ai_deriv_scaled>("Ai_deriv_scaled");
  addGSLDOUBLEFunc<gsl_sf_airy_Bi_deriv_scaled>("Bi_deriv_scaled");
  addGSLIntFunc<gsl_sf_airy_zero_Ai>("zero_Ai");
  addGSLIntFunc<gsl_sf_airy_zero_Bi>("zero_Bi");
  addGSLIntFunc<gsl_sf_airy_zero_Ai_deriv>("zero_Ai_deriv");
  addGSLIntFunc<gsl_sf_airy_zero_Bi_deriv>("zero_Bi_deriv");
  
  // Bessel functions
  addGSLRealFunc<gsl_sf_bessel_J0>("J0");
  addGSLRealFunc<gsl_sf_bessel_J1>("J1");
  addGSLIntRealFunc<gsl_sf_bessel_Jn>("Jn");
  addGSLRealFunc<gsl_sf_bessel_Y0>("Y0");
  addGSLRealFunc<gsl_sf_bessel_Y1>("Y1");
  addGSLIntRealFunc<gsl_sf_bessel_Yn>("Yn");
  addGSLRealFunc<gsl_sf_bessel_I0>("I0");
  addGSLRealFunc<gsl_sf_bessel_I1>("I1");
  addGSLIntRealFunc<gsl_sf_bessel_In>("I");
  addGSLRealFunc<gsl_sf_bessel_I0_scaled>("I0_scaled");
  addGSLRealFunc<gsl_sf_bessel_I1_scaled>("I1_scaled");
  addGSLIntRealFunc<gsl_sf_bessel_In_scaled>("I_scaled");
  addGSLRealFunc<gsl_sf_bessel_K0>("K0");
  addGSLRealFunc<gsl_sf_bessel_K1>("K1");
  addGSLIntRealFunc<gsl_sf_bessel_Kn>("K");
  addGSLRealFunc<gsl_sf_bessel_K0_scaled>("K0_scaled");
  addGSLRealFunc<gsl_sf_bessel_K1_scaled>("K1_scaled");
  addGSLIntRealFunc<gsl_sf_bessel_Kn_scaled>("K_scaled");
  addGSLRealFunc<gsl_sf_bessel_j0>("j0");
  addGSLRealFunc<gsl_sf_bessel_j1>("j1");
  addGSLRealFunc<gsl_sf_bessel_j2>("j2");
  addGSLIntRealFunc<gsl_sf_bessel_jl>("j","l");
  addGSLRealFunc<gsl_sf_bessel_y0>("y0");
  addGSLRealFunc<gsl_sf_bessel_y1>("y1");
  addGSLRealFunc<gsl_sf_bessel_y2>("y2");
  addGSLIntRealFunc<gsl_sf_bessel_yl>("y","l");
  addGSLRealFunc<gsl_sf_bessel_i0_scaled>("i0_scaled");
  addGSLRealFunc<gsl_sf_bessel_i1_scaled>("i1_scaled");
  addGSLRealFunc<gsl_sf_bessel_i2_scaled>("i2_scaled");
  addGSLIntRealFunc<gsl_sf_bessel_il_scaled>("i_scaled","l");
  addGSLRealFunc<gsl_sf_bessel_k0_scaled>("k0_scaled");
  addGSLRealFunc<gsl_sf_bessel_k1_scaled>("k1_scaled");
  addGSLRealFunc<gsl_sf_bessel_k2_scaled>("k2_scaled");
  addGSLIntRealFunc<gsl_sf_bessel_kl_scaled>("k_scaled","l");
  addGSLRealRealFunc<gsl_sf_bessel_Jnu>("J");
  addGSLRealRealFunc<gsl_sf_bessel_Ynu>("Y");
  addGSLRealRealFunc<gsl_sf_bessel_Inu>("I");
  addGSLRealRealFunc<gsl_sf_bessel_Inu_scaled>("I_scaled");
  addGSLRealRealFunc<gsl_sf_bessel_Knu>("K");
  addGSLRealRealFunc<gsl_sf_bessel_lnKnu>("lnK");
  addGSLRealRealFunc<gsl_sf_bessel_Knu_scaled>("K_scaled");
  addGSLIntFunc<gsl_sf_bessel_zero_J0>("zero_J0");
  addGSLIntFunc<gsl_sf_bessel_zero_J1>("zero_J1");
  addGSLRealIntFunc<gsl_sf_bessel_zero_Jnu>("zero_J");
  
  // Clausen functions
  addGSLRealFunc<gsl_sf_clausen>("clausen");
  
  // Coulomb functions
  addGSLRealRealFunc<gsl_sf_hydrogenicR_1>("hydrogenicR_1","Z","r");
  addGSLIntIntRealRealFunc<gsl_sf_hydrogenicR>("hydrogenicR","n","l","Z",
                                               "r");
  // Missing: F_L(eta,x), G_L(eta,x), C_L(eta)
  
  // Coupling coefficients
  addGSLIntIntIntIntIntIntFunc<gsl_sf_coupling_3j>("coupling_3j","two_ja",
                                                   "two_jb","two_jc","two_ma",
                                                   "two_mb","two_mc");
  addGSLIntIntIntIntIntIntFunc<gsl_sf_coupling_6j>("coupling_6j","two_ja",
                                                   "two_jb","two_jc","two_jd",
                                                   "two_je","two_jf");
  addGSLIntIntIntIntIntIntIntIntIntFunc<gsl_sf_coupling_9j>("coupling_9j",
                                                            "two_ja","two_jb",
                                                            "two_jc","two_jd",
                                                            "two_je","two_jf",
                                                            "two_jg","two_jh",
                                                            "two_ji");
  // Dawson function
  addGSLRealFunc<gsl_sf_dawson>("dawson");
  
  // Debye functions
  addGSLRealFunc<gsl_sf_debye_1>("debye_1");
  addGSLRealFunc<gsl_sf_debye_2>("debye_2");
  addGSLRealFunc<gsl_sf_debye_3>("debye_3");
  addGSLRealFunc<gsl_sf_debye_4>("debye_4");
  addGSLRealFunc<gsl_sf_debye_5>("debye_5");
  addGSLRealFunc<gsl_sf_debye_6>("debye_6");
  
  // Dilogarithm
  addGSLRealFunc<gsl_sf_dilog>("dilog");
  // Missing: complex dilogarithm
  
  // Elementary operations
  // we don't support errors at the moment
  
  // Elliptic integrals
  addGSLDOUBLEFunc<gsl_sf_ellint_Kcomp>("K","k");
  addGSLDOUBLEFunc<gsl_sf_ellint_Ecomp>("E","k");
  addGSLDOUBLE2Func<gsl_sf_ellint_Pcomp>("P","k","n");
  addGSLDOUBLE2Func<gsl_sf_ellint_F>("F");
  addGSLDOUBLE2Func<gsl_sf_ellint_E>("E");
  addGSLDOUBLE3Func<gsl_sf_ellint_P>("P","phi","k","n");
  addGSLDOUBLE3Func<gsl_sf_ellint_D>("D","phi","k","n");
  addGSLDOUBLE2Func<gsl_sf_ellint_RC>("RC","x","y");
  addGSLDOUBLE3Func<gsl_sf_ellint_RD>("RD","x","y","z");
  addGSLDOUBLE3Func<gsl_sf_ellint_RF>("RF","x","y","z");
  addGSLDOUBLE4Func<gsl_sf_ellint_RJ>("RJ","x","y","z","p");
  
  // Elliptic functions (Jacobi)
  // to be implemented
  
  // Error functions
  addGSLRealFunc<gsl_sf_erf>("erf");
  addGSLRealFunc<gsl_sf_erfc>("erfc");
  addGSLRealFunc<gsl_sf_log_erfc>("log_erfc");
  addGSLRealFunc<gsl_sf_erf_Z>("erf_Z");
  addGSLRealFunc<gsl_sf_erf_Q>("erf_Q");
  addGSLRealFunc<gsl_sf_hazard>("hazard");
  
  // Exponential functions
  addGSLRealRealFunc<gsl_sf_exp_mult>("exp_mult","x","y");
//  addGSLRealFunc<gsl_sf_expm1>("expm1");
  addGSLRealFunc<gsl_sf_exprel>("exprel");
  addGSLRealFunc<gsl_sf_exprel_2>("exprel_2");
  addGSLIntRealFunc<gsl_sf_exprel_n>("exprel","n","x");
  
  // Exponential integrals
  addGSLRealFunc<gsl_sf_expint_E1>("E1");
  addGSLRealFunc<gsl_sf_expint_E2>("E2");
//  addGSLIntRealFunc<gsl_sf_expint_En>("En","n","x");
  addGSLRealFunc<gsl_sf_expint_Ei>("Ei");
  addGSLRealFunc<gsl_sf_Shi>("Shi");
  addGSLRealFunc<gsl_sf_Chi>("Chi");
  addGSLRealFunc<gsl_sf_expint_3>("Ei3");
  addGSLRealFunc<gsl_sf_Si>("Si");
  addGSLRealFunc<gsl_sf_Ci>("Ci");
  addGSLRealFunc<gsl_sf_atanint>("atanint");
  
  // Fermi--Dirac function
  addGSLRealFunc<gsl_sf_fermi_dirac_m1>("FermiDiracM1");
  addGSLRealFunc<gsl_sf_fermi_dirac_0>("FermiDirac0");
  addGSLRealFunc<gsl_sf_fermi_dirac_1>("FermiDirac1");
  addGSLRealFunc<gsl_sf_fermi_dirac_2>("FermiDirac2");
  addGSLIntRealFunc<gsl_sf_fermi_dirac_int>("FermiDirac","j","x");
  addGSLRealFunc<gsl_sf_fermi_dirac_mhalf>("FermiDiracMHalf");
  addGSLRealFunc<gsl_sf_fermi_dirac_half>("FermiDiracHalf");
  addGSLRealFunc<gsl_sf_fermi_dirac_3half>("FermiDirac3Half");
  addGSLRealRealFunc<gsl_sf_fermi_dirac_inc_0>("FermiDiracInc0","x","b");
  
  // Gamma and beta functions
  addGSLRealFunc<gsl_sf_gamma>("gamma");
  addGSLRealFunc<gsl_sf_lngamma>("lngamma");
  addGSLRealFunc<gsl_sf_gammastar>("gammastar");
  addGSLRealFunc<gsl_sf_gammainv>("gammainv");
  addGSLIntFunc<gsl_sf_fact>("fact");
  addGSLIntFunc<gsl_sf_doublefact>("doublefact");
  addGSLIntFunc<gsl_sf_lnfact>("lnfact");
  addGSLIntFunc<gsl_sf_lndoublefact>("lndoublefact");
  addGSLUnsignedUnsignedFunc<gsl_sf_choose>("choose","n","m");
  addGSLUnsignedUnsignedFunc<gsl_sf_lnchoose>("lnchoose","n","m");
  addGSLIntRealFunc<gsl_sf_taylorcoeff>("taylorcoeff","n","x");
  addGSLRealRealFunc<gsl_sf_poch>("poch","a","x");
  addGSLRealRealFunc<gsl_sf_lnpoch>("lnpoch","a","x");
  addGSLRealRealFunc<gsl_sf_pochrel>("pochrel","a","x");
  addGSLRealRealFunc<gsl_sf_gamma_inc>("gamma","a","x");
  addGSLRealRealFunc<gsl_sf_gamma_inc_Q>("gamma_Q","a","x");
  addGSLRealRealFunc<gsl_sf_gamma_inc_P>("gamma_P","a","x");
  addGSLRealRealFunc<gsl_sf_beta>("beta","a","b");
  addGSLRealRealFunc<gsl_sf_lnbeta>("lnbeta","a","b");
  addGSLRealRealRealFunc<gsl_sf_beta_inc>("beta","a","b","x");
  
  // Gegenbauer functions
  addGSLRealRealFunc<gsl_sf_gegenpoly_1>("gegenpoly_1","lambda","x");
  addGSLRealRealFunc<gsl_sf_gegenpoly_2>("gegenpoly_2","lambda","x");
  addGSLRealRealFunc<gsl_sf_gegenpoly_3>("gegenpoly_3","lambda","x");
  addGSLIntRealRealFunc<gsl_sf_gegenpoly_n>("gegenpoly","n","lambda","x");
  
  // Hypergeometric functions
  addGSLRealRealFunc<gsl_sf_hyperg_0F1>("hy0F1","c","x");
  addGSLIntIntRealFunc<gsl_sf_hyperg_1F1_int>("hy1F1","m","n","x");
  addGSLRealRealRealFunc<gsl_sf_hyperg_1F1>("hy1F1","a","b","x");
  addGSLIntIntRealFunc<gsl_sf_hyperg_U_int>("U","m","n","x");
  addGSLRealRealRealFunc<gsl_sf_hyperg_U>("U","a","b","x");
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1>("hy2F1","a","b","c","x");
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1_conj>("hy2F1_conj","aR","aI","c",
                                                     "x");
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1_renorm>("hy2F1_renorm","a","b",
                                                       "c","x");
  addGSLRealRealRealRealFunc<gsl_sf_hyperg_2F1_conj_renorm>("hy2F1_conj_renorm",
                                                            "aR","aI","c","x");
  addGSLRealRealRealFunc<gsl_sf_hyperg_2F0>("hy2F0","a","b","x");
  
  // Laguerre functions
  addGSLRealRealFunc<gsl_sf_laguerre_1>("L1","a","x");
  addGSLRealRealFunc<gsl_sf_laguerre_2>("L2","a","x");
  addGSLRealRealFunc<gsl_sf_laguerre_3>("L3","a","x");
  addGSLIntRealRealFunc<gsl_sf_laguerre_n>("L","n","a","x");
  
  // Lambert W functions
  addGSLRealFunc<gsl_sf_lambert_W0>("W0");
  addGSLRealFunc<gsl_sf_lambert_Wm1>("Wm1");
  
  // Legendre functions and spherical harmonics
  addGSLRealFunc<gsl_sf_legendre_P1>("P1");
  addGSLRealFunc<gsl_sf_legendre_P2>("P2");
  addGSLRealFunc<gsl_sf_legendre_P3>("P3");
  addGSLIntRealFunc<gsl_sf_legendre_Pl>("Pl","l");
  addGSLRealFunc<gsl_sf_legendre_Q0>("Q0");
  addGSLRealFunc<gsl_sf_legendre_Q1>("Q1");
  addGSLIntRealFunc<gsl_sf_legendre_Ql>("Ql","l");
  addGSLIntIntRealFunc<gsl_sf_legendre_Plm>("Plm","l","m","x");
  addGSLIntIntRealFunc<gsl_sf_legendre_sphPlm>("sphPlm","l","m","x");
  addGSLRealRealFunc<gsl_sf_conicalP_half>("conicalP_half","lambda","x");
  addGSLRealRealFunc<gsl_sf_conicalP_mhalf>("conicalP_mhalf","lambda","x");
  addGSLRealRealFunc<gsl_sf_conicalP_0>("conicalP_0","lambda","x");
  addGSLRealRealFunc<gsl_sf_conicalP_1>("conicalP_1","lambda","x");
  addGSLIntRealRealFunc<gsl_sf_conicalP_sph_reg>("conicalP_sph_reg","l",
                                                 "lambda","x");
  addGSLIntRealRealFunc<gsl_sf_conicalP_cyl_reg>("conicalP_cyl_reg","m",
                                                 "lambda","x");
  addGSLRealRealFunc<gsl_sf_legendre_H3d_0>("H3d0","lambda","eta");
  addGSLRealRealFunc<gsl_sf_legendre_H3d_1>("H3d1","lambda","eta");
  addGSLIntRealRealFunc<gsl_sf_legendre_H3d>("H3d","l","lambda","eta");
  
  // Logarithm and related functions
  addGSLRealFunc<gsl_sf_log_abs>("logabs");
//  addGSLRealFunc<gsl_sf_log_1plusx>("log1p");
  addGSLRealFunc<gsl_sf_log_1plusx_mx>("log1pm");
  
  // Matthieu functions
  // to be implemented
  
  // Power function
  addGSLRealSignedFunc<gsl_sf_pow_int>("pow","x","n");
  
  // Psi (digamma) function
  addGSLSignedFunc<gsl_sf_psi_int>("psi","n");
  addGSLRealFunc<gsl_sf_psi>("psi");
  addGSLRealFunc<gsl_sf_psi_1piy>("psi_1piy","y");
  addGSLSignedFunc<gsl_sf_psi_1_int>("psi1","n");
  addGSLRealFunc<gsl_sf_psi_1>("psi1","x");
  addGSLIntRealFunc<gsl_sf_psi_n>("psi","n","x");
  
  // Synchrotron functions
  addGSLRealFunc<gsl_sf_synchrotron_1>("synchrotron_1");
  addGSLRealFunc<gsl_sf_synchrotron_2>("synchrotron_2");
  
  // Transport functions
  addGSLRealFunc<gsl_sf_transport_2>("transport_2");
  addGSLRealFunc<gsl_sf_transport_3>("transport_3");
  addGSLRealFunc<gsl_sf_transport_4>("transport_4");
  addGSLRealFunc<gsl_sf_transport_5>("transport_5");
  
  // Trigonometric functions
  addGSLRealFunc<gsl_sf_sinc>("sinc");
  addGSLRealFunc<gsl_sf_lnsinh>("lnsinh");
  addGSLRealFunc<gsl_sf_lncosh>("lncosh");
  
  // Zeta functions
  addGSLSignedFunc<gsl_sf_zeta_int>("zeta","n");
  addGSLRealFunc<gsl_sf_zeta>("zeta","s");
  addGSLSignedFunc<gsl_sf_zetam1_int>("zetam1","n");
  addGSLRealFunc<gsl_sf_zetam1>("zetam1","s");
  addGSLRealRealFunc<gsl_sf_hzeta>("hzeta","s","q");
  addGSLSignedFunc<gsl_sf_eta_int>("eta","n");
  addGSLRealFunc<gsl_sf_eta>("eta","s");
#endif
  
#ifdef STRUCTEXAMPLE
  dummyRecord *fun=createDummyRecord(ve, "test");
  addFunc(fun->e.ve,realReal<sin>,primReal(),"f",formal(primReal(),"x"));
  addVariable<Int>(fun->e.ve,1,primInt(),"x");
#endif
  
  addFunc(ve,writestring,primVoid(),"write",
          formal(primFile(),"file",true),
          formal(primString(),"s"),
          formal(voidFileFunction(),"suffix",true));
  
  addWrite(ve,write<transform>,primTransform(),transformArray());
  addWrite(ve,write<guide *>,primGuide(),guideArray());
  addWrite(ve,write<pen>,primPen(),penArray());
  addFunc(ve,arrayArrayOp<pen,equals>,booleanArray(),"==",
          formal(penArray(),"a"),formal(penArray(),"b"));
  addFunc(ve,arrayArrayOp<pen,notequals>,booleanArray(),"!=",
          formal(penArray(),"a"),formal(penArray(),"b"));

  addFunc(ve,arrayFunction,realArray(),"map",
          formal(realPairFunction(),"f"),
          formal(pairArray(),"a"));
  addFunc(ve,arrayFunction,IntArray(),"map",
          formal(IntRealFunction(),"f"),
          formal(realArray(),"a"));
  
  addConstant<Int>(ve, Int_MAX, primInt(), "intMax");
  addConstant<Int>(ve, Int_MIN, primInt(), "intMin");
  addConstant<double>(ve, HUGE_VAL, primReal(), "inf");
  addConstant<double>(ve, run::infinity, primReal(), "infinity");
  addConstant<double>(ve, DBL_MAX, primReal(), "realMax");
  addConstant<double>(ve, DBL_MIN, primReal(), "realMin");
  addConstant<double>(ve, DBL_EPSILON, primReal(), "realEpsilon");
  addConstant<Int>(ve, DBL_DIG, primInt(), "realDigits");
  addConstant<Int>(ve, RAND_MAX, primInt(), "randMax");
  addConstant<double>(ve, PI, primReal(), "pi");
  addConstant<string>(ve, string(settings::VERSION)+string(SVN_REVISION),
                      primString(),"VERSION");
  
  addVariable<pen>(ve, &processData().currentpen, primPen(), "currentpen");

#ifdef OPENFUNCEXAMPLE
  addOpenFunc(ve, openFunc, primInt(), "openFunc");
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
