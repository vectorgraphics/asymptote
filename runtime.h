/*****
 * runtime.h
 * Andy Hammerlindl 2002/7/31
 *
 * Defines all of the runtime functions that are used by the vm::stack
 * machine.
 *****/

#ifndef RUNTIME_H
#define RUNTIME_H

#include <sstream>
#include <boost/lexical_cast.hpp>
#include <cfloat>

#include "inst.h"
#include "stack.h"
#include "fileio.h"

extern "C" double hypot(double x, double y) throw();
extern "C" double drem(double x, double y) throw();

using vm::pop;

namespace run {
  
// Math
template <double (*func)(double)>
void realReal(vm::stack *s) 
{
  s->push(func(s->template pop<double>()));
}

void intZero(vm::stack *s);
void realZero(vm::stack *s);
void boolFalse(vm::stack *s);
void boolTrue(vm::stack *s);

void intIntMod(vm::stack *s);
void realRealMod(vm::stack *s);
void realFmod(vm::stack *s);
void intIntPow(vm::stack *s);
void realRealPow(vm::stack *s);
void realIntPow(vm::stack *s);
void realAtan2(vm::stack *s);
void realHypot(vm::stack *s);
void realDrem(vm::stack *s);
void intAbs(vm::stack *s);
void intCeil(vm::stack *s);
void intFloor(vm::stack *s);
void intRound(vm::stack *s);
void intRand(vm::stack *s);
void intSrand(vm::stack *s);
void realPi(vm::stack *s);
void intIntMax(vm::stack *s);
void realInfinity(vm::stack *s);
void realRealMax(vm::stack *s);
void realRealMin(vm::stack *s);
void realRealEpsilon(vm::stack *s);
void intRandMax(vm::stack *s);
void boolDeconstruct(vm::stack *s);
void pairExpi(vm::stack *s);
void pairConj(vm::stack *s);

// String concatenation
void concat(vm::stack *s);

// Array operations
void emptyArray(vm::stack *s);
void newArray(vm::stack *s);
void newDeepArray(vm::stack *s);
void newInitializedArray(vm::stack *s);
void arrayRead(vm::stack *s);
void arrayArrayRead(vm::stack *s);
void arrayWrite(vm::stack *s);
void arrayLength(vm::stack *s);
void arrayPush(vm::stack *s);
void arrayPushHelper(vm::stack *s);
void arrayAlias(vm::stack *s);
void arrayConditional(vm::stack *s);
void arrayIntArray(vm::stack *s);
void arraySequence(vm::stack *s);
void intSequence(vm::stack *s);
void arrayFunction(vm::stack *s);
void arrayFind(vm::stack *s);
void arrayAll(vm::stack *s);
void arrayBoolNegate(vm::stack *s);
void arrayBoolSum(vm::stack *s);
void arrayCopy(vm::stack *s);
void array2Copy(vm::stack *s);
void array2Transpose(vm::stack *s);
void pairArrayFFT(vm::stack *s);

// Null operations
void pushNullArray(vm::stack *s);
void pushNullRecord(vm::stack *s);
void pushNullFunction(vm::stack *s);

// Casts
void pairToGuide(vm::stack *s);
void pathToGuide(vm::stack *s);
void guideToPath(vm::stack *s);

// Pair operations
void pairZero(vm::stack *s);
void realRealToPair(vm::stack *s);
void pairNegate(vm::stack *s);
void pairXPart(vm::stack *s);
void pairYPart(vm::stack *s);
void pairLength(vm::stack *s);
void pairAngle(vm::stack *s);
void pairDegrees(vm::stack *s);
void pairUnit(vm::stack *s);
void realDir(vm::stack *s);

// Transform operations
void transformIdentity(vm::stack *s);
void transformInverse(vm::stack *s);
void transformShift(vm::stack *s);
void transformXscale(vm::stack *s);
void transformYscale(vm::stack *s);
void transformScale(vm::stack *s);
void transformScaleInt(vm::stack *s);
void transformScalePair(vm::stack *s);
void transformSlant(vm::stack *s);
void transformRotate(vm::stack *s);
void transformReflect(vm::stack *s);
  
void transformTransformMult(vm::stack *s);
void transformTransformMult(vm::stack *s);
void transformPairMult(vm::stack *s);
void transformPathMult(vm::stack *s);
void transformPenMult(vm::stack *s);
void transformFrameMult(vm::stack *s);
void transformPow(vm::stack *s);

// Path operations
void nullPath(vm::stack *s);
void pathIntPoint(vm::stack *s);
void pathRealPoint(vm::stack *s);
void pathIntPrecontrol(vm::stack *s);
void pathRealPrecontrol(vm::stack *s);
void pathIntPostcontrol(vm::stack *s);
void pathRealPostcontrol(vm::stack *s);
void pathIntDirection(vm::stack *s);
void pathRealDirection(vm::stack *s);
void pathReverse(vm::stack *s);
void pathSubPath(vm::stack *s);
void pathSubPathReal(vm::stack *s);
void pathLength(vm::stack *s);
void pathCyclic(vm::stack *s);
void pathStraight(vm::stack *s);
void pathArcLength(vm::stack *s);
void pathArcTimeOfLength(vm::stack *s);
void pathDirectionTime(vm::stack *s);
void pathIntersectionTime(vm::stack *s);
void pathSize(vm::stack *s);
void pathMax(vm::stack *s);
void pathMin(vm::stack *s);
void pathConcat(vm::stack *s);

// Guide operations
void nullGuide(vm::stack *s);
void newJoin(vm::stack *s);
void newCycle(vm::stack *s);
void newDirguide(vm::stack *s);

// String operations
void emptyString(vm::stack *s);
void stringLength(vm::stack *s);
void stringFind(vm::stack *s);
void stringRfind(vm::stack *s);
void stringSubstr(vm::stack *s);
void stringReverse(vm::stack *s);
void stringInsert(vm::stack *s);
void stringErase(vm::stack *s);
void stringReplace(vm::stack *s);
void stringFormatReal(vm::stack *s);
void stringFormatInt(vm::stack *s);
void stringTime(vm::stack *s);
  
// Pen operations
void defaultpen(vm::stack *s);
void invisiblepen(vm::stack *s);
void rgb(vm::stack *s);
void cmyk(vm::stack *s);
void gray(vm::stack *s);
void linetype(vm::stack *s);
void linewidth(vm::stack *s);
void defaultLineWidth(vm::stack *s);
void fontsize(vm::stack *s);
void defaultFontSize(vm::stack *s);
void penFontsize(vm::stack *s);
void boolPenEq(vm::stack *s);
void penPenPlus(vm::stack *s);
void realPenTimes(vm::stack *s);
void penRealTimes(vm::stack *s);
void penMax(vm::stack *s);
void penMin(vm::stack *s);

// Picture operations
void nullFrame(vm::stack *s);
void frameMax(vm::stack *s);
void frameMin(vm::stack *s);
void draw(vm::stack *s);
void fill(vm::stack *s);
void clip(vm::stack *s);
void add(vm::stack *s);

void postscript(vm::stack *s);
void tex(vm::stack *s);
void texPreamble(vm::stack *s);
void label(vm::stack *s);
void overwrite(vm::stack *s);

void shipout(vm::stack *s);
void stringFilePrefix(vm::stack *s);

// Interactive mode
void suppressOutput(vm::stack *s);
void upToDate(vm::stack *s);

// Execute a shell command  
void system(vm::stack *s);
void abort(vm::stack *s);
void exit(vm::stack *s);
  
// Merge output files  
void merge(vm::stack *s);
  
// Execute an asymptote file
void execute(vm::stack *s);
  
// I/O Routines
void nullFile(vm::stack *s);
void fileOpenOut(vm::stack *s);
void fileOpenIn(vm::stack *s);
void fileOpenXOut(vm::stack *s);
void fileOpenXIn(vm::stack *s);

void fileEof(vm::stack *s);
void fileEol(vm::stack *s);
void fileError(vm::stack *S);
void fileClear(vm::stack *S);
void fileClose(vm::stack *s);
void filePrecision(vm::stack *s);
void fileFlush(vm::stack *s);
void fileDimension1(vm::stack *s);
void fileDimension2(vm::stack *s);
void fileDimension3(vm::stack *s);
void fileCSVMode(vm::stack *s);
void fileLineMode(vm::stack *s);
void fileArray1(vm::stack *s);
void fileArray2(vm::stack *s);
void fileArray3(vm::stack *s);

void readChar(vm::stack *s);

using vm::read;

inline bool checkArray(vm::stack *s, vm::array *a)
{
  if(a == 0) vm::error(s,"dereference of null array");
  return true;
}

inline size_t checkArrays(vm::stack *s, vm::array *a, vm::array *b) 
{
  checkArray(s,a);
  checkArray(s,b);
  
  size_t asize=(size_t) a->size();
  if(asize != (size_t) b->size())
    vm::error(s,"array operation attempted on arrays of different lengths.");
  return asize;
}
  
template<class T, class S>
void cast(vm::stack *s)
{
  s->push((S) s->pop<T>());
}

template<class T>
void stringCast(vm::stack *s)
{
  std::ostringstream buf;
  buf.precision(DBL_DIG);
  buf << s->pop<T>();
  s->push(std::string(buf.str()));
}

template<class T>
void castString(vm::stack *s)
{
  try {
    s->push(boost::lexical_cast<T>(s->pop<std::string>()));
  } catch (boost::bad_lexical_cast&) {
    vm::error(s,"invalid cast.");
  }
}

template<class T, class S>
void arrayToArray(vm::stack *s)
{
  vm::array *a = s->pop<vm::array*>();
  checkArray(s,a);
  unsigned int size=(unsigned int) a->size();
  vm::array *c=new vm::array(size);
  for(unsigned i=0; i < size; i++)
    (*c)[i]=(S) read<T>(a,i);
  s->push(c);
}

template<class T>
void read(vm::stack *s)
{
  camp::file *f = pop<camp::file*>(s);
  T val;
  f->read(val);
  s->push(val);
}

inline void eof(vm::stack *s, camp::file *f, int count) 
{
  std::ostringstream buf;
  buf << "EOF after reading " << count
      << " values from file '" << f->filename() << "'.";
  error(s,buf.str().c_str());
}

inline int Limit(int nx) {return nx == 0 ? INT_MAX : nx;}

template<class T>
void readArray(vm::stack *s)
{
  camp::file *f = s->pop<camp::file*>();
  vm::array *c=new vm::array(0);
  int nx=f->Nx();
  if(nx == -2) {f->read(nx); if(nx == 0) {s->push(c); return;}}
  int ny=f->Ny();
  if(ny == -2) {f->read(ny); if(ny == 0) {s->push(c); return;}}
  int nz=f->Nz();
  if(nz == -2) {f->read(nz); if(nz == 0) {s->push(c); return;}}
  T v;
  if(nx >= 0) {
    for(int i=0; i < Limit(nx); i++) {
      if(ny >= 0) {
	vm::array *ci=new vm::array(0);
	c->push(ci);
	for(int j=0; j < Limit(ny); j++) {
	  if(nz >= 0) {
	    vm::array *cij=new vm::array(0);
	    ci->push(cij);
	    for(int k=0; k < Limit(nz); k++) {
	      f->read(v);
	      if(f->error()) {
		if(nx && ny && nz) eof(s,f,(i*ny+j)*nz+k); s->push(c); return;
	      }
	      cij->push(v);
	    }
	  } else {
	    f->read(v);
	    if(f->error()) {if(nx && ny) eof(s,f,i*ny+j); s->push(c); return;}
	    ci->push(v);
	  }
	}
      } else {
	f->read(v);
	if(f->error()) {if(nx) eof(s,f,i); s->push(c); return;}
	c->push(v);
      }
    }
  } else {
    for(;;) {
      f->read(v);
      if(f->error()) break;
      c->push(v);
      if(f->LineMode() && f->eol()) break;
    }
  }
  s->push(c);
}

inline vm::array *copyArray(vm::stack *s)
{
  vm::array *a=pop<vm::array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  vm::array *c=new vm::array(size);
  for(size_t i=0; i < size; i++) 
    (*c)[i]=(*a)[i];
  return c;
}

inline vm::array *copyArray2(vm::stack *s)
{
  vm::array *a=pop<vm::array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  vm::array *c=new vm::array(size);
  for(size_t i=0; i < size; i++) {
    vm::array *ai=read<vm::array *>(a,i);
    checkArray(s,ai);
    size_t aisize=(size_t) ai->size();
    vm::array *ci=new vm::array(aisize);
    (*c)[i]=ci;
    for(size_t j=0; j < aisize; j++) 
      (*ci)[j]=(*ai)[j];
  }
  return c;
}

} // namespace run

#endif
