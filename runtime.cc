/*****
 * runtime.cc
 * Andy Hammerlindl 2002/7/31
 *
 * Defines some runtime functions used by the stack machine.
 *
 *****/

#include <cassert>
#include <cstdio>
#include <string>
#include <cfloat>
#include <cmath>
#include <sstream>
#include <iostream>
#include <cassert>
#include <sstream>
#include <time.h>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ostringstream;
using std::string;

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "util.h"
#include "pow.h"
#include "errormsg.h"
#include "runtime.h"
#include "settings.h"
#include "guideflags.h"
#include "stack.h"

#include "angle.h"
#include "pair.h"
#include "transform.h"
#include "path.h"
#include "pen.h"
#include "guide.h"
#include "picture.h"
#include "drawpath.h"
#include "drawfill.h"
#include "drawclipbegin.h"
#include "drawclipend.h"
#include "drawlabel.h"
#include "drawverbatim.h"
#include "drawlayer.h"
#include "fileio.h"
#include "genv.h"
#include "builtin.h"

#ifdef HAVE_LIBFFTW3
#include "fftw++.h"
#endif

using namespace vm;
using namespace camp;
using namespace settings;

namespace run {
  
using vm::stack;
using camp::pair;
using camp::transform;

// Math
  
void intZero(stack *s)
{
  s->push(0);
}

void realZero(stack *s)
{
  s->push(0.0);
}

void boolFalse(stack *s)
{
  s->push(false);  
}

void boolTrue(stack *s)
{
  s->push(true);  
}

void boolXor(stack *s)
{
  bool b = s->pop<bool>();
  bool a = s->pop<bool>();
  s->push(a^b ? true : false);  
}

void intIntMod(stack *s)
{
  int y = s->pop<int>();
  int x = s->pop<int>();
  s->push(trans::mod<int>()(x,y,s,0));
}

void realRealMod(stack *s)
{
  double y = s->pop<double>();
  double x = s->pop<double>();
  s->push(trans::mod<double>()(x,y,s,0));
}

void realFmod(stack *s)
{
  double y = s->pop<double>();
  double x = s->pop<double>();
  if (y == 0.0)
    error(s,"Division by zero");
  double val = fmod(x,y);
  s->push(val);
}

void intIntPow(stack *s)
{
  int y = s->pop<int>();
  int x = s->pop<int>();
  s->push(trans::power<int>()(x,y,s,0));
}

void realIntPow(stack *s)
{
  int y = s->pop<int>();
  double x = s->pop<double>();
  s->push(pow(x,y));
}

void realRealPow(stack *s)
{
  double y = s->pop<double>();
  double x = s->pop<double>();
  s->push(pow(x,y));
}

void realAtan2(stack *s)
{ 
  s->push(atan2(s->pop<double>(),s->pop<double>()));
}  

void realHypot(stack *s)
{ 
  double x = s->pop<double>();
  double y = s->pop<double>();
  
  double sx;
  sx = hypot(x,y);
  s->push(sx);
}  

void realDrem(stack *s)
{ 
  double x = s->pop<double>();
  double y = s->pop<double>();
  
  double sx;
  sx = drem(y,x);
  s->push(sx);
}  

void intAbs(stack *s)
{ 
  s->push(abs(s->pop<int>()));
}  

static inline int round(double x) 
{
  return int(x+((x >= 0) ? 0.5 : -0.5));
}

void intCeil(stack *s)
{ 
  double x = s->pop<double>();
  int sx = round(ceil(x));
  s->push(sx);
}  

void intFloor(stack *s)
{ 
  double x = s->pop<double>();
  int sx = round(floor(x));
  s->push(sx);
}  

void intRound(stack *s)
{ 
  double x = s->pop<double>();
  int sx = round(x);
  s->push(sx);
}  

void intSgn(stack *s)
{ 
  double x = s->pop<double>();
  int sx = (x == 0.0 ? 0 : (x > 0.0 ? 1 : -1));
  s->push(sx);
}  

void intRand(stack *s)
{ 
  s->push(rand());
}  

void intSrand(stack *s)
{ 
  int seed = s->pop<int>();
  srand(seed);
}  

void realPi(stack *s)
{ 
  s->push(PI);
}  

void intIntMax(stack *s)
{ 
  s->push(INT_MAX);
}

void realInfinity(stack *s)
{ 
  s->push(HUGE_VAL);
}  

void realRealMax(stack *s)
{ 
  s->push(DBL_MAX);
}

void realRealMin(stack *s)
{ 
  s->push(DBL_MIN);
}  

void realRealEpsilon(stack *s)
{ 
  s->push(DBL_EPSILON);
}  

void intRandMax(stack *s)
{ 
  s->push(RAND_MAX);
}

void boolDeconstruct(stack *s)
{ 
  s->push(settings::deconstruct != 0.0);
}



// Create an empty array.
void emptyArray(stack *s)
{
  s->push(new array(0));
}

// Helper function to create deep arrays.
static array* deepArray(int depth, int *dims)
{
  assert(depth > 0);
  
  if (depth == 1) {
    return new array(dims[0]);
  } else {
    int length = dims[0];
    depth--; dims++;

    array *a = new array(length);

    for (int index = 0; index < length; index++) {
      (*a)[index] = deepArray(depth, dims);
    }
    return a;
  }
}
 

// Create a new array (technically a vector).
// This array will be multidimensional.  First the number of dimensions
// is popped off the stack, followed by each dimension in reverse order.
// The array itself is technically a one dimensional array of one
// dimension arrays and so on.
void newDeepArray(stack *s)
{
  int depth = s->pop<int>();
  assert(depth > 0);

  int *dims = new int[depth];

  for (int index = depth-1; index >= 0; index--)
    dims[index] = s->pop<int>();

  s->push(deepArray(depth, dims));
  delete [] dims;
}

// Creates an array with elements already specified.  First, the number
// of elements is popped off the stack, followed by each element in
// reverse order.
void newInitializedArray(stack *s)
{
  int n = s->pop<int>();
  assert(n >= 0);

  array *a = new array(n);

  for (int index = n-1; index >= 0; index--)
    (*a)[index] = s->pop();

  s->push(a);
}

static void outOfBounds(stack *s, const char *op, int len, int n)
{
    ostringstream buf;
    buf << op << " array of length " << len << " with out-of-bounds index "
	<< n;
    error(s,buf.str().c_str());
}

// Read an element from an array. Checks for initialization & bounds.
void arrayRead(stack *s)
{
  int n = s->pop<int>();
  int n0 = n;
  array *a = s->pop<array*>();

  checkArray(s,a);
  int len=(int) a->size();
  if (n < 0) n += len; // Map indices [-len,-1] to [0,len-1]
  if (n >= 0 && n < len) {
    item i = (*a)[(unsigned) n];
    if (i.empty()) {
      ostringstream buf;
      buf << "read uninitialized value from array at index " << n0;
      error(s,buf.str().c_str());
    }
    s->push(i);
  } else outOfBounds(s,"reading",len,n0);
}

// Read an element from an array of arrays. Checks bounds and initialize
// as necessary.
void arrayArrayRead(stack *s)
{
  int n = s->pop<int>();
  int n0 = n;
  array *a = s->pop<array*>();

  checkArray(s,a);
  int len=(int) a->size();
  if (n < 0) n += len; // Map indices [-len,-1] to [0,len-1]
  if (n >= 0 && n < len) {
    item i = (*a)[(unsigned) n];
    if (i.empty()) i=new array(0);
    s->push(i);
  } else outOfBounds(s,"reading",len,n0);
}

// Write an element to an array.  Increases size if necessary.
void arrayWrite(stack *s)
{
  int n = s->pop<int>();
  array *a = s->pop<array*>();
  item value = s->pop();

  checkArray(s,a);
  int len=(int) a->size();
  if (n < 0) n += len; // Map indices [-len,-1] to [0,len-1]
  if (n < 0) outOfBounds(s,"writing",len,n-len);
  if (a->size() <= (size_t) n)
    a->resize(n+1);
  (*a)[n] = value;
  s->push(value);
}

// Returns the length of an array.
void arrayLength(stack *s)
{
  array *a = s->pop<array*>();
  checkArray(s,a);
  s->push((int)a->size());
}

// Returns the push method for an array.
void arrayPush(stack *s)
{
  array *a = s->pop<array *>();
  checkArray(s,a);
  s->push((callable*)new vm::thunk(new vm::bfunc(arrayPushHelper),a));
}

// The helper function for the push method that does the actual operation.
void arrayPushHelper(stack *s)
{
  array *a = s->pop<array *>();
  item i = s->pop();

  checkArray(s,a);
  a->push(i);
}

void arrayAlias(stack *s)
{
  array *b=pop<array *>(s);
  array *a=pop<array *>(s);
  s->push(a==b);
}

// construct vector obtained by replacing those elements of b for which the
// corresponding elements of a are false by the corresponding element of c.
void arrayConditional(stack *s)
{
  array *c=pop<array *>(s);
  array *b=pop<array *>(s);
  array *a=pop<array *>(s);
  size_t size=(size_t) a->size();
  array *r=new array(size);
  if(b && c) {
    checkArrays(s,a,b);
    checkArrays(s,b,c);
    for(size_t i=0; i < size; i++)
      (*r)[i]=read<bool>(a,i) ? (*b)[i] : (*c)[i];
  } else {
    r->clear();
    if(b) {
      checkArrays(s,a,b);
    for(size_t i=0; i < size; i++)
      if(read<bool>(a,i)) r->push((*b)[i]);
    } else if(c) {
      checkArrays(s,a,c);
      for(size_t i=0; i < size; i++)
      if(!read<bool>(a,i)) r->push((*c)[i]);
    }
  }
  
  s->push(r);
}
  
// Return array formed by indexing array a with elements of integer array b
void arrayIntArray(stack *s)
{
  array *b=pop<array *>(s);
  array *a=pop<array *>(s);
  checkArray(s,a);
  checkArray(s,b);
  size_t asize=(size_t) a->size();
  size_t bsize=(size_t) b->size();
  array *r=new array(bsize);
  for(size_t i=0; i < bsize; i++) {
    int index=read<int>(b,i);
    if(index < 0) index += (int) asize;
    if(index < 0 || index >= (int) asize)
      error(s,"reading out-of-bounds index from array");
    (*r)[i]=(*a)[index];
  }
  s->push(r);
}

// Generate the sequence {f_i : i=0,1,...n-1} given a function f and integer n
void arraySequence(stack *s)
{
  int n=pop<int>(s);
  callable* f = pop<callable*>(s);
  if(n < 0) n=0;
  array *a=new array(n);
  for(int i=0; i < n; ++i) {
    s->push<int>(i);
    f->call(s);
    (*a)[i]=s->pop();
  }
  s->push(a);
}

// Return the array {0,1,...n}
void intSequence(stack *s)
{
  int n=pop<int>(s);
  if(n < 0) n=0;
  array *a=new array(n);
  for(int i=0; i < n; ++i) {
    (*a)[i]=i;
  }
  s->push(a);
}

// Apply a function to each element of an array
void arrayFunction(stack *s)
{
  array *a=pop<array *>(s);
  callable* f = pop<callable*>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; ++i) {
    s->push((*a)[i]);
    f->call(s);
    (*a)[i]=s->pop();
  }
  s->push(a);
}

// In a boolean array, find the index of the nth true value or -1 if not found
// If n is negative, search backwards.
void arrayFind(stack *s)
{
  int n=pop<int>(s);
  array *a=pop<array *>(s);
  checkArray(s,a);
  int size=(int) a->size();
  int j=-1;
  if(n > 0)
    for(int i=0; i < size; i++)
      if(read<bool>(a,i)) {
	n--; if(n == 0) {j=i; break;}
      }
  if(n < 0)
    for(int i=size-1; i >= 0; i--)
      if(read<bool>(a,i)) {
	n++; if(n == 0) {j=i; break;}
      }
  s->push(j);
}

void arrayAll(stack *s)
{
  array *a = s->pop<array*>();
  checkArray(s,a);
  unsigned int size=(unsigned int) a->size();
  bool c=true;
  for(unsigned i=0; i < size; i++)
    if(!boost::any_cast<bool>((*a)[i])) {c=false; break;}
  s->push(c);
}

void arrayBoolNegate(stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=!read<bool>(a,i);
  s->push(c);
}

void arrayBoolSum(stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  int sum=0;
  for(size_t i=0; i < size; i++)
    sum += read<bool>(a,i) ? 1 : 0;
  s->push(sum);
}

void arrayCopy(vm::stack *s)
{
  s->push(copyArray(s));
}

void array2Copy(vm::stack *s)
{
  s->push(copyArray2(s));
}

void array2Transpose(vm::stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t asize=(size_t) a->size();
  array *c=new array(0);
  for(size_t i=0; i < asize; i++) {
    size_t ip=i+1;
    array *ai=read<array *>(a,i);
    checkArray(s,ai);
    size_t aisize=(size_t) ai->size();
    size_t csize=(size_t) c->size();
    if(csize < aisize) {
      c->resize(aisize);
      for(size_t j=csize; j < aisize; j++) {
	(*c)[j]=new array(ip);
      }
    }
    for(size_t j=0; j < aisize; j++) {
    array *cj=read<array *>(c,j);
    if(cj->size() < ip) cj->resize(ip);
    (*cj)[i]=(*ai)[j];
    }
  }
  s->push(c);
}

#ifdef HAVE_LIBFFTW3
// Compute the fast Fourier transform of a pair array
void pairArrayFFT(vm::stack *s)
{
  int sign = s->pop<int>() > 0 ? 1 : -1;
  array *a=pop<array *>(s);
  checkArray(s,a);
  unsigned n=(unsigned) a->size();
  Complex *f=FFTWComplex(n);
  fft1d Forward(n,sign,f);
  
  for(size_t i=0; i < n; i++) {
    pair z=read<pair>(a,i);
    f[i]=Complex(z.getx(),z.gety());
  }
  Forward.fft(f);
  
  array *c=new array(n);
  for(size_t i=0; i < n; i++) {
    Complex z=f[i];
    (*c)[i]=pair(z.real(),z.imag());
  }
  FFTWdelete(f);
  s->push(c);
}
#endif //  HAVE_LIBFFTW3

// Null operations

void pushNullArray(stack *s)
{
  s->push<array *>(0);
}

void pushNullRecord(stack *s)
{
  s->push<vm::frame>(vm::frame());
}

void pushNullFunction(stack *s)
{
  s->push(nullfunc::instance());
}

//Casts

void pairToGuide(stack *s) {
  pair z = s->pop<pair>();
  guide *g = new pairguide(z);
  s->push(g);
}

void pathToGuide(stack *s) {
  path p = s->pop<path>();
  guide *g = new pathguide(p);
  s->push(g);
}

void guideToPath(stack *s) {
  guide *g = s->pop<guide*>();
  path p = g->solve();
  s->push(p);
}

// Pair operations.
void pairZero(stack *s)
{
  s->push(pair(0,0));
}

void realRealToPair(stack *s)
{
  double y = s->pop<double>();
  double x = s->pop<double>();
  pair z(x, y);
  s->push(z);
}

void pairNegate(stack *s)
{
  s->push(-s->pop<pair>());
}

void pairXPart(stack *s)
{
  s->push(s->pop<pair>().getx());
}

void pairYPart(stack *s)
{
  s->push(s->pop<pair>().gety());
}

void pairLength(stack *s)
{
  s->push(s->pop<pair>().length());
}

void pairAngle(stack *s)
{
  s->push(s->pop<pair>().angle());
}

// Return the angle of z in degrees (between 0 and 360)
void pairDegrees(stack *s)
{
  double deg=degrees(s->pop<pair>().angle());
  if(deg < 0) deg += 360; 
  s->push(deg);
}

void pairUnit(stack *s)
{
  s->push(unit(s->pop<pair>()));
}

void realDir(stack *s)
{
  s->push(expi(radians(s->pop<double>())));
}

void pairExpi(stack *s)
{
  s->push(expi(s->pop<double>()));
}

void pairConj(stack *s)
{
  s->push(conj(s->pop<pair>()));
}

void pairDot(stack *s)
{
  pair b = s->pop<pair>();
  pair a = s->pop<pair>();
  s->push(a.getx()*b.getx()+a.gety()*b.gety());
}

void transformIdentity(stack *s)
{
  s->push(new transform(identity()));
}

void transformInverse(stack *s)
{
  transform *t = s->pop<transform*>();
  s->push(new transform(inverse(*t)));
}

void transformShift(stack *s)
{
  pair z = s->pop<pair>();
  s->push(new transform(shift(z)));
}

void transformXscale(stack *s)
{
  double x = s->pop<double>();
  s->push(new transform(xscale(x)));
}

void transformYscale(stack *s)
{
  double x = s->pop<double>();
  s->push(new transform(yscale(x)));
}

void transformScale(stack *s)
{
  double x = s->pop<double>();
  s->push(new transform(scale(x)));
}

void transformScaleInt(stack *s)
{
  double x = (double) s->pop<int>();
  s->push(new transform(scale(x)));
}

void transformScalePair(stack *s)
{
  pair z = s->pop<pair>();
  s->push(new transform(scale(z)));
}

void transformSlant(stack *s)
{
  double x = s->pop<double>();
  s->push(new transform(slant(x)));
}

void transformRotate(stack *s)
{
  pair z = s->pop<pair>();
  double x = s->pop<double>();
  s->push(new transform(rotatearound(z,radians(x))));
}

void transformReflect(stack *s)
{
  pair w = s->pop<pair>();
  pair z = s->pop<pair>();
  s->push(new transform(reflectabout(z,w)));
}

void transformTransformMult(stack *s)
{
  transform *t2 = s->pop<transform*>();
  transform *t1 = s->pop<transform*>();
  s->push(new transform(*t1 * *t2));
}

void transformPairMult(stack *s)
{
  pair z = s->pop<pair>();
  transform *t = s->pop<transform*>();
  s->push((*t)*z);
}

void transformPathMult(stack *s)
{
  path p = s->pop<path>();
  transform *t = s->pop<transform*>();
  s->push(transformed(*t,p));
}

void transformPenMult(stack *s)
{
  pen *p = s->pop<pen*>();
  transform *t = s->pop<transform*>();
  s->push(new pen(transformed(t,*p)));
}

void transformFrameMult(stack *s)
{
  picture *p = s->pop<picture *>();
  transform *t = s->pop<transform*>();
  s->push(transformed(*t,p));
}

void transformPow(stack *s)
{
  int n = s->pop<int>();
  transform *t = s->pop<transform*>();
  transform *T=new transform(identity());
  bool alloc=false;
  if(n < 0) {
    n=-n;
    t=new transform(inverse(*t));
    alloc=true;
  }
  for(int i=0; i < n; i++) (*T)=(*T) * (*t);
  s->push(T);
  if(alloc) delete t;
}

void emptyString(stack *s)
{
  s->push((string) "");
}

void stringLength(stack *s)
{
  string a = s->pop<string>();
  s->push((int) a.length());
}

void stringFind(stack *s)
{
  size_t pos=s->pop<int>();
  string b = s->pop<string>();
  string a = s->pop<string>();
  s->push((int) a.find(b,pos));
}

void stringRfind(stack *s)
{
  size_t pos=s->pop<int>();
  string b = s->pop<string>();
  string a = s->pop<string>();
  s->push((int) a.rfind(b,pos));
}

void stringSubstr(stack *s)
{
  size_t n=s->pop<int>();
  size_t pos=s->pop<int>();
  string a = s->pop<string>();
  if(pos < a.length()) s->push(a.substr(pos,n));
  else s->push((string)"");
}

void stringReverse(stack *s)
{
  string a = s->pop<string>();
  reverse(a.begin(),a.end());
  s->push(a);
}

void stringInsert(stack *s)
{
  string b = s->pop<string>();
  size_t pos=s->pop<int>();
  string a = s->pop<string>();
  if(pos < a.length()) s->push(a.insert(pos,b));
  else s->push(a);
}

void stringErase(stack *s)
{
  size_t n=s->pop<int>();
  size_t pos=s->pop<int>();
  string a = s->pop<string>();
  size_t len=a.length();
  if(pos < len) s->push(a.erase(pos,n));
  else s->push(a);
}

// returns a string constructed by translating all occurrences of the string
// from in an array of string pairs {from,to} to the string to in string s.
void stringReplace(stack *s)
{
  array *translate=s->pop<array*>();
  string S=s->pop<string>();
  checkArray(s,translate);
  size_t size=translate->size();
  for(size_t i=0; i < size; i++) {
    array *a=read<array *>(translate,i);
    checkArray(s,a);
  }
  const char *p=S.c_str();
  ostringstream buf;
  while(*p) {
    for(size_t i=0; i < size;) {
      array *a=read<array *>(translate,i);
      string from=read<string>(a,0);
      size_t len=from.length();
      if(strncmp(p,from.c_str(),len) != 0) {i++; continue;}
      buf << read<string>(a,1);
      p += len;
      if(*p == 0) {s->push(buf.str()); return;}
      i=0;
    }
    buf << *(p++);
  }
  s->push(buf.str());
}

void stringFormatInt(stack *s) 
{
  int x=s->pop<int>();
  string format=s->pop<string>();
  int size=snprintf(NULL,0,format.c_str(),x)+1;
  char *buf=new char[size];
  snprintf(buf,size,format.c_str(),x);
  s->push(string(buf));
  delete [] buf;
}

void stringFormatReal(stack *s) 
{
  double x=s->pop<double>();
  string format=s->pop<string>();
  const char *beginScientific="\\!\\times\\!10^{";
  const char *phantom="\\phantom{+}";
  const char *endScientific="}";
  int size=snprintf(NULL,0,format.c_str(),x)+1
    +(int) (strlen(beginScientific)+strlen(phantom)+strlen(endScientific));
  char *buf=new char[size];
  snprintf(buf,size,format.c_str(),x);

  const char *p0=format.c_str();
  const char *p=p0;
  while (*p != 0) {
    if(*p == '%' && *(p+1) != '%') break;
    p++;
  }
  char *q=buf+(p-p0); // beginning of formatted number
  
  // Ignore any spaces
  while(*q != 0) {
    if(*q != ' ') break;
    q++;
  }
  
  // Remove any spurious minus sign
  if(*q == '-') {
    p=q+1;
    bool zero=true;
    while(*p != 0) {
      if(!isdigit(*p) && *p != '.') break;
      if(isdigit(*p) && *p != '0') {zero=false; break;}
      p++;
    }
    if(zero) remove(q,1);
  }
  
  
  char *r=q;
  bool dp=false;
  while(*r != 0 && (isdigit(*r) || *r == '.' || *r == '+' || *r == '-')) {
    if(*r == '.') dp=true;
    r++;
  }
  if(dp) { // Remove trailing zeros and/or decimal point
    r--;
    unsigned int n=0;
    while(r > q && *r == '0') {r--; n++;}
    if(*r == '.') {r--; n++;}
    if(n > 0) remove(r+1,n);
  }
  
  bool zero=(r == q && *r == '0');
  
  // Translate "E+/E-/e+/e-" exponential notation to TeX
  while(*q != 0) {
    if((*q == 'E' || *q == 'e') && (*(q+1) == '+' || *(q+1) == '-')) {
      if(!zero) q=insert(q,beginScientific);
      bool plus=(*(q+1) == '+');
      remove(q,plus ? 2 : 1);
      if(*q == '-') q++;
      while(*q == '0' && (zero || isdigit(*(q+1)))) remove(q,1);
      while(isdigit(*q)) q++;
      if(!zero) {
	if(plus) q=insert(q,phantom);
	insert(q,endScientific);
      }
      break;
    }
    q++;
  }
  s->push(string(buf));
  delete [] buf;
}

void stringTime(stack *s)
{
  static const size_t n=256;
  static char Time[n]="";
#ifdef HAVE_STRFTIME
  string format = s->pop<string>();
  const time_t bintime=time(NULL);
  strftime(Time,n,format.c_str(),localtime(&bintime));
#else
  s->pop<string>();
#endif  
  s->push((string) Time);
}

// Path operations.

void nullPath(stack *s)
{
  s->push(path());
}

void pathIntPoint(stack *s)
{
  int n = s->pop<int>();
  path p = s->pop<path>();
  s->push(p.point(n));
}

void pathRealPoint(stack *s)
{
  double t = s->pop<double>();
  path p = s->pop<path>();
  s->push(p.point(t));
}

void pathIntPrecontrol(stack *s)
{
  int n = s->pop<int>();
  path p= s->pop<path>();
  s->push(p.precontrol(n));
}

void pathRealPrecontrol(stack *s)
{
  double t = s->pop<double>();
  path p= s->pop<path>();
  s->push(p.precontrol(t));
}

void pathIntPostcontrol(stack *s)
{
  int n = s->pop<int>();
  path p= s->pop<path>();
  s->push(p.postcontrol(n));
}

void pathRealPostcontrol(stack *s)
{
  double t = s->pop<double>();
  path p= s->pop<path>();
  s->push(p.postcontrol(t));
}

void pathIntDirection(stack *s)
{
  int n = s->pop<int>();
  path p= s->pop<path>();
  s->push(unit(p.direction(n)));
}

void pathRealDirection(stack *s)
{
  double t = s->pop<double>();
  path p= s->pop<path>();
  s->push(unit(p.direction(t)));
}

void pathReverse(stack *s)
{
  s->push(s->pop<path>().reverse());
}

void pathSubPath(stack *s)
{
  int e = s->pop<int>();
  int b = s->pop<int>();
  s->push(s->pop<path>().subpath(b,e));
}

void pathSubPathReal(stack *s)
{
  double e = s->pop<double>();
  double b = s->pop<double>();
  s->push(s->pop<path>().subpath(b,e));
}

void pathLength(stack *s)
{
  path p = s->pop<path>();
  s->push(p.length());
}

void pathCyclic(stack *s)
{
  path p = s->pop<path>();
  s->push(p.cyclic());
}

void pathStraight(stack *s)
{
  int i = s->pop<int>();
  path p = s->pop<path>();
  s->push(p.straight(i));
}

void pathArcLength(stack *s)
{
  s->push(s->pop<path>().arclength());
}

void pathArcTimeOfLength(stack *s)
{
  double dval = s->pop<double>();
  path p = s->pop<path>();
  s->push(p.arctime(dval));
}

void pathDirectionTime(stack *s)
{
  pair z = s->pop<pair>();
  path p = s->pop<path>();
  s->push(p.directiontime(z));
}

void pathIntersectionTime(stack *s)
{
  path y = s->pop<path>();
  path x = s->pop<path>();
  s->push(intersectiontime(x,y));
}

void pathSize(stack *s)
{
  path p = s->pop<path>();
  s->push(p.size());
}

void pathConcat(stack *s)
{
  path y = s->pop<path>();
  path x = s->pop<path>();
  s->push(camp::concat(x, y));
}

void pathMin(stack *s)
{
  path p = s->pop<path>();
  s->push(p.bounds().Min());
}

void pathMax(stack *s)
{
  path p = s->pop<path>();
  s->push(p.bounds().Max());
}

// Guide operations.

void nullGuide(stack *s)
{
  s->push((guide*)new pathguide(path()));
}

void newJoin(stack *s)
{
  guide *right = s->pop<guide*>();

  // Read flags to see what goodies come with the join
  int flags = s->pop<int>();
  pair leftGiven, rightGiven;
  double leftCurl=0.0, rightCurl=0.0;
  double leftTension=0.0, rightTension=0.0;
  pair leftCont, rightCont;
  if (flags & RIGHT_CURL) {
    rightCurl = s->pop<double>();
  }
  if (flags & RIGHT_GIVEN) {
    rightGiven = s->pop<pair>();
  }
  if (flags & RIGHT_CONTROL) {
    rightCont = s->pop<pair>();
  }
  if (flags & LEFT_CONTROL) {
    leftCont = s->pop<pair>();
  }
  if (flags & RIGHT_TENSION) {
    rightTension = s->pop<double>();
  }
  if (flags & LEFT_TENSION) {
    leftTension = s->pop<double>();
  }
  if (flags & LEFT_CURL) {
    leftCurl = s->pop<double>();
  }
  if (flags & LEFT_GIVEN) {
    leftGiven = s->pop<pair>();
  }

  guide *left = s->pop<guide*>();

  join *g = new join(left, right);

  if (flags & RIGHT_CURL) {
    g->curlin(rightCurl);
  }
  if (flags & RIGHT_GIVEN) {
    g->dirin(rightGiven);
  }
  if (flags & LEFT_CONTROL) {
    if (flags & RIGHT_CONTROL) {
      g->controls(leftCont, rightCont);
    } else {
      g->controls(leftCont, leftCont);
    } 
  }
  if (flags & LEFT_TENSION) {
    if (flags & TENSION_ATLEAST) {
       g->tensionAtleast();
    }
    if (flags & RIGHT_TENSION) {
      g->tension(leftTension, rightTension);
    } else {
      g->tension(leftTension, leftTension);
    } 
  }
  if (flags & LEFT_CURL) {
    g->curlout(leftCurl);
  }
  if (flags & LEFT_GIVEN) {
    g->dirout(leftGiven);
  }

  s->push((guide*)g);
}

void newCycle(stack *s)
{
  guide *g = new cycle;
  s->push(g);
}

void newDirguide(stack *s)
{
  // Read flags to see what the dirtag is
  int flags = s->pop<int>();
  pair rightGiven;
  double rightCurl=0.0;
  if (flags & RIGHT_CURL) {
    rightCurl = s->pop<double>();
  }
  if (flags & RIGHT_GIVEN) {
    rightGiven = s->pop<pair>();
  }

  guide *base = s->pop<guide*>();

  dirguide *g = new dirguide(base);

  if (flags & RIGHT_CURL) {
    g->curl(rightCurl);
  }
  if (flags & RIGHT_GIVEN) {
    g->dir(rightGiven);
  }


  s->push((guide*)g);
}

// Pen operations.

void newPen(stack *s)
{
  s->push(new pen());
}

void defaultPen(stack *)
{
  defaultpen=startupdefaultpen;
}

void setDefaultPen(stack *s)
{
  pen *p=s->pop<pen*>();
  defaultpen=pen(p->stroke(),p->width(),p->size(),p->colorspace(),
		 p->red(),p->green(),p->blue(),p->gray(),"",
		 p->cap(),p->join(),p->Overwrite(),0);
}

void invisiblePen(stack *s)
{
  s->push(new pen(transparentpen));
}

void rgb(stack *s)
{
  double b = s->pop<double>();
  double g = s->pop<double>();
  double r = s->pop<double>();
  s->push(new pen(r,g,b));
}

void cmyk(stack *s)
{
  double k = s->pop<double>();
  double y = s->pop<double>();
  double m = s->pop<double>();
  double c = s->pop<double>();
  s->push(new pen(c,m,y,k));  
}

void gray(stack *s)
{
  s->push(new pen(s->pop<double>()));  
}

void colors(stack *s)
{  
  pen *p=s->pop<pen*>();
  int n=ColorComponents[p->colorspace()];
  array *a=new array(n);
  
  switch(n) {
  case 0:
    break;
  case 1: 
    (*a)[0]=p->gray(); 
    break;
  case 3:
    (*a)[0]=p->red(); 
    (*a)[1]=p->blue(); 
    (*a)[2]=p->green(); 
    break;
  case 4:
    (*a)[0]=p->cyan();
    (*a)[1]=p->magenta(); 
    (*a)[2]=p->yellow(); 
    (*a)[3]=p->black();
    break;
  default:
    ostringstream buf;
    buf << "Undefined colorspace index: " << n;
    error(s,buf.str().c_str());
    break;
  }
  s->push(a);
}

void pattern(stack *s)
{
  s->push(new pen(setpattern,s->pop<string>()));  
}

void lineType(stack *s)
{
  string t = s->pop<string>();
  s->push(new pen(t));  
}

void lineCap(stack *s)
{
  int n = s->pop<int>();
  s->push(new pen(setlinecap,n >= 0 && n < nCap ? n : DEFCAP));
}

void lineJoin(stack *s)
{
  int n = s->pop<int>();
  s->push(new pen(setlinejoin,n >= 0 && n < nJoin ? n : DEFJOIN));
}

void lineWidth(stack *s)
{
  double x = s->pop<double>();
  s->push(new pen(setlinewidth,x >= 0.0 ? x : DEFWIDTH));
}

void penLineWidth(stack *s)
{
  pen *p=s->pop<pen*>();
  s->push(p->width());  
}

void fontSize(stack *s)
{
  double x = s->pop<double>();
  s->push(new pen(setfontsize,x > 0.0 ? x : 0.0));
}

void penFontSize(stack *s)
{
  pen *p=s->pop<pen*>();
  s->push(p->size());  
}

void overWrite(stack *s)
{
  int n = s->pop<int>();
  s->push(new pen(setoverwrite,n >= 0 && n < nOverwrite ? (overwrite_t) n 
		  : DEFWRITE));
}

void boolPenEq(stack *s)
{
  pen *b = s->pop<pen*>();
  pen *a = s->pop<pen*>();
  s->push((*a) == (*b));
}

void penPenPlus(stack *s)
{
  pen *b = s->pop<pen*>();
  pen *a = s->pop<pen*>();
  s->push(new pen((*a) + (*b)));
}

void realPenTimes(stack *s)
{
  pen *b = s->pop<pen*>();
  double a = s->pop<double>();
  s->push(new pen(a * (*b)));
}

void penRealTimes(stack *s)
{
  double b = s->pop<double>();
  pen *a = s->pop<pen*>();
  s->push(new pen(b * (*a)));
}

void penMax(stack *s)
{
  pen *p = s->pop<pen*>();
  s->push(p->bounds().Max());
}

void penMin(stack *s)
{
  pen *p = s->pop<pen*>();
  s->push(p->bounds().Min());
}

// Picture operations.

void nullFrame(stack *s)
{
  s->push(new picture());
}

void frameMax(stack *s)
{
  picture *pic = s->pop<picture *>();
  s->push(pic->bounds().Max());
}

void frameMin(stack *s)
{
  picture *pic = s->pop<picture *>();
  s->push(pic->bounds().Min());
}


void draw(stack *s)
{
  pen *n = s->pop<pen*>();
  path p = s->pop<path>();
  picture *pic = s->pop<picture*>();

  drawPath *d = new drawPath(p,*n);
  pic->append(d);
}

void fill(stack *s)
{
  double rb = s->pop<double>();
  pair b = s->pop<pair>();
  pen *penb = s->pop<pen*>();
  double ra = s->pop<double>();
  pair a = s->pop<pair>();
  pen *pena = s->pop<pen*>();
  path p = s->pop<path>();
  picture *pic = s->pop<picture*>();
  drawFill *d = new drawFill(p,*pena,a,ra,*penb,b,rb);
  pic->append(d);
}
 
void clip(stack *s)
{
  path p = s->pop<path>();
  picture *pic = s->pop<picture*>();
  clip(*pic, p);
}
  
void add(stack *s)
{
  picture *from = s->pop<picture*>();
  picture *to = s->pop<picture*>();
  to->add(*from);
}

void postscript(stack *s)
{
  string t = s->pop<string>();
  picture *pic = s->pop<picture*>();
  drawVerbatim *d = new drawVerbatim(PostScript,t);
  pic->append(d);
}
  
void tex(stack *s)
{
  string t = s->pop<string>();
  picture *pic = s->pop<picture*>();
  drawVerbatim *d = new drawVerbatim(TeX,t);
  pic->append(d);
}
  
void texPreamble(stack *s)
{
  camp::TeXpreamble.push_back(s->pop<string>()+"\n");
}
  
void layer(stack *s)
{
  picture *pic = s->pop<picture*>();
  drawLayer *d = new drawLayer();
  pic->append(d);
}
  
void label(stack *s)
{
  pen *p = s->pop<pen*>();
  pair a = s->pop<pair>();
  pair z = s->pop<pair>();
  double r = s->pop<double>();
  string t = s->pop<string>();
  picture *pic = s->pop<picture*>();
  drawLabel *d = new drawLabel(t,r,z,a,p);
  pic->append(d);
}
  
void shipout(stack *s)
{
  bool wait = s->pop<bool>();
  string format = s->pop<string>();
  const picture *preamble = s->pop<picture*>();
  picture *pic = s->pop<picture*>();
  string prefix = s->pop<string>();
  pic->shipout(*preamble,prefix == "" ? outname : prefix,format,wait);
}

void stringFilePrefix(stack *s)
{
  s->push((string) outname);
}

// Interactive mode

void interact(stack *s)
{
  bool interaction=s->pop<bool>();
  if(interact::interactive) settings::suppressStandard=!interaction;
}

void upToDate(stack *s)
{
  bool val=s->pop<bool>();
  if(settings::suppressStandard && val == false) return;
  settings::upToDate=val;
}

void boolUpToDate(stack *s)
{
  s->push(interact::interactive && (settings::suppressStandard || 
				    settings::upToDate));
}

// System commands

void system(stack *s)
{
  string str = s->pop<string>();
  
  if(settings::suppressStandard) {s->push(0); return;}
  
  if(safe){
    em->runtime(s->getPos());
    *em << "system() call disabled; override with option -unsafe";
    em->sync();
    s->push(-1);
  }
  else s->push(System(str.c_str()));
}

void abort(stack *s)
{
  string msg = s->pop<string>();
  error(s,msg.c_str());
}

void exit(stack *s)
{
  _exit(s->pop<int>());
}

// Merge output files  

void merge(stack *s)
{
  int ret;
  bool keep = s->pop<bool>();
  string format = s->pop<string>();
  string args = s->pop<string>();
  
  if(settings::suppressStandard) {s->push(0); return;}
  
  if(!checkFormatString(format)) return;
  
  ostringstream cmd,remove;
  cmd << "convert "+args;
  remove << "rm";
  while(!outnameStack->empty()) {
    string name=outnameStack->front();
    cmd << " " << name;
    remove << " " << name;
    outnameStack->pop_front();
  }
  
  string name=buildname(outname,format.c_str());
  cmd << " " << name;
  ret=System(cmd);
  
  if(ret == 0) {
    if(settings::verbose > 0) cout << "Wrote " << name << endl;
    if(!keep && !settings::keep) System(remove);
  }
  
  s->push(ret);
}

void execute(stack *s)
{
  string str = s->pop<string>();
  symbol *id = symbol::trans(str);
  string Outname=outname;
  outname=str;
  size_t p=findextension(outname,suffix);
  if (p < string::npos) outname.erase(p);
  trans::genv ge;
  trans::record *m = ge.loadModule(id);
  if (em->errors() == false) {
    if (m) {
      lambda *l = ge.bootupModule(m);
      assert(l);
      stack s(0);
      s.run(l);
    }
  }
  outname=Outname;
}

// I/O Operations

void nullFile(stack *s)
{
  file *f=&camp::Stdout;
  s->push(f);
}

void fileOpenIn(stack *s)
{
  bool check=s->pop<bool>();
  string filename=s->pop<string>();
  file *f=new ifile(filename,check);
  f->open();
  s->push(f);
}

void fileOpenOut(stack *s)
{
  bool check=s->pop<bool>();
  string filename=s->pop<string>();
  file *f=new ofile(filename,check);
  f->open();
  s->push(f);
}

void fileOpenXIn(stack *s)
{
  bool check=s->pop<bool>();
  string filename=s->pop<string>();
#ifdef HAVE_RPC_RPC_H
  file *f=new ixfile(filename,check);
  s->push(f);
#else  
  error(s,"XDR support not enabled");
#endif
}

void fileOpenXOut(stack *s)
{
  bool check=s->pop<bool>();
  string filename=s->pop<string>();
#ifdef HAVE_RPC_RPC_H
  file *f=new oxfile(filename,check);
  s->push(f);
#else  
  error(s,"XDR support not enabled");
#endif
}

void fileEof(stack *s)
{
  file *f = s->pop<file*>();
  s->push(f->eof());
}

void fileEol(stack *s)
{
  file *f = s->pop<file*>();
  s->push(f->eol());
}

void fileError(stack *s)
{
  file *f = s->pop<file*>();
  s->push(f->error());
}

void fileClear(stack *s)
{
  file *f = s->pop<file*>();
  f->clear();
}

void fileClose(stack *s)
{
  file *f = s->pop<file*>();
  f->close();
}

void filePrecision(stack *s) 
{
  int val = s->pop<int>();
  file *f = s->pop<file*>();
  f->precision(val);
}

void fileFlush(stack *s) 
{
   file *f = s->pop<file*>();
   f->flush();
}

void readChar(stack *s)
{
  file *f = s->pop<file*>();
  char c;
  if(f->isOpen()) f->read(c);
  static char str[1];
  str[0]=c;
  s->push(string(str));
}

// Set file dimensions
void fileDimension1(stack *s) 
{
  int nx = s->pop<int>();
  file *f = s->pop<file*>();
  f->dimension(nx);
  s->push(f);
}

void fileDimension2(stack *s) 
{
  int ny = s->pop<int>();
  int nx = s->pop<int>();
  file *f = s->pop<file*>();
  f->dimension(nx,ny);
  s->push(f);
}

void fileDimension3(stack *s) 
{
  int nz = s->pop<int>();
  int ny = s->pop<int>();
  int nx = s->pop<int>();
  file *f = s->pop<file*>();
  f->dimension(nx,ny,nz);
  s->push(f);
}

// Set file to read comma-separated values
void fileCSVMode(stack *s) 
{
  file *f = s->pop<file*>();
  f->CSVMode(true);
  s->push(f);
}

// Set file to read arrays in line-at-a-time mode
void fileLineMode(stack *s) 
{
  file *f = s->pop<file*>();
  f->LineMode(true);
  s->push(f);
}

// Set file to read an array1 (1 int size followed by a 1-d array)
void fileArray1(stack *s) 
{
  file *f = s->pop<file*>();
  f->dimension(-2);
  s->push(f);
}

// Set file to read an array2 (2 int sizes followed by a 2-d array)
void fileArray2(stack *s) 
{
  file *f = s->pop<file*>();
  f->dimension(-2,-2);
  s->push(f);
}

// Set file to read an array1 (3 int sizes followed by a 3-d array)
void fileArray3(stack *s) 
{
  file *f = s->pop<file*>();
  f->dimension(-2,-2,-2);
  s->push(f);
}

} // namespace run
