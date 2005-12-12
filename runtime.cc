/*****
 * runtime.cc
 * Andy Hammerlindl 2002/7/31
 *
 * Defines some runtime functions used by the stack machine.
 *
 *****/

#include <cassert>
#include <cstdio>
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
#include "callable.h"

#include "angle.h"
#include "pair.h"
#include "triple.h"
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
#include "drawgsave.h"
#include "drawgrestore.h"
#include "drawlayer.h"
#include "drawimage.h"
#include "drawgroup.h"
#include "fileio.h"
#include "genv.h"
#include "builtin.h"
#include "texfile.h"
#include "pipestream.h"
#include "parser.h"

#include "dec.h"

#ifdef HAVE_LIBFFTW3
#include "fftw++.h"
#endif

using namespace vm;
using namespace camp;
using namespace settings;

const int camp::ColorComponents[]={0,0,1,3,4,0};

namespace vm {
const char *arraymismatch=
  "operation attempted on arrays of different lengths.";
}

namespace loop {
  void doIRunnable(absyntax::runnable *r, bool embedded=false);
  void doITree(absyntax::block *tree, bool embedded=false);
}
  
namespace run {
  
using vm::stack;
using vm::frame;
using camp::pair;
using camp::triple;
using camp::transform;
using mem::string;

// Math
  
void dividebyzero(size_t i=0)
{
  std::ostringstream buf;
  if(i > 0) buf << "array element " << i << ": ";
  buf << "Divide by zero";
  error(buf.str().c_str());
}
  
void integeroverflow(size_t i=0)
{
  std::ostringstream buf;
  if(i > 0) buf << "array element " << i << ": ";
  buf << "Integer overflow";
  error(buf.str().c_str());
}
  
void boolDeconstruct(stack *s)
{ 
  s->push(settings::deconstruct != 0.0);
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
  int depth = pop<int>(s);
  assert(depth > 0);

  int *dims = new int[depth];

  for (int index = depth-1; index >= 0; index--)
    dims[index] = pop<int>(s);

  s->push(deepArray(depth, dims));
  delete [] dims;
}

// Creates an array with elements already specified.  First, the number
// of elements is popped off the stack, followed by each element in
// reverse order.
void newInitializedArray(stack *s)
{
  int n = pop<int>(s);
  assert(n >= 0);

  array *a = new array(n);

  for (int index = n-1; index >= 0; index--)
    (*a)[index] = pop(s);

  s->push(a);
}

// Similar to newInitializedArray, but after the n elements, append another
// array to it.
void newAppendedArray(stack *s)
{
  int n = pop<int>(s);
  assert(n >= 0);

  array *tail = pop<array *>(s);

  array *a = new array(n);

  for (int index = n-1; index >= 0; index--)
    (*a)[index] = pop(s);
  
  copy(tail->begin(), tail->end(), back_inserter(*a));

  s->push(a);
}

#ifdef HAVE_LIBFFTW3
// Compute the fast Fourier transform of a pair array
void pairArrayFFT(stack *s)
{
  int sign = pop<int>(s) > 0 ? 1 : -1;
  array *a=pop<array*>(s);
  checkArray(a);
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
  
void quadraticRoots(stack *s)
{
  double c = pop<double>(s);
  double b = pop<double>(s);
  double a = pop<double>(s);
  quadraticroots q(a,b,c);
  array *roots=new array(q.roots);
  if(q.roots >= 1) (*roots)[0]=q.t1;
  if(q.roots == 2) (*roots)[1]=q.t2;
  s->push(roots);
}
  
void cubicRoots(stack *s)
{
  double d = pop<double>(s);
  double c = pop<double>(s);
  double b = pop<double>(s);
  double a = pop<double>(s);
  cubicroots q(a,b,c,d);
  array *roots=new array(q.roots);
  if(q.roots >= 1) (*roots)[0]=q.t1;
  if(q.roots >= 2) (*roots)[1]=q.t2;
  if(q.roots == 3) (*roots)[2]=q.t3;
  s->push(roots);
}
  
// Null operations

void pushNullArray(stack *s)
{
  s->push<array*>(0);
}

void pushNullRecord(stack *s)
{
  s->push<frame*>(0);
}

void pushNullFunction(stack *s)
{
  s->push(nullfunc::instance());
}

// Default operations

default_t def;

// Put the default value token on the stack (in place of an argument when
// making a function call).
void pushDefault(stack *s)
{
  s->push(&def);
}

// Test if the value on the stack is the default value token.
void isDefault(stack *s)
{
  s->push(isdefault(pop(s)));
}

// Casts

void pairToGuide(stack *s) {
  pair z = pop<pair>(s);
  guide *g = new pairguide(z);
  s->push(g);
}

void pathToGuide(stack *s) {
  path p = pop<path>(s);
  guide *g = new pathguide(p);
  s->push(g);
}

void guideToPath(stack *s) {
  guide *g = pop<guide*>(s);
  path p = g->solve();
  s->push(p);
}

// Pair operations
void pairZero(stack *s)
{
  static pair zero;
  s->push(&zero);
}

void realRealToPair(stack *s)
{
  double y = pop<double>(s);
  double x = pop<double>(s);
  s->push(new pair(x,y));
}

void pairNegate(stack *s)
{
  s->push(-pop<pair>(s));
}

void pairXPart(stack *s)
{
  s->push(pop<pair>(s).getx());
}

void pairYPart(stack *s)
{
  s->push(pop<pair>(s).gety());
}

// Triple operations

void tripleZero(stack *s)
{
  static triple zero;
  s->push(&zero);
}

void realRealRealToTriple(stack *s)
{
  double z = pop<double>(s);
  double y = pop<double>(s);
  double x = pop<double>(s);
  s->push(new triple(x,y,z));
}

void tripleXPart(stack *s)
{
  s->push(pop<triple>(s).getx());
}

void tripleYPart(stack *s)
{
  s->push(pop<triple>(s).gety());
}

void tripleZPart(stack *s)
{
  s->push(pop<triple>(s).getz());
}

void realTripleMult(stack *s)
{
  triple v = pop<triple>(s);
  double x = pop<double>(s);
  s->push(x*v);
}

void tripleRealMult(stack *s)
{
  double x = pop<double>(s);
  triple v = pop<triple>(s);
  s->push(x*v);
}

void tripleRealDivide(stack *s)
{
  double x = pop<double>(s);
  triple v = pop<triple>(s);
  s->push(v/x);
}

// Transforms
  
void transformIdentity(stack *s)
{
  s->push(new transform(identity()));
}

void transformTransformMult(stack *s)
{
  transform *t2 = pop<transform*>(s);
  transform *t1 = pop<transform*>(s);
  s->push(new transform(*t1 * *t2));
}

void transformPairMult(stack *s)
{
  pair z = pop<pair>(s);
  transform *t = pop<transform*>(s);
  s->push((*t)*z);
}

void transformPathMult(stack *s)
{
  path p = pop<path>(s);
  transform *t = pop<transform*>(s);
  s->push(transformed(*t,p));
}

void transformPenMult(stack *s)
{
  pen *p = pop<pen*>(s);
  transform *t = pop<transform*>(s);
  s->push(new pen(transformed(t,*p)));
}

void transformFrameMult(stack *s)
{
  picture *p = pop<picture*>(s);
  transform *t = pop<transform*>(s);
  s->push(transformed(*t,p));
}

void transformPow(stack *s)
{
  int n = pop<int>(s);
  transform *t = pop<transform*>(s);
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

void transformXPart(stack *s)
{
  s->push(pop<transform>(s).getx());
}

void transformYPart(stack *s)
{
  s->push(pop<transform>(s).gety());
}

void transformXXPart(stack *s)
{
  s->push(pop<transform>(s).getxx());
}

void transformXYPart(stack *s)
{
  s->push(pop<transform>(s).getxy());
}
  
void transformYXPart(stack *s)
{
  s->push(pop<transform>(s).getyx());
}
  
void transformYYPart(stack *s)
{
  s->push(pop<transform>(s).getyy());
}
  
void real6ToTransform(stack *s)
{
  double yy = pop<double>(s);
  double yx = pop<double>(s);
  double xy = pop<double>(s);
  double xx = pop<double>(s);
  double y = pop<double>(s);
  double x = pop<double>(s);
  s->push(new transform(x,y,xx,xy,yx,yy));
}

string emptystring;
void emptyString(stack *s)
{
  s->push(&emptystring);
}

// returns a string constructed by translating all occurrences of the string
// from in an array of string pairs {from,to} to the string to in string s.
void stringReplace(stack *s)
{
  array *translate=pop<array*>(s);
  string *S=pop<string*>(s);
  checkArray(translate);
  size_t size=translate->size();
  for(size_t i=0; i < size; i++) {
    array *a=read<array*>(translate,i);
    checkArray(a);
  }
  const char *p=S->c_str();
  ostringstream buf;
  while(*p) {
    for(size_t i=0; i < size;) {
      array *a=read<array*>(translate,i);
      string* from=read<string*>(a,0);
      size_t len=from->length();
      if(strncmp(p,from->c_str(),len) != 0) {i++; continue;}
      buf << read<string>(a,1);
      p += len;
      if(*p == 0) {s->push<string>(buf.str()); return;}
      i=0;
    }
    buf << *(p++);
  }
  s->push<string>(buf.str());
}

void stringFormatInt(stack *s) 
{
  int x=pop<int>(s);
  string *format=pop<string*>(s);
  int size=snprintf(NULL,0,format->c_str(),x)+1;
  if(size < 1) size=255; // Workaround for non-C99 compliant systems.
  char *buf=new char[size];
  snprintf(buf,size,format->c_str(),x);
  s->push<string>(buf);
  delete [] buf;
}

void stringFormatReal(stack *s) 
{
  ostringstream out;
  
  double x=pop<double>(s);
  string *format=pop<string*>(s);
  
  const char *phantom="\\phantom{+}";
  const char *p0=format->c_str();
  
  const char *p=p0;
  const char *start=NULL;
  while (*p != 0) {
    if(*p == '%') {
      p++;
      if(*p != '%') {start=p-1; break;}
    }
    out << *(p++);
  }
  
  if(!start) {s->push<string>(out.str()); return;}
  
  // Allow at most 1 argument  
  while (*p != 0) {
    if(*p == '*' || *p == '$') {s->push<string>(out.str()); return;}
    if(isupper(*p) || islower(*p)) {p++; break;}
    p++;
  }
  
  const char *tail=p;
  string f=format->substr(start-p0,tail-start);
  int size=snprintf(NULL,0,f.c_str(),x)+1;
  if(size < 1) size=255; // Workaround for non-C99 compliant systems.
  char *buf=new char[size];
  snprintf(buf,size,f.c_str(),x);

  bool trailingzero=f.find("#") < string::npos;
  bool plus=f.find("+") < string::npos;
  bool space=f.find(" ") < string::npos;
  
  char *q=buf; // beginning of formatted number

  if(*q == ' ') {
    out << phantom;
    q++;
  }
  
  // Remove any spurious sign
  if(*q == '-' || *q == '+') {
    p=q+1;
    bool zero=true;
    while(*p != 0) {
      if(!isdigit(*p) && *p != '.') break;
      if(isdigit(*p) && *p != '0') {zero=false; break;}
      p++;
    }
    if(zero) {
      q++;
      if(plus || space) out << phantom;
    }
  }
  
  const char *r=p=q;
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
    while(q <= r) out << *(q++);
    if(!trailingzero) q += n;
  }
  
  bool zero=(r == p && *r == '0') && !trailingzero;
  
  // Translate "E+/E-/e+/e-" exponential notation to TeX
  while(*q != 0) {
    if((*q == 'E' || *q == 'e') && (*(q+1) == '+' || *(q+1) == '-')) {
      if(!zero) out << "\\!\\times\\!10^{";
      bool plus=(*(q+1) == '+');
      q++;
      if(plus) q++;
      if(*q == '-') out << *(q++);
      while(*q == '0' && (zero || isdigit(*(q+1)))) q++;
      while(isdigit(*q)) out << *(q++);
      if(!zero) {
	if(plus) out << phantom;
	out << "}";
      }
      break;
    }
    out << *(q++);
  }
  
  while(*tail != 0) 
    out << *(tail++);
  
  delete [] buf;
  s->push<string>(out.str());
}

void stringTime(stack *s)
{
  static const size_t n=256;
  static char Time[n]="";
#ifdef HAVE_STRFTIME
  string *format = pop<string*>(s);
  const time_t bintime=time(NULL);
  strftime(Time,n,format->c_str(),localtime(&bintime));
#else
  pop<string*>(s);
#endif  
  s->push<string>(Time);
}

// Path operations.

void nullPath(stack *s)
{
  static path *nullpath=new path();
  s->push(nullpath);
}

void pathSize(stack *s)
{
  path p = pop<path>(s);
  s->push(p.size());
}

void pathConcat(stack *s)
{
  path y = pop<path>(s);
  path x = pop<path>(s);
  s->push(camp::concat(x, y));
}

void pathMin(stack *s)
{
  path p = pop<path>(s);
  s->push(p.bounds().Min());
}

void pathMax(stack *s)
{
  path p = pop<path>(s);
  s->push(p.bounds().Max());
}
  
// Guide operations.

void nullGuide(stack *s)
{
  s->push<guide *>(new pathguide(path()));
}

void dotsGuide(stack *s)
{
  array *a=pop<array*>(s);

  guidevector v;
  size_t size=a->size();
  for (size_t i=0; i < size; ++i)
    v.push_back(a->read<guide*>(i));

  s->push((guide *) new multiguide(v));
}

void dashesGuide(stack *s)
{
  static camp::curlSpec curly;
  static specguide curlout(&curly, camp::OUT);
  static specguide curlin(&curly, camp::IN);

  array *a=pop<array*>(s);
  size_t n=a->size();

  // a--b is equivalent to a{curl 1}..{curl 1}b
  guidevector v;
  if (n > 0)
    v.push_back(a->read<guide*>(0));

  if (n==1) {
    v.push_back(&curlout);
    v.push_back(&curlin);
  }
  else
    for (size_t i=1; i<n; ++i) {
      v.push_back(&curlout);
      v.push_back(&curlin);
      v.push_back(a->read<guide*>(i));
    }

  s->push((guide *) new multiguide(v));
}

void cycleGuide(stack *s)
{
  s->push((guide *) new cycletokguide());
}
      

void dirSpec(stack *s)
{
  camp::side d=(camp::side) pop<int>(s);
  camp::dirSpec *sp=new camp::dirSpec(angle(pop<pair>(s)));

  s->push((guide *) new specguide(sp, d));
}

void curlSpec(stack *s)
{
  camp::side d=(camp::side) pop<int>(s);
  camp::curlSpec *sp=new camp::curlSpec(pop<double>(s));

  s->push((guide *) new specguide(sp, d));
}

void realRealTension(stack *s)
{
  bool atleast=pop<bool>(s);
  tension  tin(pop<double>(s), atleast),
          tout(pop<double>(s), atleast);

  s->push((guide *) new tensionguide(tout, tin));
}

void pairPairControls(stack *s)
{
  pair  zin=pop<pair>(s),
       zout=pop<pair>(s);

  s->push((guide *) new controlguide(zout, zin));
}

// Pen operations.

void newPen(stack *s)
{
  s->push(new pen());
}

void boolPenEq(stack *s)
{
  pen *b = pop<pen*>(s);
  pen *a = pop<pen*>(s);
  s->push((*a) == (*b));
}

void boolPenNeq(stack *s)
{
  pen *b = pop<pen*>(s);
  pen *a = pop<pen*>(s);
  s->push((*a) != (*b));
}

void penPenPlus(stack *s)
{
  pen *b = pop<pen*>(s);
  pen *a = pop<pen*>(s);
  s->push(new pen((*a) + (*b)));
}

void realPenTimes(stack *s)
{
  pen *b = pop<pen*>(s);
  double a = pop<double>(s);
  s->push(new pen(a * (*b)));
}

void penRealTimes(stack *s)
{
  double b = pop<double>(s);
  pen *a = pop<pen*>(s);
  s->push(new pen(b * (*a)));
}

void penMax(stack *s)
{
  pen *p = pop<pen*>(s);
  s->push(p->bounds().Max());
}

void penMin(stack *s)
{
  pen *p = pop<pen*>(s);
  s->push(p->bounds().Min());
}

// Picture operations.

void newFrame(stack *s)
{
  s->push(new picture());
}

void boolNullFrame(stack *s)
{
  picture *b = pop<picture*>(s);
  s->push(b->null());
}

void frameMax(stack *s)
{
  picture *pic = pop<picture*>(s);
  s->push(pic->bounds().Max());
}

void frameMin(stack *s)
{
  picture *pic = pop<picture*>(s);
  s->push(pic->bounds().Min());
}

void fill(stack *s)
{
  pen *n = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  checkArray(p);
  pic->append(new drawFill(p,*n));
}
 
void latticeShade(stack *s)
{
  array *pens=copyArray(pop<array*>(s));
  pen *n = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  checkArray(p);
  checkArray(pens);
  pic->append(new drawLatticeShade(p,*n,pens));
}
 
void axialShade(stack *s)
{
  pair b = pop<pair>(s);
  pen *penb = pop<pen*>(s);
  pair a = pop<pair>(s);
  pen *pena = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  checkArray(p);
  pic->append(new drawAxialShade(p,*pena,a,*penb,b));
}
 
void radialShade(stack *s)
{
  double rb = pop<double>(s);
  pair b = pop<pair>(s);
  pen *penb = pop<pen*>(s);
  double ra = pop<double>(s);
  pair a = pop<pair>(s);
  pen *pena = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  checkArray(p);
  pic->append(new drawRadialShade(p,*pena,a,ra,*penb,b,rb));
}
 
void gouraudShade(stack *s)
{
  array *edges=copyArray(pop<array*>(s));
  array *vertices=copyArray(pop<array*>(s));
  array *pens=copyArray(pop<array*>(s));
  pen *n = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  checkArray(p);
  checkArrays(pens,vertices);
  checkArrays(vertices,edges);
  pic->append(new drawGouraudShade(p,*n,pens,vertices,edges));
}
 
// Clip a picture to a superpath using the given fill rule.
// Subsequent additions to the picture will not be affected by the clipping.
void clip(stack *s)
{
  pen *n = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  pic->enclose(new drawClipBegin(p,*n),new drawClipEnd());
}
  
void beginClip(stack *s)
{
  pen *n = pop<pen*>(s);
  array *p=copyArray(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  pic->append(new drawClipBegin(p,*n,false));
}

void inside(stack *s)
{
  pen *n = pop<pen*>(s);
  pair z = pop<pair>(s);
  array *p=copyArray(pop<array*>(s));
  checkArray(p);
  size_t size=p->size();
  int count=0;
  for(size_t i=0; i < size; i++) 
    count += read<path *>(p,i)->inside(z);
  s->push(n->inside(count));
}
 
void postscript(stack *s)
{
  string *t = pop<string*>(s);
  picture *pic = pop<picture*>(s);
  drawVerbatim *d = new drawVerbatim(PostScript,*t);
  pic->append(d);
}
  
void tex(stack *s)
{
  string *t = pop<string*>(s);
  picture *pic = pop<picture*>(s);
  drawVerbatim *d = new drawVerbatim(TeX,*t);
  pic->append(d);
}
  
void texPreamble(stack *s)
{
  string t = pop<string>(s)+"\n";
  camp::TeXpipepreamble.push_back(t);
  camp::TeXpreamble.push_back(t);
}
  
void layer(stack *s)
{
  picture *pic = pop<picture*>(s);
  drawLayer *d = new drawLayer();
  pic->append(d);
}
  
void image(stack *s)
{
  pair final = pop<pair>(s);
  pair initial = pop<pair>(s);
  array *p=copyArray(pop<array*>(s));
  array *a=copyArray2(pop<array*>(s));
  picture *pic = pop<picture*>(s);
  drawImage *d = new drawImage(a,p,matrix(initial,final));
  pic->append(d);
}
  
void shipout(stack *s)
{
  array *GUIdelete=pop<array*>(s);
  array *GUItransform=pop<array*>(s);
  bool quiet = pop<bool>(s);
  bool wait = pop<bool>(s);
  string *format = pop<string*>(s);
  const picture *preamble = pop<picture*>(s);
  picture *pic = pop<picture*>(s);
  string prefix = pop<string>(s);
  if(prefix.empty()) prefix=outname;
  
  size_t size=checkArrays(GUItransform,GUIdelete);
  
  if(settings::deconstruct || size) {
    picture *result=new picture;
    unsigned level=0;
    unsigned i=0;
    nodelist::iterator p;
    for(p = pic->nodes.begin(); p != pic->nodes.end(); ++p) {
      bool Delete;
      transform t;
      if(i < size) {
	t=*(read<transform*>(GUItransform,i));
	Delete=read<bool>(GUIdelete,i);
      } else {
	t=identity();
	Delete=false;
      }
      picture *group=new picture;
// Ignore unclosed begingroups but not spurious endgroups.
      const char *nobegin="endgroup without matching begingroup";
      assert(*p);
      if((*p)->endgroup()) error(nobegin);
      if((*p)->begingroup()) {
	++level;
	while(p != pic->nodes.end() && level) {
	  drawElement *e=t.isIdentity() ? *p : (*p)->transformed(t);
	  group->append(e);
	  ++p;
	  if(p == pic->nodes.end()) break;
	  assert(*p);
	  if((*p)->begingroup()) ++level;
	  if((*p)->endgroup()) if(level) --level;
	  else error(nobegin);
	}
      }
      if(p == pic->nodes.end()) break;
      assert(*p);
      drawElement *e=t.isIdentity() ? *p : (*p)->transformed(t);
      group->append(e);
      if(!group->empty()) {
	if(settings::deconstruct) {
	  ostringstream buf;
	  buf << prefix << "_" << i;
	  group->shipout(*preamble,buf.str(),"tgif",false,true,Delete);
	}
	++i;
      }
      if(size && !Delete) result->add(*group);
    }
    if(size) pic=result;
  }

  pic->shipout(*preamble,prefix,*format,wait,quiet);
}

// System commands

void cleanup()
{
  defaultpen=camp::pen::startupdefaultpen();
  if(!interact::interactive) settings::scrollLines=0;
  
  if(TeXinitialized) {
    camp::TeXpipepreamble.clear();
    camp::TeXpreamble.clear();
    camp::tex.pipeclose();
    TeXinitialized=false;
  }
}

extern callable *atExitFunction;

void exitFunction(stack *s)
{
  if(atExitFunction && !nullfunc::instance()->compare(atExitFunction)) {
    atExitFunction->call(s);
    atExitFunction=NULL;
  }
  cleanup();
}
  
void updateFunction(stack *s)
{
  if(atExitFunction && !nullfunc::instance()->compare(atExitFunction))
    atExitFunction->call(s);
}

// Wrapper for the stack::load() method.
void loadModule(stack *s)
{
  string *index= pop<string*>(s);
  s->load(*index);
}

void changeDirectory(stack *s)
{
  string *d=pop<string*>(s);
  int rc=setPath(d->c_str());
  if(rc != 0) {
    ostringstream buf;
    buf << "Cannot change to directory '" << *d << "'";
    error(buf.str().c_str());
  }
  char *p=getPath();
  if(p && interact::interactive) 
    cout << p << endl;
  s->push<string>(p);
}

void scrollLines(stack *s)
{
  int n=pop<int>(s);
  settings::scrollLines=n;
}

// I/O Operations

void standardOut(stack *s)
{
  file *f=&camp::Stdout;
  s->push(f);
}

void nullFile(stack *s)
{
  file *f=&camp::nullfile;
  s->push(f);
}

void fileOpenIn(stack *s)
{
  string *comment=pop<string*>(s);
  bool check=pop<bool>(s);
  string *filename=pop<string*>(s);
  char c=*comment == "" ? (char) 0 : (*comment)[0];
  file *f=new ifile(*filename,check,c);
  f->open();
  s->push(f);
}

void fileOpenOut(stack *s)
{
  bool append=pop<bool>(s);
  string *filename=pop<string*>(s);
  file *f=new ofile(*filename,append);
  f->open();
  s->push(f);
}

void fileOpenXIn(stack *s)
{
#ifdef HAVE_RPC_RPC_H
  bool check=pop<bool>(s);
  string *filename=pop<string*>(s);
  file *f=new ixfile(*filename,check);
  s->push(f);
#else  
  error("XDR support not enabled");
#endif
}

void fileOpenXOut(stack *s)
{
#ifdef HAVE_RPC_RPC_H
  bool append=pop<bool>(s);
  string *filename=pop<string*>(s);
  file *f=new oxfile(*filename,append);
  s->push(f);
#else  
  error("XDR support not enabled");
#endif
}

void fileEof(stack *s)
{
  file *f = pop<file*>(s);
  s->push(f->eof());
}

void fileEol(stack *s)
{
  file *f = pop<file*>(s);
  s->push(f->eol());
}

void fileError(stack *s)
{
  file *f = pop<file*>(s);
  s->push(f->error());
}

void fileClear(stack *s)
{
  file *f = pop<file*>(s);
  f->clear();
}

void fileClose(stack *s)
{
  file *f = pop<file*>(s);
  f->close();
}

void filePrecision(stack *s) 
{
  int val = pop<int>(s);
  file *f = pop<file*>(s);
  f->precision(val);
}

void fileFlush(stack *s) 
{
   file *f = pop<file*>(s);
   f->flush();
}

void readChar(stack *s)
{
  file *f = pop<file*>(s);
  char c;
  if(f->isOpen()) f->read(c);
  static char str[1];
  str[0]=c;
  s->push<string>(str);
}

void writestring(stack *s)
{
  callable *suffix=pop<callable *>(s,NULL);
  string S=pop<string>(s);
  vm::item it=pop(s);
  bool defaultfile=isdefault(it);
  camp::file *f=defaultfile ? &camp::Stdout : vm::get<camp::file*>(it);
  if(!f->isOpen()) return;
  if(S != "") f->write(S);
  if(f->text()) {
    if(suffix) {
      s->push(f);
      suffix->call(s);
    } else if(defaultfile) f->writeline();
  }
}

// Set file dimensions
void fileDimension1(stack *s) 
{
  int nx = pop<int>(s);
  file *f = pop<file*>(s);
  f->dimension(nx);
  s->push(f);
}

void fileDimension2(stack *s) 
{
  int ny = pop<int>(s);
  int nx = pop<int>(s);
  file *f = pop<file*>(s);
  f->dimension(nx,ny);
  s->push(f);
}

void fileDimension3(stack *s) 
{
  int nz = pop<int>(s);
  int ny = pop<int>(s);
  int nx = pop<int>(s);
  file *f = pop<file*>(s);
  f->dimension(nx,ny,nz);
  s->push(f);
}

// Set file to read comma-separated values
void fileCSVMode(stack *s) 
{
  bool b = pop<bool>(s);
  file *f = pop<file*>(s);
  f->CSVMode(b);
  s->push(f);
}

// Set file to read arrays in line-at-a-time mode
void fileLineMode(stack *s) 
{
  bool b = pop<bool>(s);
  file *f = pop<file*>(s);
  f->LineMode(b);
  s->push(f);
}

// Set file to read/write single-precision XDR values.
void fileSingleMode(stack *s) 
{
  bool b = pop<bool>(s);
  file *f = pop<file*>(s);
  f->SingleMode(b);
  s->push(f);
}

// Set file to read an array1 (1 int size followed by a 1-d array)
void fileArray1(stack *s) 
{
  file *f = pop<file*>(s);
  f->dimension(-2);
  s->push(f);
}

// Set file to read an array2 (2 int sizes followed by a 2-d array)
void fileArray2(stack *s) 
{
  file *f = pop<file*>(s);
  f->dimension(-2,-2);
  s->push(f);
}

// Set file to read an array3 (3 int sizes followed by a 3-d array)
void fileArray3(stack *s) 
{
  file *f = pop<file*>(s);
  f->dimension(-2,-2,-2);
  s->push(f);
}

// Utilities



} // namespace run
