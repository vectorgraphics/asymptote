/*****
 * env.cc
 * Andy Hammerlindl 2002/6/20
 *
 * Keeps track of the namespaces of variables and types when traversing
 * the abstract syntax.
 *****/

#include <utility>

#include "errormsg.h"
#include "env.h"
#include "genv.h"
#include "entry.h"

using namespace sym;
using namespace types;

namespace trans {

namespace {
function *inittype();
function *bootuptype();
}

// The dummy environment of the global environment.
// Used purely for global variables and static code blocks of file
// level modules.
env::env(genv &ge, modifier sord)
  : level(new frame(0, 0)),
    recordLevel(0),
    recordType(0),
    l(new vm::lambda),
    funtype(bootuptype()),
    ge(ge),
    parent(0),
    sord(sord),
    perm(READONLY),
    te(ge.te),
    ve(ge.ve),
    me(ge.me),
    program(),
    numLabels(0)
{
  sord_stack.push(sord);
}

// Defines a new function environment.
env::env(function *t, env &parent, modifier sord)
  : level(new frame(parent.getFrame(), t->sig.getNumFormals())),
    recordLevel(parent.recordLevel),
    recordType(parent.recordType),
    l(new vm::lambda),
    funtype(t),
    ge(parent.ge),
    parent(&parent),
    sord(sord),
    perm(READONLY),
    te(ge.te),
    ve(ge.ve),
    me(ge.me),
    program(),
    numLabels(0)
{
  sord_stack.push(sord);
}

// Start encoding the body of the record.  The function being encoded
// is the record's initializer.
env::env(record *t, env &parent, modifier sord)
  : level(new frame(t->getLevel(), 0)),
    recordLevel(t->getLevel()),
    recordType(t),
    l(t->getInit()),
    funtype(inittype()),
    ge(parent.ge),
    parent(&parent),
    sord(sord),
    perm(READONLY),
    te(ge.te),
    ve(ge.ve),
    me(ge.me),
    program(),
    numLabels(0)
{
  sord_stack.push(sord);
}

record *env::getModule(symbol *id)
{
  record *m = ge.getModule(id);
  if (m) {
    return m;
  }
  else {
    return ge.loadModule(id);
  }
}

env env::newFunction(function *t)
{
  return env(t, *this);
}

record *env::newRecord(symbol *id)
{
  frame *underlevel = getFrame();

  frame *level = new frame(underlevel, 0);
  
  vm::lambda *init = new vm::lambda;

  record *r = new record(id, level, init);

  return r;
}

env env::newRecordInit(record *r)
{
  return env(r, *this);
}


bool env::encode(frame *f)
{
  frame *toplevel = getFrame();
  
  if (f == 0) {
    encode(inst::constpush);
    encode(0);
  }
  else if (f == toplevel) {
    encode(inst::pushclosure);
  }
  else {
    encode(inst::varpush);
    encode(0);
    
    frame *level = toplevel->getParent();
    while (level != f) {
      if (level == 0)
	// Frame request was in an improper scope.
	return false;

      encode(inst::fieldpush);
      encode(0);

      level = level->getParent();
    }
  }

  return true;
}

bool env::encode(frame *dest, frame *top)
{
  //std::cerr << "env::encode()\n";
  
  if (dest == 0) {
    encode(inst::pop);
    encode(inst::constpush);
    encode(0);
  }
  else {
    frame *level = top;
    while (level != dest) {
      if (level == 0) {
	// Frame request was in an improper scope.
	std::cerr << "failed\n";
	
	return false;
      }

      encode(inst::fieldpush);
      encode(0);

      level = level->getParent();
    }
  }

  //std::cerr << "succeeded\n";
  return true;
}

// Prints out error messages for the cast methods.
static inline void castError(position pos, ty *target, ty *source)
{
  em->error(pos);
  *em << "cannot convert \'" << *source
      << "\' to \'" << *target << "\'";
}

bool env::implicitCast(position pos, ty *target, ty *source)
{
  access *a = types::cast(target, source);
  if (a) {
    a->encodeCall(pos, *this);
    return true;
  }
  else {
    castError(pos, target, source);
    return false;
  }
}

bool env::explicitCast(position pos, ty *target, ty *source)
{
  access *a = types::explicitCast(target, source);
  if (a) {
    a->encodeCall(pos, *this);
    return true;
  }
  else {
    castError(pos, target, source);
    return false;
  }
}

int env::defLabel()
{
  if (isStatic())
    return parent->defLabel();
  
  //defs.insert(std::make_pair(numLabel,program.size()));
  return defLabel(numLabels++);
}

int env::defLabel(int label)
{
  if (isStatic())
    return parent->defLabel(label);
  
  assert(label >= 0 && label < numLabels);

  defs.insert(std::make_pair(label,program.end()));

  std::multimap<int,vm::program::label>::iterator p = uses.lower_bound(label);
  while (p != uses.upper_bound(label)) {
    p->second->label = program.end();
    ++p;
  }

  return label;
}

void env::useLabel(int label)
{
  if (isStatic())
    return parent->useLabel(label);
  
  std::map<int,vm::program::label>::iterator p = defs.find(label);
  if (p != defs.end()) {
    inst i; i.label = p->second;
    program.encode(i);
  } else {
    // Not yet defined
    uses.insert(std::make_pair(label,program.end()));
    program.encode(inst());
  }
}

int env::fwdLabel()
{
  if (isStatic())
    return parent->fwdLabel();
  
  // Create a new label without specifying its position.
  return numLabels++;
}

void env::markPos(position pos)
{
  if (isStatic())
    parent->markPos(pos);
  else
    l->pl.push_back(lambda::instpos(program.end(), pos));
}

namespace {
function *inittype()
{
  static function t(types::primVoid());
  return &t;
}

function *bootuptype()
{
  static function t(types::primVoid());
  return &t;
}
} // private

} // namespace env

