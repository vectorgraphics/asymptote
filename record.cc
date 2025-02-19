/*****
 * record.cc
 * Tom Prince 2004/07/15
 *
 * The type for records and modules in the language.
 *****/

#include "record.h"
#include "inst.h"
#include "runtime.h"
#include "coder.h"



namespace types {

record::record(symbol name, frame *level)
  : ty(ty_record),
    name(name),
    level(level),
    init(new vm::lambda),
    e()
{
  assert(init);
#ifdef DEBUG_FRAME
  init->name = "struct "+string(name);
#endif
}

record::~record()
{}

record *record::newRecord(symbol id, bool statically)
{
  frame *underlevel = getLevel(statically);
  assert(underlevel);

  frame *level = new frame(id, underlevel, 0);

  record *r = new record(id, level);
  return r;
}

// Initialize to null by default.
trans::access *record::initializer() {
  static trans::bltinAccess a(run::pushNullRecord);
  return &a;
}

mem::pair<ty*, ty*> computeKVTypes(trans::venv& ve, const position& pos)
{
  mem::pair<ty*, ty*> errorPair(primError(), primError());

  // TODO: Make the lookup more efficient. (See DEFSYMBOL in camp.l.)
  ty* getTy= ve.getType(symbol::trans("[]"));
  ty* setTy= ve.getType(symbol::trans("[=]"));
  if (getTy == nullptr) {
    if (setTy != nullptr) {
      em.error(pos);
      em << "operator[=] defined without operator[]";
    }
    return errorPair;
  }

  // Find the keytype and valuetype based on operator[].
  if (getTy->isOverloaded()) {
    em.error(pos);
    em << "multiple operator[] definitions in one struct";
    return errorPair;
  }
  if (getTy->kind != ty_function) {
    em.error(pos);
    em << "operator[] is not a function";
    return errorPair;
  }
  types::function* get= static_cast<types::function*>(getTy);
  types::ty* valTy= get->getResult();
  signature* getSig= get->getSignature();
  // TODO: Can we get the position of the definition of operator[] rather than
  // the end of the struct?
  if (getSig->hasRest() || getSig->getNumFormals() != 1) {
    em.error(pos);
    em << "operator[] must have exactly one parameter";
    return errorPair;
  }
  ty* keyTy= getSig->getFormal(0).t;

  if (setTy != nullptr) {
    // Find the keytype and valuetype based on operator[=].
    if (setTy->isOverloaded()) {
      em.error(pos);
      em << "multiple operator[=] definitions in one struct";
      return errorPair;
    }
    if (setTy->kind != ty_function) {
      em.error(pos);
      em << "operator[=] is not a function";
      return errorPair;
    }
    types::function* set= static_cast<types::function*>(setTy);
    types::ty* setResult= set->getResult();
    if (setResult->kind != ty_void) {
      em.error(pos);
      em << "operator[=] must return void";
      return errorPair;
    }
    signature* setSig= set->getSignature();
    if (setSig->hasRest() || setSig->getNumFormals() != 2) {
      em.error(pos);
      em << "operator[=] must have exactly two parameters";
      return errorPair;
    }
    ty* setKeyTy= setSig->getFormal(0).t;
    ty* setValTy= setSig->getFormal(1).t;

    // Check that they agree.
    if (!keyTy->equiv(setKeyTy) || !setKeyTy->equiv(keyTy)) {
      em.error(pos);
      em << "first parameter of operator[] and operator[=] must match";
      return errorPair;
    }
    if (!valTy->equiv(setValTy) || !setValTy->equiv(valTy)) {
      em.error(pos);
      em << "return type of operator[] and second parameter of operator[=] "
            "must match";
      return errorPair;
    }
  }

  return mem::pair<ty*, ty*>(keyTy, valTy);
}

void record::computeKVTypes(const position& pos)
{
  std::tie(kType, vType)= types::computeKVTypes(e.ve, pos);
}

ty *record::keyType() {
  if (kType != nullptr) {
    return kType;
  }
  // TODO: Use an actual position.
  return types::computeKVTypes(e.ve, nullPos).first;
}

ty *record::valType() {
  if (vType != nullptr) {
    return vType;
  }
  // TODO: Use an actual position.
  return types::computeKVTypes(e.ve, nullPos).second;
}

dummyRecord::dummyRecord(symbol name)
  : record(name, new frame(name, 0,0))
{
  // Encode the instructions to put an placeholder instance of the record
  // on the stack.
  trans::coder c(nullPos, this, 0);
  c.closeRecord();
}

dummyRecord::dummyRecord(string s)
  : record (symbol::trans(s), new frame(s,0,0))
{
  // Encode the instructions to put an placeholder instance of the record
  // on the stack.
  trans::coder c(nullPos, this, 0);
  c.closeRecord();
}

void dummyRecord::add(string name, ty *t, trans::access *a,
                      trans::permission perm) {
  e.addVar(symbol::trans(name),
           new trans::varEntry(t, a, perm, this, this, nullPos));
}

void dummyRecord::add(string name, function *t, vm::bltin f,
                      trans::permission perm) {
  REGISTER_BLTIN(f, name);
  add(name, t, new trans::bltinAccess(f), perm);
}

} // namespace types
