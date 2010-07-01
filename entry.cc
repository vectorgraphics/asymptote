/*****
 * entry.cc
 * Andy Hammerlindl 2002/08/29
 *
 * All variables, built-in functions and user-defined functions reside
 * within the same namespace.  To keep track of all these, table of
 * "entries" is used.
 *****/

#include <iostream>

#include <cmath>
#include <utility>
#include "entry.h"
#include "coder.h"

using types::ty;
using types::signature;
using types::overloaded;
using types::ty_vector;
using types::ty_iterator;

namespace trans {

bool entry::pr::check(action act, coder &c) {
  // We assume PUBLIC permissions and one's without an associated record are not
  // stored.
  assert(perm!=PUBLIC && r!=0);
  return c.inTranslation(r->getLevel()) ||
    (perm == RESTRICTED && act != WRITE);
}

void entry::pr::report(action act, position pos, coder &c) {
  if (!c.inTranslation(r->getLevel())) {
    if (perm == PRIVATE) {
      em.error(pos);
      em << "accessing private field outside of structure";
    }
    else if (perm == RESTRICTED && act == WRITE) {
      em.error(pos);
      em << "modifying non-public field outside of structure";
    }
  }
}

entry::entry(entry &e1, entry &e2) : where(e2.where), pos(e2.pos) {
  perms.insert(perms.end(), e1.perms.begin(), e1.perms.end());
  perms.insert(perms.end(), e2.perms.begin(), e2.perms.end());
}

entry::entry(entry &base, permission perm, record *r)
  : where(base.where), pos(base.pos) {
  perms.insert(perms.end(), base.perms.begin(), base.perms.end());
  addPerm(perm, r);
}

bool entry::checkPerm(action act, coder &c) {
  for (mem::list<pr>::iterator p=perms.begin(); p != perms.end(); ++p)
    if (!p->check(act, c))
      return false;
  return true;
}

void entry::reportPerm(action act, position pos, coder &c) {
  for (mem::list<pr>::iterator p=perms.begin(); p != perms.end(); ++p)
    p->report(act, pos, c);
}


varEntry::varEntry(varEntry &qv, varEntry &v)
  : entry(qv,v), t(v.t),
    location(new qualifiedAccess(qv.location, qv.getLevel(), v.location)) {}

frame *varEntry::getLevel() {
  record *r=dynamic_cast<record *>(t);
  assert(r);
  return r->getLevel();
}

void varEntry::encode(action act, position pos, coder &c) {
  reportPerm(act, pos, c);
  getLocation()->encode(act, pos, c);
}

void varEntry::encode(action act, position pos, coder &c, frame *top) {
  reportPerm(act, pos, c);
  getLocation()->encode(act, pos, c, top);
}

varEntry *qualifyVarEntry(varEntry *qv, varEntry *v)
{
  return qv ? (v ? new varEntry(*qv,*v) : qv) : v;
}


bool tenv::add(symbol dest,
               names_t::value_type &x, varEntry *qualifier, coder &c)
{
  if (!x.second.empty()) {
    tyEntry *ent=x.second.front();
    if (ent->checkPerm(READ, c)) {
      enter(dest, qualifyTyEntry(qualifier, ent));
      return true;
    }
  }
  return false;
}

void tenv::add(tenv& source, varEntry *qualifier, coder &c) {
  // Enter each distinct (unshadowed) name,type pair.
  for(names_t::iterator p = source.names.begin(); p != source.names.end(); ++p)
    add(p->first, *p, qualifier, c);
}

bool tenv::add(symbol src, symbol dest,
               tenv& source, varEntry *qualifier, coder &c) {
  names_t::iterator p = source.names.find(src);
  if (p != source.names.end())
    return add(dest, *p, qualifier, c);
  else
    return false;
}

#ifdef NOHASH //{{{
/*NOHASH*/ void venv::add(venv& source, varEntry *qualifier, coder &c)
/*NOHASH*/ {
/*NOHASH*/   // Enter each distinct (unshadowed) name,type pair.
/*NOHASH*/   for(names_t::iterator p = source.names.begin();
/*NOHASH*/       p != source.names.end();
/*NOHASH*/       ++p)
/*NOHASH*/     add(p->first, p->first, source, qualifier, c);
/*NOHASH*/ }
/*NOHASH*/ 
/*NOHASH*/ bool venv::add(symbol src, symbol dest,
/*NOHASH*/                venv& source, varEntry *qualifier, coder &c)
/*NOHASH*/ {
/*NOHASH*/   bool added=false;
/*NOHASH*/   name_t &list=source.names[src];
/*NOHASH*/   types::overloaded set; // To keep track of what is shadowed.
/*NOHASH*/   bool special = src.special();
/*NOHASH*/ 
/*NOHASH*/   for(name_iterator p = list.begin();
/*NOHASH*/       p != list.end();
/*NOHASH*/       ++p) {
/*NOHASH*/     varEntry *v=*p;
/*NOHASH*/     if (!equivalent(v->getType(), &set)) {
/*NOHASH*/       set.addDistinct(v->getType(), special);
/*NOHASH*/       if (v->checkPerm(READ, c)) {
/*NOHASH*/         enter(dest, qualifyVarEntry(qualifier, v));
/*NOHASH*/         added=true;
/*NOHASH*/       }
/*NOHASH*/     }
/*NOHASH*/   }
/*NOHASH*/   
/*NOHASH*/   return added;
/*NOHASH*/ }
/*NOHASH*/ 
/*NOHASH*/ varEntry *venv::lookByType(symbol name, ty *t)
/*NOHASH*/ {
/*NOHASH*/   // Find first applicable function.
/*NOHASH*/   name_t &list = names[name];
/*NOHASH*/   for(name_iterator p = list.begin();
/*NOHASH*/       p != list.end();
/*NOHASH*/       ++p) {
/*NOHASH*/     if (equivalent((*p)->getType(), t))
/*NOHASH*/       return *p;
/*NOHASH*/   }
/*NOHASH*/   return 0;
/*NOHASH*/ }
/*NOHASH*/ 
/*NOHASH*/ void venv::list(record *module)
/*NOHASH*/ {
/*NOHASH*/   bool where=settings::getSetting<bool>("where");
/*NOHASH*/   // List all functions and variables.
/*NOHASH*/   for(names_t::iterator N = names.begin(); N != names.end(); ++N) {
/*NOHASH*/     symbol s=N->first;
/*NOHASH*/     name_t &list=names[s];
/*NOHASH*/     for(name_iterator p = list.begin(); p != list.end(); ++p) {
/*NOHASH*/       if(!module || (*p)->whereDefined() == module) {
/*NOHASH*/         if(where) cout << (*p)->getPos();
/*NOHASH*/         (*p)->getType()->printVar(cout, s);
/*NOHASH*/         cout << ";\n";
/*NOHASH*/       }
/*NOHASH*/     }
/*NOHASH*/   }
/*NOHASH*/   flush(cout);
/*NOHASH*/ }
/*NOHASH*/ 
/*NOHASH*/ ty *venv::getType(symbol name)
/*NOHASH*/ {
/*NOHASH*/   types::overloaded set;
/*NOHASH*/ 
/*NOHASH*/   // Find all applicable functions in scope.
/*NOHASH*/   name_t &list = names[name];
/*NOHASH*/   bool special = name.special();
/*NOHASH*/   
/*NOHASH*/   for(name_iterator p = list.begin();
/*NOHASH*/       p != list.end();
/*NOHASH*/       ++p) {
/*NOHASH*/     set.addDistinct((*p)->getType(), special);
/*NOHASH*/   }
/*NOHASH*/ 
/*NOHASH*/   return set.simplify();
/*NOHASH*/ }
// }}}
#else

ostream& operator<< (ostream& out, const venv::key &k) {
  if(k.special)
    k.u.t->printVar(out, k.name);
  else {
    out << k.name;
    if (k.u.sig)
      out << *k.u.sig;
  }
  return out;
}

#if TEST_COLLISION
bool venv::keyeq::operator()(const key k, const key l) const {
  keyhash kh;
  if (kh(k)==kh(l)) {
    if (base(k,l))
      return true;
    else {
      cerr << "collision: " << endl;
      cerr << "  " << k << " -> " << kh(k) << endl;
      cerr << "  " << l << " -> " << kh(l) << endl;
    }
  }
  return false;
}
#else
bool venv::keyeq::operator()(const key k, const key l) const {
  return k.name==l.name &&
    (k.special ? equivalent(k.u.t, l.u.t) :
                 equivalent(k.u.sig, l.u.sig));
}
#endif

#ifdef CALLEE_SEARCH
size_t numFormals(ty *t) {
  signature *sig = t->getSignature();
  return sig ? sig->getNumFormals() : 0;
}
#endif

void venv::checkName(symbol name)
{
  // TODO: test maxFormals

  // Get the type, and make it overloaded if it is not (for uniformity).
  overloaded o;
  ty *t = getType(name);
  if (!t)
    t = &o;
  if (!t->isOverloaded()) {
    o.add(t);
    t = &o;
  }
  assert(t->isOverloaded());

  size_t size = 0;
  for (ty_iterator i = t->begin(); i != t->end(); ++i) {
    varEntry *v = lookByType(name, *i);
    assert(v);
    assert(equivalent(v->getType(), *i));
    ++size;
  }

  size_t matches = 0;
  for (keymap::iterator p = all.begin(); p != all.end(); ++p) {
    if (p->first.name == name) {
      ++matches;

      varEntry *v=p->second.v;
      assert(v);
      assert(equivalent(t, v->getType()));
    }
  }
  assert(matches == size);
}
    
void rightKind(ty *t) {
  if (t && t->isOverloaded()) {
    ty_vector& set=((overloaded *)t)->sub;
    assert(set.size() > 1);
  }
}

#ifdef DEBUG_CACHE
#define RIGHTKIND(t) (rightKind(t))
#define CHECKNAME(name) (checkName(name))
#else
#define RIGHTKIND(t) (void)(t)
#define CHECKNAME(name) (void)(name)
#endif

void venv::namevalue::addType(ty *s) {
  RIGHTKIND(t);

#ifdef DEBUG_CACHE
  assert(!s->isOverloaded());
#endif

  if (t == 0) {
#if CALLEE_SEARCH
    maxFormals = numFormals(s);
#endif
    t = s;
  } else {
    if (!t->isOverloaded())
      t = new overloaded(t);

#ifdef DEBUG_CACHE
    assert(t->isOverloaded());
    assert(!equivalent(t, s));
#endif

    ((overloaded *)t)->add(s);

#if CALLEE_SEARCH
    size_t n = numFormals(s);
    if (n > maxFormals)
      maxFormals = n;
#endif
  }

  RIGHTKIND(t);
}

void venv::namevalue::replaceType(ty *new_t, ty *old_t) {
#ifdef DEBUG_CACHE
  assert(t != 0);
  RIGHTKIND(t);
#endif
  
  if (t->isOverloaded()) {
    for (ty_iterator i = t->begin(); i != t->end(); ++i) {
      if (equivalent(old_t, *i)) {
        *i = new_t;
        return;
      }
    }

    // An error, the type was not found.
    assert("unreachable code" == 0);

  } else {
#ifdef DEBUG_CACHE
    assert(equivalent(old_t, t));
#endif
    t = new_t;
  }

#ifdef DEBUG_CACHE
  assert(t != 0);
  RIGHTKIND(t);
#endif
}

#ifdef DEBUG_CACHE
void venv::namevalue::popType(ty *s)
#else
void venv::namevalue::popType()
#endif
{
#ifdef DEBUG_CACHE
  assert(t);
  RIGHTKIND(t);
  assert(!s->isOverloaded());
#endif

  if (t->isOverloaded()) {
    ty_vector& set=((overloaded *)t)->sub;

#ifdef DEBUG_CACHE
    assert(set.size() > 0);
    assert(equivalent(set.back(), s));
#endif

    // We are relying on the fact that this was the last type added to t, and
    // that type are added by pushing them on the end of the vector.
    set.pop_back();

    if (set.size() == 1)
      t = set.front();
  } else {
#ifdef DEBUG_CACHE
    assert(equivalent(t, s));
#endif
    t = 0;
  }

  RIGHTKIND(t);

  // Don't try to reduce numFormals as I doubt it is worth the cost of
  // recalculating. 
}

void venv::remove(const addition& a) {
  CHECKNAME(a.k.name);

  value &val=all[a.k];

#ifdef DEBUG_CACHE
  assert(val);
#endif

  if (a.shadowed) {
    // Unshadow the previously shadowed varEntry.
    names[a.k.name].replaceType(a.shadowed->getType(), val.v->getType());
    val.v = a.shadowed;
  } else {
    // Remove the (name,sig) key completely.
#if DEBUG_CACHE
    names[a.k.name].popType(val.v->getType());
#else
    names[a.k.name].popType();
#endif
    all.erase(a.k);
  }

  CHECKNAME(a.k.name);
}

void venv::beginScope() {
  if (all.empty()) {
    assert(scopesizes.empty());
    ++empty_scopes;
  } else {
    scopesizes.push(additions.size());
  }
}

void venv::endScope() {
  if (scopesizes.empty()) {
    // The corresponding beginScope happened when the venv was empty, so
    // clear the hash tables to return to that state.
    all.clear();
    names.clear();

    assert(empty_scopes > 0);
    --empty_scopes;
  } else {
    size_t scopesize = scopesizes.top();
    assert(additions.size() >= scopesize);
    while (additions.size() > scopesize) {
      remove(additions.top());
      additions.pop();
    }
    scopesizes.pop();
  }
}

// Adds the definitions of the top-level scope to the level underneath,
// and then removes the top scope.
void venv::collapseScope() {
  if (scopesizes.empty()) {
    // Collapsing an empty scope.
    assert(empty_scopes > 0);
    --empty_scopes;
  } else {
    scopesizes.pop();
  }
}


void venv::enter(symbol name, varEntry *v)
{
  CHECKNAME(name);

  key k(name, v);

  value &slot=all[k];
  if (slot.v) {
    // The new value shadows an old value.  They have the same signature, but
    // possibly different return types.  If necessary, update the type stored
    // by name.
    names[name].replaceType(v->getType(), slot.v->getType());

    // Replace the old value, but store its now-shadowed varEntry.
    if (!scopesizes.empty())
      additions.push(addition(k, slot.v));

    slot.v = v;

  } else {
    // Add to the names hash table.
    names[name].addType(v->getType());

    if (!scopesizes.empty())
      additions.push(addition(k, 0));

    slot.v=v;
  }

  CHECKNAME(name);
}


varEntry *venv::lookBySignature(symbol name, signature *sig) {
#ifdef CALLEE_SEARCH
  // Rest arguments are complicated and rare.  Don't handle them here.
  if (sig->hasRest()) {
#if 0
    if (lookByType(key(name, sig)))
      cout << "FAIL BY REST ARG" << endl;
    else
      cout << "FAIL BY REST ARG AND NO-MATCH" << endl;
#endif
    return 0;
  }

  // Likewise with the special operators.
  if (name.special()) {
    //cout << "FAIL BY SPECIAL" << endl;
    return 0;
  }

  namevalue& nv = names[name];

  // Avoid ambiguities with default parameters.
  if (nv.maxFormals != sig->getNumFormals()) {
#if 0
    if (lookByType(key(name, sig)))
      cout << "FAIL BY NUMARGS" << endl;
    else
      cout << "FAIL BY NUMARGS AND NO-MATCH" << endl;
#endif
    return 0;
  }

  // At this point, any function with an equivalent an signature will be equal
  // to the result of the normal overloaded function resolution.  We may
  // safely return it.
  varEntry *result = lookByType(key(name, sig));
#if 0
  if (!result) {
    cout << "FAIL BY NO-MATCH" << endl;
  }
#endif
  return result;
#else
  // The maxFormals field is necessary for this optimization.
  return 0;
#endif
}

void venv::add(venv& source, varEntry *qualifier, coder &c)
{
  // Enter each distinct (unshadowed) name,type pair.
  for(keymap::iterator p = source.all.begin(); p != source.all.end(); ++p) {
    varEntry *v=p->second.v;
    if (v->checkPerm(READ, c))
      enter(p->first.name, qualifyVarEntry(qualifier, v));
  }
}

bool venv::add(symbol src, symbol dest,
               venv& source, varEntry *qualifier, coder &c)
{
  ty *t=source.getType(src);
  if (!t)
    return false;
  else if (t->isOverloaded()) {
    bool added=false;
    for (ty_iterator i = t->begin(); i != t->end(); ++i) {
      varEntry *v=source.lookByType(src, *i);
      if (v->checkPerm(READ, c)) {
        enter(dest, qualifyVarEntry(qualifier, v));
        added=true;
      }
    }
    return added;
  } else {
    varEntry *v=source.lookByType(src, t);
    if (v->checkPerm(READ, c)) {
      enter(dest, qualifyVarEntry(qualifier, v));
      return true;
    }
    return false;
  }
}


ty *venv::getType(symbol name)
{
  return names[name].t;
}

void listValue(symbol name, varEntry *v, record *module)
{
  if (!module || v->whereDefined() == module)
  {
    if (settings::getSetting<bool>("where"))
      cout << v->getPos();

    v->getType()->printVar(cout, name);

    cout << ";\n";
  }
}

void venv::listValues(symbol name, record *module)
{
  ty *t=getType(name);

  if (t->isOverloaded())
    for (ty_iterator i = t->begin(); i != t->end(); ++i)
      listValue(name, lookByType(name, *i), module);
  else
    listValue(name, lookByType(name, t), module);

  flush(cout);
}

void venv::list(record *module)
{
  // List all functions and variables.
  for (namemap::iterator N = names.begin(); N != names.end(); ++N)
    listValues(N->first, module);
}

void venv::completions(mem::list<symbol >& l, string start)
{
  for(namemap::iterator N = names.begin(); N != names.end(); ++N)
    if (prefix(start, N->first) && N->second.t)
      l.push_back(N->first);
}

#endif

} // namespace trans
