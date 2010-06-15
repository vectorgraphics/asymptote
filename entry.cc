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

using types::signature;

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
void venv::add(venv& source, varEntry *qualifier, coder &c)
{
  // Enter each distinct (unshadowed) name,type pair.
  for(names_t::iterator p = source.names.begin();
      p != source.names.end();
      ++p)
    add(p->first, p->first, source, qualifier, c);
}

bool venv::add(symbol src, symbol dest,
               venv& source, varEntry *qualifier, coder &c)
{
  bool added=false;
  name_t &list=source.names[src];
  types::overloaded set; // To keep track of what is shadowed.
  bool special = src.special();

  for(name_iterator p = list.begin();
      p != list.end();
      ++p) {
    varEntry *v=*p;
    if (!equivalent(v->getType(), &set)) {
      set.addDistinct(v->getType(), special);
      if (v->checkPerm(READ, c)) {
        enter(dest, qualifyVarEntry(qualifier, v));
        added=true;
      }
    }
  }
  
  return added;
}

varEntry *venv::lookByType(symbol name, ty *t)
{
  // Find first applicable function.
  name_t &list = names[name];
  for(name_iterator p = list.begin();
      p != list.end();
      ++p) {
    if (equivalent((*p)->getType(), t))
      return *p;
  }
  return 0;
}

void venv::list(record *module)
{
  bool where=settings::getSetting<bool>("where");
  // List all functions and variables.
  for(names_t::iterator N = names.begin(); N != names.end(); ++N) {
    symbol s=N->first;
    name_t &list=names[s];
    for(name_iterator p = list.begin(); p != list.end(); ++p) {
      if(!module || (*p)->whereDefined() == module) {
        if(where) cout << (*p)->getPos();
        (*p)->getType()->printVar(cout, s);
        cout << ";\n";
      }
    }
  }
  flush(cout);
}

ty *venv::getType(symbol name)
{
  types::overloaded set;

  // Find all applicable functions in scope.
  name_t &list = names[name];
  bool special = name.special();
  
  for(name_iterator p = list.begin();
      p != list.end();
      ++p) {
    set.addDistinct((*p)->getType(), special);
  }

  return set.simplify();
}
// }}}
#else // {{{

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

void venv::remove(key k) {
  //cerr << "removing: " << k << endl;
  value *&val=all[k];
  assert(val);
  if (val->next) {
    val->next->shadowed=false;
    value *temp=val->next;
    val->next=0;
    val=temp;
  }
  else
    all.erase(k);

  // Don't erase it from scopes.top() as that will be popped of the stack at
  // the end of endScope anyway.

  names[k.name].pop_front();
}

#ifdef CALLEE_SEARCH
size_t numArgs(ty *t) {
  signature *sig = t->getSignature();
  return sig ? sig->getNumFormals() : 0;
}

void checkMaxArgs(venv *ve, symbol name, size_t expected) {
  size_t maxFormals = 0;
  ty *t = ve->getType(name);
  if (types::overloaded *o=dynamic_cast<types::overloaded *>(t)) {
    for (types::ty_vector::iterator i=o->sub.begin(); i != o->sub.end(); ++i)
    {
      size_t n = numArgs(*i);
      if (n > maxFormals)
        maxFormals = n;
    }
  } else {
    maxFormals = numArgs(t);
  }
  if (expected != maxFormals) {
    cout << "expected: " << expected << " computed: " << maxFormals << endl;
    cout << "types: " << endl;
    if (types::overloaded *o=dynamic_cast<types::overloaded *>(t)) {
      cout << " overloaded" << endl;
      for (types::ty_vector::iterator i=o->sub.begin(); i != o->sub.end();
          ++i)
      {
        cout << "  " << **i << endl;
      }
    } else {
      cout << " non-overloaded" << endl;
      cout << "  " << *t << endl;
    }
    cout.flush();
  }
  assert(expected == maxFormals);
}
#endif

void venv::enter(symbol name, varEntry *v) {
  assert(!scopes.empty());
  key k(name, v);
#ifdef DEBUG_ENTRY
  cout << "entering: " << k << endl;
#endif
  value *val=new value(v);

#ifdef DEBUG_ENTRY
  keymap::iterator p=all.find(k);
  if (p!=all.end()) {
    cout << "  over: " << p->first << endl;
  }
#endif
  
  val->next=all[k];
  if (val->next)
    val->next->shadowed=true;

  all[k]=val;
  scopes.top().insert(keymultimap::value_type(k,val));

#ifdef CALLEE_SEARCH
  // I'm not sure if this works properly with rest arguments.
  signature *sig = v->getSignature();
  size_t newmax = sig ? sig->getNumFormals() : 0;
  mem::list<value *>& namelist = names[k.name];
  if (!namelist.empty()) {
    size_t oldmax = namelist.front()->maxFormals;
    if (oldmax > newmax)
      newmax = oldmax;
  }
  val->maxFormals = newmax;
  namelist.push_front(val);

  // A sanity check, disabled for speed reasons.
#ifdef DEBUG_CACHE
  checkMaxArgs(this, k.name, val->maxFormals);
#endif
#else
  names[k.name].push_front(val);
#endif
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

  mem::list<value *>& namelist = names[name];
  if (namelist.empty()) {
    // No variables of this name.
    //cout << "FAIL BY EMPTY" << endl;
    return 0;
  }



  // Avoid ambiguities with default parameters.
  if (namelist.front()->maxFormals != sig->getNumFormals()) {
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
  if (!result)
    cout << "FAIL BY NO-MATCH" << endl;
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
    varEntry *v=p->second->v;
    if (v->checkPerm(READ, c))
      enter(p->first.name, qualifyVarEntry(qualifier, v));
  }
}

bool venv::add(symbol src, symbol dest,
               venv& source, varEntry *qualifier, coder &c)
{
  bool added=false;
  values &list=source.names[src];

  for (values::iterator p=list.begin(); p!=list.end(); ++p)
    if (!(*p)->shadowed) {
      varEntry *v=(*p)->v;
      if (v->checkPerm(READ, c)) {
        enter(dest, qualifyVarEntry(qualifier, v));
        added=true;
      }
    }

  return added;
}


ty *venv::getType(symbol name)
{
#if 0
  cout << "getType: " << name << endl;
#endif
  types::overloaded set;
  values &list=names[name];

  for (values::iterator p=list.begin(); p!=list.end(); ++p)
    if (!(*p)->shadowed)
      set.add((*p)->v->getType());

  return set.simplify();
}

void venv::listValues(symbol name, values &vals, record *module)
{
  ostream& out=cout;

  bool where=settings::getSetting<bool>("where");
  for(values::iterator p = vals.begin(); p != vals.end(); ++p) {
    if(!module || (*p)->v->whereDefined() == module) {
      if(where) out << (*p)->v->getPos();
      if ((*p)->shadowed)
        out << "  <shadowed> ";
      (*p)->v->getType()->printVar(out, name);
      out << ";\n";
    }
  }
  flush(out);
}

void venv::list(record *module)
{
  // List all functions and variables.
  for(namemap::iterator N = names.begin(); N != names.end(); ++N)
    listValues(N->first, N->second,module);
}

void venv::completions(mem::list<symbol >& l, string start)
{
  for(namemap::iterator N = names.begin(); N != names.end(); ++N)
    if (prefix(start, N->first) && !N->second.empty())
      l.push_back(N->first);
}

#endif // }}}

} // namespace trans
