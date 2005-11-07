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

using std::ostream;
using std::cerr;
using std::endl;

using types::signature;

namespace trans {

bool entry::pr::check(action act, coder &c) {
  // We assume PUBLIC permissions and one's without an associated record are not
  // stored.
  return c.inTranslation(r->getLevel()) ||
         (perm == READONLY && act != WRITE);
}

void entry::pr::report(action act, position pos, coder &c) {
  if (!c.inTranslation(r->getLevel())) {
    if (perm == PRIVATE) {
      em->error(pos);
      *em << "accessing private field outside of structure";
    }
    else if (perm == READONLY && act == WRITE) {
      em->error(pos);
      *em << "modifying non-public field outside of structure";
    }
  }
}

entry::entry(entry &e1, entry &e2) {
  perms.insert(perms.end(), e1.perms.begin(), e1.perms.end());
  perms.insert(perms.end(), e2.perms.begin(), e2.perms.end());
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
  : entry(qv,v), t(v.t)
{
  record *r=dynamic_cast<record *>(qv.t);
  assert(r);
  location = new qualifiedAccess(qv.location, r->getLevel(), v.location);
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

tyEntry *qualifyTyEntry(varEntry *qv, tyEntry *ent)
{
  // Records need a varEntry that refers back to the qualifier qv.  Ie. in
  // the last new of the code
  //   struct A {
  //     struct B {}
  //   }
  //   A a=new A;
  //   use a;
  //   new B;
  // we need to put a's frame on the stack before allocating an instance of
  // B.
  // NOTE: A possible optimization could be to only qualify the varEntry if
  // the type is a record, as other types don't use the varEntry.
  return new tyEntry(ent->t, qualifyVarEntry(qv, ent->v));
}

void tenv::add(tenv& source, varEntry *qualifier, coder &c)
{
  // Enter each distinct (unshadowed) name,type pair.
  for(names_t::iterator p = source.names.begin(); p != source.names.end(); ++p)
    if (!p->second.empty()) {
      tyEntry *ent=p->second.front();
      if (ent->checkPerm(READ, c))
        enter(p->first, qualifyTyEntry(qualifier, ent));
    }
}

#ifdef NOHASH //{{{
venv::venv()
{
}

varEntry *venv::lookByType(symbol *name, ty *t)
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

void venv::list()
{
  // List all functions.
  for(names_t::iterator N = names.begin(); N != names.end(); ++N) {
    symbol *s=N->first;
    name_t &list=names[s];
    for(name_iterator p = list.begin(); p != list.end(); ++p) {
      (*p)->getType()->printVar(std::cout, s);
      std::cout << ";" << std::endl;
    }
  }
}

varEntry *venv::lookInTopScope(symbol *name, ty *t)
{
  scope_t &scope = scopes.front();
  for (scope_iterator p = scope.lower_bound(name);
       p != scope.upper_bound(name);
       ++p) {
    if (name == p->first &&
        equivalent(t, p->second->getType(), name->special))
      return p->second;
  }
  return 0;
}

ty *venv::getType(symbol *name)
{
  types::overloaded set;

  // Find all applicable functions in scope.
  name_t &list = names[name];
  
  for(name_iterator p = list.begin();
      p != list.end();
      ++p) {
      set.addDistinct((*p)->getType(), name->special);
  }

  return set.simplify();
}
// }}}
#else // {{{

ostream& operator<< (ostream& out, const venv::key &k) {
  k.t->printVar(out, k.name);
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
#if 0
  cerr << "k.t = " << k.t << " " << k << endl;
  cerr << "l.t = " << l.t << " " << l << endl;
#endif
  return k.name==l.name &&
    (k.name->special ? equivalent(k.t, l.t) :
                       equivalent(k.t->getSignature(),
                                  l.t->getSignature()));
}
#endif

void venv::remove(key k) {
  //cerr << "removing: " << k << endl;
  value *&val=all[k];
  assert(val);
  if (val->next) {
#if SHADOWING
    val->next->shadowed=false;
#endif
    val=val->next;
  }
  else
    all.erase(k);

  // Don't erase it from scopes.top() as that will be popped of the stack at
  // the end of endScope anyway.

  names[k.name].pop_front();
}

void venv::enter(symbol *name, varEntry *v) {
  assert(!scopes.empty());
  key k(name, v);
  //cerr << "entering: " << k << " (t=" << k.t << ")" << endl;
  value *val=new value(v);

#if 0
  keymap::iterator p=all.find(k);
  if (p!=all.end()) {
    cerr << "  over: " << p->first << endl;
  }
#endif
  
  val->next=all[k];
#if SHADOWING
  if (val->next)
    val->next->shadowed=true;
#endif

  all[k]=val;
  scopes.top().insert(keymultimap::value_type(k,val));
  names[k.name].push_front(val);
}

void venv::add(venv& source, varEntry *qualifier, coder &c)
{
  // Enter each distinct (unshadowed) name,type pair.
  for(keymap::iterator p = source.all.begin(); p != source.all.end(); ++p) {
    varEntry *v=p->second->v;
    if (v->checkPerm(READ, c))
      enter(p->first.name, qualifyVarEntry(qualifier, p->second->v));
  }
}

ty *venv::getType(symbol *name)
{
  //cerr << "getType: " << *name << endl;
  types::overloaded set;
  values &list=names[name];

  for (values::iterator p=list.begin(); p!=list.end(); ++p) {
#if SHADOWING
    if (!(*p)->shadowed)
      set.add((*p)->v->getType());
#else
    set.addDistinct((*p)->v->getType());
#endif
  }

  return set.simplify();
}

void venv::listValues(symbol *name, values &vals) {
  ostream& out=std::cout;

  for(values::iterator p = vals.begin(); p != vals.end(); ++p) {
    if ((*p)->shadowed)
      out << "  <shadowed> ";
    (*p)->v->getType()->printVar(out, name);
    out << ";" << std::endl;
  }
}

void venv::list()
{
  // List all variables.
  for(namemap::iterator N = names.begin(); N != names.end(); ++N)
    listValues(N->first, N->second);
}

#endif // }}}

} // namespace trans
