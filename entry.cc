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

void varEntry::checkPerm(action act, position pos, coder &c) {
  if (r && !c.inTranslation(r->getLevel())) {
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

void varEntry::encode(action act, position pos, coder &c) {
  checkPerm(act, pos, c);
  getLocation()->encode(act, pos, c);
}

void varEntry::encode(action act, position pos, coder &c, frame *top) {
  checkPerm(act, pos, c);
  getLocation()->encode(act, pos, c, top);
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
