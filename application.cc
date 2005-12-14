/*****
 * application.cc
 * Andy Hammerlindl 2005/05/20
 *
 * An application is a matching of arguments in a call expression to formal
 * parameters of a function.  Since the language allows default arguments,
 * keyword arguments, rest arguments, and anything else we think of, this
 * is not a simple mapping.
 *****/

#include "application.h"
#include "exp.h"
#include "coenv.h"
#include "runtime.h"

using namespace types;
using absyntax::varinit;
using absyntax::arrayinit;
using absyntax::arglist;

namespace trans {

// Lower scores are better.  Packed is added onto the other qualifiers so
// we may score both exact and casted packed arguments.
const score FAIL=0, EXACT=1, CAST=2;
const score PACKED=2;

bool castable(env &e, formal& target, formal& source) {
  return target.Explicit ? equivalent(target.t,source.t)
    : e.castable(target.t,source.t, symbol::castsym);
}

score castScore(env &e, formal& target, formal& source) {
  return equivalent(target.t,source.t) ? EXACT :
         (!target.Explicit &&
          e.castable(target.t,source.t, symbol::castsym)) ? CAST : FAIL;
}

defaultArg::defaultArg(types::ty *t)
  : arg(new absyntax::callExp(position(), 
            new absyntax::varEntryExp(position(),
                new function(t),
                run::pushDefault)),
        t)
{}

void restArg::transMaker(coenv &e, int size, bool rest) {
  // Push the number of cells and call the array maker.
  e.c.encode(inst::intpush, size);
  e.c.encode(inst::builtin, rest ? run::newAppendedArray :
                                   run::newInitializedArray);
}

void restArg::trans(coenv &e, temp_vector &temps)
{
  // Push the values on the stack.
  for (list<arg *>::iterator p = inits.begin(); p != inits.end(); ++p)
    (*p)->trans(e, temps);

  if (rest)
    rest->trans(e, temps);
  
  transMaker(e, (int)inits.size(), (bool)rest);
}

class maximizer {
  app_list l;

  // Tests if x is as good (or better) an application as y.
  bool asgood(application *x, application *y) {
    // Test if each score in x is no higher than the corresponding score in
    // y.
    return std::equal(x->scores.begin(), x->scores.end(), y->scores.begin(),
                      std::less_equal<score>());
  }

  bool better(application *x, application *y) {
    return asgood(x,y) && !asgood(y,x);
  }

  // Add an application that has already been determined to be maximal.
  // Remove any other applications that are now not maximal because of its
  // addition.
  void addMaximal(application *x) {
    app_list::iterator y=l.begin();
    while (y!=l.end())
      if (better(x,*y))
        y=l.erase(y);
      else
        ++y;
    l.push_front(x);
  }
  
  // Tests if x is maximal.
  bool maximal(application *x) {
    for (app_list::iterator y=l.begin(); y!=l.end(); ++y)
      if (better(*y,x))
        return false;
    return true;
  }

public:
  maximizer() {}

  void add(application *x) {
    if (maximal(x))
      addMaximal(x);
  }

  app_list result() {
    return l;
  }
};


void application::initRest() {
  types::formal& f=sig->getRest();
  if (f.t) {
    types::array *a=dynamic_cast<types::array *>(f.t);
    if(!a)
      vm::error("formal rest argument must be an array");

    rest=new restArg();
    rf=types::formal(a->celltype, 0, 0, f.Explicit);
  }
}

//const int REST=-1; 
const int NOMATCH=-2;

int application::find(symbol *name) {
  formal_vector &f=sig->formals;
  for (size_t i=index; i<f.size(); ++i)
    if (f[i].name==name && args[i]==0)
      return (int)i;
  return NOMATCH;
}

bool application::matchDefault() {
  if (index==args.size())
    return false;
  else {
    formal &target=getTarget();
    if (target.defval) {
      args[index]=new defaultArg(target.t);
      advanceIndex();
      return true;
    }
    else
      return false;
  }
}

bool application::matchArgumentToRest(env &e, formal &source,
                                      varinit *a, size_t evalIndex)
{
  if (rest) {
    score s=castScore(e, rf, source);
    if (s!=FAIL) {
      rest->add(seq.addArg(a, rf.t, evalIndex));
      scores.push_back(s+PACKED);
      return true;
    }
  }
  return false;
}

bool application::matchAtSpot(size_t spot, env &e, formal &source,
                              varinit *a, size_t evalIndex)
{
  formal &target=sig->getFormal(spot);
  score s=castScore(e, target, source);
  if (s!=FAIL) {
    // The argument matches.
    args[spot]=seq.addArg(a, target.t, evalIndex);
    if (spot==index)
      advanceIndex();
    scores.push_back(s);
    return true;
  }
  else
    return false;
}

bool application::matchArgument(env &e, formal &source,
                                varinit *a, size_t evalIndex)
{
  assert(source.name==0);

  if (index==args.size())
    // Try to pack into the rest array.
    return matchArgumentToRest(e, source, a, evalIndex);
  else
    // Match here, or failing that use a default and try to match at the next
    // spot.
    return matchAtSpot(index, e, source, a, evalIndex) ||
           (matchDefault() && matchArgument(e, source, a, evalIndex));
}

bool application::matchNamedArgument(env &e, formal &source,
                                     varinit *a, size_t evalIndex)
{
  assert(source.name!=0);

  int spot=find(source.name);
  return spot!=NOMATCH && matchAtSpot(spot, e, source, a, evalIndex);
}

bool application::complete() {
  if (index==args.size())
    return true;
  else if (matchDefault())
    return complete();
  else
    return false;
}

bool application::matchRest(env &e, formal &source, varinit *a) {
  if (index==args.size())
    // Match rest to rest.
    if (rest) {
      formal &target=sig->getRest();
      score s=castScore(e, target, source);
      if (s!=FAIL) {
        rest->addRest(new arg(a, target.t));
        scores.push_back(s);
        return true;
      }
    }
  return false;
}
  
bool application::matchSignature(env &e, types::signature *source,
                                 arglist &al) {
  formal_vector &f=source->formals;

  // First, match all of the named (non-rest) arguments.
  for (size_t i=0; i<f.size(); ++i)
    if (f[i].name)
      if (!matchNamedArgument(e, f[i], al[i].val, i))
        return false;

  // Then, the unnamed.
  for (size_t i=0; i<f.size(); ++i)
    if (!f[i].name)
      if (!matchArgument(e, f[i], al[i].val, i))
        return false;

  // Then, the rest argument.
  if (source->hasRest())
    if (!matchRest(e, source->getRest(), al.rest.val))
      return false;

  // Fill in any remaining arguments with their defaults.
  return complete();
}
       
#if 0
application *application::matchHelper(env &e,
                                      application *app,
                                      signature *source)
{
  return app->matchSignature(e, source) ? app : 0;
}

application *application::match(env &e, signature *target, signature *source) {
  application *app=new application(target);
  return matchHelper(e, app, source);
}
#endif

application *application::match(env &e, function *t, signature *source,
                                arglist &al) {
  assert(t->kind==ty_function);
  application *app=new application(t);
  return app->matchSignature(e, source, al) ? app : 0;
}

void application::transArgs(coenv &e) {
  temp_vector temps;

  for(arg_vector::iterator a=args.begin(); a != args.end(); ++a)
    (*a)->trans(e,temps);

  if (rest)
    rest->trans(e,temps);
}

app_list multimatch(env &e,
                    types::overloaded *o,
                    types::signature *source,
                    arglist &al)
{
  assert(source);

  app_list l;

  for(ty_vector::iterator t=o->sub.begin(); t!=o->sub.end(); ++t) {
    if ((*t)->kind==ty_function) {
      application *a=application::match(e, (function *)(*t), source, al);
      if (a)
        l.push_back(a);
    }
  }

  if (l.size() > 1) {
    // Return the most specific candidates.
    maximizer m;
    for (app_list::iterator x=l.begin(); x!=l.end(); ++x) {
      assert(*x);
      m.add(*x);
    }
    return m.result();
  }
  else
    return l;
}

#if 0
app_list resolve(env &e,
                 types::ty *t,
                 types::signature *source)
{
  if (t->kind == ty_overloaded)
    return multimatch(e, (overloaded *)t, source);
  else {
    overloaded o;
    o.add(t);
    return multimatch(e, &o, source);
  }
}
#endif

} // namespace trans
