/*****
 * application.h
 * Andy Hammerlindl 2005/05/20
 *
 * An application is a matching of arguments in a call expression to formal
 * parameters of a function.  Since the language allows default arguments,
 * keyword arguments, rest arguments, and anything else we think of, this is not
 * a simple mapping.
 *****/

#ifndef APPLICATION_H
#define APPLICATION_H

#include <list>

#include "memory.h"
#include "types.h"
#include "coenv.h"
#include "exp.h"

using mem::vector;

using absyntax::arglist;
using absyntax::varinit;
using absyntax::arrayinit;
using absyntax::tempExp;

// This is mid-way between trans an absyntax.
namespace trans {

typedef int score;

typedef vector<score> score_vector;

// This is used during the translation of arguments to store temporary
// expresssions for arguments that need to be translated for side-effects at a
// certain point but used later on.  The invariant maintained is that if the
// vector has n elements, then the side-effects for the first n arguments have
// been translated.  Null is pushed onto the vector to indicate that the
// expression was evaluated directly onto the stack, without the use of a
// temporary.
typedef vector<tempExp *> temp_vector;

struct arg : public gc {
  virtual ~arg() {}
  varinit *v;
  types::ty *t;

  arg(varinit *v, types::ty *t)
    : v(v), t(t) {}

  virtual void trans(coenv &e, temp_vector &) {
    v->transToType(e, t);
  }
};

// This class generates sequenced args, args whose side-effects occur in order
// according to their index, regardless of the order they are called.  This is
// used to ensure left-to-right order of evaluation of keyword arguments, even
// if they are given out of the order specified in the declaration.
class sequencer {
  struct sequencedArg : public arg {
    sequencer &parent;
    size_t i;
    sequencedArg(varinit *v, types::ty *t, sequencer &parent, size_t i)
      : arg(v, t), parent(parent), i(i) {}

    void trans(coenv &e, temp_vector &temps) {
      parent.trans(e, i, temps);
    }
  };

  typedef vector<sequencedArg *> sa_vector;
  sa_vector args;

#if 0
  // This is used during the translation to store temporary expresssions for
  // arguments that need to be translated but used later.  The invariant
  // maintained is that if the vector has n elements, then the side-effects for
  // the first n arguments have been translated.  Null is pushed onto the
  // vector to indicate that the expression was evaluated directly onto the
  // stack, without the use of a temporary.
  typedef vector<tempExp *> temp_vector;
  temp_vector temps;
#endif

  // Makes a temporary for the next argument in the sequence.
  void alias(coenv &e, temp_vector &temps) {
    size_t n=temps.size();
    assert(n < args.size());
    sequencedArg *sa=args[n];

    temps.push_back(new tempExp(e, sa->v, sa->t));
  }

  // Get in a state to translate the i-th argument, aliasing any arguments that
  // occur before it in the sequence.
  void advance(coenv &e, size_t i, temp_vector &temps) {
    while (temps.size() < i)
      alias(e,temps);
  }

  void trans(coenv &e, size_t i, temp_vector &temps) {
    if (i < temps.size()) {
      // Already translated, use the alias.
      assert(temps[i]);
      temps[i]->trans(e);
    }
    else {
      // Alias earlier args if necessary.
      advance(e, i, temps);

      // Translate using the base method.
      args[i]->arg::trans(e,temps);

      // Push null to indicate the argument has been translated.
      temps.push_back(0);
    }
  }

public:
  sequencer(size_t size)
    : args(size) {}

  arg *addArg(varinit *v, types::ty *t, size_t i) {
    return args[i]=new sequencedArg(v, t, *this, i);
  }
};


class application : public gc {
  types::signature *sig;
  types::function *t;

  // Sequencer to ensure given argument are evaluated in the proper order.  Use
  // of this sequencer means that transArgs can only be called once.
  sequencer seq;

  typedef mem::vector<arg *> arg_vector;
  arg_vector args;
  arrayinit *rest;

  // Target formal to match with arguments to be packed into the rest array.
  types::formal rf;

  // During the matching of arguments to an application, this stores the index
  // of the first unmatched formal.
  size_t index;

  // To resolve which is best of application in case multiple matches of
  // overloaded functions, a score is kept for every source argument matched,
  // and an application with higher-scoring matches is chosen.
  score_vector scores;

  void initRest();

  application(types::signature *sig)
    : sig(sig),
      t(0),
      seq(sig->formals.size()),
      args(sig->formals.size()),
      rest(0),
      rf(0),
      index(0)
    { assert(sig); initRest(); }

  application(types::function *t)
    : sig(t->getSignature()),
      t(t),
      seq(sig->formals.size()),
      args(sig->formals.size()),
      rest(0),
      rf(0),
      index(0)
    { assert(sig); initRest(); }

  types::formal &getTarget() {
    return sig->getFormal(index);
  }

  // Move the index forward one, then keep going until we're at an unmatched
  // argument.
  void advanceIndex() {
    do {
      ++index;
    } while (index < args.size() && args[index]!=0);
  }

  // Finds the first unmatched formal of the given name, returning the index.
  // May return REST or FAIL if it matches the rest parameter or nothing
  // (respectively).
  int find(symbol *name);

  // Match the formal at index to its default argument (if it has one).
  bool matchDefault();

  // Match the argument to be packed into the rest array, if possible.
  bool matchArgumentToRest(env &e, types::formal& f, varinit *a);

  // Matches the argument to a formal in the target signature (possibly causing
  // other formals in the target to be matched to default values), and updates
  // the matchpoint accordingly. 
  bool matchArgument(env &e, types::formal& f,
                     varinit *a, size_t evalIndex);

  // Match an argument bound to a name, as in f(index=7). 
  bool matchNamedArgument(env &e, types::formal& f,
                          varinit *a, size_t evalIndex);

  // After all source formals have been matched, checks if the match is complete
  // (filling in final default values if necessary).
  bool complete();

  // Match a rest argument in the calling expression.
  bool matchRest(env &e, types::formal& f, varinit *a);
 
  // Match the argument represented in signature to the target signature.  On
  // success, all of the arguments in args will be properly set up.
  bool matchSignature(env &e, types::signature *source, arglist &al);

  friend class maximizer;
public:
  // Attempt to build an application given the target signature and the source
  // signature (representing the given arguments).  Return 0 if they cannot be
  // matched.
  static application *match(env &e,
                            types::function *t,
                            types::signature *source,
                            arglist &al);

  // Translate the arguments so they appear in order on the stack in preparation
  // for a call.
  void transArgs(coenv &e);

  types::function *getType() {
    return t;
  }
};

typedef list<application *> app_list;

// Given an overloaded list of types, determines which type to call.  If none
// are applicable, returns an empty vector, if there is ambiguity, several will
// be returned.
app_list multimatch(env &e,
                    types::overloaded *o,
                    types::signature *source,
                    arglist &al);

}  // namespace trans

#endif
