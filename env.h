/*****
 * env.h
 * Andy Hammerlindl 2002/6/20
 *
 * Keeps track of the namespaces of variables and types when traversing
 * the abstract syntax.
 *****/

#ifndef ENV_H
#define ENV_H

#include <list>
#include <map>
#include <stack>

#include "errormsg.h"
#include "entry.h"
#include "types.h"
#include "cast.h"
#include "record.h"
#include "frame.h"
#include "inst.h"
#include "import.h"
#include "util.h"

namespace trans {

using std::list;

using sym::symbol;
using types::ty;
using types::function;
using types::record;

using vm::bltin;
using vm::inst;
using vm::item;

class genv;

enum modifier {
  DEFAULT_STATIC,
  DEFAULT_DYNAMIC,
  EXPLICIT_STATIC,
  EXPLICIT_DYNAMIC
};

class env {
  // The frame of the function we are currently encoding.  This keeps
  // track of local variables, and parameters with respect to the stack.
  frame *level;

  // The frame of the enclosing record that the "this" expression yields.  ie.
  // the highest frame that is a record, not a function. 
  frame *recordLevel;

  // The type of the enclosing record.  Also needed for the "this" expression.
  record *recordType;
  
  // The lambda being constructed. In some cases, this lambda is needed
  // before full translation of the function, so it is stored,
  // incomplete, here.
  vm::lambda *l;

  // The type of the function being translated.
  const function *funtype;

  // The global environment - keeps track of modules.
  genv &ge;

  // The enclosing environment.  Null if this is a file-level module.
  env *parent;

  // The mode of encoding, either static or dynamic. sord is used as an
  // acronym for Static OR Dynamic.
  // Once something is static, no amount of dynamic modifiers can change
  // that, so once a stack is EXPLICIT_STATIC, additional modifiers will
  // be pushed on as EXPLICIT_STATIC.
  modifier sord;
  std::stack<modifier> sord_stack;

  // What permissions will be given to a new access.
  // TODO: Ensure private fields don't show up calling lookup for a
  // record.
  permission perm;
  

  // These tables keep track of type, variable definitions, and of
  // imported modules.
  tenv &te;
  venv &ve;
  menv &me;

  // The function code as its being written.  Code points to next place in
  // array to write.
  // NOTE: As the program array is only given a finite size at its
  // allocation, it may overflow for a large program.  Check on fixing
  // this.
  vm::program program;

  // Keeps track of labels and writes in memory addresses as they're defined.
  // This way a label can be used before its address is known.
  std::map<int,vm::program::label> defs;
  std::multimap<int,vm::program::label> uses;
  int numLabels;

  // Loops need to store labels to where break and continue statements
  // should pass control to.  Since loops can be nested, this needs to
  // be stored as a stack.
  std::stack<int> breakLabels;
  std::stack<int> continueLabels;

private:
  friend class genv;
  // Start encoding the body of a file-level record.
  env(genv &ge, modifier sord = DEFAULT_DYNAMIC);
  
  // Define a new function environment.
  env(function *t, env &parent, modifier sord = DEFAULT_DYNAMIC);

  // Start encoding the body of the record.  The function being encoded
  // is the record's initializer.
  env(record *t, env &parent, modifier sord = DEFAULT_DYNAMIC);

public:
  env(const env&);
  
  /* Add a static or dynamic modifier. */
  void pushModifier(modifier s)
  {
    /* Default setting should only be used in the constructor. */
    assert(s != DEFAULT_STATIC && s != DEFAULT_DYNAMIC);

    /* Non-default static overrules. */
    if (sord != EXPLICIT_STATIC)
      sord = s;

    sord_stack.push(sord);
  }

  /* Tests if encoding mode is currently static. */
  bool isStatic()
  {
    switch(sord) {
      case DEFAULT_STATIC:
      case EXPLICIT_STATIC:
	return true;
      case DEFAULT_DYNAMIC:
      case EXPLICIT_DYNAMIC:
	return false;
      default:
	assert(False);
	return false;
    }
  }


  /* Remove a modifier. */
  void popModifier()
  {
    assert(!sord_stack.empty());
    sord_stack.pop();

    assert(!sord_stack.empty());
    sord = sord_stack.top();
  }

  /* Set/get/clear permissions. */
  void setPermission(permission p)
  {
    perm = p;
  }
  permission getPermission()
  {
    return perm;
  }
  void clearPermission()
  {
    perm = READONLY;
  }
    

  // Says what the return type of the function is.
  ty *getReturnType() {
    return funtype->result;
  }

  void beginScope()
  {
    te.beginScope(); ve.beginScope(); me.beginScope();
  }
  void endScope()
  {
    te.endScope(); ve.endScope(); me.endScope();
  }

  ty *lookupType(symbol *s)
  {
    // Search in local types.
    ty *t = te.look(s);
    if (t)
      return t;

    // Search in modules.
    t = me.lookupType(s);
    if (t)
      return t;
    
    // Search module names.
    import *i = me.look(s);
    if (i)
      return i->getModule();
    
    // No luck.
    return 0;
  }

  // Returns the import in which the type is contained.
  import *lookupTypeImport(symbol *s)
  {
    // If the typename is in the local environment, it is not imported.
    if (te.look(s))
      return 0;

    // Search in modules.
    import *i = me.lookupTypeImport(s);
    if (i)
      return i;

    // Search the module name, if it is module, it is its own import?
    // NOTE: Types in this fashion should not be allocated!
    i = me.look(s);
    if (i)
      return i;

    // Error!
    assert(False);
    return 0;
  }

  varEntry *lookupExactVar(symbol *name, signature *sig)
  {
    // Search in local vars.
    varEntry *v = ve.lookExact(name, sig);
    if (v)
      return v;

    // Search in modules.
    v = me.lookupExactVar(name, sig);
    if (v)
      return v;
    
    // Search module name.
    import *i = me.look(name);
    if (i)
      return i->getVarEntry();

    // No luck.
    return 0;
  }

  ty *varGetType(symbol *name)
  {
    // NOTE: This overhead seems unnecessarily slow.
    types::overloaded o;
    
    ty *t = ve.getType(name);
    if (t)
      o.add(t);

    t = me.varGetType(name);
    if (t)
      o.addDistinct(t);

    import *i = me.look(name);
    if (i)
      o.addDistinct(i->getModule());

    return o.simplify();
  }

  void addType(position pos, symbol *name, ty *desc)
  {
    if (te.lookInTopScope(name)) {
      em->error(pos);
      *em <<  "type \'" << *name << "\' previously declared";
    }
    te.enter(name, desc);
  }
  
  void addVar(position pos, symbol *name, varEntry *desc, bool ignore=false)
  {
    signature *sig = desc->getSignature();
    if (ve.lookInTopScope(name, sig)) {
      if(ignore) return;
      em->error(pos);
      if (sig) {
        *em << "function variable \'" << *name << *sig
            << "\' previously declared";
      }
      else {
        *em << "variable '" << *name <<  "' previously declared";
      }
    }
    ve.enter(name, desc);
  }

  void addImport(position pos, symbol *name, import *i)
  {
    if (me.lookInTopScope(name)) {
      em->error(pos);
      *em << "multiple imports under name '" << *name << "'";
      return;
    }
    if(settings::verbose > 1) std::cerr << "Importing " <<  *name << std::endl;
    me.enter(name, i);
  }

  record *getModule(symbol *id);
  
  // Creates a new environment to handle the translation of a new function.
  env newFunction(function *t);

  // Creates a new record type.
  record *newRecord(symbol *id);

  // Create an environment for the initializer of the record.
  env newRecordInit(record *r);


  frame *getFrame()
  {
    if (isStatic()) {
      assert(parent);
      return parent->getFrame();
    }
    else
      return level;
  }

  // Allocates space in the function or record frame for a new local variable.
  access *allocLocal()
  {
    return getFrame()->allocLocal(perm);
  }

  // Get the access in the frame for a specified formal parameter.
  access *accessFormal(int index)
  {
    return level->accessFormal(index);
  }


  // The encode functions add instructions and operands on to the code array.
  void encode(inst i)
  {
    if (isStatic()) {
      assert(parent);
      parent->encode(i);
    }
    else {
      program.encode(i);
    }
  }

  void encode(inst::opcode op)
  {
    inst i; i.op = op;
    encode(i);
  }

  void encode(int val)
  {
    inst i; i.val = val;
    encode(i);
  }
  void encode(item it)
  {
    inst i; i.ref = it;
    encode(i);
  }
  void encode(bltin func)
  {
    inst i; i.bfunc = func;
    encode(i);
  }
  void encode(vm::lambda *l)
  {
    inst i; i.lfunc = l;
    encode(i);
  }

  // Puts the requested frame on the stack.  If the frame is not that of
  // this environment or its parents, false is returned.
  bool encode(frame *f);

  // Puts the frame corresponding to the expression "this" on the stack.
  bool encodeThis()
  {
    assert(recordLevel);
    return encode(recordLevel);
  }

  // Returns the type of the enclosing record.
  record *thisType()
  {
    return recordType;
  }

  // Puts the 'dest' frame on the stack, assuming the frame 'top' is on
  // top of the stack.  If 'dest' is not a ancestor frame of 'top',
  // false is returned.
  bool encode(frame *dest, frame *top);

  // This is used when an expression of type source needs to be an
  // expression of type target.
  // If it is allowed, the casting instructions (if any) will be added.
  // Otherwise, an appropriate error message will be printed.
  bool implicitCast(position pos, ty *target, ty *source);

  // Similar to implicitCast, but allows the narowing conversion of
  // real to int.
  bool explicitCast(position pos, ty *target, ty *source);


  // Assigns a handle to the current point in the list of stack
  // instructions and returns that handle.
  int defLabel();

  // Sets the handle given by label to the current point in the list of
  // instructions.
  int defLabel(int label);

  // Encodes the address pointed to by the handle label into the
  // sequence of instructions.  This is useful for jump instruction to
  // jump to where a label was defined.
  void useLabel(int label);

  // If an address has to be used for a jump instruction before it is
  // actually encoded, a handle can be given to it by this function.
  // When that handle's label is later defined, the proper address will
  // be inserted into the code where the handle was used. 
  int fwdLabel();

  void pushBreak(int label) {
    breakLabels.push(label);
  }
  void pushContinue(int label) {
    continueLabels.push(label);
  }
  void popBreak() {
    breakLabels.pop();
  }
  void popContinue() {
    continueLabels.pop();
  }
  bool encodeBreak() {
    if (breakLabels.empty())
      return false;
    else {
      encode(inst::jmp);
      useLabel(breakLabels.top());
      return true;
    }
  }
  bool encodeContinue() {
    if (continueLabels.empty())
      return false;
    else {
      encode(inst::jmp);
      useLabel(continueLabels.top());
      return true;
    }
  }
  
  // Adds an entry into the position list, linking the given point in the
  // source code to the current position in the virtual machine code.  This is
  // used to print positions at runtime.
  void markPos(position pos);

  // When translating the function is finished, this ties up loose ends
  // and returns the lambda.
  vm::lambda *close() {
    // These steps must be done dynamically, not statically.
    sord = EXPLICIT_DYNAMIC;
    sord_stack.push(sord);
    
    // Add a return for void types; may be redundant.
    if (funtype->result->kind == types::ty_void)
      encode(inst::ret);

    l->code = program;
    l->maxStackSize = 10; // NOTE: To be implemented.
    l->params = level->getNumFormals();
    l->vars = level->size();

    sord_stack.pop();
    sord = sord_stack.top();

    return l;
  }
private: // Non-copyable
  void operator=(const env&);
};

} // namespace trans

#endif
