/*****
 * stack.cc
 * Andy Hammerlindl 2002/06/27
 *
 * The general stack machine used to run compiled camp code.
 *****/

#include <fstream>
#include <sstream>

#include "stack.h"
#include "program.h"
#include "callable.h"
#include "errormsg.h"
#include "util.h"
#include "runtime.h"

#include "profiler.h"

#ifdef DEBUG_STACK
#include <iostream>

namespace vm {
void draw(ostream& out, frame *v);
}
#endif

namespace run {
void breakpoint(vm::stack *Stack, absyntax::runnable *r);
}

namespace vm {

mem::list<bpinfo> bplist;
  
namespace {
position curPos = nullPos;
const program::label nulllabel;
}

#ifdef DEBUG_FRAME
inline stack::vars_t stack::make_frame(string name,
                                       size_t size, vars_t closure)
{
  vars_t vars = new frame(name, size, 1+size);
  (*vars)[size] = closure;
  return vars;
}
#else
inline stack::vars_t stack::make_frame(size_t size, vars_t closure)
{
  vars_t vars = new frame(1+size);
  (*vars)[size] = closure;
  return vars;
}
#endif

void run(lambda *l)
{
  func f;
  f.body = l;
  stack s;
  s.run(&f);
}

void stack::marshall(size_t args, vars_t vars)
{
  for (size_t i = args; i > 0; --i)
    (*vars)[i-1] = pop();
}

#ifdef PROFILE

#ifndef DEBUG_FRAME
#warning "profiler needs DEBUG_FRAME for function names"
#endif
#ifndef DEBUG_BLTIN
#warning "profiler needs DEBUG_BLTIN for builtin function names"
#endif

profiler prof;

void dumpProfile() {
  std::ofstream out("asyprof");
  if (!out.fail())
    prof.dump(out);
}
#endif

void assessClosure(lambda *body) {
  // If we have already determined if it needs closure, just return.
  if (body->closureReq != lambda::MAYBE_NEEDS_CLOSURE)
    return;

  for (program::label l = body->code->begin(); l != body->code->end(); ++l)
    if (l->op == inst::pushclosure ||
        l->op == inst::pushframe) {
      body->closureReq = lambda::NEEDS_CLOSURE;
      return;
    }

  body->closureReq = lambda::DOESNT_NEED_CLOSURE;
}

void stack::run(func *f)
{
  lambda *body = f->body;

#ifdef PROFILE
  prof.beginFunction(body);
#endif

#ifdef DEBUG_STACK
#ifdef DEBUG_FRAME
  cout << "running lambda " + body->name + ": \n";
#else
  cout << "running lambda: \n";
#endif
  print(cout, body->code);
  cout << endl;
#endif

  runWithOrWithoutClosure(body, 0, f->closure);

#ifdef PROFILE
  prof.endFunction(body);
#endif
}

void stack::breakpoint(absyntax::runnable *r) 
{
  lastPos=curPos;
  indebugger=true;
  ::run::breakpoint(this,r);
  string s=vm::pop<string>(this);
  debugOp=(s.length() > 0) ? s[0] : (char) 0;
  indebugger=false;
}
  
void stack::debug() 
{
  if(!curPos) return;
  if(indebugger) {em.clear(); return;}
  
  switch(debugOp) {
    case 'i': // inst
      breakpoint();
      break;
    case 's': // step
      if((!curPos.match(lastPos.filename()) || !curPos.match(lastPos.Line())))
        breakpoint();
      break;
    case 'n': // next
      if(curPos.match(lastPos.filename()) && !curPos.match(lastPos.Line()))
        breakpoint();
      break;
    case 'f': // file
      if(!curPos.match(lastPos.filename()))
        breakpoint();
      break;
    case 'r': // return
      if(curPos.match(breakPos.filename()))
        breakpoint();
      break;
    case 'c': // continue
    default:
      for(mem::list<bpinfo>::iterator p=bplist.begin(); p != bplist.end(); ++p) {
        if(curPos.match(p->f.name()) && curPos.match(p->f.line()) &&
           (newline || !curPos.match(breakPos.filename()) ||
            !curPos.match(breakPos.Line()))) {
          breakPos=curPos;
          breakpoint(p->r);
          newline=false;
          break;
        }
        if(!newline && 
           (curPos.match(lastPos.filename()) && !curPos.match(lastPos.Line())))
          newline=true;
      }
      break;
  }
}

void stack::runWithOrWithoutClosure(lambda *l, vars_t vars, vars_t parent)
{
  // Link to the variables, be they in a closure or on the stack.
  mem::vector<item>* varlink;
  Int frameStart = 0;

  // The size of the frame (when running without closure).
  size_t frameSize = l->params;

#define SET_VARLINK assert(vars); varlink = &vars->vars
#define VAR(n) ( (*varlink)[(n) + frameStart] )

  // Set up the closure, if necessary.
  if (vars == 0)
  {
    assessClosure(l);
    if (l->closureReq == lambda::NEEDS_CLOSURE)
    {
      /* make new activation record */
#ifdef DEBUG_FRAME
      assert(!l->name.empty());
      vars = make_frame(l->name, l->params, parent);
#else
      vars = make_frame(l->params, parent);
#endif
      assert(vars);
    }
    else 
    {
      assert(l->closureReq == lambda::DOESNT_NEED_CLOSURE);

      // Use the stack to store variables.
      varlink = &theStack;

      // Record where the parameters start on the stack.
      frameStart = theStack.size() - frameSize;

      // Add the parent's closure to the frame.
      push(parent);
      ++frameSize;
    }
  }

  if (vars) {
      marshall(l->params, vars);

      SET_VARLINK;
  }

  /* start the new function */
  program::label ip = l->code->begin();

  try {
    for (;;) {
      const inst &i = *ip;
      curPos = i.pos;
      
#ifdef PROFILE
      prof.recordInstruction();
#endif

#ifdef DEBUG_STACK
      printInst(cout, ip, l->code->begin());
      cout << "    (";
			i.pos.printTerse(cout);
			cout << ")\n";
#endif

      if(settings::verbose > 4) em.trace(curPos);
      
      if(!bplist.empty()) debug();
      
      if(errorstream::interrupt) throw interrupted();
      
      switch (i.op)
        {
          case inst::varpush:
            push(VAR(get<Int>(i)));
            break;

          case inst::varsave:
            VAR(get<Int>(i)) = top();
            break;
        
#ifdef COMBO
          case inst::varpop:
            VAR(get<Int>(i)) = pop();
            break;
#endif

          case inst::ret: {
            if (vars == 0)
              // Delete the frame from the stack.
              theStack.erase(theStack.begin() + frameStart,
                             theStack.begin() + frameStart + frameSize);
            return;
          }

          case inst::alloc: {
            if (vars)
              vars->extend(get<Int>(i));
            else {
              // Insert more frame space into the stack, moving operands on
              // the stack, if there are any.
              size_t newFrameSize = (size_t)get<Int>(i);
              if (newFrameSize <= frameSize)
                break;

              theStack.insert(theStack.begin() + frameStart + frameSize,
                              newFrameSize - frameSize,
                              item());
              frameSize = newFrameSize;
            }
            break;
          }

          case inst::pushframe:
          {
            assert(vars);
#ifdef DEBUG_FRAME
            vars=make_frame("<pushed frame>", 0, vars);
#else
            vars=make_frame(0, vars);
#endif
            SET_VARLINK;

            break;
          }

          case inst::popframe:
          {
            assert(vars);
            vars=get<frame *>((*vars)[0]);

            SET_VARLINK;

            break;
          }

          case inst::pushclosure:
            assert(vars);
            push(vars);
            break; 

          case inst::nop:
            break;

          case inst::pop:
            pop();
            break;
        
          case inst::intpush:
          case inst::constpush:
            push(i.ref);
            break;
        
          case inst::fieldpush: {
            vars_t frame = pop<vars_t>();
            if (!frame)
              error("dereference of null pointer");
            push((*frame)[get<Int>(i)]);
            break;
          }
        
          case inst::fieldsave: {
            vars_t frame = pop<vars_t>();
            if (!frame)
              error("dereference of null pointer");
            (*frame)[get<Int>(i)] = top();
            break;
          }

#if COMBO
          case inst::fieldpop: {
            vars_t frame = pop<vars_t>();
            if (!frame)
              error("dereference of null pointer");
            (*frame)[get<Int>(i)] = pop();
            break;
          }
#endif
        
        
          case inst::builtin: {
            bltin func = get<bltin>(i);
#ifdef PROFILE
            prof.beginFunction(func);
#endif
            func(this);
#ifdef PROFILE
            prof.endFunction(func);
#endif
            break;
          }

          case inst::jmp:
            ip = get<program::label>(i);
            continue;

          case inst::cjmp:
            if (pop<bool>()) { ip = get<program::label>(i); continue; }
            break;

          case inst::njmp:
            if (!pop<bool>()) { ip = get<program::label>(i); continue; }
            break;

          case inst::jump_if_not_default:
            if (!isdefault(pop())) { ip = get<program::label>(i); continue; }
            break;

#ifdef COMBO
          case inst::gejmp: {
            Int y = pop<Int>();
            Int x = pop<Int>();
            if (x>=y)
              { ip = get<program::label>(i); continue; }
            break;
          }

#if 0
          case inst::jump_if_func_eq: {
            callable * b=pop<callable *>();
            callable * a=pop<callable *>();
            if (a->compare(b))
              { ip = get<program::label>(i); continue; }
            break;
          }

          case inst::jump_if_func_neq: {
            callable * b=pop<callable *>();
            callable * a=pop<callable *>();
            if (!a->compare(b))
              { ip = get<program::label>(i); continue; }
            break;
          }
#endif
#endif

          case inst::push_default:
            push(Default);
            break;

          case inst::popcall: {
            /* get the function reference off of the stack */
            callable* f = pop<callable*>();
            f->call(this);
            break;
          }

          case inst::makefunc: {
            func *f = new func;
            f->closure = pop<vars_t>();
            f->body = get<lambda*>(i);

            push((callable*)f);
            break;
          }
        
          default:
            error("Internal VM error: Bad stack operand");
        }

#ifdef DEBUG_STACK
      draw(cerr);
      vm::draw(cerr,vars);
      cerr << "\n";
#endif
            
      ++ip;
    }
  } catch (bad_item_value&) {
    error("Trying to use uninitialized value.");
  }

#undef SET_VARLINK
#undef VAR
}

void stack::load(string index) {
  frame *inst=instMap[index];
  if (inst)
    push(inst);
  else {
    func f;
    assert(initMap);
    f.body=(*initMap)[index];
    assert(f.body);
    run(&f);
    instMap[index]=get<frame *>(top());
  }
}


#ifdef DEBUG_STACK

const size_t MAX_ITEMS=20;

void stack::draw(ostream& out)
{
//  out.setf(out.hex);

  out << "operands:";
  stack_t::const_iterator left = theStack.begin();
  if (theStack.size() > MAX_ITEMS) {
    left = theStack.end()-MAX_ITEMS;
    out << " ...";
  }
  else
    out << " ";
  
  while (left != theStack.end())
    {
      if (left != theStack.begin())
        out << " | " ;
      out << *left;
      left++;
    }
  out << "\n";
}

void draw(ostream& out, frame* v)
{
  out << "vars:" << endl;
  
  while (!!v) {
    item link=(*v)[v->getParentIndex()];

    out << "  " <<  v->getName() << ":  ";

    for (size_t i = 0; i < MAX_ITEMS && i < v->size(); i++) {
      if (i > 0)
        out << " | ";
      out << i << ": ";

      if (i == v->getParentIndex()) {
        try {
          frame *parent = get<frame *>(link);
          out << (parent ? "link" :  "----");
        } catch (bad_item_value&) {
          out << "non-link " << (*v)[0];
        }
      } else {
        out << (*v)[i];
      }
    }

    if (v->size() > MAX_ITEMS)
      out << "...";
    out << "\n";


    frame *parent;
    try {
      parent = get<frame *>(link);
    } catch (bad_item_value&) {
      parent = 0;
    }

    v = parent;
  }
}
#endif // DEBUG_STACK

position getPos() {
  return curPos;
}

void errornothrow(const char* message)
{
  em.error(curPos);
  em << message;
  em.sync();
}
  
void error(const char* message)
{
  errornothrow(message);
  throw handled_error();
}
  
void error(const ostringstream& message)
{
  error(message.str().c_str());
}

interactiveStack::interactiveStack()
#ifdef DEBUG_FRAME
  : globals(new frame("globals", 0, 0)) {}
#else
  : globals(new frame(0)) {}
#endif


void interactiveStack::run(lambda *codelet) {
  stack::runWithOrWithoutClosure(codelet, globals, 0);
}

} // namespace vm

