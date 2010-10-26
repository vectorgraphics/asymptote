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
  vars_t vars = new frame(name, 1+size);
  (*vars)[0] = closure;
  return vars;
}
#else
inline stack::vars_t stack::make_frame(size_t size, vars_t closure)
{
  vars_t vars = new frame(1+size);
  (*vars)[0] = closure;
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
    (*vars)[i] = pop();
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
  
  /* make new activation record */
#ifdef DEBUG_FRAME
  assert(!body->name.empty());
  vars_t vars = make_frame(body->name, body->params, f->closure);
#else
  vars_t vars = make_frame(body->params, f->closure);
#endif
  marshall(body->params, vars);

  run(body->code, vars);

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


  
void stack::run(program *code, vars_t vars)
{
  /* start the new function */
  program::label ip = code->begin();

  try {
    for (;;) {
      const inst &i = *ip;
      curPos = i.pos;
      
#ifdef PROFILE
      prof.recordInstruction();
#endif

#ifdef DEBUG_STACK
      cerr << curPos << "\n";
      printInst(cerr, ip, code->begin());
      cerr << "\n";
#endif

      if(settings::verbose > 4) em.trace(curPos);
      
      if(!bplist.empty()) debug();
      
      if(errorstream::interrupt) throw interrupted();
      
      switch (i.op)
        {
          case inst::nop:
            break;

          case inst::pop:
            pop();
            break;
        
          case inst::intpush:
          case inst::constpush:
            push(i.ref);
            break;
        
          case inst::varpush:
            push((*vars)[get<Int>(i)]);
            break;

          case inst::varsave:
            (*vars)[get<Int>(i)] = top();
            break;
        
#ifdef COMBO
          case inst::varpop:
            (*vars)[get<Int>(i)] = pop();
            break;

          case inst::fieldpop: {
            vars_t frame = pop<vars_t>();
            if (!frame)
              error("dereference of null pointer");
            (*frame)[get<Int>(i)] = pop();
            break;
          }
#endif
        
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
#endif

          case inst::push_default:
            push(Default);
            break;

          case inst::popcall: {
            /* get the function reference off of the stack */
            callable* f = pop<callable*>();

#if 0
            cout << "popcall ";
            if (dynamic_cast<nullfunc *>(f))
              cout << "nullfunc";
            if (dynamic_cast<func *>(f))
              cout << "func";
            if (dynamic_cast<bfunc *>(f))
              cout << "bfunc";
            if (dynamic_cast<thunk *>(f))
              cout << "thunk";
            cout << '\n';
#endif

            f->call(this);
            break;
          }

          case inst::pushclosure:
            push(vars);
            break; 

          case inst::makefunc: {
            func *f = new func;
            f->closure = pop<vars_t>();
            f->body = get<lambda*>(i);

            push((callable*)f);
            break;
          }
        
          case inst::ret: {
            return;
          }

          case inst::alloc: {
            vars->extend(get<Int>(i));
            break;
          }

          case inst::pushframe: {
#ifdef DEBUG_FRAME
            vars=make_frame("<pushed frame>", 0, vars);
#else
            vars=make_frame(0, vars);
#endif
            break;
          }

          case inst::popframe: {
            vars=get<frame *>((*vars)[0]);
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
    out << "  " <<  v->getName() << ":";

    frame *parent=0;
    item link=(*v)[0];
    try {
      parent = get<frame *>(link);
      out << (parent ? "  link" :  "  ----");
    } catch (bad_item_value&) {
      out << "  " << (*v)[0];
    }

    for (size_t i = 1; i < MAX_ITEMS && i < v->size(); i++) {
      out << " | " << i << ": " << (*v)[i];
    }
    if (v->size() > MAX_ITEMS)
      out << "...";
    out << "\n";

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
  : globals(new frame("globals", 0)) {}
#else
  : globals(new frame(0)) {}
#endif


void interactiveStack::run(lambda *codelet) {
  stack::run(codelet->code, globals);
}

} // namespace vm

