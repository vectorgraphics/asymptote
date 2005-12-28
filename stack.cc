/*****
 * stack.cc
 * Andy Hammerlindl 2002/06/27
 *
 * The general stack machine used to run compiled camp code.
 *****/

#include "stack.h"
#include "program.h"
#include "callable.h"
#include "errormsg.h"
#include "util.h"

//#define DEBUG_STACK

#ifdef DEBUG_STACK
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace vm {
void draw(ostream& out, frame *v);
}
#endif

namespace vm {

namespace {
position curPos = position::nullPos();
const program::label nulllabel;
}

inline stack::vars_t stack::make_frame(size_t size, vars_t closure)
{
  vars_t vars = new frame(1+size);
  (*vars)[0] = closure;
  return vars;
}

stack::stack()
{}

stack::~stack()
{}

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

void stack::run(func *f)
{
  lambda *body = f->body;

#ifdef DEBUG_STACK
  cout << "running lambda: \n";
  print(cout, body->code);
  cout << endl;
#endif
  
  /* make new activation record */
  vars_t vars = make_frame(body->params, f->closure);
  marshall(body->params, vars);

  run(body->code, vars);
}

void stack::run(program *code, vars_t vars)
{
  /* start the new function */
  program::label ip = code->begin();

  em->Pending(settings::verbose > 4);
  
  try {
    for (;;) {
      const inst &i = *ip;
      curPos = i.pos;
      
#ifdef DEBUG_STACK
      cerr << curPos << "\n";
      printInst(cerr, ip, body->code->begin());
      cerr << "\n";
#endif

      if(em->Pending()) em->process(curPos);
      
      switch (i.op)
        {
          case inst::pop:
            pop();
            break;
        
          case inst::intpush:
          case inst::constpush:
            push(i.ref);
            break;
        
          case inst::varpush:
            push((*vars)[get<int>(i)]);
            break;

          case inst::varsave:
            (*vars)[get<int>(i)] = top();
            break;
        
          case inst::fieldpush: {
            vars_t frame = pop<vars_t>();
            if (!frame)
	      error("dereference of null pointer");
            push((*frame)[get<int>(i)]);
            break;
          }
        
          case inst::fieldsave: {
            vars_t frame = pop<vars_t>();
            if (!frame)
	      error("dereference of null pointer");
            (*frame)[get<int>(i)] = top();
            break;
          }
	
          case inst::builtin: {
            bltin func = get<bltin>(i);
            func(this);
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

          case inst::popcall: {
            /* get the function reference off of the stack */
            callable* f = pop<callable*>();
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
            vars->extend(get<int>(i));
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

void stack::load(mem::string index) {
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
#if __GNUC__
#include <cxxabi.h>
string demangle(const char *s)
{
  int status;
  char *demangled = abi::__cxa_demangle(s,NULL,NULL,&status);
  if (status == 0 && demangled) {
    string str(demangled);
    free(demangled);
    return str;
  } else if (status == -2) {
    free(demangled);
    return s;
  } else {
    free(demangled);
    return string("Unknown(") + s + ")";
  }
};
#else
string demangle(const char* s)
{
  return s;
}
#endif 

void stack::draw(ostream& out)
{
//  out.setf(out.hex);

  out << "operands:";
  stack_t::const_iterator left = theStack.begin();
  if (theStack.size() > 10) {
    left = theStack.end()-10;
    out << " ...";
  }
  
  while (left != theStack.end())
    {
      out << " " << demangle(left->type().name());
      left++;
    }
  out << "\n";
}

void draw(ostream& out, frame* v)
{
  out << "vars:    ";
  
  if (!!v) {
    out << (!get<frame*>((*v)[0]) ? " 0" : " link");
    for (size_t i = 1; i < 10 && i < v->size(); i++)
      out << " " << demangle((*v)[i].type().name());
    if (v->size() > 10)
      out << "...";
    out << "\n";
  }
  else
    out << "\n";
}
#endif // DEBUG_STACK

position getPos() {
  return curPos;
}

void error(const char* message)
{
  em->error(curPos);
  *em << message;
  em->sync();
  throw handled_error();
}

interactiveStack::interactiveStack()
  : globals(new frame(1)) {}

void interactiveStack::run(lambda *codelet) {
  stack::run(codelet->code, globals);
}

} // namespace vm
