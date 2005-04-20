/*****
 * stack.cc
 * Andy Hammerlindl 2002/06/27
 *
 * The general stack machine that will be used to run compiled camp
 * code.
 *****/

#include "stack.h"
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

void stack::marshall(int args, vars_t vars)
{
  for (int i = args; i > 0; --i)
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
  
  /* alias the variables */
  
  /* start the new function */
  program::label ip = body->code.begin();
  /* make new activation record */
  vars_t vars = make_frame(body->params, f->closure);
  marshall(body->params, vars);

  em->Pending(settings::verbose > 4);
  
  try {
    for (;;) {
      const inst &i = *ip;
      curPos = i.pos;
      
#ifdef DEBUG_STACK
      cerr << curPos << "\n";
      printInst(cerr, ip, body->code.begin());
      cerr << "\n";
#endif

      switch (i.op)
        {
          case inst::pop:
            pop();
            break;
        
          case inst::intpush:
            push(i.val);
            break;
        
          case inst::constpush:
            push(i.ref);
            break;
        
          case inst::varpush:
            push((*vars)[i.val]);
            break;

          case inst::varsave:
            (*vars)[i.val] = top();
            break;
        
          case inst::fieldpush: {
            vars_t frame = pop<vars_t>();
            if (!frame)
	      error("dereference of null pointer");
            push((*frame)[i.val]);
            break;
          }
        
          case inst::fieldsave: {
            vars_t frame = pop<vars_t>();
            if (!frame)
	      error("dereference of null pointer");
            (*frame)[i.val] = top();
            break;
          }
	
          case inst::builtin: {
            bltin func = i.bfunc;
            func(this);
            em->checkCamp(curPos);
            break;
          }

          case inst::jmp:
            ip = i.label;
            continue;

          case inst::cjmp:
            if (pop<bool>()) { ip = i.label; continue; }
            break;

          case inst::njmp:
            if (!pop<bool>()) { ip = i.label; continue; }
            break;

          case inst::popcall: {
            /* get the function reference off of the stack */
            callable* f = pop<callable*>();
            f->call(this);
            em->checkCamp(curPos);
            break;
          }

          case inst::pushclosure:
            push(vars);
            break; 

          case inst::makefunc: {
            func *f = new func;
            f->closure = pop<vars_t>();
            f->body = i.lfunc;

            push((callable*)f);
            break;
          }
        
          case inst::ret: {
            return;
          }

          case inst::alloc: {
            vars->extend(i.val);
            break;
          }
	
          default:
	    error("Internal VM error: Bad stack operand");
        }

#ifdef DEBUG_STACK
      draw(cerr);
      draw(cerr,vars);
      cerr << "\n";
#endif
            
      if(em->Pending()) em->process(curPos);
      ++ip;
    }
  } catch (bad_item_value&) {
    error("Trying to use uninitialized value.");
  }
}

#ifdef DEBUG_STACK
#if __GNUC__
#include <cxxabi.h>
std::string demangle(const char *s)
{
  int status;
  char *demangled = abi::__cxa_demangle(s,NULL,NULL,&status);
  if (status == 0 && demangled) {
    std::string str(demangled);
    free(demangled);
    return str;
  } else if (status == -2) {
    free(demangled);
    return s;
  } else {
    free(demangled);
    return std::string("Unknown(") + s + ")";
  }
};
#else
std::string demangle(const char* s)
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
    for (int i = 1; i < 10 && i < v->size(); i++)
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

} // namespace vm
