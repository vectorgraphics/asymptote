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
#endif

namespace vm {

namespace {
const program::label nulllabel;
}

inline stack::vars_t stack::make_frame(size_t size, vars_t closure)
{
  vars_t vars(new item[size]);
  vars[0] = closure;
  return vars;
}

stack::stack()
  : curPos(position::nullPos())
{}

stack::~stack()
{}

void stack::run(lambda *l)
{
  func f;
  f.body = l;
    
  run(&f);
}

void stack::marshall(int args, vars_t vars)
{
  for (int i = args; i > 0; --i)
    vars[i] = pop();
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
  vars_t vars = make_frame(body->vars, f->closure);
  marshall(body->params, vars);

  /* for binops */
  vars_t u, v;

  try {
    for (;;) {
      if (!!ip->pos) curPos = ip->pos;
      
#ifdef DEBUG_STACK
      cerr << getPos() << "\n";
      printInst(cerr, ip, body->code.begin());
      cerr << "\n";
#endif

      if(errorstream::interrupt) throw interrupted();
      
      switch (ip->op)
        {
          case inst::pop:
            pop();
            break;
        
          case inst::intpush:
            push(ip->val);
            break;
        
          case inst::constpush:
            push(ip->ref);
            break;
        
          case inst::varpush:
            push(vars[ip->val]);
            break;
        
          case inst::varsave:
            vars[ip->val] = top();
            break;
        
          case inst::fieldpush: {
            vars_t frame = pop<vars_t>();
            if (!frame) {
	      error(this,"dereference of null pointer");
            }
            push(frame[ip->val]);
            break;
          }
        
          case inst::fieldsave: {
            vars_t frame = pop<vars_t>();
            if (!frame) {
	      error(this,"dereference of null pointer");
            }
            frame[ip->val] = top();
            break;
          }
	
          case inst::builtin: {
            bltin func = ip->bfunc;
            func(this);
            
            em->checkCamp(getPos());
            break;
          }

          case inst::jmp:
            ip = ip->label;
            continue;

          case inst::cjmp:
            if (pop<bool>()) { ip = ip->label; continue; }
            break;

          case inst::njmp:
            if (!pop<bool>()) { ip = ip->label; continue; }
            break;

          case inst::popcall: {
            /* get the function reference off of the stack */
            callable* f = pop<callable*>();
            
            f->call(this);

            em->checkCamp(getPos());
            
            break;
          }

          case inst::pushclosure:
            push(vars);
            break; 

          case inst::makefunc: {
            func *f = new func;
            f->closure = pop<vars_t>();
            f->body = ip->lfunc;

            push((callable*)f);
            break;
          }
        
          case inst::ret: {
            return;
          }
	
          default:
	    error(this,"Internal VM error: Bad stack operand");
        }

#ifdef DEBUG_STACK
      draw(cerr,vars,body->vars);
      cerr << "\n";
#endif
            
      ++ip;
    }
  } catch (boost::bad_any_cast&) {
    error(this,"Trying to use uninitialized value.");
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

void stack::draw(ostream& out, vars_t vars, size_t nvars)
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

  out << "vars:    ";
  vars_t v = vars;
  
  if (!!v) {
    out << (!get<vars_t>(v[0]) ? " 0" : " link");
    for (int i = 1; i < 10 && i < nvars; i++)
      out << " " << demangle(v[i].type().name());
    if (nvars > 10)
      out << "...";
    out << "\n";
  }
  else
    out << "\n";
}
#endif // DEBUG_STACK

position stack::getPos()
{
  return curPos;
}

void error(stack *s, const char* message)
{
  em->error(s->getPos());
  *em << message;
  em->sync();
  throw handled_error();
}

} // namespace vm
