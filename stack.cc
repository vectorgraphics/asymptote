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

inline stack::vars_t stack::make_frame(size_t size)
{
  return frame(new item[size]);
}

stack::stack(int numGlobals)
  : numGlobals(numGlobals), vars()
{
  ip = nulllabel;
  globals = make_frame(numGlobals);
}

stack::~stack()
{}

void stack::run(lambda *l)
{
  func f;
  f.body = l;
    
  run(&f);
}

#define UNALIAS                                 \
  {                                             \
    this->ip = ip;                              \
    this->vars = vars;                          \
    this->body = body;                          \
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
  vars_t vars = vars = make_frame(body->vars);

  vars[0] = f->closure;
  for (int i = body->params; i > 0; --i)
    vars[i] = pop();

  /* for binops */
  vars_t u, v;

  try {
    for (;;) {
#ifdef DEBUG_STACK
      UNALIAS;
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
        
          case inst::globalpush:
            push(globals[ip->val]);
            break;
        
          case inst::globalsave:
            globals[ip->val] = top();
            break;
        
          case inst::fieldpush: {
            vars_t frame = pop<vars_t>();
            if (!frame) {
              UNALIAS;
	      error(this,"dereference of null pointer");
            }
            push(frame[ip->val]);
            break;
          }
        
          case inst::fieldsave: {
            vars_t frame = pop<vars_t>();
            if (!frame) {
              UNALIAS;
	      error(this,"dereference of null pointer");
            }
            frame[ip->val] = top();
            break;
          }
	
          case inst::builtin: {
            bltin func = ip->bfunc;
            func(this);
            
            UNALIAS;
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
            UNALIAS;
            
            f->call(this);

            UNALIAS;
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
		      
          case inst::alloc: {
            // Get the record's enclosing frame off the stack.
            vars_t frame = pop<vars_t>();
	
            lambda *init = ip->lfunc;

            // Call the initializer.
            func f;
            f.body = init; f.closure = frame;

            run(&f);
            break;
          }
	
          default:
            UNALIAS;
	    error(this,"Internal VM error: Bad stack operand");
        }

#ifdef DEBUG_STACK
      UNALIAS;
      draw(cerr);
      cerr << "\n";
#endif
            
      ++ip;
    }
  } catch (boost::bad_any_cast&) {
    error(this,"Trying to use uninitialized value.");
  }
}

#undef UNALIAS

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

  out << "vars:    ";
  vars_t v = vars;
  
  if (!!v) {
    out << (!get<vars_t>(v[0]) ? " 0" : " link");
    for (int i = 1; i < 10 && i < body->vars; i++)
      out << " " << demangle(v[i].type().name());
    if (body->vars > 10)
      out << "...";
    out << "\n";
  }
  else
    out << "\n";

  out << "globals: ";
  vars_t g = globals;
  for (int i = 0; i < 10 && i < numGlobals; i++)
    out << " " << demangle(g[i].type().name());
  if (numGlobals > 10)
    out << " ...\n";
  else
    out << " \n";
}
#endif // DEBUG_STACK

position stack::getPos()
{
  return body ? body->pl.getPos(ip) : position::nullPos();
}

} // namespace vm
