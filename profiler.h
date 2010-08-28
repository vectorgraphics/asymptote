/*****
 * profiler.h
 * Andy Hammerlindl 2010/07/24
 *
 * Profiler for the execution of the virtual machine bytecode.
 *****/

#ifndef PROFILER_H
#define PROFILER_H

#include "inst.h"

namespace vm {

inline position positionFromLambda(lambda *func) {
  if (func == 0)
    return position();

  program& code = *func->code;

  // Check for empty program.
  if (code.begin() == code.end())
    return position();

  return code.begin()->pos;
}

class profiler : public gc {
  // To do call graph analysis, each call stack that occurs in practice is
  // represented by a node.  For instance, if f and g are functions, then
  //   f -> g -> g
  // is represented by a node and
  //   g -> f -> g
  // is represented by a different one.
  struct node {
    // The top-level function of the call stack.
    lambda *func;

    // The number of times the top-level function has been called resulting in
    // this specific call stack.
    int calls;

    // The number of bytecode instructions executed with this exact call stack.
    // It does not include time spent in called function.
    int instructions;

    // Call stacks resulting from calls during this call stack.
    mem::vector<node> children;

    node(lambda *func)
      : func(func), calls(0), instructions(0) {}

    // Return the call stack resulting from a call to func when this call
    // stack is current.
    node *getChild(lambda *func) {
      size_t n = children.size();
      for (size_t i = 0; i < n; ++i)
        if (children[i].func == func)
          return &children[i];

      // Not found, create a new one.
      children.push_back(node(func));
      return &children.back();
    }

    void dump() {
#ifdef DEBUG_FRAME
      string name = func ? func->name : "<top level>";
#else
      string name = "";
#endif

      cout << "dict(\n"
           << "    name = '" << name << " " << func << "',\n"
           << "    pos = '" << positionFromLambda(func) << "',\n"
           << "    calls = " << calls << ",\n"
           << "    instructions = " << instructions << ",\n"
           << "    children = [\n";

      size_t n = children.size();
      for (size_t i = 0; i < n; ++i) {
        children[i].dump();
        cout << ",\n";
      }

      cout << "    ])\n";
    }
  };

  // An empty call stack.
  node emptynode;

  // All of the callstacks.
  std::stack<node *> callstack;

  node &topnode() {
    return *callstack.top();
  }


public:
  profiler();

  void beginFunction(lambda *func);
  void endFunction(lambda *func);
  void recordInstruction();

  // TODO: Add position, type of instruction info to profiling.

  // Dump all of the data out to stdout.  This can be interpreted by a
  // different program.  In fact, it is formatted to be a Python data
  // structure that can be read in.
  void dump();
};

inline profiler::profiler()
  : emptynode(0)
{
    callstack.push(&emptynode);
}

inline void profiler::beginFunction(lambda *func) {
  //cout << "begin " << func->name << endl;
  assert(func);
  assert(!callstack.empty());
  callstack.push(topnode().getChild(func));
  ++topnode().calls;
}

inline void profiler::endFunction(lambda *func) {
  //cout << "end" << endl;
  assert(func);
  assert(!callstack.empty());
  assert(topnode().func == func);
  callstack.pop();
}

inline void profiler::recordInstruction() {
  assert(!callstack.empty());
  ++topnode().instructions;
}

inline void profiler::dump() {
  cout << "profile = ";
  emptynode.dump();
}

} // namespace vm

#endif // PROFILER_H
