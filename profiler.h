/*****
 * profiler.h
 * Andy Hammerlindl 2010/07/24
 *
 * Profiler for the execution of the virtual machine bytecode.
 *****/

#ifndef PROFILER_H
#define PROFILER_H

#include <unistd.h>
#if _POSIX_TIMERS
#define HAVE_TIMER
#include <time.h>
#endif 

#include <iostream>

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

    // Number of instructions spent in this function or its children.  This is
    // computed by computeTotals.
    int instTotal;

    // The number of real-time nanoseconds spent in this node.  WARNING: May
    // be wildly inaccurate.
    long long nsecs;

    // Total including children.
    long long nsecsTotal;

    // Call stacks resulting from calls during this call stack.
    mem::vector<node> children;

    node(lambda *func)
      : func(func), calls(0),
        instructions(0), instTotal(0),
        nsecs(0), nsecsTotal(0) {}

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

    void computeTotals() {
      instTotal = 0;
      nsecsTotal = 0;
      size_t n = children.size();
      for (size_t i = 0; i < n; ++i) {
        children[i].computeTotals();
        instTotal += children[i].instTotal;
        nsecsTotal += children[i].nsecsTotal;
      }
    }



    void dump(ostream& out) {
#ifdef DEBUG_FRAME
      string name = func ? func->name : "<top level>";
#else
      string name = "";
#endif

      out << "dict(\n"
           << "    name = '" << name << " " << func << "',\n"
           << "    pos = '" << positionFromLambda(func) << "',\n"
           << "    calls = " << calls << ",\n"
           << "    instructions = " << instructions << ",\n"
           << "    nsecs = " << nsecs << ",\n"
           << "    children = [\n";

      size_t n = children.size();
      for (size_t i = 0; i < n; ++i) {
        children[i].dump(out);
        out << ",\n";
      }

      out << "    ])\n";
    }
  };

  // An empty call stack.
  node emptynode;

  // All of the callstacks.
  std::stack<node *> callstack;

  node &topnode() {
    return *callstack.top();
  }

  // Arc representing one function calling another.  Used only for building
  // the output for kcachegrind.
  struct arc : public gc {
    int calls;
    int instTotal;
    long long nsecsTotal;

    arc() : calls(0), instTotal(0), nsecsTotal(0) {}

    void add(node n) {
      calls += n.calls;
      instTotal += n.instTotal;
      nsecsTotal += n.nsecsTotal;
    }
  };

  // Representing one function and its calls to other functions.
  struct fun : public gc {
    int instructions;
    long long nsecs;
    mem::map<lambda *, arc> arcs;

    fun() : instructions(0), nsecs(0) {}
  };

#ifdef HAVE_TIMER
  // The timestamp when the stack was last changed.  Used to compute the
  // realtime number of nanoseconds spent by each function.
  struct timespec timestamp;

  static const clockid_t CLOCK_TYPE = CLOCK_REALTIME;

  void startLap() {
    clock_gettime(CLOCK_TYPE, &timestamp);
  }

  // Measure and return the time since the last timestamp, and reset the
  // timestamp to now.
  long long timeAndResetLap() {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    long long nsecs = 10000000000LL * (now.tv_sec - timestamp.tv_sec) +
                 (now.tv_nsec - timestamp.tv_nsec);
    timestamp = now;
    return nsecs;
  }
#else
  // No fast timer support, just return zero for all time intervals.
  void startLap() {
    std::cerr << "warning: profiling without timer support" << std::endl;
  }
  long long timeAndResetLap() {
    return 0;
  }
#endif

public:
  profiler();

  void beginFunction(lambda *func);
  void endFunction(lambda *func);
  void recordInstruction();

  // TODO: Add position, type of instruction info to profiling.

  // Dump all of the data out to stdout.  This can be interpreted by a
  // different program.  In fact, it is formatted to be a Python data
  // structure that can be read in.
  void dump(ostream &out);
};

inline profiler::profiler()
  : emptynode(0)
{
    callstack.push(&emptynode);
    startLap();
}

inline void profiler::beginFunction(lambda *func) {
  //cout << "begin " << func->name << endl;
  assert(func);
  assert(!callstack.empty());

  // As the node is about to change, record the time spent in the node.
  topnode().nsecs += timeAndResetLap();

  callstack.push(topnode().getChild(func));
  ++topnode().calls;
}

inline void profiler::endFunction(lambda *func) {
  //cout << "end" << endl;
  assert(func);
  assert(!callstack.empty());
  assert(topnode().func == func);

  // As the node is about to change, record the time spent in the node.
  topnode().nsecs += timeAndResetLap();

  callstack.pop();
}

inline void profiler::recordInstruction() {
  assert(!callstack.empty());
  ++topnode().instructions;
}

inline void profiler::dump(ostream& out) {
  out << "profile = ";
  emptynode.dump(out);
}

} // namespace vm

#endif // PROFILER_H
