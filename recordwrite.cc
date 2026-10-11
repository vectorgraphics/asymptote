//
// recordwrite.cc
//
// Writes the values that the heterogeneous write() runtime (write_var in
// arrayop.h) cannot write as scalars: structs, through their own writer if
// they have one and field by field otherwise, and values with no textual
// form, as a placeholder naming their type.
//

#include "stack.h"
#include "types.h"
#include "record.h"
#include "entry.h"
#include "access.h"
#include "callable.h"
#include "fileio.h"
#include "util.h"
#include "array.h"
#include "frame.h"
#include "settings.h"
#include "pair.h"
#include "triple.h"
#include "pen.h"
#include "transform.h"
#include "path.h"
#include "guide.h"

#include <algorithm>

namespace run {

// A no-op suffix: pops the file argument and does nothing.
static void noSuffixBltin(vm::stack *s)
{
  (void)vm::pop<camp::file *>(s);
}

bool isWriteSuffixType(types::ty *t)
{
  static types::function *suffixType = new types::function(
    types::primVoid(), types::formal(types::primFile()));
  return t->kind == types::ty_function && types::equivalent(t, suffixType);
}

// Calls the writer that belongs to the struct type r, if it has one, to write
// the instance `frame` to f.  Returns false if r has no writer.
bool callRecordWriter(vm::stack *s, vm::vmFrame *frame,
                      types::record *r, camp::file *f);

namespace {

types::function *suffixType()
{
  static types::function *t = new types::function(
    types::primVoid(), types::formal(types::primFile()));
  return t;
}

// A writer that belongs to a struct type, so that it can be found from the
// type alone, wherever a value of that type is written.
struct RecordWriter {
  trans::localAccess *loc = nullptr;
  // False for a method   void write(file, void(file)),
  // true for a static (typically autounravel) function
  //                      void write(file, T, void(file)).
  bool takesValue = false;
};

RecordWriter findRecordWriter(types::record *r)
{
  static symbol writeSym = symbol::trans("write");
  RecordWriter w;

  types::function method(types::primVoid(),
                         types::formal(types::primFile()),
                         types::formal(suffixType()));
  types::function statik(types::primVoid(),
                         types::formal(types::primFile()),
                         types::formal(r),
                         types::formal(suffixType()));

  // A private writer is an implementation detail of the struct, not a
  // statement of how its values are to be displayed.
  trans::varEntry *ve = r->e.lookupVarByType(writeSym, &method);
  if (!ve || ve->isPrivate()) {
    ve = r->e.lookupVarByType(writeSym, &statik);
    w.takesValue = true;
  }
  if (ve && !ve->isPrivate())
    w.loc = dynamic_cast<trans::localAccess *>(ve->getLocation());
  return w;
}

// Returns the frame at the given level, starting from an instance of r.
vm::vmFrame *frameAtLevel(vm::vmFrame *frame, types::record *r,
                          trans::frame *target)
{
  trans::frame *level = r->getLevel();
  while (frame && level && level != target) {
    vm::item& parent = (*frame)[level->parentIndex()];
    frame = parent.empty() ? nullptr : vm::get<vm::vmFrame *>(parent);
    level = level->getParent();
  }
  return level == target ? frame : nullptr;
}

// A data field of a struct instance.
struct Field {
  Int offset;
  symbol name;
  types::ty *t;
};

// Returns the fields stored in each instance of r, in the order they were
// declared.  Methods, meaning fields introduced by a function definition, are
// left out: they would clutter the output without saying anything about the
// value.  A variable of function type is kept.
mem::vector<Field> dataFields(types::record *r)
{
  mem::vector<Field> fields;
  trans::frame *level = r->getLevel();
  r->e.ve.forEach([&](symbol name, trans::varEntry *v) {
    types::ty *t = v->getType();
    if (!t || v->isFunctionDefinition())
      return;
    auto *loc = dynamic_cast<trans::localAccess *>(v->getLocation());
    if (loc && loc->getLevel() == level)
      fields.push_back({loc->getOffset(), name, t});
  });
  std::sort(fields.begin(), fields.end(),
            [](const Field& a, const Field& b) { return a.offset < b.offset; });
  return fields;
}

// Writes descriptions of values, within a budget of characters.  When the
// budget runs out, an ellipsis is written and everything further is dropped
// except for closing brackets, so that brackets always match.
class Describer {
  vm::stack *s;
  camp::file *f;
  Int maxDepth;
  bool limited;
  size_t remaining;
  bool truncated = false;
  // The structs whose fields are currently being written, outermost first.
  mem::vector<vm::vmFrame *> enclosing;

  void truncate()
  {
    f->write(string("..."));
    truncated = true;
  }

  // Writes text that counts against the budget.  If it does not fit, it is
  // dropped, unless it is splittable, in which case as much as fits is kept.
  void emit(const string& text, bool splittable = false)
  {
    if (truncated)
      return;
    if (limited && text.size() > remaining) {
      if (splittable && remaining > 0)
        f->write(text.substr(0, remaining));
      remaining = 0;
      truncate();
      return;
    }
    f->write(text);
    if (limited)
      remaining -= text.size();
  }

  // Writes a closing bracket or quote, which is never dropped.
  void close(const char *bracket)
  {
    string text(bracket);
    f->write(text);
    if (limited)
      remaining -= std::min(remaining, text.size());
  }

  // Writes a string in quotes.  A string that does not fit is cut short, but
  // the quotes always match.
  void emitQuoted(const string& text)
  {
    emit("\"");
    if (truncated)
      return;
    emit(text, true);
    close("\"");
  }

  template<class T>
  void emitValue(const T& value)
  {
    ostringstream out;
    out.precision(settings::getSetting<Int>("digits"));
    out << value;
    emit(out.str());
  }

  void emitType(types::ty *t)
  {
    ostringstream out;
    out << "<" << *t << ">";
    emit(out.str());
  }

  void describeArray(types::array *t, vm::item val, Int depth)
  {
    vm::array *a = vm::get<vm::array *>(val);
    if (!a) {
      emit("null");
      return;
    }
    if (t->celltype->kind == types::ty_function) {
      ostringstream out;
      out << "<" << *t << " of length " << a->size() << ">";
      emit(out.str());
      return;
    }
    emit("{");
    for (size_t i = 0; i < a->size() && !truncated; ++i) {
      if (i > 0)
        emit(", ");
      describe(t->celltype, (*a)[i], depth);
    }
    close("}");
  }

  void describeRecord(types::record *r, vm::item val, Int depth)
  {
    vm::vmFrame *frame = vm::get<vm::vmFrame *>(val);
    if (!frame) {
      emit("null");
      return;
    }
    if (truncated)
      return;
    // The output of a struct's own writer is not counted against the budget.
    if (callRecordWriter(s, frame, r, f))
      return;
    if (depth > maxDepth) {
      emitType(r);
      return;
    }
    // A struct that contains itself is not written again.
    if (std::find(enclosing.begin(), enclosing.end(), frame) !=
        enclosing.end()) {
      ostringstream out;
      out << "<" << *r << ", cyclic>";
      emit(out.str());
      return;
    }
    enclosing.push_back(frame);
    emit("(");
    bool first = true;
    for (const Field& field : dataFields(r)) {
      if (truncated)
        break;
      if (!first)
        emit(", ");
      first = false;
      emit((string) field.name + "=");
      describe(field.t, (*frame)[field.offset], depth + 1);
    }
    close(")");
    enclosing.pop_back();
  }

public:
  Describer(vm::stack *s, camp::file *f)
    : s(s), f(f), maxDepth(settings::getSetting<Int>("structdepth"))
  {
    Int limit = settings::getSetting<Int>("structlimit");
    limited = limit > 0;
    remaining = limited ? (size_t) limit : 0;
  }

  // Writes val, of type t, as it appears at the given depth of nesting of
  // structs (1 for a value that is not inside any struct).
  void describe(types::ty *t, vm::item val, Int depth)
  {
    if (truncated)
      return;
    if (val.empty()) {
      emit("<uninitialized>");
      return;
    }
    switch (t->kind) {
      case types::ty_null:
        emit("null");
        break;
      case types::ty_boolean:
        emit(vm::get<bool>(val) ? "true" : "false");
        break;
      case types::ty_Int:
        emitValue(vm::get<Int>(val));
        break;
      case types::ty_real:
        emitValue(vm::get<double>(val));
        break;
      case types::ty_pair:
        emitValue(vm::get<camp::pair>(val));
        break;
      case types::ty_triple:
        emitValue(vm::get<camp::triple>(val));
        break;
      case types::ty_string:
        emitQuoted(vm::get<string>(val));
        break;
      case types::ty_pen:
        emitValue(vm::get<camp::pen>(val));
        break;
      case types::ty_transform:
        emitValue(vm::get<camp::transform>(val));
        break;
      case types::ty_guide:
        emitValue(*vm::get<camp::guide *>(val));
        break;
      case types::ty_path:
        emitValue(*vm::get<camp::path *>(val));
        break;
      case types::ty_array:
        describeArray(static_cast<types::array *>(t), val, depth);
        break;
      case types::ty_record:
        describeRecord(static_cast<types::record *>(t), val, depth);
        break;
      default:
        // Functions, files and anything else with no textual form.
        emitType(t);
        break;
    }
  }
};

} // namespace

bool callRecordWriter(vm::stack *s, vm::vmFrame *frame,
                      types::record *r, camp::file *f)
{
  // Note: access control is not checked against the context of the write()
  // call.  The writer need only not be private.
  RecordWriter w = findRecordWriter(r);
  if (!w.loc)
    return false;

  vm::vmFrame *home = frameAtLevel(frame, r, w.loc->getLevel());
  if (!home)
    return false;
  vm::item& writerItem = (*home)[w.loc->getOffset()];
  if (writerItem.empty())
    return false;
  vm::callable *writer = vm::get<vm::callable *>(writerItem);
  if (!writer)
    return false;

  // Call the writer with a no-op suffix.
  static vm::bfunc noSuffix(noSuffixBltin);
  s->push((vm::item) f);
  if (w.takesValue)
    s->push((vm::item) frame);
  s->push((vm::item) (vm::callable *) &noSuffix);
  writer->call(s);
  return true;
}

void describeValue(vm::stack *s, camp::file *f, types::ty *t, vm::item val)
{
  Describer(s, f).describe(t, val, 1);
}

} // namespace run
