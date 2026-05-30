/*****
 * pluginloader.cc
 * Loader for C++ plugin modules (asybind) — Phase 2.
 *
 * Phase 2 adds, on top of Phase 1:
 *   - make_function_type:  cache and return a types::function* for use as
 *                          an asy function type in signatures.
 *   - push_callable/pop_callable/invoke_callable: marshal and invoke
 *                          asy callables (vm::callable*) from C++.
 *   - result_class:        synthesize (and cache) a `result_T` record
 *                          carrying readonly `bool found` and `T value`.
 *   - push_result:         construct and push a `result_T` instance.
 *****/

#include "pluginloader.h"

#include "common.h"
#include "errormsg.h"
#include "locate.h"
#include "record.h"
#include "types.h"
#include "entry.h"
#include "access.h"
#include "builtin.h"
#include "vm.h"
#include "stack.h"
#include "frame.h"
#include "memory.h"
#include "inst.h"
#include "program.h"
#include "callable.h"
#include "virtualfieldaccess.h"
#include "coder.h"

#include "asybind/abi.h"

#include <array>
#include <atomic>
#include <cstring>
#include <utility>

#ifdef PACKAGE_VERSION
#define ASYBIND_HOST_VERSION_STRING PACKAGE_VERSION
#else
#define ASYBIND_HOST_VERSION_STRING "unknown"
#endif

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

using namespace types;

namespace asybind {

namespace {

// =====================================================================
//  Slot registry for plugin thunks. Each registered plugin function is
//  assigned a unique static wrapper bltin drawn from a fixed-size pool.
// =====================================================================

constexpr int kMaxPluginFns = 1024;

struct PluginSlot {
  asybind_thunk_t thunk = nullptr;
};

std::array<PluginSlot, kMaxPluginFns> g_slots{};
std::atomic<int> g_nextSlot{0};

extern const asybind_host_api_v1 g_host_api;

template <int N>
void plugin_wrapper(vm::stack* s) {
  auto& slot = g_slots[N];
  slot.thunk(reinterpret_cast<asybind_stack_ptr>(s), &g_host_api);
}

template <int... Is>
constexpr std::array<vm::bltin, sizeof...(Is)>
make_wrappers(std::integer_sequence<int, Is...>) {
  return { &plugin_wrapper<Is>... };
}

const auto g_wrappers =
    make_wrappers(std::make_integer_sequence<int, kMaxPluginFns>{});

vm::bltin allocate_bltin_for(asybind_thunk_t thunk) {
  int slot = g_nextSlot.fetch_add(1);
  if (slot >= kMaxPluginFns) {
    em.compiler(nullPos);
    em << "asybind: exceeded plugin-function pool (limit "
       << kMaxPluginFns << ")";
    em.sync(true);
    return nullptr;
  }
  g_slots[slot].thunk = thunk;
  return g_wrappers[slot];
}

// =====================================================================
//  Method-getter pool. Each method registered via add_method needs a
//  separate small bltin that pops the receiver and pushes a thunk-bound
//  callable so that `f = obj.method;` produces a usable function value.
// =====================================================================

constexpr int kMaxMethodGetters = 1024;

struct GetterSlot {
  vm::bltin caller = nullptr;
};

std::array<GetterSlot, kMaxMethodGetters> g_getter_slots{};
std::atomic<int> g_nextGetter{0};

// Marker type so opaque plugin-instance pointers travel through vm::item
// with a stable type tag, regardless of the underlying C++ class.
struct PluginInstance {};

template <int N>
void method_getter_wrapper(vm::stack* s) {
  PluginInstance* recv = vm::pop<PluginInstance*>(s);
  vm::bltin caller = g_getter_slots[N].caller;
  s->push<vm::callable*>(
      new vm::thunk(new vm::bfunc(caller),
                    vm::item(recv)));
}

template <int... Is>
constexpr std::array<vm::bltin, sizeof...(Is)>
make_getter_wrappers(std::integer_sequence<int, Is...>) {
  return { &method_getter_wrapper<Is>... };
}

const auto g_getter_wrappers =
    make_getter_wrappers(std::make_integer_sequence<int, kMaxMethodGetters>{});

vm::bltin allocate_getter_bltin(vm::bltin caller) {
  int slot = g_nextGetter.fetch_add(1);
  if (slot >= kMaxMethodGetters) {
    em.compiler(nullPos);
    em << "asybind: exceeded method-getter pool (limit "
       << kMaxMethodGetters << ")";
    em.sync(true);
    return nullptr;
  }
  g_getter_slots[slot].caller = caller;
  return g_getter_wrappers[slot];
}

// =====================================================================
//  pluginRecord: a record subclass that exposes its venv via the
//  virtualField path. The asy compiler treats virtual fields by pushing
//  the receiver onto the stack and then encoding the access, which is
//  exactly what our bltins expect. Without this override, asy would
//  treat field accesses as record-frame field loads and consume the
//  receiver via inst::pop, leaving nothing for the bltin.
// =====================================================================

class pluginRecord : public record {
public:
  explicit pluginRecord(symbol name)
    : record(name, new frame(name, 0, 0))
  {
    // Trivial init lambda so `Type t;` (which would emit
    // makefunc(init)+popcall) doesn't crash. The recommended
    // construction syntax is `T t = T();` (the explicit constructor
    // registered separately by the plugin).
    vm::lambda* initLambda = getInit();
    initLambda->code = new vm::program();
    vm::inst i;
    i.op = vm::inst::ret;
    i.pos = nullPos;
    initLambda->code->encode(i);

    trans::coder c(nullPos, this, 0);
    c.closeRecord();
  }

  trans::varEntry* virtualField(symbol id, signature* sig) override {
    // Prefer a type-based lookup so that field access (sig == null) and
    // unambiguous method names both resolve cleanly. Fall back to a
    // signature-based lookup for overloaded methods.
    ty* t = e.varGetType(id);
    if (t && t->kind != ty_overloaded)
      return e.ve.lookByType(id, t);
    if (sig)
      return e.ve.lookBySignature(id, sig);
    return nullptr;
  }

  ty* virtualFieldGetType(symbol id) override {
    return e.varGetType(id);
  }
};

// =====================================================================
//  Type-spec translation.
// =====================================================================

ty* spec_to_type(const asybind_type_spec& spec) {
  switch (spec.tag) {
    case ASYBIND_VOID:     return primVoid();
    case ASYBIND_INT:      return primInt();
    case ASYBIND_REAL:     return primReal();
    case ASYBIND_BOOL:     return primBoolean();
    case ASYBIND_STRING:   return primString();
    case ASYBIND_USERPTR:  return reinterpret_cast<pluginRecord*>(spec.cls);
    case ASYBIND_FUNCTION: return reinterpret_cast<function*>(spec.fnty);
    default:               return nullptr;
  }
}

// =====================================================================
//  Host-API implementation.
// =====================================================================

vm::stack* as_stack(asybind_stack_ptr p) {
  return reinterpret_cast<vm::stack*>(p);
}

void host_push_int(asybind_stack_ptr s, long long v) {
  as_stack(s)->push<Int>(static_cast<Int>(v));
}
void host_push_real(asybind_stack_ptr s, double v) {
  as_stack(s)->push<double>(v);
}
void host_push_bool(asybind_stack_ptr s, int v) {
  as_stack(s)->push<bool>(v != 0);
}
void host_push_string(asybind_stack_ptr s, const char* data, size_t len) {
  mem::string gcstr(data, len);
  as_stack(s)->push<mem::string>(gcstr);
}

long long host_pop_int(asybind_stack_ptr s) {
  return static_cast<long long>(vm::pop<Int>(as_stack(s)));
}
double host_pop_real(asybind_stack_ptr s) {
  return vm::pop<double>(as_stack(s));
}
int host_pop_bool(asybind_stack_ptr s) {
  return vm::pop<bool>(as_stack(s)) ? 1 : 0;
}
void host_pop_string(asybind_stack_ptr s, const char** out, size_t* outlen) {
  mem::string* str = vm::pop<mem::string*>(as_stack(s));
  *out    = str->data();
  *outlen = str->size();
}

void host_raise(const char* msg) {
  em.runtime(vm::getPos());
  em << msg;
  em.sync(true);
  throw handled_error();
}

void host_add_func(asybind_module_ptr module, const char* name,
                   asybind_thunk_t thunk,
                   asybind_type_spec restype,
                   int nargs, const asybind_type_spec* argtypes) {
  auto* rec = reinterpret_cast<record*>(module);

  ty* result = spec_to_type(restype);
  if (!result) {
    em.compiler(nullPos);
    em << "asybind: invalid result type tag " << restype.tag
       << " for function '" << name << "'";
    em.sync(true);
    return;
  }

  vm::bltin wrapper = allocate_bltin_for(thunk);
  if (!wrapper) return;

  if (nargs > 8) {
    em.compiler(nullPos);
    em << "asybind: function '" << name << "' has " << nargs
       << " arguments; Phase 1 supports at most 8";
    em.sync(true);
    return;
  }

  formal f[8] = {
    trans::noformal, trans::noformal, trans::noformal, trans::noformal,
    trans::noformal, trans::noformal, trans::noformal, trans::noformal,
  };
  for (int i = 0; i < nargs; ++i) {
    ty* t = spec_to_type(argtypes[i]);
    if (!t || argtypes[i].tag == ASYBIND_VOID) {
      em.compiler(nullPos);
      em << "asybind: invalid argument type tag for '" << name << "'";
      em.sync(true);
      return;
    }
    f[i] = formal(t, symbol::nullsym, /*defval=*/false, /*Explicit=*/false);
  }

  trans::addFunc(rec->e.ve, wrapper, result, symbol::trans(name),
                 f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}

asybind_class_ptr host_create_class(asybind_module_ptr module,
                                    const char* name) {
  auto* rec = reinterpret_cast<record*>(module);
  sym::symbol id = symbol::trans(name);
  auto* cls = new pluginRecord(id);

  // Register the class as a type in the module's type env so asy code
  // can name it.
  rec->e.addType(id,
                 new trans::tyEntry(cls, /*v=*/nullptr,
                                    /*where=*/rec, nullPos));
  return reinterpret_cast<asybind_class_ptr>(cls);
}

// Build an asy types::function from spec arrays (used by add_method to
// build the method's UNBOUND function type).
function* build_function_type(const asybind_type_spec& restype,
                              int nargs,
                              const asybind_type_spec* argtypes,
                              const char* what) {
  ty* result = spec_to_type(restype);
  if (!result) {
    em.compiler(nullPos);
    em << "asybind: invalid result type tag " << restype.tag
       << " for " << what;
    em.sync(true);
    return nullptr;
  }
  function* fn = new function(result);
  for (int i = 0; i < nargs; ++i) {
    ty* t = spec_to_type(argtypes[i]);
    if (!t || argtypes[i].tag == ASYBIND_VOID) {
      em.compiler(nullPos);
      em << "asybind: invalid argument type tag for " << what;
      em.sync(true);
      return nullptr;
    }
    fn->add(formal(t, symbol::nullsym, /*defval=*/false, /*Explicit=*/false));
  }
  return fn;
}

void host_add_method(asybind_class_ptr cls, const char* name,
                     asybind_thunk_t thunk,
                     asybind_type_spec restype,
                     int nargs, const asybind_type_spec* argtypes) {
  auto* rec = reinterpret_cast<pluginRecord*>(cls);

  // The caller bltin pops the receiver and then the args (last-first).
  vm::bltin caller = allocate_bltin_for(thunk);
  if (!caller) return;

  // The getter bltin pops the receiver and pushes a callable bound to
  // it; this is what gives us `f = obj.method;`.
  vm::bltin getter = allocate_getter_bltin(caller);
  if (!getter) return;

  function* fnTy = build_function_type(restype, nargs, argtypes, name);
  if (!fnTy) return;

  auto* access = new trans::virtualFieldAccess(getter, /*setter=*/0, caller);
  rec->e.addVar(symbol::trans(name),
                new trans::varEntry(fnTy, access, trans::PUBLIC,
                                    rec, rec, nullPos));
}

void host_add_readonly_field(asybind_class_ptr cls, const char* name,
                             asybind_thunk_t getter_thunk,
                             asybind_type_spec type) {
  auto* rec = reinterpret_cast<pluginRecord*>(cls);

  vm::bltin getter = allocate_bltin_for(getter_thunk);
  if (!getter) return;

  ty* fieldTy = spec_to_type(type);
  if (!fieldTy) {
    em.compiler(nullPos);
    em << "asybind: invalid type for readonly field '" << name << "'";
    em.sync(true);
    return;
  }

  auto* access = new trans::virtualFieldAccess(getter);
  rec->e.addVar(symbol::trans(name),
                new trans::varEntry(fieldTy, access, trans::PUBLIC,
                                    rec, rec, nullPos));
}

void* host_alloc_obj(size_t size) {
  // GC-allocated, conservatively scanned (objects may hold pointers to
  // other GC values).
  return GC_MALLOC(size);
}

void host_push_obj(asybind_stack_ptr s, void* obj) {
  as_stack(s)->push<PluginInstance*>(static_cast<PluginInstance*>(obj));
}

void* host_pop_obj(asybind_stack_ptr s) {
  return static_cast<void*>(vm::pop<PluginInstance*>(as_stack(s)));
}

// =====================================================================
//  Phase 2: function-type cache.
//
//  We dedupe function types by structural equivalence so that the asy
//  type-comparison machinery (e.g. checking that a callable argument
//  matches a registered formal) sees the same `function*` instance for
//  repeated registrations of the same signature.
// =====================================================================

struct FuncTyCacheEntry {
  function* fn;
};
mem::list<FuncTyCacheEntry> g_funcTyCache;

asybind_funty_ptr host_make_function_type(asybind_type_spec restype,
                                          int nargs,
                                          const asybind_type_spec* argtypes) {
  function* fn = build_function_type(restype, nargs, argtypes,
                                     "callable type");
  if (!fn) return nullptr;
  // Dedupe via structural equivalence.
  for (auto& e : g_funcTyCache) {
    if (equivalent(e.fn, fn)) return reinterpret_cast<asybind_funty_ptr>(e.fn);
  }
  g_funcTyCache.push_back({fn});
  return reinterpret_cast<asybind_funty_ptr>(fn);
}

// =====================================================================
//  Phase 2: asy callable marshalling and dispatch.
// =====================================================================

asybind_callable_ptr host_pop_callable(asybind_stack_ptr s) {
  vm::callable* c = vm::pop<vm::callable*>(as_stack(s));
  return reinterpret_cast<asybind_callable_ptr>(c);
}

void host_push_callable(asybind_stack_ptr s, asybind_callable_ptr c) {
  as_stack(s)->push<vm::callable*>(reinterpret_cast<vm::callable*>(c));
}

void host_invoke_callable(asybind_stack_ptr s, asybind_callable_ptr c) {
  reinterpret_cast<vm::callable*>(c)->call(as_stack(s));
}

// =====================================================================
//  Phase 2: result<T> synthesis.
//
//  A result_class is a pluginRecord with two readonly virtual fields:
//    bool found;
//    T    value;
//  Each instance is a small GC-allocated `ResultInstance` carrying the
//  found flag and a vm::item holding the value (preserving its dynamic
//  type tag, so that re-pushing it during a field read yields a stack
//  item that the asy compiler will accept at the field's declared type).
// =====================================================================

struct ResultInstance {
  bool       found;
  vm::item   value;
};

// Each result class gets two dedicated getter bltins that close over the
// known field offset. To avoid per-class static thunk slots we just use
// `g_slots` (the regular plugin-thunk pool) for getters that don't need
// any per-instance state beyond the receiver: we generate one C++ stub
// per (class, field) and store it in the pool.

void result_getter_found(asybind_stack_ptr s,
                         const asybind_host_api_v1* api) {
  ResultInstance* r = static_cast<ResultInstance*>(api->pop_obj(s));
  api->push_bool(s, r->found ? 1 : 0);
}

void result_getter_value(asybind_stack_ptr s,
                         const asybind_host_api_v1* api) {
  ResultInstance* r = static_cast<ResultInstance*>(api->pop_obj(s));
  // Re-push the stored vm::item with its original type tag intact. If
  // !found the stored item is default-constructed; the asy side must
  // check `found` before reading `value`.
  as_stack(s)->push(r->value);
}

// Cache key for result classes: composite of element spec fields.
struct ResultCacheKey {
  int tag;
  void* cls_or_fnty;  // class_ptr for USERPTR, fnty for FUNCTION, else null
  bool operator==(const ResultCacheKey& o) const {
    return tag == o.tag && cls_or_fnty == o.cls_or_fnty;
  }
};

struct ResultCacheEntry {
  ResultCacheKey key;
  asybind_class_ptr cls;
};
mem::list<ResultCacheEntry> g_resultCache;

asybind_class_ptr host_result_class(asybind_type_spec elem) {
  void* extra = nullptr;
  if (elem.tag == ASYBIND_USERPTR)       extra = elem.cls;
  else if (elem.tag == ASYBIND_FUNCTION) extra = elem.fnty;
  ResultCacheKey key{elem.tag, extra};
  for (auto& e : g_resultCache) {
    if (e.key == key) return e.cls;
  }

  ty* elemTy = spec_to_type(elem);
  if (!elemTy || elem.tag == ASYBIND_VOID) {
    em.compiler(nullPos);
    em << "asybind: invalid element type for result_class";
    em.sync(true);
    return nullptr;
  }

  // Use a synthetic, stable name. The name is not really observable to
  // asy code (result types are typically used via `var r = f(...)` and
  // accessed by field), but it shows up in error messages.
  static int seq = 0;
  std::string nameStd = "result_" + std::to_string(++seq);
  sym::symbol id = symbol::trans(nameStd.c_str());
  auto* cls = new pluginRecord(id);

  // Register `bool found` and `T value` as readonly virtual fields.
  {
    vm::bltin getter = allocate_bltin_for(&result_getter_found);
    auto* access = new trans::virtualFieldAccess(getter);
    cls->e.addVar(symbol::trans("found"),
                  new trans::varEntry(primBoolean(), access, trans::PUBLIC,
                                      cls, cls, nullPos));
  }
  {
    vm::bltin getter = allocate_bltin_for(&result_getter_value);
    auto* access = new trans::virtualFieldAccess(getter);
    cls->e.addVar(symbol::trans("value"),
                  new trans::varEntry(elemTy, access, trans::PUBLIC,
                                      cls, cls, nullPos));
  }

  auto handle = reinterpret_cast<asybind_class_ptr>(cls);
  g_resultCache.push_back({key, handle});
  return handle;
}

void host_push_result(asybind_stack_ptr s, asybind_class_ptr /*result_cls*/,
                      int found) {
  ResultInstance* r =
      static_cast<ResultInstance*>(GC_MALLOC(sizeof(ResultInstance)));
  new (r) ResultInstance{};
  r->found = (found != 0);
  if (found) {
    // Pop the value off the top, preserving its dynamic type tag.
    r->value = as_stack(s)->pop();
  }
  // Reuse the same opaque-pointer tagging as class instances so the asy
  // compiler's USERPTR path works uniformly.
  as_stack(s)->push<PluginInstance*>(reinterpret_cast<PluginInstance*>(r));
}

const asybind_host_api_v1 g_host_api = {
  &host_push_int,
  &host_push_real,
  &host_push_bool,
  &host_push_string,
  &host_pop_int,
  &host_pop_real,
  &host_pop_bool,
  &host_pop_string,
  &host_raise,
  &host_add_func,
  &host_create_class,
  &host_add_method,
  &host_add_readonly_field,
  &host_alloc_obj,
  &host_push_obj,
  &host_pop_obj,
  &host_make_function_type,
  &host_pop_callable,
  &host_push_callable,
  &host_invoke_callable,
  &host_result_class,
  &host_push_result,
};

// =====================================================================
//  Filename resolution and dl wrapper.
// =====================================================================

string platform_lib_name(const string& base) {
#if defined(_WIN32)
  return base + ".dll";
#elif defined(__APPLE__)
  return "lib" + base + ".dylib";
#else
  return "lib" + base + ".so";
#endif
}

string locate_plugin(const string& filename) {
  string libname = platform_lib_name(filename);
  return settings::locateFile(libname, /*full=*/true, /*suffix=*/"");
}

struct DlHandle {
#if defined(_WIN32)
  HMODULE h = nullptr;
  ~DlHandle() = default;
  void* sym(const char* name) {
    return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
  }
#else
  void* h = nullptr;
  ~DlHandle() = default;
  void* sym(const char* name) {
    return h ? dlsym(h, name) : nullptr;
  }
#endif
};

bool dl_open(DlHandle& out, const string& path, string& err) {
#if defined(_WIN32)
  out.h = LoadLibraryA(path.c_str());
  if (!out.h) { err = "LoadLibrary failed"; return false; }
#else
  out.h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!out.h) {
    const char* e = dlerror();
    err = e ? e : "dlopen failed";
    return false;
  }
#endif
  return true;
}

}  // anonymous namespace

// =====================================================================
//  Public entry point.
// =====================================================================

types::record* tryLoadPlugin(sym::symbol id, mem::string filename) {
  string path = locate_plugin(filename);
  if (path.empty()) return nullptr;

  DlHandle dl;
  string err;
  if (!dl_open(dl, path, err)) {
    em.sync();
    em << "error: failed to load plugin '" << path << "': " << err << "\n";
    em.sync(true);
    return nullptr;
  }

  using init_fn_t = const asybind_module_descriptor* (*)(void);
  init_fn_t initFn = reinterpret_cast<init_fn_t>(dl.sym("asybind_init_v1"));
  if (!initFn) {
    em.sync();
    em << "error: plugin '" << path
       << "' does not export 'asybind_init_v1'\n";
    em.sync(true);
    return nullptr;
  }

  const asybind_module_descriptor* desc = initFn();
  if (!desc) {
    em.sync();
    em << "error: plugin '" << path
       << "' returned a null module descriptor\n";
    em.sync(true);
    return nullptr;
  }
  if (desc->abi_version != ASYBIND_ABI_VERSION) {
    em.sync();
    em << "error: plugin '" << path << "' has ABI version "
       << desc->abi_version << " but host expects "
       << ASYBIND_ABI_VERSION
       << "; recompile the plugin against the current asy headers\n";
    em.sync(true);
    return nullptr;
  }

  record* rec = new record(id, new frame(id, 0, 0));

  // Trivial init lambda (see Phase 0 notes): without this the VM
  // dereferences a null `program*` and crashes.
  vm::lambda* initLambda = rec->getInit();
  initLambda->code = new vm::program();
  {
    vm::inst i;
    i.op = vm::inst::ret;
    i.pos = nullPos;
    initLambda->code->encode(i);
  }

  desc->populate(reinterpret_cast<asybind_module_ptr>(rec),
                 &g_host_api);

  if (em.errors()) return nullptr;
  return rec;
}

}  // namespace asybind
