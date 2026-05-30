/*****
 * pluginloader.cc
 * Loader for C++ plugin modules (asybind) — Phase 0.
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

// =====================================================================
//  Slot registry: each registered plugin function is assigned a unique
//  static wrapper bltin (a void(*)(vm::stack*) function pointer) drawn
//  from a fixed-size pool. The wrapper looks up the plugin's thunk in
//  `g_slots` and forwards the call along with the global host-API table.
// =====================================================================

namespace {

constexpr int kMaxPluginFns = 1024;

struct PluginSlot {
  asybind_thunk_t thunk = nullptr;
};

std::array<PluginSlot, kMaxPluginFns> g_slots{};
std::atomic<int> g_nextSlot{0};

// Forward decl for the API table; defined below.
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
//  Tag-to-type translation.
// =====================================================================

ty* tag_to_type(int tag) {
  switch (tag) {
    case ASYBIND_VOID:   return primVoid();
    case ASYBIND_INT:    return primInt();
    case ASYBIND_REAL:   return primReal();
    case ASYBIND_BOOL:   return primBoolean();
    case ASYBIND_STRING: return primString();
    default:             return nullptr;
  }
}

// =====================================================================
//  Host-API implementation (called from inside plugin thunks).
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
  // mem::string is GC-allocated; push by value to trigger item's
  // copy-into-GC constructor.
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
  // Strings travel on the stack as mem::string* (per runstring.cc).
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
                   int restype, int nargs, const int* argtypes) {
  auto* rec = reinterpret_cast<record*>(module);
  ty* result = tag_to_type(restype);
  if (!result) {
    em.compiler(nullPos);
    em << "asybind: invalid result type tag " << restype
       << " for function '" << name << "'";
    em.sync(true);
    return;
  }

  vm::bltin wrapper = allocate_bltin_for(thunk);
  if (!wrapper) return;

  // Build formals (up to 8 supported in Phase 0; matches plugin SDK limits).
  if (nargs > 8) {
    em.compiler(nullPos);
    em << "asybind: function '" << name << "' has " << nargs
       << " arguments; Phase 0 supports at most 8";
    em.sync(true);
    return;
  }

  formal f[8] = {
    trans::noformal, trans::noformal, trans::noformal, trans::noformal,
    trans::noformal, trans::noformal, trans::noformal, trans::noformal,
  };
  for (int i = 0; i < nargs; ++i) {
    ty* t = tag_to_type(argtypes[i]);
    if (!t || argtypes[i] == ASYBIND_VOID) {
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
};

// =====================================================================
//  Filename resolution.
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
  // mungeFileName in locate.cc already strips/adds a suffix; pass the
  // platform-specific filename and the empty suffix to avoid double-
  // appending ".asy".
  string libname = platform_lib_name(filename);
  string found = settings::locateFile(libname, /*full=*/true, /*suffix=*/"");
  return found;
}

// =====================================================================
//  dlopen / LoadLibrary wrapper.
// =====================================================================

struct DlHandle {
#if defined(_WIN32)
  HMODULE h = nullptr;
  ~DlHandle() = default;  // intentionally leak: see header
  void* sym(const char* name) {
    return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
  }
#else
  void* h = nullptr;
  ~DlHandle() = default;  // intentionally leak
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

  // Construct a fresh record to receive the plugin's bindings. Use the
  // same shape as a parsed .asy module (record + frame).
  record* rec = new record(id, new frame(id, 0, 0));

  // The record constructor allocates an empty vm::lambda but leaves its
  // `code` pointer null. The runtime executes the module initializer
  // when the module is first accessed, so we give it a trivial program
  // that simply returns. Without this the VM dereferences a null
  // `program*` and crashes.
  vm::lambda* initLambda = rec->getInit();
  initLambda->code = new vm::program();
  {
    vm::inst i;
    i.op = vm::inst::ret;
    i.pos = nullPos;
    initLambda->code->encode(i);
  }

  // Run populate. Cast our record* into the opaque module handle.
  desc->populate(reinterpret_cast<asybind_module_ptr>(rec),
                 &g_host_api);

  if (em.errors()) return nullptr;
  return rec;
}

}  // namespace asybind
