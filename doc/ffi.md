
# Asymptote FFI -- Getting Started

The Asymptote Foreign Function Interface (FFI) lets you write plugins as
native shared libraries (`.so` / `.dll`) that register functions and
manipulate Asymptote objects (items, paths, pictures, records, and more)
directly from C/C++.

The entire public API lives in the single header `asyffi.h`
(Apache License 2.0, see `LICENSE-APACHE.TXT`). If you only need to *write*
a plugin, that is the only file you need from the Asymptote source tree.

---

## 1. Supported platforms

| Platform | ABI notes |
|----------|-----------|
| Windows (x86-64 / x86) | Exports use `__declspec(dllexport)`; 32-bit builds use `__cdecl` (the `LNK_CALL` macro handles this) |
| Linux (x86-64, or x86) | Exports use `[[gnu::visibility("default")]]`; 32-bit x86 uses `__cdecl` (the `LNK_CALL` macro handles this) |
| macOS (arm64 / x86-64) | Same as Linux |

Any other platform will make `asyffi.h` emit a compile error.

Because the API is based on C++ abstract interfaces with virtual dispatch,
your plugin must be compiled with a compatible C++ ABI (same compiler
family / library as the Asymptote host, e.g. MSVC for MSVC builds,
GCC/Clang libstdc++ or libc++ accordingly).

**C++ standard:** `asyffi.h` itself only requires C++17. (The separate
`asymptote-ffi-helper-lib` convenience layer needs C++20 -- see section 7.)

---

## 2. How plugins are loaded

From Asymptote code, a plugin is loaded with the ordinary `import`
statement:

```asy
import myplugin;      // or: import "mymodule";
myplugin.hello();
```

When Asymptote cannot find `mymodule.asy`, it looks for a dynamic library:

1. `mymodule.so` (or `mymodule.dll` on Windows), then
2. `libmymodule.so` on non-Windows.

The search uses the standard Asymptote path (`asy path`). The import name
is used as the library key

When the library is found, Asymptote looks up the exported symbol

```c
extern "C" void registerAsymptotePlugin(IAsyContext* context,
                                        IAsyFfiRegisterer* registerer);
```

and calls it. Your plugin registers its functions through `registerer`;
the imported module then exposes them as ordinary Asymptote functions.

The header provides convenience macros:

```c
#define DECLARE_REGISTER_FN            // declares the entry point (exported)
#define ASY_FOREIGN_FUNC_SIG(name)     // declares a foreign function signature
#define ASYFFI_FN_NAME_AND_ADDR(name)  // "name", &name helper for registration
```

---

## 3. Minimal example plugin

### Recommended: use the helper library (type-inferred registration)

For most plugins, the [asymptote-ffi-helper-lib](https://github.com/vectorgraphics/asymptote-ffi-helper-lib)
provides a type-inferred `registerFunction` that eliminates all hand-written
metadata. The simplest possible plugin is:

```cpp
// myplugin.cc
#include <asyffihelper.h>
using namespace AsyFfi;
DECLARE_REGISTER_FN;

REGISTER_FN_SIG
{
    // int(int) -> int.  Types are inferred from the lambda signature.
    registerFunction(registerer, "doubleIt", [](int64_t x) -> int64_t { return 2 * x; });

    // pair(pair, pair) -> pair
    registerFunction(
        registerer, "midpoint", [](AsyFfi::pair a, AsyFfi::pair b) -> AsyFfi::pair {
            return AsyFfi::pair{(a.x + b.x) / 2.0, (a.y + b.y) / 2.0};
        }
    );

    // real(real[]) -> real
    registerFunction(
        registerer, "addAll", [](std::vector<double> xs) -> double {
            double s = 0; for (double v : xs) s += v; return s;
        }
    );

    // void(int)  -- a leading AsyFfi::Context param gives host access; it is
    // excluded from the inferred Asy signature (still void(int)).
    registerFunction(
        registerer, "logIt", [](Context ctx, int64_t x) -> void {
            ctx->reportWarning("logIt called");
        }
    );
}
```

Build it with the CMake helper (see section 7):

```cmake
asyffi_add_plugin(mymodule myplugin.cc)
```

Now in Asymptote:

```asy
import mymodule;
label(string(mymodule.doubleIt(21)));              // 42
label(string(mymodule.midpoint((0,0),(4,6)).x));   // 2
label(string(mymodule.addAll(new real[]{1,2,3})));  // 6
```

The return and argument types are inferred from the C++ lambda's signature.
See section 7 for the full helper-library reference, and the helper repo's
`tests/fnsSample/fnsSample.cc` for a buildable example that also demonstrates
calling back into Asymptote from C++ via `AsyFfi::call`.

### If you do not want the helper library

You can write a plugin using only `asyffi.h` (no C++20 required, C++17
suffices). Every function requires hand-built type metadata:

```cpp
// myplugin.cc  (raw asyffi.h, C++17)
#include "asyffi.h"

DECLARE_REGISTER_FN;

REGISTER_FN_SIG
{
  // Register: real doubleIt(real x)
  Asy::TypeInfo realType{ Asy::BaseTypes::Real, {} };

  Asy::FnArgMetadata const arg{ realType, "x", /*optional*/ false,
                                /*explicitArgs*/ false };

  Asy::FunctionTypeMetadata const meta{ realType, /*numArgs*/ 1, &arg };

  registerer->registerFunction("doubleIt",
      [](IAsyContext* ctx, IAsyStackContext* stack,
          IAsyArgs* args, IAsyItem* ret) {
        double const x = args->getNumberedArg(0)->asDouble();
        ret->setDoubleValue(2.0 * x);
      },
      meta);
}
```

Compile it as a shared library (the function pointer must be a plain
function or a no-capture lambda cast to `TAsyForeignFunction`):

```bash
# Linux / macOS (clang works on macOS; same -shared -fPIC flags)
g++ -std=c++17 -shared -fPIC -I/path/to/asymptote \
    -o mymodule.so myplugin.cc

# Windows (MSVC)
cl /std:c++17 /LD /I path\to\asymptote myplugin.cc /Fe:mymodule.dll
```

Now in Asymptote:

```asy
import mymodule;
real y = mymodule.doubleIt(21);   // 42
```

### Foreign function signature

Every registered function has this shape:

```c
typedef void (*TAsyForeignFunction)(
    IAsyContext*    ctx,       // global API context (see below)
    IAsyStackContext* stack,   // call Asy functions / run embedded code
    IAsyArgs*       args,      // the arguments passed by the caller
    IAsyItem*       ret        // set your return value here; nullptr for void
);
```

### Type metadata

Registration takes an `Asy::FunctionTypeMetadata`:

- `returnType` -- an `Asy::TypeInfo`; its `baseType` is one of
  `Asy::BaseTypes` (`Void`, `Real`, `Integer`, `Pair`, `Triple`, `Boolean`,
  `Str`, `Transform`, `Guide`, `Path`, `Path3`, `ArrayType`, `FunctionType`,
  `Record`, ...). For `ArrayType`, `extraData.arrayTypeInfo` must be filled in
  (element type + dimension); for `Record`, `extraData.recordPtr`.
- `numArgs` / `argInfoPtr` -- a C array of `Asy::FnArgMetadata`
  (type, name, `optional`, `explicitArgs`).
- If the return type is `Void`, `ret` will be `nullptr` when your function
  runs -- do not write to it.

> **Tip:** the metadata arrays (`argInfoPtr`, `returnType`) are read at
> registration time, so file-scope or static locals are fine.

### Type mapping (C++ type -> Asymptote type)

When using the helper library's `TypeInfoOf<T>` trait (or the type-inferred
`registerFunction`), the following C++ types map to Asymptote types:

| C++ type | Asy type | `BaseTypes` |
|----------|----------|-------------|
| `void` | `void` | `Void` |
| `double` | `real` | `Real` |
| `int64_t` | `int` | `Integer` |
| `int` | `int` (convenience) | `Integer` |
| `bool` | `bool` | `Boolean` |
| `std::string` | `string` | `Str` |
| `std::vector<T>` | `T[]` (1-D array) | `ArrayType` |
| `AsyFfi::pair` | `pair` | `Pair` |
| `AsyFfi::triple` | `triple` | `Triple` |
| `AsyFfi::transform` | `transform` | `Transform` |
| `AsyFfi::path` | `path` | `Path` |
| `AsyFfi::path3` | `path3` | `Path3` |
| `AsyFfi::picture` | `picture` | `Picture` |
| `AsyFfi::pen` | `pen` | `Pen` |

> **Note on integers:** Asymptote's `Int` is a 64-bit type. In C++, use
> `int64_t` (not `int`) to match it exactly. The `TypeInfoOf` trait accepts
> `int` as a convenience, but `int64_t` is the canonical mapping.

---

## 4. Core interfaces

### `IAsyItem` -- a value of any type

The basic box every argument and return value lives in:

```cpp
ret->setDoubleValue(3.14);   // real
ret->setInt64Value(42);      // int
ret->setBooleanValue(true);  // bool
ret->setRawPointer(th);      // any opaque Asy value (string, path, ...)
```

Reading: `asDouble()`, `asInt64()`, `asBoolean()`, `asRawPointer()`,
`isDefault()` (true if the caller left an optional argument at its default).

### `IAsyArgs` -- the function arguments

```cpp
size_t n   = args->getArgumentCount();
IAsyItem* a0 = args->getNumberedArg(0);   // out-of-range is a hard error
```

### `IAsyContext` -- the global workbench

Everything else hangs off this. Highlights:

| Area | Methods |
|------|---------|
| Memory | `malloc`, `mallocAtomic` -- use these for anything the Asy GC must track |
| GC (multithreaded plugins) | `isGcSupported`, `getGcStackBase`, `registerThreadWithGc`, `unregisterThreadWithGc` |
| Strings | `createNewAsyString`, `updateAsyString(Sized)`, `getStringLength`, `copyString` (opaque `THAsyString`, assign into an item with `setRawPointer`) |
| Arrays | `createNewArray`, then `IAsyArray` (`getItem`, `setItem`, `setSize`, `pushItem`, `popItem`, ...); out-of-range access is undefined behavior |
| Tuples | `createPair`, `createTriple`; access via `IAsyTuple::getIndexedValue` / `setIndexedValue` |
| Transforms | `createNewTransform(x, y, xx, xy, yx, yy)`, `createNewIdentityTransform`; `IAsyTransform::apply(in, out)` |
| Paths | `createAsyPath`, `createAsyPath3`, `createSolvedKnot2D/3D` |
| Pens | `createNewPen(...)` (colors, widths, caps/joins, transparency, ...) |
| Pictures & drawing | `createPicture(deconstruct)`, `createDrawElementFromPath/Path3`, `createDrawElementForPixel/Fill/Label/Verbatim/...` (all shade types), `createDrawElementForBeginClip/EndClip` |
| Error reporting | `reportError` (user-catchable), `reportWarning`, `reportFatal` |
| Misc | `getVersion`, `isCompactBuild`, `isSimpleFrameBuild`, `getSetting(name)`, `createAsyType` |

### `IAsyStackContext` -- For calling asymptote functions at the C++ level

Available from any foreign function:

```cpp
// call an existing Asymptote builtin, e.g. sqrt
Asy::TypeInfo realType{ Asy::BaseTypes::Real };
IAsyCallable* sqrtFn = stack->getBuiltin(nullptr, "sqrt", realType);
IAsyItem* const callArgs[] = { args->getNumberedArg(0) };
IAsyItem* result = stack->callReturning(sqrtFn, 1, callArgs);

// or run raw Asymptote code (interactive stacks only)
if (stack->isInteractive())
  stack->runStringEmbedded("settings.outformat:=\"svg\";");
```

There are three call variants: `callVoid`, `callReturning`, and
`callReturningToExistingItem` (stores into an item you own).

### Deeper access

For advanced plugins, `asyffi.h` also exposes `IAsyRecord`,
`IAsyProtoEnvironment`, `IAsyGlobalEnvironment` (module imports),
`IAsyAccess` / `IAsyLocalAccess` / `IAsyBuiltinAccess` (env manipulation),
`IAsyLambda`, `IAsyVarFrame`, `IAsyPen`, `IAsyBbox(3)`, `IAsyGuide`,
`IAsyPath(3)`, and `IAsyDrawElement` info structs. Browse the header -- every
method has doxygen comments.

---

## 5. Compact vs. full builds

Most Asymptote builds (including the official releases) are **compact
builds**: values are stored in a single 64-bit word, and a couple of the
highest `Int` values are reserved for *default*, *undefined*, and
boolean truth/falsity. The header defines these magic values:

```cpp
ASY_COMPACT_DEFAULT_VALUE
ASY_COMPACT_UNDEFINED_VALUE
ASY_COMPACT_BOOL_TRUTH_VALUE
ASY_COMPACT_BOOL_FALSE_VALUE
```

You rarely need to touch them directly: the `IAsyItem` accessors
(`setBooleanValue`, `isDefault`, ...) handle it. Check
`ctx->isCompactBuild()` if you need to branch. On non-compact builds,
typed values use `setValueWithTypeId` (a no-op difference from
`setRawPointer` on compact builds).

---

## 6. Error handling

Use the context instead of `abort`/exceptions:

```cpp
if (ptr == nullptr)
  ctx->reportError("myplugin: bad pointer");   // returns control to Asy
```

`reportError` throws a C++ exception back into the Asymptote VM, so code
after it in your function is not executed. Asymptote-tracked memory is
reclaimed by the GC; because this is an exception and not a non-local jump,
C++ destructors on your plugin's stack still run during unwinding.
`reportFatal` is for unrecoverable situations (the host terminates).

### Helper-library error messages

When using the helper library's `AsyFfi::call` (C++ calling back into
Asymptote), a failed function resolution produces a diagnostic that names
the module, the function, and the expected return type. For example, calling
a non-existent built-in:

```
asy::call: could not resolve built-in 'foo' in module '(top-level)' as a function returning real
```

Calling a built-in with the wrong expected return type:

```
asy::call: could not resolve built-in 'sqrt' in module '(top-level)' as a function returning pair
```

This makes it straightforward to identify typos, wrong module names, or
signature mismatches during development. The raw `asyffi.h` API does not
provide equivalent diagnostics (a `nullptr` from `getBuiltin` must be
checked manually).

---

## 7. C++ helper library (recommended for real plugins)

For anything beyond a toy example, use
[asymptote-ffi-helper-lib](https://github.com/vectorgraphics/asymptote-ffi-helper-lib),
an Apache-license C++20 layer on top of `asyffi.h` that removes most of the
boilerplate. It is the **recommended** way to write a plugin:

- **Type-inferred registration** (`AsyFfi::registerFunction`) -- the
  function's Asymptote signature is deduced from the C++ lambda's type.
  No hand-written `FunctionTypeMetadata`, no manual argument unpacking.
  See section 3 for the minimal example.
- **Injected host context** (`AsyFfi::Context`) -- a plugin lambda may take a
  leading `AsyFfi::Context` parameter to reach the Asy host (via `ctx->`) and
  to call back into Asy; it is excluded from the inferred Asy signature.
- **Typed value conversion** (`AsyFfi::toItem` / `AsyFfi::fromItem`) --
  write/read any supported C++ type to/from an `IAsyItem` without choosing
  the right `set*Value` / `as*` overload.
- **C++ -> Asymptote calls** (`AsyFfi::call`) -- call Asymptote built-ins
  or imported-module functions from C++ with typed arguments and a typed
  return value, in a one-liner.
- **Typed value proxies** (`AsyFfi::pair`, `AsyFfi::triple`,
  `AsyFfi::path`, `AsyFfi::path3`, `AsyFfi::picture`, `AsyFfi::pen`,
  `AsyFfi::transform`) -- lightweight C++ types for structural Asy values,
  each with `TypeInfoOf` and `ItemFunctions` specializations.
- **Type builders** (`AsyFfiHelpers::TypeObjects`) -- for the
  explicit-metadata (escape-hatch) path, build `Asy::TypeInfo` trees
  declaratively instead of hand-filling C structs.
- **RAII GC thread registration** (`asyffihelpers/threads.h`) --
  `registerThreadWithGc` / `unregisterThreadWithGc` automatically on
  thread start/end.
- **Managed records** (`AsyFfiHelpers::Structs::ManagedRecord`) --
  RAII wrapper over `IAsyRecord` for working with Asymptote structs.
- Pen construction and other context helpers
  (`pen.h`, `contextFuncs.h`, `array.h`, `args.h`).

### Building a plugin (`asyffi_add_plugin`)

The helper library ships a one-function CMake helper that builds a
loadable plugin with the correct flags, include dirs, and output name:

```cmake
# In a consumer project that has the asyffi_helper target available:
add_subdirectory(asyffi-helper-lib helper)   # provides target: asyffi_helper
include(asyffi-helper-lib/cmake/asyffi_plugin.cmake)

asyffi_add_plugin(mymodule myplugin.cc)      # -> libmymodule.so / mymodule.dll
```

Options:
- `LIBNAME <name>` -- set the output file name (leading `lib` is stripped).
- `ASY_PATH_DIR <dir>` -- copy the built library into an Asymptote
  search-path directory so you can `import <name>;` directly.

### Buildable examples

The helper repo ships complete, buildable plugins: self-contained example
projects under `examples/` (each its own CMake project with a `test.asy`), and
round-trip ctest targets under `tests/`:

| Example | Demonstrates |
|---------|--------------|
| `examples/minimal/minimal.cc` | The minimal type-inferred `registerFunction` plugin ("hello world") |
| `examples/picture/picture.cc` | Creating pictures / draw elements, with the injected `AsyFfi::Context` |
| `examples/clip/clip.cc` | Clip begin/end draw elements |
| `examples/thread/thread.cc` | Multi-threaded plugins with GC registration |
| `examples/struct/struct.cc` | Returning an Asy struct instance, loading a `.asy` module (escape-hatch overload) |
| `tests/fnsSample/fnsSample.cc` | Type-inferred `registerFunction` + `AsyFfi::call` (C++ -> Asy) in one plugin |

The `tests/` targets are verified by round-trip ctests that build the plugin,
load it in `asy`, and abort on a wrong return value. Each `examples/` project
builds its `.so` next to a `test.asy` that you run with `asy -V test`.

### Building the helper library

```sh
cd asyffi-helper-lib
cmake -S . -B build -DCMAKE_CXX_STANDARD=20 \
      -DASYFFI_HEADER_DOWNLOAD_URL=/path/to/asymptote/asyffi.h
cmake --build build -j
ctest --test-dir build
```

---

## 8. Checklist for a new plugin

1. Write a `.cc` that `#include`s `asyffi.h` (the only dependency).
2. Define your function(s) with `ASY_FOREIGN_FUNC_SIG` (or matching the
   `TAsyForeignFunction` signature directly).
3. Export the entry point with `DECLARE_REGISTER_FN` and call
   `registerer->registerFunction(...)` for each function, supplying correct
   `FunctionTypeMetadata` (wrong metadata -> confusing runtime errors).
4. Build a shared library: Linux/macOS -> `-shared -fPIC` (name it
   `mymodule.so` or `libmymodule.so`); Windows -> DLL named `mymodule.dll`.
5. Put the library on the Asymptote path (`asy path`), then `import mymodule;`
   in your `.asy` file.
6. Debug tips: `import "mymodule"` with a quoted string gives you the file
   used as the library key; `unloadLib("mymodule")` reload-safe re-import
   during interactive sessions; `ctx->reportWarning(...)` is a handy
   print-to-console.

External (companion repo, Apache-2.0):

- [asymptote-ffi-helper-lib](https://github.com/vectorgraphics/asymptote-ffi-helper-lib)
  -- C++20 convenience layer over the FFI plus complete example plugins
  (structs, pictures, clipping, threads).

