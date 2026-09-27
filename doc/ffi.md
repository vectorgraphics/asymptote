
# Asymptote FFI -- Getting Started

The Asymptote Foreign Function Interface (FFI) lets you write **plugins as
native shared libraries** (`.so` / `.dll`) that register functions and
manipulate Asymptote objects (items, paths, pictures, records, and more)
directly from C/C++.

The entire public API lives in the single header **`asyffi.h`**
(Apache License 2.0, see `LICENSE-APACHE.TXT`). If you only need to *write*
a plugin, that is the only file you need from the Asymptote source tree.

---

## 1. Supported platforms

| Platform | ABI notes |
|----------|-----------|
| Windows (x64 / 32-bit x86) | Exports use `__declspec(dllexport)`; 32-bit builds use `__cdecl` (the `LNK_CALL` macro handles this) |
| Linux (LP64, or 32-bit x86) | Exports use `[[gnu::visibility("default")]]`; 32-bit x86 uses `__cdecl` (the `LNK_CALL` macro handles this) |
| macOS (arm64 / x86_64) | Same as Linux |

Any other platform will make `asyffi.h` emit a compile error.

Because the API is based on C++ abstract interfaces with virtual dispatch,
your plugin must be compiled with a compatible C++ ABI (same compiler
family / library as the Asymptote host, e.g. MSVC for MSVC builds,
GCC/Clang libstdc++ or libc++ accordingly).

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

The search uses the standard Asymptote path (`asy path`). The **import name
is used as the library key**; you can later unload it with the built-in

```asy
unloadLib("mymodule");
```

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

```c
// myplugin.cc
#include "asyffi.h"

extern "C" ASY_FFI_EXPORT
void registerAsymptotePlugin(IAsyContext* ctx, IAsyFfiRegisterer* reg)
{
  // Register: real doubleIt(real x)
  Asy::TypeInfo realType;
  realType.baseType = Asy::BaseTypes::Real;

  Asy::FnArgMetadata const arg{ realType, "x", /*optional*/ false,
                                /*explicitArgs*/ false };

  Asy::FunctionTypeMetadata const meta{ realType, /*numArgs*/ 1, &arg };

  reg->registerFunction("doubleIt",
      +[](IAsyContext* ctx, IAsyStackContext* stack,
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
| Memory | `malloc`, `mallocAtomic` -- **use these, not your own allocator**, for anything the Asy GC must track |
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

### `IAsyStackContext` -- talking back to Asymptote

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

---

## 7. C++ helper library (recommended for real plugins)

For anything beyond a toy example, look at the companion project
**[asymptote-ffi-helper-lib](https://github.com/vectorgraphics/asymptote-ffi-helper-lib)**
(Apache-2.0). It is a thin C++20 layer on top of `asyffi.h` that removes
most of the boilerplate:

- **Type builders** (`AsyFfiHelpers::TypeObjects`) -- build the
  `Asy::FunctionTypeMetadata` / `Asy::TypeInfo` trees for
  `registerFunction` declaratively instead of hand-filling C structs.
  Types are built as `TypeObject`s (`Primitive`, `Array`, `Function`,
  `Record`) and a function's signature is produced with
  `TO::Function::builder(returnTypeObj).build(arg1, arg2, ...)`, where
  each `argN` is a `TO::Function::Argument`.
- **Template-based item access** (`AsyFfiHelpers::Item`) -- typed
  get/set helpers for `IAsyItem`, so you can `AI::setItem(ptr, 3.14)`
  without thinking about which `set*Value` overload applies.
- **RAII GC thread registration** (`asyffihelpers/threads.h`) -- wrappers
  that call `IAsyContext::registerThreadWithGc` / `unregisterThreadWithGc`
  automatically on thread start/end.
- **Managed records** (`AsyFfiHelpers::Structs::ManagedRecord`) --
  RAII wrapper over `IAsyRecord` for working with Asymptote structs.
- Pen construction and other context helpers
  (`pen.h`, `contextFuncs.h`, `array.h`, `args.h`).

The repo's `examples/` directory has complete, buildable plugins:

| Example | Demonstrates |
|---------|--------------|
| `structSample.cc` (+ `structSample_recfile.asy`) | Creating instances of an Asymptote `struct`, setting fields, returning the var frame; loading the `.asy` module via `IAsyGlobalEnvironment::loadExistingModule` |
| `pictureSample.cc` | Creating pictures / draw elements |
| `clipSample.cc` | Clip begin/end draw elements |
| `threadSample.cc` | Multi-threaded plugins with GC registration |

### Building the helper library

It requires a C++20 compiler (MSVC on Windows; gcc  16 on POSIX, e.g.
`CXX=g++-16`). It is *not* self-contained -- it needs `asyffi.h` from the
Asymptote source tree:

- **CMake (recommended):** set the cache variable
  `ASYFFI_HEADER_DOWNLOAD_URL` to either a local path to your
  `asyffi.h` (e.g. `ASYFFI_HEADER_DOWNLOAD_URL=/path/to/asymptote/asyffi.h`)
  or a URL. Without it, CMake downloads the header from a default URL
  that may be outdated.
- **Autotools:** `autoconf && ./configure` with either
  `--with-asyffi-header-location=/path/to/asyffi.h` or
  `--with-asyffi-header-url=https://...` (requires `wget`);
  `CXX=g++-16` recommended on Linux/macOS.

If you are building Asymptote from source (as in this repo), point it at
`./asyffi.h` here -- that guarantees your plugin matches the host's ABI.

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

## 9. Where to look in the source

| File | Contents |
|------|----------|
| `asyffi.h` | **The public API** (interfaces, macros, metadata structs) |
| `asyffiimpl.h` / `asyffiimpl.cc` | Reference implementation of every interface (great example code) |
| `dlmanager.h` / `dlmanager.cc` | Shared-library loading (RAII, refcounted) |
| `dynlib.h` / `dynlib.cc` | Bridge: locating the file, calling the entry point, dispatching calls |
| `rundynlib.in` | Asymptote-level `unloadLib` |
| `genv.cc` -> `genv::loadModule` | Where `import` falls through to dynamic-library loading |

External (companion repo, Apache-2.0):

- [asymptote-ffi-helper-lib](https://github.com/vectorgraphics/asymptote-ffi-helper-lib)
  -- C++20 convenience layer over the FFI plus complete example plugins
  (structs, pictures, clipping, threads).

