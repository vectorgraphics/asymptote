# Working with the garbage collector — a guide for C++ plugin authors

asy embeds the Boehm–Demers–Weiser (BDW) conservative garbage
collector. Every value the asy interpreter creates (numbers boxed into
`vm::item`s, strings, arrays, user-defined structs, etc.) lives on
the GC heap and is collected when it becomes unreachable.

C++ plugins built on the asybind SDK share that GC. This is mostly
invisible — `ay::gc_new<T>`, `ay::mem::vector<...>`, and the
`asybind/abi.h` thunk machinery do the right thing as long as you
follow a handful of rules. This document spells those rules out:

* [Quick checklist](#quick-checklist) — the bare minimum, suitable for
  copy-pasting into a code review.
* [Best practices](#best-practices) — the recommended patterns.
* [When things actually go wrong](#when-things-actually-go-wrong) — a
  deeper, mechanism-level explanation of which mistakes cause silent
  use-after-free bugs and which are merely awkward.
* [Reference: how the SDK pieces interact with the GC](#reference-how-the-sdk-pieces-interact-with-the-gc).

The audience is someone writing a `.cc` file that exposes a templated
C++ struct to asy via `ASY_TEMPLATED_MODULE` / `ASY_MODULE`. Familiarity
with `class_<T>`, `ay::Any`, and `ay::callable<...>` is assumed.


## Quick checklist

1. **All long-lived heap objects must be allocated with `ay::gc_new<T>`
   or via `class_<T>::def(ay::init<>())`.** Never `new` or `malloc` a
   struct that asy will hold a pointer to.
2. **Anything that *contains* a GC-managed reference must itself live
   in GC memory.** A struct member of type `ay::Any`, `ay::callable<…>`,
   `T*` (pointing to a `gc_new`-allocated `T`), or
   `ay::mem::vector<…>` is a GC-managed reference for this purpose.
3. **For containers, use `ay::mem::vector`, `ay::mem::list`,
   `ay::mem::string`, `ay::mem::map`, etc.** — never plain `std::vector`
   or `std::string` if the container will outlive the current C call.
4. **Make GC-allocated types trivially destructible** (or accept that
   their destructors will not run). `ay::gc_new` does not register a
   finalizer.
5. **Never put a GC-managed reference into a process-wide `static` or
   into a thread-local.** `static inline ay::callable<…> g;` is a bug
   waiting to happen.
6. **It is fine to hold a `T*`, `ay::Any`, or `ay::callable<…>` as a
   local variable on the C stack** for the duration of one thunk
   invocation — the BDW collector scans the C stack conservatively.


## Best practices

### Allocate every persistent object with `ay::gc_new<T>`

```cpp
struct Node {                  // POD-ish; no virtuals, no destructor.
    ay::Any            payload;
    Node*              next = nullptr;
    ay::mem::vector<int> tags; // ay::mem container — see below
};

Node* n = ay::gc_new<Node>();  // ✔ scanned by GC
```

`ay::gc_new` calls the host's `alloc_obj` thunk, which returns
zero-initialized memory from the same heap asy itself uses. The
returned pointer is reachable from:

* anywhere asy code can reach it (because the wrapping `class_<T>`
  glue stores it inside a scanned `vm::item`),
* any C-stack frame currently executing inside this plugin
  (conservative scan),
* any GC-allocated object that has it as a member.

If none of those hold, the next collection cycle will free the storage.

### Hold long-lived state in a `class_<T>`, not in `static` variables

`class_<T>::def(ay::init<>())` arranges for instances to be allocated
via `alloc_obj`, so all members of `T` are automatically scanned. The
recommended pattern is:

```cpp
struct MySetCore_T {
    ay::callable<bool(ay::Any, ay::Any)> equiv;   // ✔ scanned
    ay::mem::vector<ay::Any>             elements; // ✔ scanned

    void reset(ay::callable<bool(ay::Any, ay::Any)> eq) { equiv = eq; }
    bool contains(ay::Any item) { /* ... */ }
};

ASY_TEMPLATED_MODULE(my_set_core, m, "T") {
    ay::class_<MySetCore_T> core(m, "MySetCore_T");
    core.def(ay::init<>());
    core.def<&MySetCore_T::reset>("reset");
    core.def<&MySetCore_T::contains>("contains");
}
```

Each `MySetCore_T` instance is allocated by the `init<>` thunk via
`alloc_obj`, so the `equiv` callable and the `elements` buffer are
both reachable from the asy-side wrapper that holds the instance.

### Containers: prefer `ay::mem::` aliases

The `<asybind/mem.h>` header (re-exported by the umbrella
`<asybind/asybind.h>`) provides:

| `ay::mem::vector<T>`  | `ay::mem::set<K>`           |
|-----------------------|-----------------------------|
| `ay::mem::list<T>`    | `ay::mem::multiset<K>`      |
| `ay::mem::deque<T>`   | `ay::mem::map<K,V>`         |
| `ay::mem::string`     | `ay::mem::multimap<K,V>`    |
|                       | `ay::mem::unordered_set<K>` |
|                       | `ay::mem::unordered_map<K,V>` (and `multi-` variants) |

These are aliases for `std::vector<T, gc_allocator<T>>`, etc. They
behave identically to their `std::` counterparts except that the
backing storage is allocated through the host's `alloc_obj`. If a
container is a member of a GC-allocated owner, every pointer inside
its buffer is found by the conservative scan.

```cpp
struct Bucket {
    ay::mem::vector<Entry*> slots;   // ✔ Entry* pointers are scanned
    ay::mem::string         label;   // ✔ string bytes live in GC heap
};
```

### Keep GC-allocated types trivially destructible

`ay::gc_new` does **not** register a finalizer. If your `T` has a
non-trivial destructor (e.g. it owns a `std::unique_ptr`, a file
handle, a `pthread_mutex_t` requiring explicit destruction), that
destructor will silently never run. Symptoms are usually a leak of
the non-GC resource rather than a crash, but the bug is real.

Recommended workarounds:

* Make `T` a plain struct of value/POD/SDK types only
  (`int`, `double`, `ay::Any`, `ay::callable<…>`, `ay::mem::vector<…>`,
  GC-allocated `U*`).
* If you genuinely need a finalizer, allocate the resource separately
  (outside the GC heap) and arrange for it to be released by an
  explicit asy-callable cleanup method.

### Don't mix allocators

`ay::mem::vector<T>` and `std::vector<T>` are *different types*.
Methods like `swap`, `assign(begin, end)`, and `operator=` work
within one family. Don't try to feed a `std::vector` into a function
that expects an `ay::mem::vector` and vice-versa — you'll get
allocator-mismatch compile errors at best, and accidental copies that
move data out of the GC heap at worst.

### Don't use thread-local or namespace-scope storage for GC references

The conservative scan can find pointers in:

* the C call stack of every thread the GC knows about,
* the registers of every thread the GC has stopped,
* the contents of any object allocated via `alloc_obj`.

It cannot reliably find pointers in:

* thread-local storage (the GC does not know about your TLS layout),
* arbitrary `static` data in the plugin's `.so` (unless that data
  itself happens to live on a scanned page, which is implementation-
  dependent and not portable).

In particular, do **not** do this:

```cpp
// BAD: callable held in a namespace-scope static.
namespace { ay::callable<void(ay::Any)> g_observer; }

ASY_MODULE(observer, m) {
    m.def("install", +[](ay::callable<void(ay::Any)> f) {
        g_observer = f;        // pointer lives in .bss — may be freed
    });
}
```

The `vm::callable*` inside `g_observer` may be reclaimed before the
next call to it, because nothing reachable to the GC keeps it alive.
Instead, store the callable inside a GC-allocated wrapper class
registered via `class_<T>`, and let the asy-side code hold the
instance.

(There is a documented exception, used by the §6 direct-binding
machinery in `base/collections/{hashset,btreegeneral}_core.cc`:
`static inline ay::type_param T_resolved;` is safe because
`ay::type_param` stores only an *index* and a `Kind` enum, with no
pointers.)


## When things actually go wrong

This section explains the underlying mechanism so you can reason
about cases the checklist doesn't cover.

### The collector's reachability model

A pointer to a GC-allocated object stays valid as long as the
collector can find that pointer in one of its roots when it runs a
collection cycle. The roots are:

* **The C stack of every registered thread.** Every word on the
  stack is examined; if it looks like a pointer into the GC heap, the
  pointed-to object is considered live.
* **CPU registers of stopped threads.** Same conservative treatment.
* **Static data segments (`.data`, `.bss`) of the main executable
  and dynamically loaded libraries.** Whether the GC scans your
  plugin's `.bss` depends on whether the loader-registered shared
  object was added to the GC's root set. **Do not rely on this.** The
  asy host registers itself and its statically linked code; plugins
  loaded later via `dlopen` are usually *not* added.
* **Objects allocated via `alloc_obj` (transitively).** Every byte of
  an `alloc_obj` block is scanned. Any word inside that looks like a
  GC pointer makes the target live.

The conservative scan means that if a pointer is bit-rotated, XOR-ed
with another value, or otherwise hidden, the GC will not find it.

### Failure mode 1 — use-after-free from missed reachability

The most common GC bug in a plugin is a `vm::callable*` (held inside
`ay::callable<…>`) or a `vm::item*` (held inside `ay::Any`) being
freed while the plugin still has a copy of the raw pointer.

Concretely, this happens when a GC-managed reference ends up in
memory that the collector does *not* scan. Examples:

* a `static`/`namespace`-scope `ay::callable<…>` in a `.so` whose
  `.bss` is not in the GC root set;
* a `std::vector<ay::Any>` (note: `std::`, not `ay::mem::`) whose
  buffer was allocated by `::operator new` — the buffer is not on the
  GC heap, and the `vm::item*` values inside it are invisible to the
  scan;
* a `T*` (pointing to a `gc_new`-allocated `T`) stored inside a
  `malloc`-allocated struct;
* an `ay::Any` value reinterpreted as an integer and shipped through a
  `long long` field;
* a `T*` stored in thread-local storage.

The symptom is intermittent crashes that depend on whether a
collection happens between the store and the next use. They are hard
to reproduce in a debugger because attaching often perturbs allocation
patterns enough to hide the bug.

**Why the SDK types avoid this:** `ay::callable<…>` and `ay::Any` are
just one-pointer wrappers, so when they sit inside a `class_<T>`
instance (which is GC-allocated) or an `ay::mem::vector` (whose buffer
is GC-allocated), the conservative scan walks right through them.

### Failure mode 2 — premature finalization (resource leak)

If you put a `std::unique_ptr<FILE, …>` (or any other RAII handle)
inside a `gc_new<T>`-allocated `T`, the destructor will not run when
the GC reclaims `T`. The file handle leaks. There is no crash; you
just exhaust file descriptors after enough allocations.

Same goes for `std::mutex`, `std::thread`, sockets, GPU resources.
None of these should be members of a GC-allocated struct.

### Failure mode 3 — false retention (no crash, but a leak)

The conservative scan can be too generous: an integer that happens
to hold the same bit pattern as a valid GC heap pointer will pin
the object. In practice this matters only if you stuff a large
`alloc_obj` allocation full of arbitrary user data and one of the
bytes happens to coincide with a live address.

This is rarely a correctness problem; it just means a few extra
objects survive longer than necessary. If you ever store *encrypted*
or *random* binary blobs in a `gc_new`-allocated buffer that is
larger than a few kilobytes, prefer placing the blob in a separate
`malloc`-allocated buffer that is held only as long as needed and
not stored anywhere the GC scans.

### Failure mode 4 — concurrent modification & invalidation

Unrelated to memory management but often confused with it:
`ay::mem::vector::push_back` can reallocate the backing buffer. Any
iterators, pointers, or references into the buffer obtained before
the reallocation become dangling — but the GC still considers them
live (they look like valid GC pointers), so you get *incorrect* but
not *invalid* memory access. Treat `ay::mem::` containers exactly as
you would treat `std::` containers w.r.t. iterator invalidation.


## Reference: how the SDK pieces interact with the GC

### `ay::gc_new<T>(args…)`

```cpp
template <class T, class... Args>
inline T* gc_new(Args&&... args) {
  void* mem = detail::current_api()->alloc_obj(sizeof(T));
  return new (mem) T(std::forward<Args>(args)...);
}
```

* Memory comes from `alloc_obj`, the host's wrapper for `GC_MALLOC`.
* The full `sizeof(T)` block is scanned. Every pointer-shaped word
  inside `T` is a potential GC root.
* No finalizer is registered — `~T()` is never called.

### `class_<T>::def(init<>())`

Registers a zero-argument constructor whose thunk does:

```cpp
void* mem = api->alloc_obj(sizeof(T));
::new (mem) T();
api->push_obj(s, mem);
```

The returned pointer is wrapped in a `vm::item` of user-pointer type
and pushed onto the asy stack, where the asy interpreter holds it for
the lifetime of the asy-side reference. Equivalent to `gc_new<T>`
plus the registration plumbing.

### `ay::Any` and `ay::callable<…>`

Both are single-pointer wrappers around opaque host handles
(`asybind_any_ptr`, `asybind_callable_ptr`). The pointed-to
`vm::item` / `vm::callable` is GC-allocated by the host. Storing
either type:

* on the C stack — safe (conservative scan finds it);
* as a member of a GC-allocated owner — safe (scan walks into the owner);
* inside an `ay::mem::` container that is itself a member of a
  GC-allocated owner — safe (allocator buffer is on the GC heap);
* in static or TLS storage — **unsafe** (see above).

### `ay::mem::gc_allocator<T>`

* `allocate(n)` ⇒ `alloc_obj(n * sizeof(T))`.
* `deallocate(p, n)` ⇒ no-op; GC reclaims.
* Stateless, always-equal, propagating — STL containers can be moved,
  swapped, and copied freely.

The backing buffer of any `ay::mem::vector`, `ay::mem::map`, etc. is
on the GC heap, so all pointer-shaped elements inside the buffer are
scanned. The *container header itself* (the three or so words holding
`begin`, `end`, `capacity`) lives wherever the container does — i.e.
in the enclosing object's memory. If the enclosing object is on the
C stack or in a GC-allocated block, all three header words are
scanned, so the buffer pointer stays live.

### `ay::raise`

`ay::raise(...)` does not return. Anything you allocated via
`gc_new` is collected as soon as no reachable pointers to it remain —
which, for state local to the throwing function, is immediately.

### `ASY_MODULE` / `ASY_TEMPLATED_MODULE` populate body

Runs once per module load. Anything allocated here goes away when
nothing reachable holds it. In particular, do not allocate the
"singleton state" of a module inside the populate body and assume it
will survive — register a `class_<T>` and let asy hold the instance.


## Further reading

* `asybind/include/asybind/mem.h` — allocator and container aliases.
* `asybind/include/asybind/module.h` — `gc_new`, `raise`, `rand`.
* `asybind/include/asybind/callable.h` — `ay::callable<…>` lifetime.
* `asybind/include/asybind/any.h` — `ay::Any` semantics.
* `base/collections/hashset_core.cc`,
  `base/collections/btreegeneral_core.cc` — realistic examples of
  GC-clean plugin design (callables, raw pointers, and
  `ay::mem::vector`s composed inside `class_<T>`-registered cores).
* `local/cpp-module-design-draft2.md` — the higher-level design that
  motivates the SDK.
