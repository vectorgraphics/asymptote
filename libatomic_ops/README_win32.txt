Most of the atomic_ops functionality is available under Win32 with
the Microsoft tools, but the build process is somewhat different from
that on Linux/Unix platforms.

To build and test the package:
1) Make sure the Microsoft command-line tools (e.g. nmake) are available.
2) Go to the src directory in the distribution and run
"nmake -f Makefile.msft check".  This should build atomic_ops.lib and
atomic_ops_gpl.lib, and execute some tests.
Alternatively, CMake could be used (e.g., see how to in README_details.txt).

To compile applications, you will need to retain or copy the following
pieces from the resulting src directory contents:
        "atomic_ops.h" - Header file defining low-level primitives.  This
                         includes files from the following folder.
        "atomic_ops" - Subdirectory containing implementation header files.
                       The atomic_ops.h implementation is entirely in the
                       header files in Win32.
        "atomic_ops.lib" - Library containing implementation of AO_pause()
                           defined in atomic_ops.c (AO_pause is needed for
                           the almost lock-free stack implementation).
        "atomic_ops_stack.h" - Header file describing almost lock-free stack.
        "atomic_ops_malloc.h" - Header file describing almost lock-free malloc.
        "atomic_ops_gpl.lib" - Library containing implementation of the
                               above two.

Note that atomic_ops_gpl.lib is covered by the GNU General Public License,
while the top 3 of these pieces allow use in proprietary code.

There are several macros a client could use to configure the build with the
Microsoft tools (except for AO_CMPXCHG16B_AVAILABLE one, others should be
rarely needed in practice):
* AO_ASM_X64_AVAILABLE - inline assembly available (only x86_64)
* AO_ASSUME_VISTA - assume Windows Server 2003, Vista or later target (only
  x86, implied if Visual Studio 2015 or older)
* AO_CMPXCHG16B_AVAILABLE - assume target is not old AMD Opteron chip (only
  x86_64)
* AO_OLD_STYLE_INTERLOCKED_COMPARE_EXCHANGE - assume ancient MS VS Win32
  headers (only arm and x86)
* AO_PREFER_GENERALIZED - prefer generalized definitions to direct
  assembly-based ones
* AO_UNIPROCESSOR - assume single-core target (only arm)
* AO_USE_INTERLOCKED_INTRINSICS - assume Win32 _Interlocked* primitives
  available as intrinsics (only arm)
* AO_USE_PENTIUM4_INSTRS - use mfence instruction instead of xchg (only x86,
  implied if SSE2 is available)
