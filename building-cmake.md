# Building Asymptote with CMake

## Dependency management

The recommended way is to use [vcpkg](https://vcpkg.io/). Clone vcpkg to your system, run bootstrap script and ensure
`VCPKG_ROOT` environment is exported as set as path to your vcpkg repository. For example,

```bash
cd ~/dev/
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
export VCPKG_ROOT=~/dev/vcpkg
```

or Windows,

```powershell
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.bat
```

Make sure Visual Studio is installed (or a C++ compiler that is compatible with vcpkg.)
Unfortunately vcpkg is not yet compatible fully with the LLVM toolchain nor does it
provide LLVM-related triplets.

If full Visual Studio editor is not need, Visual Studio Build Tools should suffice.
Visual Studio or its build tools can be retrieved [here](https://visualstudio.microsoft.com/downloads/).

## Linux-specific dependency

Make sure flex and bison is available in path, if not, install them manually first.

```bash
# This is specific to arch linux, other distributions might use a different name
sudo pacman -S flex bison
```

For OpenGL, make sure all relevant dependencies for freeglut is installed. The relevant dependencies
should be shown during vcpkg install

## Using CMake

### Quick start (Linux)

Make sure `ninja` and `cmake`, `python3` and `perl` are installed, as well as `gcc`.
Then run

```bash
mkdir -p cmake-build-linux/release
cmake --preset linux/release 
cmake --build --preset linux/release --target asy-with-basefiles
```

The asymptote binary should be available in `cmake-build-linux/release` directory.


### Quick start (Windows with MSVC toolchain)

Firstly, make sure `ninja` and `cmake` is installed, as well as Visual Studio or its build tools.
Visual Studio or its build tools can be found at [here](https://visualstudio.microsoft.com/downloads/).

Additionally, building Asymptote requires `perl` and Python 3 to be installed.

- Perl can be found at [Strawberry Perl](https://strawberryperl.com/).
  We do not recommend ActiveState perl because of licensing issues.
- Python 3 can be found at https://www.python.org/downloads/.

#### Installing GCC-compatible C++ compiler

Additionally, we (highly) suggest installing a GCC-compatible C++ compiler for preprocessing.
Our recommendation is to use clang/LLVM tools, available [here](https://releases.llvm.org/).
Once your compiler is installed, set `GCCCOMPAT_CXX_COMPILER_FOR_MSVC` environment variable to
your GCC-compatible C++ compiler. For example,

```powershell
$env:GCCCOMPAT_CXX_COMPILER_FOR_MSVC="<LLVM install location>/bin/clang++.exe"
```

#### Building steps

After that, run cmake with 
```powershell
cmake --preset msvc/release 
cmake --build --preset msvc/release --target asy-with-basefiles
```

The Asymptote binary is available at `cmake-build-msvc/release/asy.exe`.


### On Debug Builds

One thing you may notice is that we do not provide a debug build preset. This is intentional
since anyone developing might want to add configurations specific to their system
(such as a particular clang they want to use for preprocessing), or for vendor-specific configurations
(e.g. selecting a particular toolchain in CLion).

Our recommendation is to create your own debug presets in `CMakeUserPresets.json` - for example,
for my (Jamie's) setup:

```json
{
  "version": 6,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 26,
    "patch": 0
  },

  "configurePresets": [
    {
      "name": "msvc/debug-clion+vs",
      "displayName": "[MSVC-x86/64] Debug (With preset environment vars)",
      "binaryDir": "${sourceDir}/cmake-build-msvc/debug",
      "inherits": ["base/buildBaseWithVcpkg", "base/debug", "base/gccCompatCacheVar", "base/windows-only"],
      "environment": {
        "GCCCOMPAT_CXX_COMPILER_FOR_MSVC": "C:\\msys64\\clang64\\bin\\clang++.exe"
      },
      "vendor": {
        "jetbrains.com/clion": {
          "toolchain": "MSVC"
        }
      }
    },
    {
      "name": "linux/debug-clion+vs",
      "displayName": "[linux-x86/64] Debug (With preset environment vars)",
      "binaryDir": "${sourceDir}/cmake-build-linux/debug",
      "inherits": [ "base/buildBaseWithVcpkg", "base/debug" ],
      "environment": {
          "VCPKG_ROOT": "$env{HOME}/dev/vcpkg"
      },
      "vendor": {
          "jetbrains.com/clion": {
              "toolchain": "WSL"
          }
      }
    }
  ]
}
```

## Testing

Asymptote unit testing is integerated into CMake's `CTest` framework.
All Asymptote `.asy` based tests are named `asy.<test dirname>.<test file name>`
excluding `*.asy` extension.

These tests can be run by CTest. For example, after building on linux/release,

```bash
ctest --test-dir cmake-build-linux/release/ -R "asy.types.*"
```
