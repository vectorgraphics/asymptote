# Building Asymptote with VCPKG and CMake

## Dependency Management

The recommended way is to use [vcpkg](https://vcpkg.io/). Clone vcpkg to your system, run bootstrap script and ensure
`VCPKG_ROOT` environment is exported as set as path to your vcpkg repository. For example,

```bash
cd ~/dev/
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
export VCPKG_ROOT=~/dev/vcpkg
```
## On Windows
See INSTALL-WIN.md for windows-specific instructions.

## Linux-specific dependency (Experimental)

Make sure flex and bison are available in PATH, if not, install them manually first.

```bash
# This is specific to arch linux, other distributions might use a different name
sudo pacman -S flex bison
```

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

### Additional build information

One can specify additional package string (this is useful for CI for denoting build revision).
To do this, add a file called `asy-pkg-version-suffix.cmake` with a cmake command 
```cmake
set(ASY_VERSION_SUFFIX "<custom version suffix>")
```

This suffix will get embedded into the final asymptote version. If this file is not specified, the default
suffix is "+debug" for debug builds, or an empty string for all other builds, including release builds

## Testing

Asymptote unit testing is integrated into CMake's `CTest` framework. The
Asymptote `.asy` tests are not registered with CTest one by one; the whole
`tests/` tree is driven by `tests/run_asy_tests.py`, which CTest runs as the
single test `bundled.asy.checktests`. The other `bundled.asy.*` tests cover the
`collections` error messages, `getExecutablePath()`, the relocatable sysdir
matrix and `wce`; they all carry the label `asy-check-tests`.

CTest never builds anything, so build the `asy-check-test-deps` target first —
`asy-with-basefiles` is enough to *run* asy, but not to run every test.

```bash
cmake --build --preset linux/release --target asy-check-test-deps
ctest --test-dir cmake-build-linux/release/                     # everything
ctest --test-dir cmake-build-linux/release/ -L asy-check-tests  # only the asy suites
```

To run just a few `.asy` tests, call the runner directly with `--tests-list`, a
semicolon-separated list of `<test dirname>/<test file name>` paths without the
`.asy` extension:

```bash
python3 tests/run_asy_tests.py \
    --asy cmake-build-linux/release/asy \
    --asy-base-dir cmake-build-linux/release/base \
    --tests-list "types/cast;types/var"
```
