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

# On Windows
See INSTALL-WIN.md for windows-specific instructions.