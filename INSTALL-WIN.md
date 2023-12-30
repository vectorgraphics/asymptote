# Building Asymptote with CMake (On Windows)

## Notes on Depedency management

The recommended way is to use [vcpkg](https://vcpkg.io/).
See `INSTALL.md` for more details.
On windows, one may run

```powershell
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.bat
```

to initialize vcpkg.
Make sure the environment VCPKG_ROOT points to where your vcpkg repository is at user or machine scope.

#### For User scope
This can be done either by Start -> "Edit environment variables for your account" and then adding
VCPKG_ROOT entry, or by Powershell,

```powershell
[Environment]::SetEnvironmentVariable('VCPKG_ROOT', '<path to vcpkg>', 'User')
```

#### For machine scope

Otherwise, you can also set VCPKG_ROOT for everyone in your machine.

#### Regarding Visual Studio

Make sure Visual Studio is installed (or a C++ compiler that is compatible with vcpkg.)
Unfortunately vcpkg is not yet compatible fully with the LLVM toolchain nor does it
provide LLVM-related triplets.

If full Visual Studio editor is not need, Visual Studio Build Tools should suffice.
Visual Studio or its build tools can be retrieved [here](https://visualstudio.microsoft.com/downloads/).


## Using CMake

### Quick start (Windows with MSVC toolchain)

Firstly, make sure `ninja` and `cmake` is installed, as well as Visual Studio or its build tools.
- Ninja can be found [here](https://ninja-build.org/)
- CMake can be found [here](https://cmake.org/), or can be installed alongside Visual Studio Build Tools.
- Visual Studio or its build tools can be found at [here](https://visualstudio.microsoft.com/downloads/).

Make sure CMake and Ninja are available in your PATH.

Additionally, building Asymptote requires `perl` and Python 3 to be installed.

- Perl can be found at [Strawberry Perl](https://strawberryperl.com/).
  We do not recommend ActiveState perl because of licensing issues.
- Python 3 can be found at https://www.python.org/downloads/.

#### Installing GCC-compatible C++ compiler

Additionally, we (highly) suggest installing a GCC-compatible C++ compiler for preprocessing.
Our recommendation is to use clang/LLVM tools, available [here](https://releases.llvm.org/).
Once your compiler is installed, there are a few options.

- (Recommended) Set `GCCCOMPAT_CXX_COMPILER_FOR_MSVC` environment variable to 
  your GCC-compatible C++ compiler. For example
  ```powershell
  $env:GCCCOMPAT_CXX_COMPILER_FOR_MSVC="<LLVM install location>/bin/clang++.exe
  ```
- If you want to make the environment variable permanent, run
  ```powershell
  [Environment]::SetEnvironmentVariable('GCCCOMPAT_CXX_COMPILER_FOR_MSVC', '<LLVM install location>/bin/clang++.exe', 'User
  ```
- Or, add your clang++.exe to `PATH` and leave `GCCCOMPAT_CXX_COMPILER_FOR_MSVC` unset.
  The build script will automatically try to locate `clang++.exe` or `g++.exe` in places
  within `PATH`. Be warned that the build script may select a different compiler depending
  on if there are other compilers available in `PATH`.


#### Even a quicker start...

If you are getting started and want a quick configuration, run `./quick-start-win32.ps1`.
This script automatically checks that you have vcpkg, and if not, clones and bootstraps vcpkg on your system.

Additionally, this script auto locates your Visual Studio installation and establishes all needed environment variables.

#### Building steps

Ensure you have `cl.exe` in your path.
The easiest way is to use Visual Studio Developer Powershell, though be careful that by default
VS Developer Powershell selects 32-bit cl.exe.

To explicitly select 64-bit Visual Studio Developer Powershell, one can use visual studio locator
alongside its developer shell script as

```powershell
$vsInfo = Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs
& "$($vsInfo.InstallLocation)\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
```

This prompt should put you in to 64-bit Visual Studio Developer Powershell.

After that, run cmake with 
```powershell
cmake --preset msvc/release 
cmake --build --preset msvc/release --target asy-with-basefiles
```

The Asymptote binary is available at `cmake-build-msvc/release/asy.exe`.
