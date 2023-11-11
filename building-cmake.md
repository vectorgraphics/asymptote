# Building Asymptote with CMake

## Dependency management

The recommended way is to use vcpkg. Clone vcpkg to your system, run bootstrap script and ensure
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

## Linux-specific dependency

Make sure flex and bison is available in path, if not, install them manually first.

```bash
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


### Quick start (Windows)

Firstly, make sure `ninja` and `cmake` is installed.

Additionally, building Asymptote requires `perl` and Python 3 to be installed.

- Perl can be found at [Strawberry Perl](https://strawberryperl.com/).
  We do not recommend ActiveState perl because of licensing issues.
- Python 3 can be found at https://www.python.org/downloads/.

Ensure both executables are available in Windows PATH.

Install clang. While [LLVM Releases page](https://releases.llvm.org/download.html) may work,
we recommend using [MSYS2](https://www.msys2.org/) to install clang64 toolchain.

For MSYS2, install the following dependencies:

```bash
pacman -S mingw-w64-clang-x86_64-toolchain
```
(Note that this list may be incomplete and more dependencies.
Please let us know if you need to install more msys2 packages).

Then, set CC and CXX environment variables to your clang/clang++ compiler, for example

```powershell
$env:CC="<msys2 install location>/clang64/bin/clang.exe"
$env:CXX="<msys2 install location>/clang64/bin/clang++.exe"
```

After that, run cmake with 
```powershell
cmake --preset clang64-win/release 
cmake --build --preset clang64-win/release --target asy-with-basefiles
```

The Asymptote binary is available at `cmake-build-clang64-win/release/asy.exe`

After that, if asymptote fails to start, you may need to copy libc++ from msys2 clang64's bin folder -
e.g. `<msys2 install location>/clang64/bin/libc++.dll` to `cmake-build-clang64-win/release/`.
