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

## Linux-specific dependency

Make sure flex and bison is available in path, if not, install them manually first.

```bash
sudo pacman -S flex bison
```

For OpenGL, make sure all relevant dependencies for freeglut is installed. The relevant dependencies
should be shown during vcpkg install

## Using CMake

### Quick start (Linux)

Make sure `ninja` and `cmake` is installed, as well as `gcc`.
Then run

```bash
mkdir -p cmake-build-linux/release
cmake --preset linux/release 
cmake --build --preset linux/release --target asy
```

The asymptote binary should be available in `cmake-build-linux/release` directory.
