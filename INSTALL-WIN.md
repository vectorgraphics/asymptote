# Building Asymptote with CMake (On Windows)

## Basic Requirements

### Required Dependencies
Ensure the following are installed:
- Visual Studio or Visual Studio build tools.
  - Both can be found at [here](https://visualstudio.microsoft.com/downloads/).
- CMake
  - (Recommended way) Visual Studio/Visual Studio Build Tools provides bundled CMake as a selectable component
  - Otherwise, CMake can be found [here](https://cmake.org/)
- Ninja 
  - (Recommended way) Ninja can be installed using winget by running `winget install Ninja-build.Ninja`.
  - Otherwise, Ninja can be found [here](https://ninja-build.org/).
    If installing this way, ensure `ninja` is accessible from `PATH`.
- Python 3+
  - Available [here](https://www.python.org/downloads/windows/).
- Perl on Windows
  - (Recommended way) Strawberry Perl is available at [Strawberry Perl](https://strawberryperl.com/).
  - (Not recommended due to license terms) ActiveState Perl is available [here](https://www.activestate.com/products/perl/).

### Optional, but highly recommended dependencies
- A GCC-compatible C++ compiler (Optional, but highly recommended. See #installing-gcc-compatible-c++-compiler)
  - (Recommended way) [here](https://releases.llvm.org/).
  - (Untested) Visual Studio also provides clang tools as an installable component.
    If installing this way, ensure that `clang++.exe` is available.
- Vcpkg (Optional, but highly recommended. See #notes-on-dependency-management)
  - Can be found [here](https://vcpkg.io/).

## For a quick start

If you are getting started and want a quick configuration, run `./quick-start-win32.ps1`.
This script automatically checks that you have vcpkg, and if not, clones and bootstraps vcpkg on your system.

Additionally, this script automatically locates your Visual Studio installation and
establishes all required environment variables.

## Notes on Dependency management

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

##### Building asymptote with base files only

If you do not intend to generate a setup file, run cmake with 
```powershell
cmake --preset msvc/release 
cmake --build --preset msvc/release --target asy-with-basefiles
```

The Asymptote binary is available at `cmake-build-msvc/release/asy.exe`.
Instructions for generating a setup file is in the next section

### Documentation generation

#### Prerequisites for documentation generation

Make sure [MikTeX](https://miktex.org/) is installed in the system.
[TeX Live](https://tug.org/texlive/windows.html) is an acceptable substitute if you are not building `asymptote.pdf`,
which requires extra steps which will be discussed further.

##### Extra considerations for `asymptote.pdf`

On Windows, `asymptote.pdf` is built using MikTeX's `texify` program, hence why TeX Live cannot be used here.
Additionally, ensure that a replacement for `texindex` is available in the system.
As of the moment, I have only tested using WSL's `texindex`.

- If you have a WSL distribution with `texindex` installed,
that may be used as a substitute for `texindex` on windows. In this case, ensure the cache variable
`WIN32_TEXINDEX` is set to `WSL`. This is the default option.
- If you have a replacement `texindex` program, ensure `WIN32_TEXINDEX` points to that file.

The target `docgen` should build all relevant documentation files. To do this, run
```powershell
cmake --build --preset msvc/release --target docgen
```

### Installation file generation 

#### Prerequisites for installation file generation

Ensure that
- Requirements for building asymptote executable
- Requirements for building documentation (excluding `asymptote.pdf`)
- At least one of the following:
  - A pre-built `asymptote.pdf` file
  - Requirements for building `asymptote.pdf` file
- NSIS installer. This can be found [here](https://nsis.sourceforge.io/Download).
- Powershell. This should come pre-installed on windows.
  - Ensure that the ability to execute unsigned scripts is enabled
- Python 3 with relevant dependencies for building GUI files (This will be discussed in a separate section)

are present in the system.

##### If using a pre-built `asymptote.pdf`

Place `asymptote.pdf` in the directory `<asymptote-repo>/extfiles/`.
That is, the file `<asymptote-repo>/extfiles/asymptote.pdf` is present.
After that, configure cmake with the preset `msvc/release-with-existing-asymptote-pdf` - that is,

```powershell
cmake --preset msvc/release-with-existing-asymptote-pdf
```

##### If generating `asymptote.pdf` as part of build process

Use the `msvc/release` build preset for cmake.

##### Dependencies for GUI files

All required dependencies for building GUI are present in `GUI/requirements.txt` and `GUI/requirements.dev.txt`.
We recommend using a virtual environment, for example

```powershell
python.exe -m virtualenv asyguibuild
./asyguibuild/Scripts/activate.ps1

cd <asymptote-repo>/GUI
pip install -r requirements.txt
pip install -r requirements.dev.txt
```
.

However, against our recommendations, the dependencies can be also installed into the system interpreter.

#### Building Asymptote install files

The cmake target `asy-pre-nsis-targets` should build everything on the `C++` side needed
for asymptote installation. 

#### Building the GUI files

The python script `GUI/buildtool.py` is used for building required files. To do this, run

```powershell
cd <asymptote-repo>/GUI
python.exe buildtool.py build
```

This should build all needed GUI files.

#### Generating the installer file

After building `asy-pre-nsis-targets`, install using CMake.
Note that this does not install into 
the program files directory, but rather, to a "local install root"
at `<asymptote-repo>/cmake-install-w32-nsis-release/`.

Due to how google test build files are written (as of currently), installing 
every components may result in an error (in particular, with `gmock.lib`).
This can be remedied by installing only the component needed for installer generation: `asy-pre-nsis`
To do this, run 

```powershell
cmake --install cmake-build-msvc/release --component asy-pre-nsis
```

After building all the needed dependencies,
navigate to the directory `<asymptote-repo>/cmake-install-w32-nsis-release/`.
There, a script called `build-asy-installer.ps1` script is present.
Run that script and it will prompt for the location of `makensis.exe` from the NSIS.
Specify the path to `makensis.exe`.

After this, the script should generate the installer file with the name `asymptote-<version>-setup.exe`.
This is the setup file ready for distribution.
