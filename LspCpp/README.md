# LspCpp

## Dependencies
`LspCpp` depends on boost, rapidjson, utfcpp, uri.

## Build

### Linux / Mac
1. On linux, install boost
   ```shell
      $ sudo apt-get install libboost-dev 
   ``` 
   On Mac, install boost on Mac
   ```shell
      $ brew install boost
   ``` 

2. Building with ``CMake``
-----------------------
	$ mkdir _build
	$ cd _build
	$ cmake -DUri_BUILD_TESTS=OFF ..
	$ make -j4

### Windows

  1. Open cmd or powershell and generate visual studio project  with ``CMake``.
  -----------------------
    mkdir _build
	cd _build
	cmake -DUri_BUILD_TESTS=OFF -DUri_USE_STATIC_CRT=OFF ..

  2. "cmake -help" is useful if you are not familiar with cmake.
  
  3. Build it with Visual Studio.
 
## Reference
 Some code from :[cquery][1]

## Projects using LspCpp:
* [JCIDE](https://www.javacardos.com/javacardforum/viewtopic.php?f=5&t=3569&sid=e01238adf55cd08696fbf495dfa6c8e5)
* [LPG-language-server](https://github.com/kuafuwang/LPG-language-server)
* [Asymptote](https://github.com/vectorgraphics)
* [chemical](https://github.com/chemicallang/chemical)
## License
   MIT
   
##  Example:
[It's here](https://github.com/kuafuwang/LspCpp/tree/master/examples)


[1]: https://github.com/cquery-project/cquery "cquery:"
[2]: https://www.javacardos.com/tools "JcKit:"
[3]: https://docs.microsoft.com/en-us/nuget/consume-packages/package-restore "Package Restore"

