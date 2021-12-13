# LspCpp

## Dependencies
`LspCpp` depends on the boost and rapidjson,utfcpp,and threadpool

## Build

### Linux
  1. Install boost
   ```shell
      $ sudo apt-get install libboost-dev 
   ``` 
 2. [Restore the submodules][4].
    ```shell
      $ git submodule init
      $ git submodule update
    ``` 
 3. Build it.
    ```shell
      $ make
    ``` 
### Windows
  1. Open project with Vistual Studio.
  2. [Restore packages][3] with Vistual Studio.
  3. [Restore the submodules][4].
     ```shell
      git submodule init
      git submodule update
     ``` 
  4. Build it with Vistual Studio.
 
## Reference
 Some code from :[cquery][1]

## Projects using LspCpp:
* [JCIDE](https://www.javacardos.com/tools)
* [LPG-language-server](https://github.com/kuafuwang/LPG-language-server)
## License
   MIT
   
##  Example:
[It's here](https://github.com/kuafuwang/LspCpp/tree/master/example)


[1]: https://github.com/cquery-project/cquery "cquery:"
[2]: https://www.javacardos.com/tools "JcKit:"
[3]: https://docs.microsoft.com/en-us/nuget/consume-packages/package-restore "Package Restore"
[4]: https://git-scm.com/book/en/v2/Git-Tools-Submodules "Git-Tools-Submodules"
