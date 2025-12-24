#pragma once

#include "asyffi.h"
#include "common.h"
#include "record.h"

namespace camp
{
class AsyArgsImpl : public IAsyArgs
{
public:
  AsyArgsImpl(size_t const& argSize);

  [[nodiscard]]
  size_t getArgumentCount() const override;

  [[nodiscard]]
  IAsyItem* getNumberedArg(size_t const& argNum) override;


  void setArgNum(size_t const& argNum, vm::item const& arg);

private:
  mem::vector<vm::item> argsStorage;
};

class AsyContextImpl : public IAsyContext
{
public:
  void* malloc(size_t const& size) override;
  void* mallocAtomic(size_t const& size) override;
};

class AsyFfiRegistererImpl : public IAsyFfiRegisterer
{
public:
  AsyFfiRegistererImpl(string const& dynlibName);
  void registerFunction(
          char const* name, TAsyForeignFunction fn, AsyTypes const& returnType,
          size_t numArgs, AsyFnArgMetadata* argInfoPtr
  ) override;

  /**
   * @remark Note that this pointer can be safely used outside the scope of
   * this class instance because it is created using gc
   */
  [[nodiscard]]
  record* getRecord() const;

private:
  string libName;

public:
  symbol sym;
  // recordVar /must/ come after sym declaration

private:
  types::dummyRecord* recordVar= nullptr;
};

ty* asyTypesEnumToTy(AsyTypes const& asyType);

types::formal asyArgInfoToFormal(AsyFnArgMetadata const& argInfo);
}// namespace camp
