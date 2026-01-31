#pragma once

#include "asyffi.h"
#include "common.h"
#include "record.h"

#include <type_traits>

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

  [[nodiscard]]
  bool isCompactBuild() const override;

  [[nodiscard]]
  const char* getVersion() const override;

  [[nodiscard]]
  const char* getAsyGlVersion() const override;

  IAsyItem* createBlankItem() override;

  void* createNewAsyString(char const* str) override;

  void* createNewAsyStringSized(char const* str, size_t const& size) override;

  void updateAsyString(void* asyStringPtr, const char* str) override;
  void updateAsyStringSized(
          void* asyStringPtr, const char* str, const size_t& size
  ) override;
  IAsyArray* createNewArray(const size_t& initialSize) override;

  IAsyTransform* createNewTransform(
          double x, double y, double xx, double xy, double yx, double yy
  ) override;

  IAsyTransform* createNewIdentityTransform() override;

  IAsyTuple* createPair(double x, double y) override;
  IAsyTuple* createTriple(double x, double y, double z) override;

  IAsyTensionSpecifier*
  createTensionSpecifierWithSameVal(double val, bool atleast) override;
  IAsyTensionSpecifier*
  createTensionSpecifier(double out, double in, bool atleast) override;

protected:
  template<typename TImpl, typename TInterface, typename... TCreationArgs>
  static TInterface* createNewItemGeneric(TCreationArgs&&... args)
  {
    static_assert(std::is_base_of_v<TInterface, TImpl>);
    static_assert(std::is_base_of_v<gc, TImpl>);
    return static_cast<TInterface*>(
            new TImpl(std::forward<TCreationArgs>(args)...)
    );
  }
};

class AsyFfiRegistererImpl : public IAsyFfiRegisterer
{
public:
  AsyFfiRegistererImpl(string const& dynlibName);
  void registerFunction(
          char const* name, TAsyForeignFunction fn,
          AsyTypeInfo const& returnType, size_t numArgs,
          AsyFnArgMetadata* argInfoPtr
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

ty* asyTypesEnumToTy(AsyTypeInfo const& asyType);
ty* processArrayTypesInfoToTy(AsyTypeInfo const& baseType);

types::formal asyArgInfoToFormal(AsyFnArgMetadata const& argInfo);
}// namespace camp
