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

  size_t getStringLength(void* asyString) override;
  void
  copyString(void* asyString, char* destination, size_t bufferSize) override;

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

  IAsyCurlSpecifier* createCurlSpecifier(double value, uint8_t s) override;

  TAsyFfiCycleToken createCycleToken() override;

  IAsyItem* getSetting(char const* name) override;

  IAsyPath* createAsyPath(
          int64_t n, bool cycles, size_t numSolvedKnots,
          const IAsySolvedKnot* const* solvedKnotsPtr
  ) override;
  IAsyPath3* createAsyPath3(
          int64_t n, bool cycles, size_t numSolvedKnots,
          const IAsySolvedKnot* const* solvedKnotsPtr
  ) override;

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

class AsyStackContextImpl : public IAsyStackContext
{
public:
  AsyStackContextImpl(vm::stack* inStack);

  void callVoid(
          IAsyCallable* callable, size_t numArgs, IAsyItem const** ptrArgs
  ) override;

  IAsyItem* callReturning(
          IAsyCallable* callable, size_t numArgs, IAsyItem const** ptrArgs
  ) override;

  void callReturningToExistingItem(
          IAsyCallable* callable, size_t numArgs, const IAsyItem** ptrArgs,
          IAsyItem* returnItem
  ) override;

  IAsyCallable* getBuiltin(
          char const* module, const char* fnName, Asy::TypeInfo typeInfo
  ) override;

protected:
  trans::access* getVariableAccess(
          char const* module, char const* fnName, Asy::TypeInfo const& typeInfo
  ) const;

private:
  vm::stack* stack;
};

class AsyFfiRegistererImpl : public IAsyFfiRegisterer
{
public:
  AsyFfiRegistererImpl(string const& dynlibName);
  void registerFunction(
          char const* name, TAsyForeignFunction fn,
          Asy::FunctionTypeMetadata const& fnTypeInfo
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

ty* asyTypesEnumToTy(Asy::TypeInfo const& asyType);
ty* processArrayTypesInfoToTy(Asy::ArrayTypeMetadata const& arrayInfo);


types::function* createFunctionTypeFromMetadata(
        Asy::FunctionTypePtrRetMetadata const& fnTypeInfo
);
types::formal asyArgInfoToFormal(Asy::FnArgMetadata const& argInfo);

ty* getArrayTypeFromBaseType(ty* baseType, size_t const& dimension);

}// namespace camp
