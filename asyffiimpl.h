#pragma once

#include "asyffi.h"
#include "common.h"
#include "record.h"
#include "genv.h"

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

  THAsyString createNewAsyString(char const* str) override;

  THAsyString
  createNewAsyStringSized(char const* str, size_t const& size) override;

  void updateAsyString(THAsyString asyStringPtr, const char* str) override;
  void updateAsyStringSized(
          THAsyString asyStringPtr, const char* str, const size_t& size
  ) override;

  size_t getStringLength(THAsyString asyString) override;
  void
  copyString(
          THAsyString asyString, char* destination, size_t bufferSize
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

  IAsySolvedKnot* createSolvedKnot2D(
          const IAsyTuple* pre, const IAsyTuple* point, const IAsyTuple* post,
          bool isStraight
  ) override;
  IAsySolvedKnot* createSolvedKnot3D(
          const IAsyTuple* pre, const IAsyTuple* point, const IAsyTuple* post,
          bool isStraight
  ) override;

  [[nodiscard]]
  bool isGcSupported() const override;
  bool getGcStackBase(void* stackBase) override;
  [[nodiscard]]
  size_t getGcStackBaseSize() const override;
  bool registerThreadWithGc(void* stackBase) const override;
  void unregisterThreadWithGc() const override;

  [[nodiscard]]
  bool isSimpleFrameBuild() const override;

  [[noreturn]]
  void reportError(const char* message) override;
  
  void reportWarning(const char* message) override;
  [[noreturn]]
  void reportFatal(const char* message) override;

  IAsyPen* createNewPen(
          const Asy::PenLineType* lineType, double lineWidth,
          IAsyPath const* pathValue, const char* font, double fontSize,
          double lineSkip, Asy::PenColorSpace colorSpace, Asy::PenColor color,
          const char* pattern, Asy::PenFillRule fillRule,
          Asy::PenTransparencyInfo const* transparency,
          Asy::PenBaseLine baseLine, Asy::PenLineCap lineCap,
          Asy::PenLineJoin lineJoin, double miterLimit,
          Asy::PenOverwrites overwriteType, IAsyTransform const* transformValue
  ) override;
  
  [[nodiscard]]
  void* createAsyType(Asy::TypeInfo typeInfo) const override;

  IAsyVarFrame* createNewVarFrame(const size_t& initialSize) override;

  IAsyPicture* createPicture(bool deconstruct) override;

  IAsyDrawElement* createDrawElementFromPath(
          IAsyPath* path, IAsyPen* pen, const char* key
  ) override;
  IAsyDrawElement* createDrawElementFromPath3(
          IAsyPath3* path3, IAsyTuple* center, double opacity,
          const Asy::Material3D& material, bool billboard, const char* key
  ) override;
  IAsyDrawElement* createDrawElementForPixel(
          IAsyTuple* point, const IAsyPen* pen, double width, const char* key
  ) override;

  IAsyDrawElement* createDrawElementForFill(
          IAsyArray const* srcPaths, bool stroke, IAsyPen* penType,
          const char* key
  ) override;

  IAsyDrawElement* createDrawElementForLatticeShade(
          IAsyArray const* srcPaths, bool stroke, IAsyPen* penType,
          IAsyArray const* pens, const IAsyTransform* transf, const char* key
  ) override;
  IAsyDrawElement* createDrawElementForAxialShade(
          IAsyArray const* srcPaths, bool stroke, IAsyPen* penType,
          IAsyTuple* pairA, bool extendA, IAsyPen* penB, IAsyTuple* pairB,
          bool extendB, const char* key
  ) override;
  IAsyDrawElement* createDrawElementForRadialShade(
          IAsyArray const* srcPaths, bool stroke, IAsyPen* penType,
          IAsyTuple* pairA, const double& ra, bool extendA, IAsyPen* penB,
          IAsyTuple* pairB, const double& rb, bool extendB, const char* key
  ) override;
  IAsyDrawElement* createDrawElementForFunctionShade(
          IAsyArray const* srcPaths, bool stroke, IAsyPen* penType,
          const char* shader, const char* key
  ) override;
  
  IAsyDrawElement* createDrawElementForGouraudShade(
          const IAsyArray* srcPaths, bool stroke, IAsyPen* penType,
          const IAsyArray* pens, const IAsyArray* pairVertices,
          const IAsyArray* intEdges, const char* key
  ) override;

  IAsyDrawElement* createDrawElementForTensorShade(
          const IAsyArray* srcPaths, bool stroke, IAsyPen* penType,
          const IAsyArray* pens, const IAsyArray* boundaries,
          const IAsyArray* z, const char* key
  ) override;

  IAsyDrawElement* createDrawElementForLabel(
          const char* label, const char* size, IAsyTransform* transf,
          IAsyTuple* pairPosition, IAsyTuple* pairAlign, IAsyPen* penType,
          const char* key
  ) override;

  IAsyDrawElement* createDrawElementForLabelPath(
          const char* label, const char* size, IAsyPath* src,
          const char* justify, IAsyTuple* pairShift, IAsyPen* penType,
          const char* key
  ) override;

  IAsyDrawElement* createDrawElementForVerbatim(
          Asy::DrawVerbatimLanguage language, const char* text,
          IAsyTuple* pairMin, IAsyTuple* pairMax
  ) override;

  void runString(const char* text, bool interactiveWrite) override;
  void runRunnable(THAsyRunnable runnableCode) override;

protected:
  static string fromCharConstOrEmpty(char const* originalStr);

  template<typename TCastedTo, typename TCastedFrom>
  static TCastedTo& castDynamicAndDereference(TCastedFrom* ptr, bool checkNotNull=true)
  {
    static_assert(std::is_base_of_v<TCastedFrom, TCastedTo>);
    auto* dynCastedPtr= dynamic_cast<TCastedTo*>(ptr);
    if (checkNotNull && dynCastedPtr == nullptr) {
      camp::reportError("Failed to dynamically cast ptr to requested type ");
    }
    return *dynCastedPtr;
  }

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

  [[nodiscard]]
  bool isInteractive() const override;

  void runStringEmbedded(const char* text) override;
  
  void runCodeEmbedded(THAsyRunnable runnableCode) override;

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
  AsyFfiRegistererImpl(string const& dynlibName, trans::genv* genv, IAsyContext* context);
  void registerFunction(
          char const* name, TAsyForeignFunction fn,
          Asy::FunctionTypeMetadata const& fnTypeInfo
  ) override;

  IAsyGlobalEnvironment* getGlobalEnvironment() override;
  /**
   * @remark Note that this pointer can be safely used outside the scope of
   * this class instance because it is created using gc
   */
  [[nodiscard]]
  record* getRecord() const;

  IAsyContext* getContext() override;

private:
  string libName;
  IAsyContext* contextPtr;

public:
  trans::genv* globalEnv;
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
