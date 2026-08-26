#include "asyffiimpl.h"

#include "absyn.h"
#include "coenv.h"
#include "common.h"
#include "drawfill.h"
#include "drawlabel.h"
#include "drawpath.h"
#include "drawpath3.h"
#include "drawverbatim.h"
#include "path3.h"
#include "picture.h"
#include "settings.h"
#include "transform.h"
#include "util.h"

#include "triple.h"
#include <array.h>
#include <callable.h>

#include <cstring>
#include <guide.h>
#include <stack.h>

namespace camp
{

using Asy::BaseTypes;

AsyArgsImpl::AsyArgsImpl(size_t const& argSize) : argsStorage(argSize) {}
void AsyArgsImpl::setArgNum(size_t const& argNum, vm::item const& arg)
{
  argsStorage[argNum]= arg;
}

size_t AsyArgsImpl::getArgumentCount() const { return argsStorage.size(); }


IAsyItem* AsyArgsImpl::getNumberedArg(const size_t& argNum)
{
  return argsStorage.data() + argNum;
}


void* AsyContextImpl::malloc(size_t const& size) { return asy_malloc(size); }
void* AsyContextImpl::mallocAtomic(size_t const& size)
{
  return asy_malloc_atomic(size);
}
bool AsyContextImpl::isCompactBuild() const
{
#if COMPACT
  return true;
#else
  return false;
#endif
}
const char* AsyContextImpl::getVersion() const { return REVISION; }
const char* AsyContextImpl::getAsyGlVersion() const { return AsyGLVersion; }

IAsyItem* AsyContextImpl::createBlankItem() { return new vm::item(); }

void* AsyContextImpl::createNewAsyString(char const* str)
{
  return new (UseGC) mem::string(str);
}

void* AsyContextImpl::createNewAsyStringSized(
        char const* str, size_t const& size
)
{
  return new (UseGC) mem::string(str, size);
}
void AsyContextImpl::updateAsyString(void* asyStringPtr, const char* str)
{
  auto* castedStr= static_cast<mem::string*>(asyStringPtr);
  castedStr->assign(str);
}
void AsyContextImpl::updateAsyStringSized(
        void* asyStringPtr, const char* str, const size_t& size
)
{
  auto* castedStr= static_cast<mem::string*>(asyStringPtr);
  castedStr->assign(str, size);
}
size_t AsyContextImpl::getStringLength(void* asyString)
{
  auto* castedStr= static_cast<mem::string*>(asyString);
  return castedStr->length();
}
void AsyContextImpl::copyString(
        void* asyString, char* destination, size_t bufferSize
)
{
  auto const* castedStr= static_cast<mem::string*>(asyString);
#ifdef _WIN32
  if (strcpy_s(destination, bufferSize, castedStr->c_str()) != 0) {
    reportError("Failed to copy string");
  }
#else
  strncpy(destination, castedStr->c_str(), bufferSize);
#endif
}

IAsyArray* AsyContextImpl::createNewArray(const size_t& initialSize)
{
  return new vm::array(initialSize);
}
IAsyTransform* AsyContextImpl::createNewTransform(
        double x, double y, double xx, double xy, double yx, double yy
)
{
  return createNewItemGeneric<transform, IAsyTransform>(x, y, xx, xy, yx, yy);
}
IAsyTransform* AsyContextImpl::createNewIdentityTransform()
{
  return createNewTransform(0, 0, 1, 0, 0, 1);
}
IAsyTuple* AsyContextImpl::createPair(double x, double y)
{
  return createNewItemGeneric<pair, IAsyTuple>(x, y);
}
IAsyTuple* AsyContextImpl::createTriple(double x, double y, double z)
{
  return createNewItemGeneric<triple, IAsyTuple>(x, y, z);
}

IAsyTensionSpecifier*
AsyContextImpl::createTensionSpecifierWithSameVal(double val, bool atleast)
{
  return new tensionSpecifier(val, atleast);
}
IAsyTensionSpecifier*
AsyContextImpl::createTensionSpecifier(double out, double in, bool atleast)
{
  return new tensionSpecifier(out, in, atleast);
}
IAsyCurlSpecifier* AsyContextImpl::createCurlSpecifier(double value, uint8_t s)
{
  return new curlSpecifier(value, static_cast<side>(s));
}
TAsyFfiCycleToken AsyContextImpl::createCycleToken() { return new cycleToken; }

IAsyItem* AsyContextImpl::getSetting(char const* name)
{
  return &settings::Setting(string(name));
}
IAsyPath* AsyContextImpl::createAsyPath(
        int64_t n, bool cycles, size_t const numSolvedKnots,
        const IAsySolvedKnot* const* solvedKnotsPtr
)
{
  vector<solvedKnot> newSolvedKnots;
  newSolvedKnots.reserve(numSolvedKnots);

  for (size_t i= 0; i < numSolvedKnots; ++i) {
    auto const* solvedKnotPtr= *(solvedKnotsPtr + i);
    auto const* castedSolvedKnotPtr=
            dynamic_cast<solvedKnot const*>(solvedKnotPtr);

    if (!castedSolvedKnotPtr) {
      reportError(
              "Invalid IAsySolvedKnot pointer specified. The IAsySolvedKnot "
              "must point to a solved knot of 2D type"
      );
      return nullptr;
    }

    newSolvedKnots.emplace_back(*castedSolvedKnotPtr);
  }

  return createNewItemGeneric<path, IAsyPath>(newSolvedKnots, n, cycles);
}
IAsyPath3* AsyContextImpl::createAsyPath3(
        int64_t n, bool cycles, size_t const numSolvedKnots,
        const IAsySolvedKnot* const* solvedKnotsPtr
)
{
  vector<solvedKnot3> newSolvedKnots;
  newSolvedKnots.reserve(numSolvedKnots);

  for (size_t i= 0; i < numSolvedKnots; ++i) {
    auto const* solvedKnotPtr= *(solvedKnotsPtr + i);
    auto const* castedSolvedKnotPtr=
            dynamic_cast<solvedKnot3 const*>(solvedKnotPtr);

    if (!castedSolvedKnotPtr) {
      reportError(
              "Invalid IAsySolvedKnot pointer specified. The IAsySolvedKnot "
              "must point to a solved knot of 3D type"
      );
      return nullptr;
    }

    newSolvedKnots.emplace_back(*castedSolvedKnotPtr);
  }

  return createNewItemGeneric<path3, IAsyPath3>(newSolvedKnots, n, cycles);
}
IAsySolvedKnot* AsyContextImpl::createSolvedKnot2D(
        const IAsyTuple* pre, const IAsyTuple* point, const IAsyTuple* post,
        bool isStraight
)
{
  auto const* preCasted= dynamic_cast<pair const*>(pre);
  if (!preCasted) {
    reportError("Invalid pre point specified");
  }
  auto const* pointCasted= dynamic_cast<pair const*>(point);
  if (!pointCasted) {
    reportError("Invalid main point specified");
  }
  auto const* postCasted= dynamic_cast<pair const*>(post);
  if (!postCasted) {
    reportError("Invalid post point specified");
  }

  auto* retSolvedKnot= new solvedKnot;
  retSolvedKnot->pre= *preCasted;
  retSolvedKnot->point= *pointCasted;
  retSolvedKnot->post= *postCasted;
  retSolvedKnot->straight= isStraight;

  return retSolvedKnot;
}
IAsySolvedKnot* AsyContextImpl::createSolvedKnot3D(
        const IAsyTuple* pre, const IAsyTuple* point, const IAsyTuple* post,
        bool isStraight
)
{
  auto const* preCasted= dynamic_cast<triple const*>(pre);
  if (!preCasted) {
    reportError("Invalid pre point specified");
  }
  auto const* pointCasted= dynamic_cast<triple const*>(point);
  if (!pointCasted) {
    reportError("Invalid main point specified");
  }
  auto const* postCasted= dynamic_cast<triple const*>(post);
  if (!postCasted) {
    reportError("Invalid post point specified");
  }

  auto* retSolvedKnot= new solvedKnot3;
  retSolvedKnot->pre= *preCasted;
  retSolvedKnot->point= *pointCasted;
  retSolvedKnot->post= *postCasted;
  retSolvedKnot->straight= isStraight;

  return retSolvedKnot;
}

bool AsyContextImpl::isGcSupported() const
{
#if defined(USEGC) && defined(HAVE_PTHREAD)
  return true;
#else
  return false;
#endif
}

bool AsyContextImpl::getGcStackBase(void* stackBase)
{
#if defined(USEGC) && defined(HAVE_PTHREAD)
  auto* stackBaseCasted= static_cast<GC_stack_base*>(stackBase);
  auto const result= GC_get_stack_base(stackBaseCasted);
  return result == GC_SUCCESS;
#else
  return false;
#endif
}

size_t AsyContextImpl::getGcStackBaseSize() const
{
#if defined(USEGC) && defined(HAVE_PTHREAD)
  return sizeof(GC_stack_base);
#else
  return 0;
#endif
}
bool AsyContextImpl::registerThreadWithGc(void* stackBase) const
{
#if defined(USEGC) && defined(HAVE_PTHREAD)
  auto* stackBaseCasted= static_cast<GC_stack_base const*>(stackBase);
  auto const ret= GC_register_my_thread(stackBaseCasted);
  return ret == GC_SUCCESS || ret == GC_DUPLICATE;
#else
  return false;
#endif
}
void AsyContextImpl::unregisterThreadWithGc() const
{
#if defined(USEGC) && defined(HAVE_PTHREAD)
  GC_unregister_my_thread();
#endif
}
bool AsyContextImpl::isSimpleFrameBuild() const
{
#ifdef SIMPLE_FRAME
  return true;
#else
  return false;
#endif
}
[[noreturn]]
void AsyContextImpl::reportError(const char* message)
{
  camp::reportError(message);
}
void AsyContextImpl::reportWarning(const char* message)
{
  camp::reportWarning(message);
}
[[noreturn]]
void AsyContextImpl::reportFatal(const char* message)
{
  camp::reportFatal(message);
}
IAsyPen* AsyContextImpl::createNewPen(
        const Asy::PenLineType* lineType, double const lineWidth,
        IAsyPath const* pathValue, const char* font, double const fontSize,
        double const lineSkip, Asy::PenColorSpace const colorSpace,
        Asy::PenColor const color, const char* pattern,
        Asy::PenFillRule const fillRule,
        Asy::PenTransparencyInfo const* transparency, Asy::PenBaseLine baseLine,
        Asy::PenLineCap const lineCap, Asy::PenLineJoin const lineJoin,
        double const miterLimit, Asy::PenOverwrites const overwriteType,
        IAsyTransform const* transformValue
)
{
  auto const convertedColorSpace=
          static_cast<ColorSpace>(static_cast<uint8_t>(colorSpace));
  auto const convertedFillRule=
          static_cast<FillRule>(static_cast<int8_t>(fillRule));
  auto const convertedBaseLine=
          static_cast<BaseLine>(static_cast<int8_t>(baseLine));
  auto const convertedOverwrite=
          static_cast<overwrite_t>(static_cast<int8_t>(overwriteType));

  auto const* castedPathPtr= dynamic_cast<path const*>(pathValue);
  auto const* castedTransformPtr=
          dynamic_cast<transform const*>(transformValue);

  return new pen(
          lineType == nullptr ? LineType() : LineType(*lineType), lineWidth,
          castedPathPtr == nullptr ? nullpath : *castedPathPtr,
          font == nullptr ? "" : font, fontSize, lineSkip, convertedColorSpace,
          color.red, color.green, color.blue, color.grey,
          pattern == nullptr ? DEFPAT : string(pattern), convertedFillRule,
          convertedBaseLine,
          transparency == nullptr ? Transparency()
                                  : Transparency(*transparency),
          static_cast<int8_t>(lineCap), static_cast<int8_t>(lineJoin),
          miterLimit, convertedOverwrite,
          castedTransformPtr == nullptr ? nullTransform : *castedTransformPtr
  );
}

THAsyType AsyContextImpl::createAsyType(Asy::TypeInfo const typeInfo) const
{
  return asyTypesEnumToTy(typeInfo);
}
IAsyVarFrame* AsyContextImpl::createNewVarFrame(const size_t& initialSize)
{
  return new vm::vmFrame(initialSize);
}
IAsyPicture* AsyContextImpl::createPicture(bool const deconstruct)
{
  return new picture(deconstruct);
}
IAsyDrawElement* AsyContextImpl::createDrawElementFromPath(
        IAsyPath* path, IAsyPen* pen, const char* key
)
{

  return new drawPath(
          castDynamicAndDereference<class path>(path),
          castDynamicAndDereference<class pen>(pen), fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementFromPath3(
        IAsyPath3* path3, IAsyTuple* center, double const opacity,
        const Asy::Material3D& material, bool const billboard, const char* key
)
{
  return new drawPath3(
          castDynamicAndDereference<class path3>(path3),
          castDynamicAndDereference<triple>(center),
          castDynamicAndDereference<vm::array const>(material.dseColors),
          opacity, material.shininess, material.metallic, material.fresnel0,
          billboard ? Interaction::BILLBOARD : Interaction::EMBEDDED,
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForPixel(
        IAsyTuple* point, const IAsyPen* pen, double const width,
        const char* key
)
{
  return new drawPixel(
          castDynamicAndDereference<triple>(point),
          castDynamicAndDereference<class pen const>(pen), width,
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForFill(
        IAsyArray const* srcPaths, bool const stroke, IAsyPen* penType,
        const char* key
)
{
  return new drawFill(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType), fromCharConstOrEmpty(key)
  );
}

IAsyDrawElement* AsyContextImpl::createDrawElementForLatticeShade(
        IAsyArray const* srcPaths, bool const stroke, IAsyPen* penType,
        IAsyArray const* pens, const IAsyTransform* transf, const char* key
)
{
  return new drawLatticeShade(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType),
          castDynamicAndDereference<vm::array const>(pens),
          transf != nullptr ? castDynamicAndDereference<transform const>(transf)
                            : identity,
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForAxialShade(
        IAsyArray const* srcPaths, bool const stroke, IAsyPen* penType,
        IAsyTuple* pairA, bool const extendA, IAsyPen* penB, IAsyTuple* pairB,
        bool const extendB, const char* key
)
{
  return new drawAxialShade(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType),
          castDynamicAndDereference<pair>(pairA), extendA,
          castDynamicAndDereference<pen>(penB),
          castDynamicAndDereference<pair>(pairB), extendB,
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForRadialShade(
        IAsyArray const* srcPaths, bool const stroke, IAsyPen* penType,
        IAsyTuple* pairA, const double& ra, bool const extendA, IAsyPen* penB,
        IAsyTuple* pairB, const double& rb, bool const extendB, const char* key
)
{
  return new drawRadialShade(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType),
          castDynamicAndDereference<pair>(pairA), ra, extendA,
          castDynamicAndDereference<pen>(penB),
          castDynamicAndDereference<pair>(pairB), rb, extendB,
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForFunctionShade(
        IAsyArray const* srcPaths, bool const stroke, IAsyPen* penType,
        const char* shader, const char* key
)
{
  return new drawFunctionShade(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType), string(shader),
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForGouraudShade(
        const IAsyArray* srcPaths, bool const stroke, IAsyPen* penType,
        const IAsyArray* pens, const IAsyArray* pairVertices,
        const IAsyArray* intEdges, const char* key
)
{
  return new drawGouraudShade(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType),
          castDynamicAndDereference<vm::array const>(pens),
          castDynamicAndDereference<vm::array const>(pairVertices),
          castDynamicAndDereference<vm::array const>(intEdges),
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForTensorShade(
        const IAsyArray* srcPaths, bool const stroke, IAsyPen* penType,
        const IAsyArray* pens, const IAsyArray* boundaries, const IAsyArray* z,
        const char* key
)
{
  return new drawTensorShade(
          castDynamicAndDereference<vm::array const>(srcPaths), stroke,
          castDynamicAndDereference<pen>(penType),
          castDynamicAndDereference<vm::array const>(pens),
          castDynamicAndDereference<vm::array const>(boundaries),
          castDynamicAndDereference<vm::array const>(z),
          fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForLabel(
        const char* label, const char* size, IAsyTransform* transf,
        IAsyTuple* pairPosition, IAsyTuple* pairAlign, IAsyPen* penType,
        const char* key
)
{
  return new drawLabel(
          string(label), string(size),
          castDynamicAndDereference<transform>(transf),
          castDynamicAndDereference<pair>(pairPosition),
          castDynamicAndDereference<pair>(pairAlign),
          castDynamicAndDereference<pen>(penType), fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForLabelPath(
        const char* label, const char* size, IAsyPath* src, const char* justify,
        IAsyTuple* pairShift, IAsyPen* penType, const char* key
)
{
  return new drawLabelPath(
          string(label), string(size), castDynamicAndDereference<path>(src),
          string(justify), castDynamicAndDereference<pair>(pairShift),
          castDynamicAndDereference<pen>(penType), fromCharConstOrEmpty(key)
  );
}
IAsyDrawElement* AsyContextImpl::createDrawElementForVerbatim(
        Asy::DrawVerbatimLanguage language, const char* text,
        IAsyTuple* pairMin, IAsyTuple* pairMax
)
{
  string const textStr(text);
  auto const languageCasted=
          static_cast<Language>(static_cast<uint8_t>(language));
  if (pairMin == nullptr) {
    return new drawVerbatim(languageCasted, textStr);
  } else {
    return new drawVerbatim(
            languageCasted, textStr, castDynamicAndDereference<pair>(pairMin),
            castDynamicAndDereference<pair>(pairMax)
    );
  }
}
void AsyContextImpl::runString(const char* text, bool const interactiveWrite)
{
  ::runString(string(text), interactiveWrite);
}
void AsyContextImpl::runRunnable(THAsyRunnable const runnableCode)
{
  auto* runnableCasted= static_cast<absyntax::runnable*>(runnableCode);
  auto* ast= new absyntax::block(runnableCasted->getPos(), false);
  ast->add(runnableCasted);

  runCode(ast);
}


string AsyContextImpl::fromCharConstOrEmpty(char const* originalStr)
{
  return {originalStr == nullptr ? "" : originalStr};
}

AsyStackContextImpl::AsyStackContextImpl(vm::stack* inStack) : stack(inStack) {}
void AsyStackContextImpl::callVoid(
        IAsyCallable* callable, size_t const numArgs, IAsyItem const** ptrArgs
)
{
  for (size_t i= 0; i < numArgs; ++i) {
    auto* ptrArg= dynamic_cast<vm::item const*>(ptrArgs[i]);
    if (!ptrArg) {
      reportError("Invalid item supplied as an argument");
      return;
    }
    stack->push(*ptrArg);
  }

  auto* fn= dynamic_cast<vm::callable*>(callable);
  if (!fn) {
    reportError("Invalid function supplied");
    return;
  }

  fn->call(stack);
}

IAsyItem* AsyStackContextImpl::callReturning(
        IAsyCallable* callable, size_t const numArgs, IAsyItem const** ptrArgs
)
{
  callVoid(callable, numArgs, ptrArgs);
  return new vm::item(stack->pop());
}
void AsyStackContextImpl::callReturningToExistingItem(
        IAsyCallable* callable, size_t numArgs, const IAsyItem** ptrArgs,
        IAsyItem* returnItem
)
{
  callVoid(callable, numArgs, ptrArgs);

  auto* retItemCasted= dynamic_cast<vm::item*>(returnItem);
  if (!retItemCasted) {
    reportError("Invalid return object specified");
    return;
  }
  *retItemCasted= stack->pop();
}

IAsyCallable* AsyStackContextImpl::getBuiltin(
        char const* module, const char* fnName, Asy::TypeInfo const typeInfo
)
{
  auto* entryLoc= getVariableAccess(module, fnName, typeInfo);

  if (auto const* builtinFnAccess=
              dynamic_cast<trans::bltinAccess*>(entryLoc)) {
    return new vm::bfunc(builtinFnAccess->getFunction());
  }

  return nullptr;
}
bool AsyStackContextImpl::isInteractive() const
{
  auto const* stackInteractive= dynamic_cast<vm::interactiveStack const*>(stack);
  return stackInteractive != nullptr;
}
void AsyStackContextImpl::runStringEmbedded(const char* text)
{
  trans::coenv* coe= stack->getEnvironment();

  if (auto* stackInteractive= dynamic_cast<vm::interactiveStack*>(stack);
      stackInteractive && coe) {
    ::runStringEmbedded(string(text), *coe, *stackInteractive);
  } else {
    camp::reportError("There is no runtime for embedded evaluation");
  }
}
void AsyStackContextImpl::runCodeEmbedded(THAsyRunnable const runnableCode)
{
  auto* runnableCasted= static_cast<absyntax::runnable*>(runnableCode);
  auto* ast= new absyntax::block(runnableCasted->getPos(), false);
  ast->add(runnableCasted);
  
  trans::coenv* coe= stack->getEnvironment();

  if (auto* stackInteractive= dynamic_cast<vm::interactiveStack*>(stack);
      stackInteractive && coe) {
    ::runCodeEmbedded(ast, *coe, *stackInteractive);
  } else {
    camp::reportError("There is no runtime for embedded evaluation");
  }
}

trans::access* AsyStackContextImpl::getVariableAccess(
        char const* module, char const* fnName, Asy::TypeInfo const& typeInfo
) const
{
  auto& env= stack->getEnvironment()->e;
  auto* tyData= asyTypesEnumToTy(typeInfo);

  auto const fnNameSym= symbol::trans(string(fnName));

  varEntry* entry= nullptr;
  if (module != nullptr) {
    record* moduleEntry= env.getLoadedModule(symbol::trans(string(module)));
    if (!moduleEntry) {
      // module not found
      return nullptr;
    }

    entry= moduleEntry->e.lookupVarByType(fnNameSym, tyData);
  } else {
    entry= env.lookupVarByType(fnNameSym, tyData);
  }

  if (!entry) {
    return nullptr;
  }

  return entry->getLocation();
}

AsyFfiRegistererImpl::AsyFfiRegistererImpl(
        string const& dynlibName, trans::genv* genv, IAsyContext* context
)
    : libName(dynlibName), contextPtr(context), globalEnv(genv),
      sym(symbol::literalTrans(dynlibName)),
      recordVar(new types::dummyRecord(sym))
{}

void AsyFfiRegistererImpl::registerFunction(
        char const* name, TAsyForeignFunction fn,
        Asy::FunctionTypeMetadata const& fnTypeInfo
)
{
  Asy::FunctionTypePtrRetMetadata const fnMetadataPtr= {
          &(fnTypeInfo.returnType), fnTypeInfo.numArgs, fnTypeInfo.argInfoPtr
  };

  types::function* functionSig= createFunctionTypeFromMetadata(fnMetadataPtr);
  recordVar->add(name, functionSig, fn);
}
IAsyGlobalEnvironment* AsyFfiRegistererImpl::getGlobalEnvironment()
{
  return globalEnv;
}

record* AsyFfiRegistererImpl::getRecord() const { return recordVar; }
IAsyContext* AsyFfiRegistererImpl::getContext() { return contextPtr; }

types::function* createFunctionTypeFromMetadata(
        Asy::FunctionTypePtrRetMetadata const& fnTypeInfo
)
{
  auto* functionSig=
          new types::function(asyTypesEnumToTy(*(fnTypeInfo.returnType)));
  for (size_t i= 0; i < fnTypeInfo.numArgs; ++i) {
    functionSig->add(asyArgInfoToFormal(fnTypeInfo.argInfoPtr[i]));
  }

  return functionSig;
}

ty* asyTypesEnumToTy(Asy::TypeInfo const& asyType)
{
  switch (asyType.baseType) {
#define PRIMITIVE(name, Name, asyName)                                         \
  case BaseTypes::Name:                                                        \
    return types::prim##Name();
#define EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#define PRIMITIVES_MACRO_ONLY
#include "primitives.h"


    DEFINE_PRIMTIVES
#undef EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#undef PRIMITIVES_MACRO_ONLY
#undef PRIMITIVE
    case BaseTypes::Integer:// handle integer case separately
      return types::primInt();
    case BaseTypes::Str:
      return types::primString();
    case BaseTypes::ArrayType:
      return processArrayTypesInfoToTy(asyType.extraData.arrayTypeInfo);
    case BaseTypes::FunctionType:
      return createFunctionTypeFromMetadata(asyType.extraData.functionTypeInfo);
    case BaseTypes::Record:
      // technically, this is a hack.
      return static_cast<record*>(asyType.extraData.recordPtr);
    default:
      reportError("Invalid argument type");
      return nullptr;
  }
}

ty* processArrayTypesInfoToTy(Asy::ArrayTypeMetadata const& arrayInfo)
{
  auto* tyInfoPtr= arrayInfo.typeOfItem;
  if (tyInfoPtr->baseType == BaseTypes::ArrayType) {
    reportWarning(
            "Array type should not contain an array type. "
            "Instead, use higher dimensions to specify multidimensional arrays."
    );
  }

  ty* baseType= asyTypesEnumToTy(*tyInfoPtr);
  return getArrayTypeFromBaseType(baseType, arrayInfo.dimension);
}

namespace
{

// To avoid re-creating array types, we can use a cache.
types::primTypeArrayCache arrayTypeCache;
bool arrayTypecacheInitialized= false;
}// namespace

ty* getArrayTypeFromBaseType(ty* baseType, size_t const& dimension)
{
  if (!arrayTypecacheInitialized) {
    types::initializeArrayTypeCache(arrayTypeCache);
    arrayTypecacheInitialized= true;
  }

  return types::getArrayType(baseType, dimension, &arrayTypeCache);
}

types::formal asyArgInfoToFormal(Asy::FnArgMetadata const& argInfo)
{
  ty* parsedArgType= asyTypesEnumToTy(argInfo.type);
  return {parsedArgType, symbol::literalTrans(argInfo.name), argInfo.optional,
          argInfo.explicitArgs};
}


}// namespace camp
