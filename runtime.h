/*****
 * runtime.h
 * Andy Hammerlindl 2002/7/31
 *
 * Defines all of the runtime functions that are used by the vm::stack
 * machine.
 *****/

#ifndef RUNTIME_H
#define RUNTIME_H

#include "array.h"

namespace vm {
class stack;
}

namespace run {
  
// Math
void intZero(vm::stack *s);
void realZero(vm::stack *s);
void boolFalse(vm::stack *s);
void boolTrue(vm::stack *s);
void boolNot(vm::stack *s);
void boolXor(vm::stack *s);
void boolMemEq(vm::stack *s);
void boolMemNeq(vm::stack *s);
void boolFuncEq(vm::stack *s);
void boolFuncNeq(vm::stack *s);

void realFmod(vm::stack *s);
void realIntPow(vm::stack *s);
void realAtan2(vm::stack *s);
void realHypot(vm::stack *s);
void realRemainder(vm::stack *s);
void realJ(vm::stack *s);
void realY(vm::stack *s);
void intQuotient(vm::stack *s);
void intAbs(vm::stack *s);
void intSgn(vm::stack *s);
void intRand(vm::stack *s);
void intSrand(vm::stack *s);
void intRandMax(vm::stack *s);
void boolDeconstruct(vm::stack *s);

// Array operations
void emptyArray(vm::stack *s);
void newArray(vm::stack *s);
void newDeepArray(vm::stack *s);
void newInitializedArray(vm::stack *s);
void newAppendedArray(vm::stack *s);
void arrayRead(vm::stack *s);
void arrayArrayRead(vm::stack *s);
void arrayWrite(vm::stack *s);
void arrayLength(vm::stack *s);
void arrayCyclicFlag(vm::stack *s);
void arrayCyclic(vm::stack *s);
void arrayCyclicHelper(vm::stack *s);
void arrayPush(vm::stack *s);
void arrayPushHelper(vm::stack *s);
void arrayPop(vm::stack *s);
void arrayPopHelper(vm::stack *s);
void arrayAppend(vm::stack *s);
void arrayAppendHelper(vm::stack *s);
void arrayAlias(vm::stack *s);
void arrayConditional(vm::stack *s);
void arrayIntArray(vm::stack *s);
void arraySequence(vm::stack *s);
void intSequence(vm::stack *s);
void arrayFunction(vm::stack *s);
void arrayFind(vm::stack *s);
void arrayAll(vm::stack *s);
void arrayBoolNegate(vm::stack *s);
void arrayBoolSum(vm::stack *s);
void arrayCopy(vm::stack *s);
void arrayConcat(vm::stack *s);
void array2Copy(vm::stack *s);
void array2Transpose(vm::stack *s);

// Null operations
void pushNullArray(vm::stack *s);
void pushNullRecord(vm::stack *s);
void pushNullFunction(vm::stack *s);

// Default operations
void pushDefault(vm::stack *s);
void isDefault(vm::stack *s);

// Casts
void pairToGuide(vm::stack *s);
void pathToGuide(vm::stack *s);
void guideToPath(vm::stack *s);

// Pair operations
void pairZero(vm::stack *s);
void realRealToPair(vm::stack *s);
void pairNegate(vm::stack *s);
void pairXPart(vm::stack *s);
void pairYPart(vm::stack *s);

// Triple operations
void tripleZero(vm::stack *s);
void realRealRealToTriple(vm::stack *s);
void tripleXPart(vm::stack *s);
void tripleYPart(vm::stack *s);
void tripleZPart(vm::stack *s);
void realTripleMult(vm::stack *s);
void tripleRealMult(vm::stack *s);
void tripleRealDivide(vm::stack *s);

void intersectcubics(vm::stack *s);

// Transform operations  
void transformIdentity(vm::stack *s);
void transformTransformMult(vm::stack *s);
void transformTransformMult(vm::stack *s);
void transformPairMult(vm::stack *s);
void transformPathMult(vm::stack *s);
void transformPenMult(vm::stack *s);
void transformFrameMult(vm::stack *s);
void transformPow(vm::stack *s);
void transformXPart(vm::stack *s);
void transformYPart(vm::stack *s);
void transformXXPart(vm::stack *s);
void transformXYPart(vm::stack *s);
void transformYXPart(vm::stack *s);
void transformYYPart(vm::stack *s);
void real6ToTransform(vm::stack *s);
void boolTransformEq(vm::stack *s);
void boolTransformNeq(vm::stack *s);

// Path operations
void nullPath(vm::stack *s);
void pathSize(vm::stack *s);
void pathMax(vm::stack *s);
void pathMin(vm::stack *s);
void pathConcat(vm::stack *s);

// Guide operations
void nullGuide(vm::stack *s);
void dotsGuide(vm::stack *s);
void dashesGuide(vm::stack *s);
void cycleGuide(vm::stack *s);
void dirSpec(vm::stack *s);
void curlSpec(vm::stack *s);
void realRealTension(vm::stack *s);
void pairPairControls(vm::stack *s);

// String operations
void emptyString(vm::stack *s);
void stringReplace(vm::stack *s);
void stringFormatReal(vm::stack *s);
void stringFormatInt(vm::stack *s);
void stringTime(vm::stack *s);
  
// Pen operations
void newPen(vm::stack *s);
void boolPenEq(vm::stack *s);
void boolPenNeq(vm::stack *s);
void penPenPlus(vm::stack *s);
void realPenTimes(vm::stack *s);
void penRealTimes(vm::stack *s);
void penMax(vm::stack *s);
void penMin(vm::stack *s);

// Picture operations
void newFrame(vm::stack *s);
void boolNullFrame(vm::stack *s);
void frameMax(vm::stack *s);
void frameMin(vm::stack *s);
void fill(vm::stack *s);
void latticeShade(vm::stack *s);
void axialShade(vm::stack *s);
void radialShade(vm::stack *s);
void gouraudShade(vm::stack *s);
void clip(vm::stack *s);
void beginClip(vm::stack *s);
void inside(vm::stack *s);

void postscript(vm::stack *s);
void tex(vm::stack *s);
void texPreamble(vm::stack *s);
void layer(vm::stack *s);
void image(vm::stack *s);
void overwrite(vm::stack *s);

// Shipout
void shipout(vm::stack *s);
void exitFunction(vm::stack *s);
void updateFunction(vm::stack *s);
void stringFilePrefix(vm::stack *s);

// System commands
void system(vm::stack *s);
void abort(vm::stack *s);
void changeDirectory(vm::stack *s);
void scrollLines(vm::stack *s);
  
// Merge output files  
void merge(vm::stack *s);
  
// Execute an Asymptote file
void loadModule(vm::stack *s);
void execute(vm::stack *s);
void evalString(vm::stack *s);
void evalAst(vm::stack *s);
void readGUI(vm::stack *s);
  
// I/O Routines
void standardOut(vm::stack *s);
void fileOpenOut(vm::stack *s);
void fileOpenIn(vm::stack *s);
void fileOpenXOut(vm::stack *s);
void fileOpenXIn(vm::stack *s);

void fileEof(vm::stack *s);
void fileEol(vm::stack *s);
void fileError(vm::stack *S);
void fileClear(vm::stack *S);
void fileClose(vm::stack *s);
void filePrecision(vm::stack *s);
void fileFlush(vm::stack *s);
void fileDimension1(vm::stack *s);
void fileDimension2(vm::stack *s);
void fileDimension3(vm::stack *s);
void fileCSVMode(vm::stack *s);
void fileLineMode(vm::stack *s);
void fileSingleMode(vm::stack *s);
void fileArray1(vm::stack *s);
void fileArray2(vm::stack *s);
void fileArray3(vm::stack *s);
void boolFileEq(vm::stack *s);
void boolFileNeq(vm::stack *s);
void nullFile(vm::stack *s);

void readChar(vm::stack *s);
void writestring(vm::stack *s);

// Utils
vm::array *copyArray(vm::stack *s);
vm::array *copyArray2(vm::stack *s);
  
// Math routines  
void pairArrayFFT(vm::stack *s);
void tridiagonal(vm::stack *s);
void quadraticRoots(vm::stack *s);  
void cubicRoots(vm::stack *s);  
  
} // namespace run

#endif
