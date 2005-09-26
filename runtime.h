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
void intCeil(vm::stack *s);
void intFloor(vm::stack *s);
void intRound(vm::stack *s);
void intSgn(vm::stack *s);
void intRand(vm::stack *s);
void intSrand(vm::stack *s);
void realPi(vm::stack *s);
void intIntMax(vm::stack *s);
void realInfinity(vm::stack *s);
void realRealMax(vm::stack *s);
void realRealMin(vm::stack *s);
void realRealEpsilon(vm::stack *s);
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
void pairLength(vm::stack *s);
void pairAngle(vm::stack *s);
void pairDegrees(vm::stack *s);
void pairUnit(vm::stack *s);
void realDir(vm::stack *s);
void pairExpi(vm::stack *s);
void pairConj(vm::stack *s);
void pairDot(vm::stack *s);

// Triple operations
void tripleZero(vm::stack *s);
void realRealRealToTriple(vm::stack *s);
void tripleXPart(vm::stack *s);
void tripleYPart(vm::stack *s);
void tripleZPart(vm::stack *s);
void realTripleMult(vm::stack *s);
void tripleRealMult(vm::stack *s);
void tripleRealDivide(vm::stack *s);
void tripleLength(vm::stack *s);
void triplePolar(vm::stack *s);
void tripleAzimuth(vm::stack *s);
void tripleCoLatitude(vm::stack *s);
void tripleLatitude(vm::stack *s);
void tripleLongitude(vm::stack *s);
void tripleUnit(vm::stack *s);
void tripleDot(vm::stack *s);
void tripleCross(vm::stack *s);
void intersectcubics(vm::stack *s);

// Transform operations
void transformIdentity(vm::stack *s);
void transformInverse(vm::stack *s);
void transformShift(vm::stack *s);
void transformXscale(vm::stack *s);
void transformYscale(vm::stack *s);
void transformScale(vm::stack *s);
void transformSlant(vm::stack *s);
void transformRotate(vm::stack *s);
void transformReflect(vm::stack *s);
  
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
void relativeDistance(vm::stack *s);

// String operations
void emptyString(vm::stack *s);
void stringReplace(vm::stack *s);
void stringFormatReal(vm::stack *s);
void stringFormatInt(vm::stack *s);
void stringTime(vm::stack *s);
  
// Pen operations
void newPen(vm::stack *s);
void resetdefaultPen(vm::stack *s);
void setDefaultPen(vm::stack *s);
void invisiblePen(vm::stack *s);
void grayPen(vm::stack *s);
void rgbPen(vm::stack *s);
void rgb(vm::stack *s);
void cmyk(vm::stack *s);
void gray(vm::stack *s);
void colors(vm::stack *s);
void pattern(vm::stack *s);
void penPattern(vm::stack *s);
void fillRule(vm::stack *s);
void penFillRule(vm::stack *s);
void baseLine(vm::stack *s);
void penBaseLine(vm::stack *s);
void lineType(vm::stack *s);
void penLineType(vm::stack *s);
void lineCap(vm::stack *s);
void penLineCap(vm::stack *s);
void lineJoin(vm::stack *s);
void penLineJoin(vm::stack *s);
void lineWidth(vm::stack *s);
void penLineWidth(vm::stack *s);
void font(vm::stack *s);
void penFont(vm::stack *s);
void fontSize(vm::stack *s);
void penFontSize(vm::stack *s);
void penLineSkip(vm::stack *s);
void overWrite(vm::stack *s);
void penOverWrite(vm::stack *s);
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

void postscript(vm::stack *s);
void tex(vm::stack *s);
void texPreamble(vm::stack *s);
void layer(vm::stack *s);
void image(vm::stack *s);
void overwrite(vm::stack *s);

void shipout(vm::stack *s);
void stringFilePrefix(vm::stack *s);

// System commands
void system(vm::stack *s);
void abort(vm::stack *s);
void atExit(vm::stack *s);
void changeDirectory(vm::stack *s);
void scrollLines(vm::stack *s);
  
// Merge output files  
void merge(vm::stack *s);
  
// Execute an asymptote file
void execute(vm::stack *s);
void eval(vm::stack *s);
  
// I/O Routines
void newFile(vm::stack *s);
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

void readChar(vm::stack *s);

void exitFunction(vm::stack *s);

// Utils
vm::array *copyArray(vm::stack *s);
vm::array *copyArray2(vm::stack *s);
  
// Math routines  
void pairArrayFFT(vm::stack *s);
void tridiagonal(vm::stack *s);
 
} // namespace run

#endif
