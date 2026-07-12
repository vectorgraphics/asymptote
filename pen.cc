/*****
 * pen.cc
 * John Bowman
 *
 *****/

#include "pen.h"
#include "drawelement.h"

namespace camp {

const char* DEFPAT="<default>";
const char* DEFLATEXFONT="\\usefont{\\ASYencoding}{\\ASYfamily}{\\ASYseries}{\\ASYshape}";
const char* DEFCONTEXTFONT="modern";
const char* DEFTEXFONT="cmr12";
const double DEFWIDTH=-1;
const Int DEFCAP=-1;
const Int DEFJOIN=-1;
const double DEFMITER=0;
const transform nullTransform=transform(0.0,0.0,0.0,0.0,0.0,0.0);
LineType::LineType(Asy::PenLineType const& lineTypeInfo)
  : pattern(lineTypeInfo.patternCount),
offset(lineTypeInfo.offset),
scale(lineTypeInfo.scale),
adjust(lineTypeInfo.adjust),
isdefault(false)
{
  for (size_t i=0; i < lineTypeInfo.patternCount; ++i) {
    pattern[i]=*(lineTypeInfo.patternPtr + i);
  }
}

const char* PSCap[]={"butt","round","square"};
const char* Cap[]={"square","round","extended"};
const Int nCap=sizeof(Cap)/sizeof(char*);
const char* Join[]={"miter","round","bevel"};
const Int nJoin=sizeof(Join)/sizeof(char*);
const char* OverwriteTag[]={"Allow","Suppress","SupressQuiet",
                            "Move","MoveQuiet"};
const Int nOverwrite=sizeof(OverwriteTag)/sizeof(char*);
const char* FillRuleTag[]={"ZeroWinding","EvenOdd"};
const Int nFill=sizeof(FillRuleTag)/sizeof(char*);
const char* BaseLineTag[]={"NoAlign","Align"};
const Int nBaseLine=sizeof(BaseLineTag)/sizeof(char*);

const char* ColorDeviceSuffix[]={"","","Gray","RGB","CMYK",""};
const unsigned nColorSpace=sizeof(ColorDeviceSuffix)/sizeof(char*);
const char* BlendMode[]={"Compatible","Normal","Multiply","Screen",
                         "Overlay","SoftLight","HardLight",
                         "ColorDodge","ColorBurn","Darken",
                         "Lighten","Difference","Exclusion",
                         "Hue","Saturation","Color","Luminosity"};
const Int nBlendMode=sizeof(BlendMode)/sizeof(char*);
void pen::setColor(Asy::PenColorSpace newColorSpace, Asy::PenColor newColor)
{
  color= static_cast<ColorSpace>(static_cast<uint8_t>(newColorSpace));
  r= newColor.red;
  g = newColor.green;
  b = newColor.blue;
  grey=newColor.grey;
}
double pen::getLineWidth() const
{
  return width();
}

void* pen::getFontName() const
{
  return new (UseGC) string(Font());
}
Transparency::Transparency(Asy::PenTransparencyInfo const& transparencyInfo)
    : blend(transparencyInfo.blendType), opacity(transparencyInfo.opacity), isdefault(false)
{}
double pen::getFontSize() const { return size(); }
double pen::getLineSkip() const { return Lineskip(); }

Asy::PenColorSpace pen::getColorSpace() const
{
  auto underlyingValue = static_cast<uint8_t>(colorspace());
  return static_cast<Asy::PenColorSpace>(underlyingValue);
  
}
bool pen::tryPromoteColorSpace(Asy::PenColorSpace newColorSpace)
{
  auto underlyingValue= static_cast<uint8_t>(newColorSpace);
  return promote(static_cast<ColorSpace>(underlyingValue));
}
double pen::getRedOrCyan() const
{
  return red();
}

double pen::getGreenOrMagenta() const
{
  return green();
}
double pen::getBlueOrYellow() const
{
  return blue();
}
double pen::getGreyOrKValue() const
{
  return gray();
}
const char* pen::getPatternName() const
{
  return pattern.empty() ? nullptr : pattern.c_str();
}
Asy::PenFillRule pen::getFillRule() const
{
  auto underlyingValue= static_cast<int8_t>(Fillrule());
  return static_cast<Asy::PenFillRule>(underlyingValue);
}

Asy::PenBaseLine pen::getBaseLine() const
{
  auto underlyingValue= static_cast<int8_t>(Baseline());
  return static_cast<Asy::PenBaseLine>(underlyingValue);
}
int64_t pen::getLineCap() const
{
  return cap();
}
int64_t pen::getLineJoin() const
{
  return join();
}
double pen::getMiterLimit() const
{
  return miter();
}
Asy::PenOverwrites pen::getOverwriteValue() const
{
  auto underlyingValue= static_cast<int8_t>(Overwrite());
  return static_cast<Asy::PenOverwrites>(underlyingValue);
}
const IAsyTransform* pen::getTransformValue() const
{
  return new transform(getTransform());
}
IAsyPen* pen::composeWithAnotherPen(const IAsyPen* other) const
{
  auto const* castedPen = static_cast<pen const*>(other);
  return new pen(*this + *castedPen);
}
IAsyPen* pen::multiplyColor(double const factor) const
{
  return new pen(factor * (*this));
}
void pen::setFontName(const char* newFont)
{
  font = newFont == nullptr ? "" : string(newFont);
}
void pen::tryConvertToGreyscale() {
  togrey();
}
void pen::setFillRule(Asy::PenFillRule fillRule)
{
  auto underlyingValue= static_cast<int8_t>(fillRule);
  fillrule = static_cast<FillRule>(underlyingValue);
}

pen drawElement::lastpen;

}
