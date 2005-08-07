/*****
 * drawfill.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a cyclic path that will outline a filled shape in a picture.
 *****/

#include "drawfill.h"

namespace camp {

void drawAxialShade::palette(psfile *out)
{
  colorspace=DEFCOLOR;
  
  colorspace=(ColorSpace) max(pentype.colorspace(),penb.colorspace());
    
  switch(colorspace) {
  case PATTERN:
    reportError("Cannot shade with pattern");
  case INVISIBLE:
    reportError("Cannot shade with invisible pen");
  case DEFCOLOR:
  case GRAYSCALE:
    break;

  case RGB:
    {
      if (pentype.grayscale()) pentype.greytorgb();
      else if (penb.grayscale()) penb.greytorgb();
      break;
    }
      
  case CMYK:
    {
      if (pentype.grayscale()) pentype.greytocmyk();
      else if (penb.grayscale()) penb.greytocmyk();
	
      if (pentype.rgb()) pentype.rgbtocmyk();
      else if (penb.rgb()) penb.rgbtocmyk();
      break;
    }
  }
  
  out->gsave();
}  
  
void drawAxialShade::fill(psfile *out)
{
    out->clip(pentype.Fillrule());
    out->shade(true,ColorDeviceSuffix[colorspace],pentype,a,0,penb,b,0);
    out->grestore();
}

void drawRadialShade::fill(psfile *out)
{
    out->clip(pentype.Fillrule());
    out->shade(false,ColorDeviceSuffix[colorspace],pentype,a,ra,penb,b,rb);
    out->grestore();
}

bool drawFill::draw(psfile *out)
{
  if(pentype.invisible() || empty()) return true;
  
  palette(out);
  writepath(out);
  fill(out);
  return true;
}
  
drawElement *drawFill::transformed(const transform& t)
{
  return new drawFill(transpath(t),transpen(t));
}
  
drawElement *drawAxialShade::transformed(const transform& t)
{
  pair A=t*a, B=t*b;
  return new drawAxialShade(transpath(t),transpen(t),A,penb,B);
}
  
drawElement *drawRadialShade::transformed(const transform& t)
{
  pair A=t*a, B=t*b;
  double RA=length(t*(a+ra)-A);
  double RB=length(t*(b+rb)-B);
  return new drawRadialShade(transpath(t),transpen(t),A,RA,penb,B,RB);
}

} // namespace camp
