/*****
 * drawfill.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a cyclic path that will outline a filled shape in a picture.
 *****/

#include "drawfill.h"

namespace camp {

void drawFill::palette(psfile *out)
{
  axial=ra == 0 && rb == 0;
  shade=a != b || !axial;
  
  colorspace=DEFCOLOR;
  
  if(shade) {
    colorspace=(ColorSpace) max(pentype.colorspace(),penb.colorspace());
    
    switch(colorspace) {
    case PATTERN:
    case INVISIBLE:
      shade=false;
      break;
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
  }
  
  if(shade) {
    out->gsave();
  } else {
    out->setpen(pentype);
    penStart(out);
    penTranslate(out);
  }
}  
  
void drawFill::fill(psfile *out)
{
  if(shade) {
    out->clip(pentype.Fillrule());
    out->shade(axial,ColorDeviceSuffix[colorspace],pentype,a,ra,penb,b,rb);
    out->grestore();
  } else {
    out->fill(pentype.Fillrule());
    penEnd(out);
  }
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
  pair A=t*a, B=t*b;
  double RA=length(t*(a+ra)-A), RB=length(t*(b+rb)-B);
  if(P)
    return new drawFill(transPath(t),transpen(t),A,RA,penb,B,RB);
  else 
    return new drawFill(transpath(t),transpen(t),A,RA,penb,B,RB);
}

} // namespace camp
