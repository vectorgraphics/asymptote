/*****
 * drawfill.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a cyclic path that will outline a filled shape in a picture.
 *****/

#include "drawfill.h"

namespace camp {

bool drawFill::draw(psfile *out)
{
  int n = p.size();
  if (n == 0 || pentype.transparent())
    return true;

  bool shade=begin != end;
  
  if(shade) out->gsave();
  
  out->setpen(pentype);

  penStart(out);
  penTranslate(out);

  out->write(p);

  Colorspace colspace;
  
  if(shade) {
    colspace=(Colorspace) max(pentype.Color(),endpen.Color());
    
    switch(colspace) {
    case PATTERN:
    case TRANSPARENT:
      shade=false;
      break;
    case DEFCOLOR:
    case GRAYSCALE:
      break;

    case RGB:
      {
	if (pentype.grayscale()) pentype.greytorgb();
	else if (endpen.grayscale()) endpen.greytorgb();
	break;
      }
      
    case CMYK:
      {
	if (pentype.grayscale()) pentype.greytocmyk();
	else if (endpen.grayscale()) endpen.greytocmyk();
	
	if (pentype.rgb()) pentype.rgbtocmyk();
	else if (endpen.rgb()) endpen.rgbtocmyk();
	break;
      }
    }
  }
  
  if(shade) {
    out->clip();
    out->verbatimline("<< /ShadingType 2");
    out->verbatimline("/ColorSpace /Device"+ColorDeviceSuffix[colspace]);
    out->verbatim("/Coords [");
    out->write(begin);
    out->write(end);
    out->verbatimline("]");
    out->verbatimline("/Extend [true true]");
    out->verbatimline("/Function");
    out->verbatimline("<< /FunctionType 2");
    out->verbatimline("/Domain [0 1]");
    out->verbatim("/C0 [");
    out->write(pentype);
    out->verbatimline("]");
    out->verbatim("/C1 [");
    out->write(endpen);
    out->verbatimline("]");
    out->verbatimline("/N 1");
    out->verbatimline(">>");
    out->verbatimline(">>");
    out->verbatimline("shfill");
    out->grestore();
  } else out->fill();
  
  penEnd(out);

  return true;
}

drawElement *drawFill::transformed(const transform& t)
{
  return new drawFill(transpath(t),transpen(t),t*begin,endpen,t*end);
}

} // namespace camp
