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

  bool axial=ra == 0 && rb == 0;
  bool shade=a != b || !axial;
  
  if(shade) out->gsave();
  
  out->setpen(pentype);

  penStart(out);
  penTranslate(out);

  out->write(p);

  ColorSpace colorspace;
  
  if(shade) {
    colorspace=(ColorSpace) max(pentype.colorspace(),penb.colorspace());
    
    switch(colorspace) {
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
    out->clip();
    out->verbatim("<< /ShadingType ");
    out->verbatimline(axial ? "2" : "3");
    out->verbatimline("/ColorSpace /Device"+ColorDeviceSuffix[colorspace]);
    out->verbatim("/Coords [");
    out->write(a);
    if(!axial) out->write(ra);
    out->write(b);
    if(!axial) out->write(rb);
    out->verbatimline("]");
    out->verbatimline("/Extend [true true]");
    out->verbatimline("/Function");
    out->verbatimline("<< /FunctionType 2");
    out->verbatimline("/Domain [0 1]");
    out->verbatim("/C0 [");
    out->write(pentype);
    out->verbatimline("]");
    out->verbatim("/C1 [");
    out->write(penb);
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
  pair A=t*a, B=t*b;
  return new drawFill(transpath(t),transpen(t),A,length(t*(a+ra)-A),
		      penb,B,length(t*(b+rb)-B));
}

} // namespace camp
