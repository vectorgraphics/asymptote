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
  colorspace=(ColorSpace) max(pentype.colorspace(),penb.colorspace());
  
  switch(colorspace) {
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
  default:
    break;
  }
  
  out->gsave();
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
  
drawElement *drawLatticeShade::transformed(const transform& t)
{
  return new drawLatticeShade(transpath(t),pentype,pens);
}

drawElement *drawAxialShade::transformed(const transform& t)
{
  pair A=t*a, B=t*b;
  return new drawAxialShade(transpath(t),pentype,A,penb,B);
}
  
drawElement *drawRadialShade::transformed(const transform& t)
{
  pair A=t*a, B=t*b;
  double RA=length(t*(a+ra)-A);
  double RB=length(t*(b+rb)-B);
  return new drawRadialShade(transpath(t),pentype,A,RA,penb,B,RB);
}

drawElement *drawGouraudShade::transformed(const transform& t)
{
  size_t size=vertices->size();
  vm::array *Vertices=new vm::array(size);
  for(size_t i=0; i < size; i++)
    (*Vertices)[i]=t*vm::read<pair>(vertices,i);

  return new drawGouraudShade(transpath(t),pentype,pens,Vertices,edges);
}

} // namespace camp
