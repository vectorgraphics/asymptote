/*****
 * drawfill.h
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

  out->setpen(pentype);

  penStart(out);
  penTranslate(out);

  out->write(p);

  out->fill();

  penEnd(out);

  return true;
}

drawElement *drawFill::transformed(const transform& t)
{
  return new drawFill(transpath(t), transpen(t));
}

} // namespace camp
