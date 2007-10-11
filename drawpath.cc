/*****
 * drawpath.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a path that has been added to a picture.
 *****/

#include <sstream>
#include <vector>
#include <cfloat>

#include "drawpath.h"
#include "psfile.h"
#include "util.h"

namespace camp {

double PatternLength(double arclength, const std::vector<double>& pat,
		     const path& p, double penwidth)
{
  double sum=0.0;
      
  size_t n=pat.size();
  for(unsigned i=0; i < n; i ++)
    sum += pat[i]*penwidth;
  
  if(sum == 0.0) return 0.0;
  
  if(n % 2 == 1) sum *= 2.0; // On/off pattern repeats after 2 cycles.
      
  // Fix bounding box resolution problem. Example:
  // asy -f pdf testlinetype; gv -scale -2 testlinetype.pdf
  if(!p.cyclic() && pat[0] == 0) sum += 1.0e-3*penwidth;
      
  double terminator=((p.cyclic() && arclength >= 0.5*sum) ? 0.0 : 
		     pat[0]*penwidth);
  int ncycle=(int)((arclength-terminator)/sum+0.5);

  return (ncycle >= 1 || terminator >= 0.75*arclength) ? 
    ncycle*sum+terminator : 0.0;
}

void drawPath::adjustdash(pen& pen0)
{
  // Adjust dash sizes to fit arclength; also compensate for linewidth.
  string stroke=pen0.stroke();
  
  if(!stroke.empty()) {
    double penwidth=pen0.linetype().scale ? pen0.width() : 1.0;
    double factor=penwidth;
    
    std::vector<double> pat;
    
    istringstream ibuf(stroke);
    double l;
    while(ibuf >> l) {
      if(l < 0) l=0;
      pat.push_back(l);
    }
      
    size_t n=pat.size();
    
    if(pen0.linetype().adjust) {
      double arclength=p.arclength();
      if(arclength) {
	if(n == 0) return;
      
	double denom=PatternLength(arclength,pat,p,penwidth);
	if(denom != 0.0) factor *= arclength/denom;
      }
    }
    
    factor=max(factor,0.1);
    
    ostringstream buf;
    buf.setf(std::ios::fixed);
    for(unsigned i=0; i < n; i++)
      buf << pat[i]*factor << " ";
    pen0.setstroke(buf.str());
    pen0.setoffset(pen0.linetype().offset*factor);
  }
}
  
// Account for square or extended pen cap contributions to bounding box.
void cap(bbox& b, double t, path p, pen pentype) {
  transform T=pentype.getTransform();  
  
  double h=0.5*pentype.width();
  pair v=p.dir(t);
  transform S=rotate(conj(v))*shiftless(T);
  double xx=S.getxx(), xy=S.getxy();
  double yx=S.getyx(), yy=S.getyy();
  double y=hypot(yx,yy);
  if(y == 0) return;
  double numer=xx*yx+xy*yy;
  double x=numer/y;
  pair z=shift(T)*p.point(t);
  
  switch(pentype.cap()) {
  case 0:
    {
      pair d=rotate(v)*pair(x,y)*h;
      b += z+d;
      b += z-d;
      break;
    }
  case 2:
    {
      transform R=rotate(v);
      double w=(xx*yy-xy*yx)/y;
      pair dp=R*pair(x+w,y)*h;
      pair dm=R*pair(x-w,y)*h;
      b += z+dp;
      b += z+dm;
      b += z-dp;
      b += z-dm;
    break;
    }
  }
}

void drawPath::bounds(bbox& b, iopipestream&, boxvector&, bboxlist&)
{
  Int l=p.length();
  if(l < 0) return;
  
  bbox penbounds=pentype.bounds();
  
  if(cyclic() || pentype.cap() == 1) {
    b += pad(p.bounds(),penbounds);
    return;
  }
  
  b += p.internalbounds(penbounds);
  
  cap(b,0,p,pentype);
  cap(b,l,p,pentype);
}

bool drawPath::draw(psfile *out)
{
  Int n = p.size();
  if (n == 0 || pentype.invisible() || pentype.width() == 0.0)
    return true;

  pen pen0=pentype;
  adjustdash(pen0);

  penSave(out);
  penTranslate(out);

  out->write(p);

  penConcat(out);

  out->setpen(pen0);
  
  out->stroke();

  penRestore(out);

  return true;
}

drawElement *drawPath::transformed(const transform& t)
{
  return new drawPath(transpath(t), transpen(t));
}

} //namespace camp
