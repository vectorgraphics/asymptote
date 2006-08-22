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

using std::ostringstream;
using std::istringstream;

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
    
    ostringstream buf;
    for(unsigned int i=0; i < n; i++)
      buf << pat[i]*factor << " ";
    pen0.setstroke(buf.str());
    pen0.setoffset(pen0.linetype().offset*factor);
  }
}
  
void drawPath::addcap(bbox& b, const path& p, double t, const pair& dir)
{
  double h=0.5*pentype.width();
  pair z=p.point(t);
  pair v=unit(p.direction(t))*h;
  b += z+dir*v;
  b += z+conj(dir)*v;
}
  
void drawPath::bounds(bbox& b, iopipestream&, boxvector&, bboxlist&)
{
  b += p.bounds(pentype.bounds());
  if(p.cyclic()) return;
  
  int l=p.length();
  switch(pentype.cap()) {
  case 0:
    {
      addcap(b,p,0,pair(0,1));
      addcap(b,p,l,pair(0,1));
      break;
    }
  case 1:
    {
      double h=0.5*pentype.width();
      pair H=pair(h,h);
      pair z0=p.point(0);
      pair zl=p.point(l);
      b += z0+H;
      b += z0-H;
      b += zl+H;
      b += zl-H;
      break;
    }
  case 2:
    {
    addcap(b,p,0,pair(-1,1));
    addcap(b,p,l,pair(1,1));
    break;
    }
  } 
}

bool drawPath::draw(psfile *out)
{
  int n = p.size();
  if (n == 0 || pentype.invisible())
    return true;

  pen pen0=pentype;
  adjustdash(pen0);
  out->setpen(pen0);

  if (pentype.width() == 0.0)
    return true;
    
  penStart(out);
  penTranslate(out);

  out->write(p);

  penConcat(out);

  out->stroke();

  penEnd(out);

  return true;
}

drawElement *drawPath::transformed(const transform& t)
{
  return new drawPath(transpath(t), transpen(t));
}

} //namespace camp
