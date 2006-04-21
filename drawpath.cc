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

double PatternLength(double arclength, std::vector<double>& pat,
		     const pen& pen0, const path& p)
{
  double sum=0.0;
  double penwidth=pen0.linetype().scale ? pen0.width() : 1.0;
      
  size_t n=pat.size();
  for(unsigned i=0; i < n; i ++) {
    pat[i] *= penwidth;
    sum += pat[i];
  }
  if(sum == 0.0) return 0.0;
  
  if(n % 2 == 1) sum *= 2.0; // On/off pattern repeats after 2 cycles.
      
  // Fix bounding box resolution problem. Example:
  // asy -f pdf testlinetype; gv -scale -2 testlinetype.pdf
  if(!p.cyclic() && pat[0] == 0) sum += 1.0e-3*penwidth;
      
  double terminator=((p.cyclic() && arclength >= 0.5*sum) ? 0.0 : pat[0]);
  int ncycle=(int)((arclength-terminator)/sum+0.5);

  return ncycle > 0 ? ncycle*sum+terminator : 0.0;
}

void drawPath::adjustdash(pen& pen0)
{
  // Adjust dash sizes to fit arclength; also compensate for linewidth.
  string stroke=pen0.stroke();
  if(!stroke.empty() && pen0.linetype().adjust) {
    double arclength=p.arclength();
    
    if(arclength) {
      std::vector<double> pat;
      {
        istringstream buf(stroke);
        double l;
        while(buf >> l) {
	  if(l < 0) l=0;
          pat.push_back(l);
        }
      }
      
      double denom=PatternLength(arclength,pat,pen0,p);
      
      if(denom == 0.0) return; // Otherwise, we know n > 0.
      
      double factor=denom != 0.0 ? arclength/denom : 1.0;
      ostringstream buf;
      size_t n=pat.size();
      for(unsigned int i=0; i < n; i++) buf << pat[i]*factor << " ";
      pen0.setstroke(buf.str());
    }
  }
}
  
void drawPath::bounds(bbox& b, iopipestream&, boxvector&, bboxlist&)
{
  b += pad(p.bounds(),pentype.bounds());
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
