/*****
 * texfile.cc
 * John Bowman 2003/03/14
 *
 * Encapsulates the writing of commands to a TeX file.
 *****/

#include <ctime>
#include <cfloat>

#include "texfile.h"
#include "errormsg.h"

using std::ofstream;
using std::fixed;
using std::setprecision;
using settings::texmode;
  
namespace camp {

stringlist TeXpipepreamble, TeXpreamble;
  
texfile::texfile(const string& texname, const bbox& box) :
    box(box)
{
  out=new ofstream(texname.c_str());
  if(!out || !*out) {
    std::cerr << "Can't write to " << texname << std::endl;
    throw handled_error();
  }
  texdocumentclass(*out);
  lastpen=pen(initialpen);
  lastpen.convert(); 
}

texfile::~texfile()
{
  if(out) delete out;  
}
  
void texfile::prologue()
{
  texdefines(*out);
  if(!texmode)
    *out << "\\pagestyle{empty}" << newl
	 << "\\textheight=2048pt" << newl
	 << "\\textwidth=\\textheight" << newl
	 << "\\begin{document}" << newl;
  *out << "\\psset{unit=1pt}" << newl;
}
    
void texfile::beginlayer(const string& psname)
{
  *out << "\\setbox\\ASYpsbox=\\hbox{\\includegraphics{" << psname << "}}%"
       << newl
       << "\\includegraphics{" << psname << "}%" << newl;
}

void texfile::endlayer()
{
  *out << "\\kern-\\wd\\ASYpsbox%" << newl;
}
  

void texfile::setpen(pen p)
{
  p.convert();
  if(p == lastpen) return;

  if(p.cmyk() && (!lastpen.cmyk() || 
		  (p.cyan() != lastpen.cyan() || 
		   p.magenta() != lastpen.magenta() || 
		   p.yellow() != lastpen.yellow() ||
		   p.black() != lastpen.black()))) {
    *out << "\\newcmykcolor{ASYcolor}{" 
	 << p.cyan() << " " << p.magenta() << " " << p.yellow() << " " 
	 << p.black() << "}\\ASYcolor" << newl;
  } else if(p.rgb() && (!lastpen.rgb() ||
			(p.red() != lastpen.red() ||
			 p.green() != lastpen.green() || 
			 p.blue() != lastpen.blue()))) {
    *out << "\\newrgbcolor{ASYcolor}{" 
	 << p.red() << " " << p.green() << " " << p.blue()
	 << "}\\ASYcolor" << newl;
  } else if(p.grayscale() && (!lastpen.grayscale() || 
			      p.gray() != lastpen.gray())) {
    *out << "\\newgray{ASYcolor}{" 
	 << p.gray()
	 << "}\\ASYcolor" << newl;
  }
  
  if(p.size() != lastpen.size() || p.Lineskip() != lastpen.Lineskip()) {
    *out << "\\fontsize{" << p.size() << "}{" << p.Lineskip()
	 << "}\\selectfont" << newl;
  }

  if(p.Font() != lastpen.Font()) {
    *out << p.Font() << "%" << newl;
  }

  offset=pair(box.right,box.bottom);
  
  lastpen=p;
}
   
void texfile::put(const string& label, double angle, const pair& z,
		  const pair& align, const pair& scale, const bbox& Box)
{
  if(label.empty()) return;
  
  for(cliplist::iterator p=clipstack.begin(); p != clipstack.end(); ++p) {
    assert(*p);
    if((*p)->rule.overlap()) {
      if(!(*p)->inside(0.5*(Box.Min()+Box.Max()))) return;
    } else {
      // Include labels exactly on boundary.
      static const double fuzz=10.0*DBL_EPSILON;
      double xfuzz=fuzz*max(fabs(Box.left),fabs(Box.right));
      double yfuzz=fuzz*max(fabs(Box.bottom),fabs(Box.top));
      if(!(*p)->inside(pair(Box.left+xfuzz,Box.bottom+yfuzz))) return;
      if(!(*p)->inside(pair(Box.right-xfuzz,Box.bottom+yfuzz))) return;
      if(!(*p)->inside(pair(Box.left+xfuzz,Box.top-yfuzz))) return;
      if(!(*p)->inside(pair(Box.right-xfuzz,Box.top-yfuzz))) return;
    }
  }
  
  static pair unscaled=pair(1.0,1.0);
  bool scaled=(scale != unscaled);
  *out << "\\ASY" << (scaled ? "scale" : "align") << fixed << setprecision(6)
       << "(" << (z.getx()-offset.getx())*ps2tex
       << "," << (z.gety()-offset.gety())*ps2tex
       << ")(" << align.getx()
       << "," << align.gety();
  if(scaled)
    *out << ")(" << scale.getx()
	 << "," << scale.gety();
  *out << "){" << angle
       << "}{" << label << "}" << newl;
}

void texfile::epilogue()
{
  if(!texmode) *out << "\\end{document}" << newl;
  out->flush();
}

} //namespace camp
