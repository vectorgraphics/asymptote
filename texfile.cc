/*****
 * texfile.cc
 * John Bowman 2003/03/14
 *
 * Encapsulates the writing of commands to a TeX file.
 *****/

#include <ctime>

#include "texfile.h"
#include "errormsg.h"

namespace camp {

using std::ofstream;
using std::fixed;
using std::setprecision;
  
bool TeXcontaminated=false;
std::list<std::string> TeXpreamble;
  
texfile::texfile(const string& texname, const bbox& box) :
    box(box)
{
  out=new ofstream(texname.c_str());
  if(!out || !*out) {
    std::cerr << "Can't write to " << texname << std::endl;
    throw handled_error();
  }
  texdocumentclass(*out);
  TeXcontaminated=false;
  lastpen=pen(initialpen);
  lastpen.convert(); 
}

texfile::~texfile()
{
  if(out) delete out;  
}
  
void texfile::prologue()
{
  texpreamble(*out);
  *out << "\\usepackage{graphicx}" << newl
       << "\\usepackage{pstricks}" << newl
       << "\\psset{unit=1pt}" << newl
       << "\\pagestyle{empty}" << newl
       << "\\textheight=2048pt" << newl
       << "\\textwidth=\\textheight" << newl
       << "\\begin{document}" << newl;
}
    
void texfile::beginlayer(const string& psname)
{
  *out << "\\setbox\\ASYbox=\\hbox{\\includegraphics{" << psname << "}}%"
       << newl
       << "\\includegraphics{" << psname << "}%" << newl;
}

void texfile::endlayer()
{
  *out << "\\kern-\\wd\\ASYbox%" << newl;
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
  } else if(p.mono() && (!lastpen.mono() || p.gray() != lastpen.gray())) {
    *out << "\\newgray{ASYcolor}{" 
	 << p.gray()
	 << "}\\ASYcolor" << newl;
  }
  
  if(p.size() != lastpen.size()) {
    *out << "\\fontsize{" << p.size() << "}{" << p.size()*1.2
	 << "}\\selectfont" << newl;
  }

  offset=pair(box.right,box.bottom);
  
  lastpen=p;
}
   
void texfile::put(const string& label, double angle, pair z)
{
  *out << "\\rput[lB]{" << setprecision(2) << angle << fixed
       << "}(" << (z.getx()-offset.getx())*ps2tex
       << "," << (z.gety()-offset.gety())*ps2tex
       << "){" << label << "}" << newl;
}

void texfile::epilogue()
{
  *out << "\\end{document}" << newl;
  out->flush();
}

} //namespace camp
