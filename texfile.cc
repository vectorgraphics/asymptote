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
using settings::getSetting;
  
namespace camp {

std::list<string> TeXpipepreamble, TeXpreamble;
  
texfile::texfile(const string& texname, const bbox& box) : box(box)
{
  out=new ofstream(texname.c_str());
  if(!out || !*out) {
    std::cerr << "Cannot write to " << texname << std::endl;
    throw handled_error();
  }
  out->setf(std::ios::fixed);
  out->precision(6);
  texdocumentclass(*out);
  lastpen=pen(initialpen);
  lastpen.convert(); 
}

texfile::~texfile()
{
  if(out) {
    delete out;  
    out=NULL;
  }
}
  
void texfile::prologue()
{
  texdefines(*out);
  if(!getSetting<bool>("inlinetex"))
    *out << "\\usepackage{color}" << newl
	 << "\\pagestyle{empty}" << newl
	 << "\\textheight=2048pt" << newl
	 << "\\textwidth=\\textheight" << newl
	 << "\\begin{document}" << newl;
}
    
void texfile::beginlayer(const string& psname)
{
  mem::string texengine=getSetting<mem::string>("tex");
  if(box.right > box.left && box.top > box.bottom) {
    *out << "\\includegraphics";
    if(!settings::pdf(texengine))
      *out << "[bb=" << box.left << " " << box.bottom << " "
	   << box.right << " " << box.top << "]";
    *out << "{" << psname << "}%" << newl;
  }
}

void texfile::endlayer()
{
  if(box.right > box.left && box.top > box.bottom)
    *out << "\\kern-" << (box.right-box.left)*ps2tex << "pt%" << newl;
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
    *out << "\\definecolor{ASYcolor}{cmyk}{" 
	 << p.cyan() << "," << p.magenta() << "," << p.yellow() << "," 
	 << p.black() << "}\\color{ASYcolor}" << newl;
  } else if(p.rgb() && (!lastpen.rgb() ||
			(p.red() != lastpen.red() ||
			 p.green() != lastpen.green() || 
			 p.blue() != lastpen.blue()))) {
    *out << "\\definecolor{ASYcolor}{rgb}{" 
	 << p.red() << "," << p.green() << "," << p.blue()
	 << "}\\color{ASYcolor}" << newl;
  } else if(p.grayscale() && (!lastpen.grayscale() || 
			      p.gray() != lastpen.gray())) {
    *out << "\\definecolor{ASYcolor}{gray}{" 
	 << p.gray()
	 << "}\\color{ASYcolor}" << newl;
  }
  
  if(p.size() != lastpen.size() || p.Lineskip() != lastpen.Lineskip()) {
    *out << "\\fontsize{" << p.size() << "}{" << p.Lineskip()
	 << "}\\selectfont" << newl;
  }

  if(p.Font() != lastpen.Font()) {
    *out << p.Font() << "%" << newl;
  }

  lastpen=p;
}
   
void texfile::gsave()
{
  *out << "\\ASYgsave" << newl;
}
  
void texfile::grestore()
{
    *out << "\\ASYgrestore" << newl;
}
  
void texfile::openclip() 
{
  *out << "\\ASYclip(" << -box.right*ps2tex
      << "," << -box.bottom*ps2tex
      << "){" << newl;
}
  
void texfile::closeclip() 
{
  *out << "}" << newl;
}
  
void texfile::put(const string& label, const transform& T, const pair& z,
		  const pair& align, const bbox& Box)
{
  mem::string texengine=settings::getSetting<mem::string>("tex");
  
  transform T0=(texengine == "pdflatex") ? inverse(T) : T;

  if(label.empty()) return;
  
  *out << "\\ASYalign"
       << "(" << (z.getx()-box.right)*ps2tex
       << "," << (z.gety()-box.bottom)*ps2tex
       << ")(" << align.getx()
       << "," << align.gety() 
       << "){" << T0.getxx() << " " << -T0.getyx()
       << " " << -T0.getxy() << " " << T0.getyy()
       << "}{" << label << "}" << newl;
}

void texfile::epilogue()
{
  if(!getSetting<bool>("inlinetex")) *out << "\\end{document}" << newl;
  out->flush();
}

} //namespace camp
