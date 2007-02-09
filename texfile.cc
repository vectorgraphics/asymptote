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
  texengine=getSetting<mem::string>("tex");
  inlinetex=getSetting<bool>("inlinetex");
  Hoffset=inlinetex ? box.right : box.left;
  out=new ofstream(texname.c_str());
  if(!out || !*out) {
    std::cerr << "Cannot write to " << texname << std::endl;
    throw handled_error();
  }
  out->setf(std::ios::fixed);
  out->precision(6);
  texdocumentclass(*out);
  resetpen();
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
  double width=box.right-box.left;
  double height=box.top-box.bottom;
  if(settings::pdf(texengine) && !inlinetex) {
    *out << "\\pdfhorigin=0bp" << newl
	 << "\\pdfvorigin=0bp" << newl;
    if(width > 0) 
      *out << "\\pdfpagewidth=" << width << "bp" << newl;
    if(height > 0)
      *out << "\\pdfpageheight=" << height << "bp" << newl;
  }
  if(settings::latex(texengine)) {
    if(inlinetex)
      *out << "\\setlength{\\unitlength}{1pt}" << newl;
    else {
      *out << "\\pagestyle{empty}" << newl
	   << "\\textheight=" << height+18.0 << "bp" << newl
	   << "\\textwidth=" << width+18.0 << "bp" << newl;
      if(settings::pdf(texengine))
	*out << "\\oddsidemargin=-17.61pt" << newl
	     << "\\evensidemargin=\\oddsidemargin" << newl
	     << "\\topmargin=-37.01pt" << newl;
      *out << "\\begin{document}" << newl;
    }
  } else {
    *out << "\\hoffset=36.6pt" << newl
	 << "\\voffset=54.0pt" << newl;
    if(settings::pdf(texengine)) {
      *out << "\\hoffset=-20pt" << newl
	   << "\\voffset=0pt" << newl;
    }
  }
}
    
void texfile::beginlayer(const string& psname)
{
  if(box.right > box.left && box.top > box.bottom) {
    *out << "\\includegraphics";
    if(!settings::pdf(texengine))
      *out << "[bb=" << box.left << " " << box.bottom << " "
	   << box.right << " " << box.top << "]";
    *out << "{" << psname << "}%" << newl;
    if(!inlinetex)
      *out << "\\kern-" << (box.right-box.left)*ps2tex << "pt%" << newl;
  }
}

void texfile::endlayer()
{
  if(inlinetex && (box.right > box.left && box.top > box.bottom))
    *out << "\\kern-" << (box.right-box.left)*ps2tex << "pt%" << newl;
}

void texfile::writeshifted(path p, bool newPath)
{
  write(p.transformed(shift(pair(-Hoffset,-box.bottom))),newPath);
}

void texfile::setlatexcolor(pen p)
{
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
}
  
void texfile::setpen(pen p)
{
  bool latex=settings::latex(texengine);
  
  p.convert();
  if(p == lastpen) return;

  if(latex) setlatexcolor(p);
  else setcolor(p,settings::beginspecial(texengine),settings::endspecial());
  
  if((p.size() != lastpen.size() || p.Lineskip() != lastpen.Lineskip()) &&
     settings::latex(texengine)) {
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
  *out << settings::beginspecial(texengine);
  psfile::gsave(true);
  *out << settings::endspecial() << newl;
}

void texfile::grestore()
{
  *out << settings::beginspecial(texengine);
  psfile::grestore(true);
  *out << settings::endspecial() << newl;
}

void texfile::beginspecial() 
{
  *out << settings::beginspecial(texengine);
}
  
void texfile::endspecial() 
{
  *out << settings::endspecial() << newl;
}
  
void texfile::beginraw() 
{
  *out << "\\ASYraw{" << newl;
}
  
void texfile::endraw() 
{
  *out << "}%" << newl;
}
  
void texfile::put(const string& label, const transform& T, const pair& z,
		  const pair& align)
{
  double sign=settings::pdf(texengine) ? 1.0 : -1.0;

  if(label.empty()) return;
  
  *out << "\\ASYalign"
       << "(" << (z.getx()-Hoffset)*ps2tex
       << "," << (z.gety()-box.bottom)*ps2tex
       << ")(" << align.getx()
       << "," << align.gety() 
       << "){" << T.getxx() << " " << sign*T.getyx()
       << " " << sign*T.getxy() << " " << T.getyy()
       << "}{" << label << "}" << newl;
}

void texfile::epilogue()
{
  if(settings::latex(texengine)) {
    if(!inlinetex)
      *out << "\\end{document}" << newl;
  } else {
      *out << "\\bye" << newl;
  }
  out->flush();
}

} //namespace camp
