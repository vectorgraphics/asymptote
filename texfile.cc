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
  if(settings::latex(texengine)) {
    if(!getSetting<bool>("inlinetex")) {
      *out << "\\pagestyle{empty}" << newl
	   << "\\textheight=2048pt" << newl
	   << "\\textwidth=\\textheight" << newl;
      if(settings::pdf(texengine))
	*out << "\\oddsidemargin=-17.61pt" << newl
	     << "\\evensidemargin=\\oddsidemargin" << newl
	     << "\\topmargin=-37.01pt" << newl
	     << "\\pdfhorigin=0bp" << newl
	     << "\\pdfvorigin=0bp" << newl
	     << "\\pdfpagewidth=" << box.right-box.left << "bp" << newl
	     << "\\pdfpageheight=" << box.top-box.bottom << "bp" << newl;
      *out << "\\begin{document}" << newl;
    }
  } else {
    *out << "\\hoffset=36.6pt" << newl
	 << "\\voffset=54.0pt" << newl
	 << "\\input graphicx" << newl
	 << "\\input picture" << newl;
    if(settings::pdf(texengine)) {
      *out << "\\hoffset=-20pt" << newl
	   << "\\voffset=0pt" << newl
	   << "\\pdfhorigin=0bp" << newl
	   << "\\pdfvorigin=0bp" << newl
	   << "\\pdfpagewidth=" << box.right-box.left << "bp" << newl
	   << "\\pdfpageheight=" << box.top-box.bottom << "bp" << newl;
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

  setcolor(p,settings::beginspecial(texengine),settings::endspecial());
  
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
		  const pair& align)
{
  double sign=settings::pdf(texengine) ? 1.0 : -1.0;

  if(label.empty()) return;
  
  *out << "\\ASYalign"
       << "(" << (z.getx()-box.right)*ps2tex
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
    if(!getSetting<bool>("inlinetex"))
      *out << "\\end{document}" << newl;
  } else {
      *out << "\\bye" << newl;
  }
  out->flush();
}

} //namespace camp
