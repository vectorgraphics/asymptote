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

texfile::texfile(const string& texname, const bbox& box, bool pipe) 
  : box(box)
{
  texengine=getSetting<string>("tex");
  inlinetex=getSetting<bool>("inlinetex");
  Hoffset=inlinetex ? box.right : box.left;
  out=new ofstream(texname.c_str());
  if(!out || !*out)
    reportError("Cannot write to "+texname);
  out->setf(std::ios::fixed);
  out->precision(6);
  texdocumentclass(*out,pipe);
  resetpen();
}

texfile::~texfile()
{
  if(out) {
    delete out;  
    out=NULL;
  }
}
  
void texfile::miniprologue()
{
  texpreamble(*out,processData().TeXpreamble,false,true);
  *out << "\\pagestyle{empty}" << newl;
  *out << "\\begin{document}" << newl;
  texfontencoding(*out);
}

void texfile::prologue()
{
  if(inlinetex) {
    string prename=auxname(getSetting<string>("outname"),"pre");
    std::ifstream exists(prename.c_str());
    std::ofstream *outpreamble=
      new std::ofstream(prename.c_str(),std::ios::app);
    bool ASYdefines=!exists;
    texpreamble(*outpreamble,processData().TeXpreamble,ASYdefines,ASYdefines);
    outpreamble->close();
  }
  
  texdefines(*out,processData().TeXpreamble,false);
  double width=box.right-box.left;
  double height=box.top-box.bottom;
  if(!inlinetex) {
    if(settings::context(texengine)) {
      *out << "\\definepapersize[asy][width=" << width << "bp,height=" 
           << height << "bp]" << newl
           << "\\setuppapersize[asy][asy]" << newl;
    } else if(settings::pdf(texengine)) {
      double voffset=0.0;
      if(settings::latex(texengine)) {
        if(height < 12.0) voffset=height-12.0;
      } else if(height < 10.0) voffset=height-10.0;

      // Work around an apparent xelatex dimension bug
      double xelatexBug=ps2tex;

      if(width > 0) 
        *out << "\\pdfpagewidth=" << width << "bp" << newl;
      *out << "\\ifx\\pdfhorigin\\undefined" << newl
           << "\\hoffset=-1in" << newl
           << "\\voffset=" << voffset-72.0*xelatexBug << "bp" << newl;
      if(height > 0)
        *out << "\\pdfpageheight=" << height*0.5*(1.0+xelatexBug) << "bp" 
             << newl;
      *out << "\\else" << newl
           << "\\pdfhorigin=0bp" << newl
           << "\\pdfvorigin=" << voffset << "bp" << newl;
      if(height > 0)
        *out << "\\pdfpageheight=" << height << "bp" << newl;
      *out << "\\fi" << newl;
    }
  }
  
  if(settings::latex(texengine)) {
    *out << "\\setlength{\\unitlength}{1pt}" << newl;
    if(!inlinetex) {
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
    if(!inlinetex) {
      if(settings::context(texengine)) {
        *out << "\\setuplayout[width=16383pt,height=16383pt,"
             << "backspace=0pt,topspace=0pt,"
             << "header=0pt,headerdistance=0pt,footer=0pt]" << newl
             << "\\starttext\\hbox{%" << newl;
      } else {
        *out << "\\footline={}" << newl;
        if(settings::pdf(texengine)) {
          *out << "\\hoffset=-20pt" << newl
               << "\\voffset=0pt" << newl;
        } else {
          *out << "\\hoffset=36.6pt" << newl
               << "\\voffset=54.0pt" << newl;
        }
      }
    }
  }
}
    
void texfile::beginlayer(const string& psname, bool postscript)
{
  const char *units=settings::texunits(texengine);
  if(box.right > box.left && box.top > box.bottom) {
    if(postscript) {
      if(settings::context(texengine))
        *out << "\\externalfigure[" << psname << "]%" << newl;
      else {
        *out << "\\includegraphics";
        if(!settings::pdf(texengine))
          *out << "[bb=" << box.left << " " << box.bottom << " "
               << box.right << " " << box.top << "]";
        *out << "{" << psname << "}%" << newl;
      }
      if(!inlinetex)
        *out << "\\kern " << (box.left-box.right)*ps2tex << units
             << "%" << newl;
    } else {
      *out << "\\leavevmode\\vbox to " << (box.top-box.bottom)*ps2tex 
           << units << "{}%" << newl;
      if(inlinetex)
        *out << "\\kern " << (box.right-box.left)*ps2tex << units
             << "%" << newl;
    }
  }
}

void texfile::endlayer()
{
  if(inlinetex && (box.right > box.left && box.top > box.bottom))
    *out << "\\kern " << (box.left-box.right)*ps2tex
         << settings::texunits(texengine) << "%" << newl;
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
  
void texfile::setfont(pen p)
{
  bool latex=settings::latex(texengine);
  
  if(latex) setlatexfont(*out,p,lastpen);
  settexfont(*out,p,lastpen,latex);
  
  lastpen=p;
}
  
void texfile::setpen(pen p)
{
  bool latex=settings::latex(texengine);
  
  p.convert();
  if(p == lastpen) return;

  if(latex) setlatexcolor(p);
  else setcolor(p,settings::beginspecial(texengine),settings::endspecial());
  
  setfont(p);
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
  
  bool trans=!T.isIdentity();
  
  *out << "\\ASYalign";
  if(trans) *out << "T";
  *out << "(" << (z.getx()-Hoffset)*ps2tex
       << "," << (z.gety()-box.bottom)*ps2tex
       << ")(" << align.getx()
       << "," << align.gety() 
       << ")";
  if(trans)
    *out << "{" << T.getxx() << " " << sign*T.getyx()
         << " " << sign*T.getxy() << " " << T.getyy() << "}";
  *out << "{" << label << "}" << newl;
}

void texfile::epilogue(bool pipe)
{
  if(settings::latex(texengine)) {
    if(!inlinetex || pipe)
      *out << "\\end{document}" << newl;
  } else {
    if(settings::context(texengine))
      *out << "}\\stoptext" << newl;
    *out << "\\bye" << newl;
  }
  out->flush();
}

} //namespace camp
