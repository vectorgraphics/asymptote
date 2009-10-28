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
using vm::array;
using vm::read;

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
  level=0;
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
  if(settings::latex(texengine)) {
    *out << "\\pagestyle{empty}" << newl
         << "\\textheight=2048pt" << newl
         << "\\textwidth=2048pt" << newl
         << "\\begin{document}" << newl;
  } else if(settings::context(texengine)) {
    *out << "\\setuppagenumbering[location=]" << newl
         << "\\usetypescript[modern]" << newl
         << "\\starttext\\hbox{%" << newl;
  }
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
             << "\\setuppagenumbering[location=]" << endl
             << "\\usetypescript[modern]" << newl
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
        *out << "\\kern " << (box.left-box.right)*ps2tex << "pt%" << newl;
    } else {
      *out << "\\leavevmode\\vbox to " << (box.top-box.bottom)*ps2tex 
           << "pt{}%" << newl;
      if(inlinetex)
        *out << "\\kern " << (box.right-box.left)*ps2tex << "pt%" << newl;
    }
  }
}

void texfile::endlayer()
{
  if(inlinetex && (box.right > box.left && box.top > box.bottom))
    *out << "\\kern " << (box.left-box.right)*ps2tex << "pt%" << newl;
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
  if(settings::latex(texengine))
    *out << "\\end{document}" << newl;
  else if(settings::context(texengine))
    *out << "}\\stoptext" << newl;
  else
    *out << "\\bye" << newl;
  out->flush();
}

#ifdef HAVE_DVISVGM_NL
string svgtexfile::nl="{?nl}";
#else
string svgtexfile::nl="";
#endif

void svgtexfile::beginspecial()
{
  out->unsetf(std::ios::fixed);
  *out << "\\catcode`\\#=11%" << newl
       << "\\special{dvisvgm:raw <g transform='matrix(1 0 0 1 "
       << (-Hoffset+1.99*settings::cm)*ps2tex << " " 
       << (1.9*settings::cm+box.top)*ps2tex 
       << ")'>" << nl;
}
    
void svgtexfile::endspecial()
{
  *out << "</g>}\\catcode`\\#=6%" << newl;
  out->setf(std::ios::fixed);
}
  
void svgtexfile::gsave(bool tex)
{
  if(clipstack.size() < 1)
    clipstack.push(0);
  else
    clipstack.push(clipcount);
  *out << "\\special{dvisvgm:raw <g>}%" << newl;
  pens.push(lastpen);
}
  
void svgtexfile::grestore(bool tex)
{
  if(pens.size() < 1 || clipstack.size() < 1)
    reportError("grestore without matching gsave");
  lastpen=pens.top();
  pens.pop();
  clipstack.pop();
  *out << "\\special{dvisvgm:raw </g>}%" << newl;
}

void svgtexfile::clippath()
{
  if(clipstack.size() > 0) {
    size_t count=clipstack.top();
    if(count > 0)
      *out << "clip-path='url(#clip" << count << ")' ";
  }
}
  
void svgtexfile::beginpath()
{
  *out << "<path ";
  clippath();
  *out << "d='";
}
  
void svgtexfile::endpath()
{
  *out << "/>" << nl;
}
  
void svgtexfile::beginclip()
{
  beginspecial();
  *out << "<clipPath ";
  clippath();
  ++clipcount;
  *out << "id='clip" << clipcount << "'>" << nl;
  beginpath();
  if(clipstack.size() >= 0)
    clipstack.pop();
  clipstack.push(clipcount);
}
  
void svgtexfile::endclip(const pen &p) 
{
  *out << "'";
  fillrule(p,"clip");
  endpath();
  *out << "</clipPath>" << nl;
  endspecial();
}

void svgtexfile::fillrule(const pen& p, const string& type)
{
  if(p.Fillrule() != lastpen.Fillrule())
    *out << " " << type << "-rule='" << 
      (p.evenodd() ? "evenodd" : "nonzero") << "'";
  lastpen=p;
}
   
void svgtexfile::color(const pen &p, const string& type)
{
  *out << "' " << type << "='#" << rgbhex(p) << "'";
  double opacity=p.opacity();
  if(opacity != 1.0)
    *out << " opacity='" << opacity << "'";
}

void svgtexfile::fill(const pen &p)
{
  color(p,"fill");
  fillrule(p);
  endpath();
  endspecial();
}

void svgtexfile::properties(const pen& p)
{
  if(p.cap() != lastpen.cap())
    *out << " stroke-linecap='" << PSCap[p.cap()] << "'";
    
  if(p.join() != lastpen.join())
    *out << " stroke-linejoin='" << Join[p.join()] << "'";
  
  if(p.miter() != lastpen.miter())
    *out << " stroke-miterlimit='" << p.miter()*ps2tex << "'";
    
  if(p.width() != lastpen.width())
    *out << " stroke-width='" << p.width()*ps2tex << "'";
  
  const LineType *linetype=p.linetype();
  const LineType *lastlinetype=lastpen.linetype();
  
  if(!(linetype->pattern == lastlinetype->pattern)) {
    size_t n=linetype->pattern.size();
    if(n > 0) {
      *out << " stroke-dasharray='";
      *out << vm::read<double>(linetype->pattern,0)*ps2tex;
      for(size_t i=1; i < n; ++i)
        *out << "," << vm::read<double>(linetype->pattern,i)*ps2tex;
      *out << "'";
    }
  }
  
  if(linetype->offset != lastlinetype->offset)
    *out << " stroke-dashoffset='" << linetype->offset*ps2tex << "'";
  
  lastpen=p;
}
  
void svgtexfile::stroke(const pen &p)
{
  color(p,"fill='none' stroke");
  properties(p);  
  endpath();
  endspecial();
}
  
void svgtexfile::strokepath()
{
  reportWarning("SVG does not support strokepath");
}

void svgtexfile::begintensorshade(const vm::array& pens,
                                  const vm::array& boundaries,
                                  const vm::array& z) 
{
  out->unsetf(std::ios::fixed);
  *out << "\\catcode`\\#=11%" << newl
       << "\\special{dvisvgm:raw <defs>" << nl;

  path g=read<path>(boundaries,0);
  pair Z[]={g.point((Int) 0),g.point((Int) 3),g.point((Int) 2),
            g.point((Int) 1)};
      
  array *pi=read<array *>(pens,0);
  if(checkArray(pi) != 4)
    reportError("specify 4 pens for each path");
  string hex[]={rgbhex(read<pen>(pi,0)),rgbhex(read<pen>(pi,3)),
                rgbhex(read<pen>(pi,2)),rgbhex(read<pen>(pi,1))};
    
  *out << "<filter id='colorAdd'>" << nl
       << "<feBlend in='SourceGraphic' in2='BackgroundImage' operator='arithmetic' k2='1' k3='1'/>" << nl
       << "</filter>";

  pair mean=0.25*(Z[0]+Z[1]+Z[2]+Z[3]);
  for(size_t k=0; k < 4; ++k) {
    pair opp=(k % 2 == 0) ? Z[(k+2) % 4] : mean;
    *out << "<linearGradient id='grad" << tensorcount << "-" << k 
         << "' gradientUnits='userSpaceOnUse'" << nl
         << " x1='" << Z[k].getx()*ps2tex << "' y1='" << -Z[k].gety()*ps2tex
         << "' x2='" << opp.getx()*ps2tex << "' y2='" << -opp.gety()*ps2tex
         << "'>" << nl
         << "<stop offset='0' stop-color='#" << hex[k] 
         << "' stop-opacity='1'/>" << nl
         << "<stop offset='1' stop-color='#" << hex[k] 
         << "' stop-opacity='0'/>" << nl
         << "</linearGradient>" << nl;
  }
  *out << "}\\catcode`\\#=6%" << newl;
}

void svgtexfile::tensorshade(const pen& pentype, const vm::array& pens,
                             const vm::array& boundaries, const vm::array& z)
{
  size_t size=pens.size();
  if(size == 0) return;
  
  *out << "' id='path" << tensorcount << "'";
  fillrule(pentype);
  endpath();
  *out << "</g>";
  *out << "</defs>";
  *out << "<g transform='matrix(1 0 0 1 "
         << (-Hoffset+1.99*settings::cm)*ps2tex << " " 
         << (1.9*settings::cm+box.top)*ps2tex 
         << ")'>" << nl;
  for(size_t i=0; i < size; i++) {
    *out << "<use xlink:href='#path" << tensorcount
         << "' fill='url(#grad" << tensorcount << "-" 
         << "0)'/>" << nl
         << "<use xlink:href='#path" << tensorcount
         << "' fill='url(#grad" << tensorcount << "-" 
         << "2)' filter='url(#colorAdd)'/>"
         << "<use xlink:href='#path" << tensorcount
         << "' fill='url(#grad" << tensorcount << "-" 
         << "1)' filter='url(#colorAdd)'/>"
         << "<use xlink:href='#path" << tensorcount 
         << "' fill='url(#grad" << tensorcount << "-"
         << "3)' filter='url(#colorAdd)'/>";
  }
  
  ++tensorcount;
  endspecial();
}

void svgtexfile::begingradientshade(bool axial, ColorSpace colorspace,
                                    const pen& pena, const pair& a, double ra,
                                    const pen& penb, const pair& b, double rb)
{
  string type=axial ? "linear" : "radial";
  out->unsetf(std::ios::fixed);
  *out << "\\catcode`\\#=11%" << newl <<
    "\\special{dvisvgm:raw <" << type << "Gradient id='grad" << gradientcount;
  if(axial) {
    *out << "' x1='" << a.getx()*ps2tex << "' y1='" << -a.gety()*ps2tex
         << "' x2='" << b.getx()*ps2tex << "' y2='" << -b.gety()*ps2tex;
  } else {
    *out << "' cx='" << b.getx()*ps2tex << "' cy='" << -b.gety()*ps2tex
         << "' r='" << rb*ps2tex;
  }
  *out <<"' gradientUnits='userSpaceOnUse'>" << nl
       << "<stop offset='0' stop-color='#" << rgbhex(pena) << "'/>" << nl
       << "<stop offset='1' stop-color='#" << rgbhex(penb) << "'/>" << nl
       << "</" << type << "Gradient>}%" << newl <<
    "\\catcode`\\#=6%" << newl;
}

void svgtexfile::gradientshade(bool axial, ColorSpace colorspace,
                            const pen& pena, const pair& a, double ra,
                            const pen& penb, const pair& b, double rb)
{
  *out << "' fill='url(#grad" << gradientcount << ")'";
  fillrule(pena);
  endpath();
  ++gradientcount;
  endspecial();
}
  
} //namespace camp
