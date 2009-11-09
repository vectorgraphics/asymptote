/*****
 * texfile.h
 * John Bowman 2003/03/14
 *
 * Encapsulates the writing of commands to a TeX file.
 *****/

#ifndef TEXFILE_H
#define TEXFILE_H

#include <fstream>
#include <iomanip>
#include <iostream>

#include "common.h"
#include "pair.h"
#include "bbox.h"
#include "pen.h"
#include "util.h"
#include "interact.h"
#include "path.h"
#include "array.h"
#include "psfile.h"
#include "settings.h"
#include "process.h"

namespace camp {

template<class T>
void texdocumentclass(T& out, bool pipe=false)
{
  if(settings::latex(settings::getSetting<string>("tex")) &&
     (pipe || !settings::getSetting<bool>("inlinetex")))
    out << "\\documentclass[12pt]{article}" << newl;
}
  
template<class T>
void texuserpreamble(T& out,
                     mem::list<string>& preamble=processData().TeXpreamble)
{
  for(mem::list<string>::iterator p=preamble.begin();
      p != preamble.end(); ++p)
    out << stripblanklines(*p);
}
  
template<class T>
void texfontencoding(T& out) 
{
  if(settings::latex(settings::getSetting<string>("tex"))) {
    out << "\\makeatletter%" << newl
        << "\\let\\ASYencoding\\f@encoding%" << newl
        << "\\let\\ASYfamily\\f@family%" << newl
        << "\\let\\ASYseries\\f@series%" << newl
        << "\\let\\ASYshape\\f@shape%" << newl
        << "\\makeatother%" << newl;
  }
}

template<class T>
void texpreamble(T& out, mem::list<string>& preamble=processData().TeXpreamble,
                 bool ASYalign=true, bool ASYbase=true)
{
  texuserpreamble(out,preamble);
  string texengine=settings::getSetting<string>("tex");
  if(ASYbase)
    out << "\\newbox\\ASYbox" << newl
        << "\\newdimen\\ASYdimen" << newl
        << "\\long\\def\\ASYbase#1#2{\\leavevmode\\setbox\\ASYbox=\\hbox{#1}"
        << "\\ASYdimen=\\ht\\ASYbox%" << newl
        << "\\setbox\\ASYbox=\\hbox{#2}\\lower\\ASYdimen\\box\\ASYbox}" << newl;
  if(ASYalign)
    out << "\\long\\def\\ASYaligned(#1,#2)(#3,#4)#5#6#7{\\leavevmode%" << newl
        << "\\setbox\\ASYbox=\\hbox{#7}%" << newl
        << "\\setbox\\ASYbox\\hbox{\\ASYdimen=\\ht\\ASYbox%" << newl
        << "\\advance\\ASYdimen by\\dp\\ASYbox\\kern#3\\wd\\ASYbox"
        << "\\raise#4\\ASYdimen\\box\\ASYbox}%" << newl
        << settings::beginput(texengine) 
        << "{#5\\wd\\ASYbox 0pt\\dp\\ASYbox 0pt\\ht\\ASYbox 0pt\\box\\ASYbox#6}"
        << settings::endput(texengine) << "}%" << newl
        << "\\long\\def\\ASYalignT(#1,#2)(#3,#4)#5#6{%" << newl
        << "\\ASYaligned(#1,#2)(#3,#4){%" << newl
        << settings::beginlabel(texengine) << "%" << newl
        << "}{%" << newl
        << settings::endlabel(texengine) << "%" << newl
        << "}{#6}}" << newl
        << "\\long\\def\\ASYalign(#1,#2)(#3,#4)#5{"
        << "\\ASYaligned(#1,#2)(#3,#4){}{}{#5}}" << newl
        << settings::rawpostscript(texengine) << newl;
}

template<class T>
void texdefines(T& out, mem::list<string>& preamble=processData().TeXpreamble,
                bool pipe=false)
{
  if(pipe || !settings::getSetting<bool>("inlinetex"))
    texpreamble(out,preamble,!pipe);

  if(pipe) {
    // Make tex pipe aware of a previously generated aux file.
    string name=auxname(settings::outname(),"aux");
    std::ifstream fin(name.c_str());
    if(fin) {
      std::ofstream fout("texput.aux");
      string s;
      while(getline(fin,s))
        fout << s << endl;
    }
  }
  texfontencoding(out);
  string texengine=settings::getSetting<string>("tex");
  if(settings::latex(texengine)) {
    if(pipe || !settings::getSetting<bool>("inlinetex")) {
      out << "\\usepackage{graphicx}" << newl;
      if(!pipe) out << "\\usepackage{color}" << newl;
    }
    if(pipe)
      out << "\\begin{document}" << newl;
  } else if(settings::context(texengine)) {
    if(!pipe && !settings::getSetting<bool>("inlinetex"))
      out << "\\usemodule[pictex]" << newl;
  } else {
    out << "\\input graphicx" << newl // Fix miniltx path parsing bug:
        << "\\makeatletter" << newl 
        << "\\def\\filename@parse#1{%" << newl
        << "  \\let\\filename@area\\@empty" << newl
        << "  \\expandafter\\filename@path#1/\\\\}" << newl
        << "\\def\\filename@path#1/#2\\\\{%" << newl
        << "  \\ifx\\\\#2\\\\%" << newl
        << "     \\def\\reserved@a{\\filename@simple#1.\\\\}%" << newl
        << "  \\else" << newl
        << "     \\edef\\filename@area{\\filename@area#1/}%" << newl
        << "     \\def\\reserved@a{\\filename@path#2\\\\}%" << newl
        << "  \\fi" << newl
        << "  \\reserved@a}" << newl
        << "\\makeatother" << newl;

    if(!pipe)
      out << "\\input picture" << newl;
  }
}
  
template<class T>
bool setlatexfont(T& out, const pen& p, const pen& lastpen)
{
  if(p.size() != lastpen.size() || p.Lineskip() != lastpen.Lineskip()) {
    out <<  "\\fontsize{" << p.size()*ps2tex << "}{" << p.Lineskip()*ps2tex
        << "}\\selectfont\n";
    return true;
  }
  return false;
}

template<class T>
bool settexfont(T& out, const pen& p, const pen& lastpen, bool latex) 
{
  string font=p.Font();
  if(font != lastpen.Font() || (!latex && p.size() != lastpen.size())) {
    out << font << "%" << newl;
    return true;
  }
  return false;
}

class texfile : public psfile {
protected:  
  bbox box;
  bool inlinetex;
  double Hoffset;
  int level;
  
public:
  string texengine;
  
  texfile(const string& texname, const bbox& box, bool pipe=false);
  virtual ~texfile();

  void prologue();
  virtual void beginpage() {}

  void epilogue(bool pipe=false);
  virtual void endpage() {}

  void setlatexcolor(pen p);
  void setpen(pen p);
  
  void setfont(pen p);
  
  void gsave();
  
  void grestore();
  
  void beginspecial();
  
  void endspecial();
  
  void beginraw();
  
  void endraw();
  
  void begingroup() {++level;}
  
  void endgroup() {--level;}
  
  bool toplevel() {return level == 0;}
  
  void beginpicture(const bbox& b);
  void endpicture(const bbox& b);
  
  void writepair(pair z) {
    *out << z;
  }
  
  void miniprologue();
  
  void writeshifted(path p, bool newPath=true);
  double hoffset() {return Hoffset;}
  
  // Draws label transformed by T at position z.
  void put(const string& label, const transform& T, const pair& z,
           const pair& Align);

  void beginlayer(const string& psname, bool postscript);
  void endlayer();
};

class svgtexfile : public texfile {
  mem::stack<size_t> clipstack;
  size_t clipcount;
  size_t gradientcount;
  size_t gouraudcount;
  size_t tensorcount;
  bool inspecial;
  static string nl;
public:  
  svgtexfile(const string& texname, const bbox& box, bool pipe=false) :
    texfile(texname,box,pipe) {
    clipcount=0;
    gradientcount=0;
    gouraudcount=0;
    tensorcount=0;
    inspecial=false;
  }
  
  void writeclip(path p, bool newPath=true) {
    write(p,false);
  }
  
  void dot(path p, pen, bool newPath=true);
  
  void writeshifted(pair z) {
    write(conj(z)*ps2tex);
  }
  
  void translate(pair z) {}
  void concat(transform t) {}
  
  void beginspecial();
  void endspecial();
  
  // Prevent unwanted page breaks.
  void beginpage() {
    beginpicture(box);
  }
  
  void endpage() {
    endpicture(box);
  }
  
  void begintransform();
  void endtransform();
  
  void clippath();
  
  void beginpath();
  void endpath();
  
  void newpath() {
    beginspecial();
    begintransform();
    beginpath();
  }
  
  void moveto(pair z) {
    *out << "M";
    writeshifted(z);
  }
  
  void lineto(pair z) {
    *out << "L";
    writeshifted(z);
  }

  void curveto(pair zp, pair zm, pair z1) {
    *out << "C";
    writeshifted(zp); writeshifted(zm); writeshifted(z1);
  }

  void closepath() {
    *out << "Z";
  }

  string rgbhex(pen p) {
    p.torgb();
    return p.hex();
  }
  
  void properties(const pen& p);
  void color(const pen &p, const string& type);
    
  void stroke(const pen &p, bool dot=false);
  void strokepath();
  
  void fillrule(const pen& p, const string& type="fill");
  void fill(const pen &p);
  
  void begingradientshade(bool axial, ColorSpace colorspace,
                          const pen& pena, const pair& a, double ra,
                          const pen& penb, const pair& b, double rb);
  void gradientshade(bool axial, ColorSpace colorspace,
                     const pen& pena, const pair& a, double ra,
                     const pen& penb, const pair& b, double rb);
  
  void gouraudshade(const pen& p0, const pair& z0,
                    const pen& p1, const pair& z1, 
                    const pen& p2, const pair& z2);
  void begingouraudshade(const vm::array& pens, const vm::array& vertices,
                         const vm::array& edges);
  void gouraudshade(const pen& pentype, const vm::array& pens,
                    const vm::array& vertices, const vm::array& edges);
  
  void begintensorshade(const vm::array& pens,
                        const vm::array& boundaries,
                        const vm::array& z);
  void tensorshade(const pen& pentype, const vm::array& pens,
                   const vm::array& boundaries, const vm::array& z);

  void beginclip();
  
  void endclip0(const pen &p);
  void endclip(const pen &p);
  
  void setpen(pen p) {if(!inspecial) texfile::setpen(p);}
  
  void gsave(bool tex=false);
  
  void grestore(bool tex=false);
};
  
} //namespace camp

#endif
