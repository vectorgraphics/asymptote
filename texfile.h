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

const double tex2ps=72.0/72.27;
const double ps2tex=1.0/tex2ps;
  
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
        << "\\put(#1,#2){#5\\wd\\ASYbox 0pt\\dp\\ASYbox 0pt\\ht\\ASYbox 0pt"
        << "\\box\\ASYbox#6}}" << newl
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
  if(settings::latex(settings::getSetting<string>("tex"))) {
    if(pipe || !settings::getSetting<bool>("inlinetex")) {
      out << "\\usepackage{graphicx}" << newl;
      if(!pipe) out << "\\usepackage{color}" << newl;
    }
    if(pipe)
      out << "\\begin{document}" << newl;
  } else {
    out << "\\input graphicx" << newl;
    if(!pipe)
      out << "\\input picture" << newl;
  }
}
  
template<class T>
bool setlatexfont(T& out, const pen& p, const pen& lastpen)
{
  if(p.size() != lastpen.size() || p.Lineskip() != lastpen.Lineskip()) {
    out <<  "\\fontsize{" << p.size() << "}{" << p.Lineskip()
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
  bbox box;
  bool inlinetex;
  double Hoffset;

public:
  string texengine;
  
  texfile(const string& texname, const bbox& box, bool pipe=false);
  ~texfile();

  void prologue();

  void epilogue(bool pipe=false);

  void setlatexcolor(pen p);
  void setpen(pen p);
  
  void setfont(pen p);
  
  void gsave();
  
  void grestore();
  
  void beginspecial();
  
  void endspecial();
  
  void beginraw();
  
  void endraw();
  
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

} //namespace camp

#endif
