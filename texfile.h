/*****
 * texfile.h
 * John Bowman 2003/03/14
 *
 * Encapsulates the writing of commands to a TeX file.
 *****/

#ifndef TEXFILE_H
#define TEXFILE_H

#include <fstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <list>

#include "pair.h"
#include "bbox.h"
#include "pen.h"
#include "util.h"
#include "interact.h"
#include "path.h"
#include "array.h"
#include "psfile.h"
#include "settings.h"

using std::string;

namespace camp {

extern std::list<string> TeXpipepreamble, TeXpreamble;

const double tex2ps=72.0/72.27;
const double ps2tex=1.0/tex2ps;
  
template<class T>
void texdocumentclass(T& out, bool pipe=false)
{
  bool latex=settings::latex(settings::getSetting<mem::string>("tex"));
  if(latex && (pipe || !settings::getSetting<bool>("inlinetex")))
    out << "\\documentclass[12pt]{article}" << newl;
}
  
template<class T>
void texpreamble(T& out, std::list<string>& preamble=TeXpreamble)
{
  for(std::list<string>::iterator p=preamble.begin(); p != preamble.end(); ++p)
    out << stripblanklines(*p);
}

template<class T>
void texdefines(T& out, std::list<string>& preamble=TeXpreamble,
		bool pipe=false)
{
  mem::string texengine=settings::getSetting<mem::string>("tex");
  texpreamble(out,preamble);
  out << "\\newbox\\ASYbox" << newl
      << "\\newdimen\\ASYdimen" << newl
      << "\\def\\ASYbase#1#2{\\setbox\\ASYbox=\\hbox{#1}"
      << "\\ASYdimen=\\ht\\ASYbox%" << newl
      << "\\setbox\\ASYbox=\\hbox{#2}\\lower\\ASYdimen\\box\\ASYbox}" << newl
      << "\\def\\ASYalign(#1,#2)(#3,#4)#5#6{\\leavevmode%" << newl
      << "\\setbox\\ASYbox=\\hbox{#6}%" << newl
      << "\\setbox\\ASYbox\\hbox{\\ASYdimen=\\ht\\ASYbox%" << newl
      << "\\advance\\ASYdimen by\\dp\\ASYbox\\kern#3\\wd\\ASYbox"
      << "\\raise#4\\ASYdimen\\box\\ASYbox}%" << newl
      << "\\put(#1,#2){%" << newl
      << settings::beginlabel(texengine) << "%" << newl
      << "\\box\\ASYbox%" << newl
      << settings::endlabel(texengine) << "%" << newl
      << "}}" << newl
      << "\\def\\ASYclip(#1,#2)#3{\\leavevmode%" << newl
      << "\\put(#1,#2){" << settings::clip(texengine) << "}}" << newl;
  
  if(settings::latex(texengine)) {
    if(pipe || !settings::getSetting<bool>("inlinetex"))
      out << "\\usepackage{graphicx}" << newl;
    if(pipe)
      out << "\\begin{document}" << newl;
    else 
      out << "\\usepackage{color}" << newl;
  } else {
    out << "\\input graphicx" << newl;
    if(!pipe)
      out << "\\input picture" << newl;
  }
}
  
class texfile : public psfile {
  bbox box;
  mem::string texengine;

public:
  texfile(const string& texname, const bbox& box);
  ~texfile();

  void prologue();

  void epilogue();

  void setlatexcolor(pen p);
  void setpen(pen p);
  
  void gsave();
  
  void grestore();
  
  void openclip();
  
  void closeclip();
  
  // Draws label transformed by T at position z.
  void put(const string& label, const transform& T, const pair& z,
	   const pair& Align);

  void beginlayer(const string& psname);
  void endlayer();
  
  void verbatim(const string& s) {
    *out << s << newl;
  }
  
};

} //namespace camp

#endif


