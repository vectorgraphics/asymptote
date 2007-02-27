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

#include "memory.h"
#include "pair.h"
#include "bbox.h"
#include "pen.h"
#include "util.h"
#include "interact.h"
#include "path.h"
#include "array.h"
#include "psfile.h"
#include "settings.h"

namespace camp {

extern mem::list<mem::string> TeXpipepreamble, TeXpreamble;

const double tex2ps=72.0/72.27;
const double ps2tex=1.0/tex2ps;
  
template<class T>
void texdocumentclass(T& out, bool pipe=false)
{
  if(settings::latex(settings::getSetting<mem::string>("tex")) &&
     (pipe || !settings::getSetting<bool>("inlinetex")))
    out << "\\documentclass[12pt]{article}" << newl;
}
  
template<class T>
void texpreamble(T& out, mem::list<mem::string>& preamble=TeXpreamble)
{
  for(mem::list<mem::string>::iterator p=preamble.begin();
      p != preamble.end(); ++p)
    out << stripblanklines(*p);
}

template<class T>
void texdefines(T& out, mem::list<mem::string>& preamble=TeXpreamble,
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
      << settings::rawpostscript(texengine) << newl;
  
  if(pipe) {
    mem::string name=auxname(settings::outname(),"aux");
    std::ifstream exists(name.c_str());
    if(exists) 
      out << "\\input " << name << newl;
  }
  if(settings::latex(texengine)) {
    if(pipe || !settings::getSetting<bool>("inlinetex")) {
      out << "\\usepackage{graphicx}" << newl;
      out << "\\usepackage{color}" << newl;
    }
    if(pipe)
      out << "\\begin{document}" << newl;
  } else {
    out << "\\input graphicx" << newl;
    if(!pipe)
      out << "\\input picture" << newl;
  }
}
  
class texfile : public psfile {
  bbox box;
  mem::string texengine;
  bool inlinetex;
  double Hoffset;

public:
  texfile(const mem::string& texname, const bbox& box);
  ~texfile();

  void prologue();

  void epilogue();

  void setlatexcolor(pen p);
  void setpen(pen p);
  
  void gsave();
  
  void grestore();
  
  void beginspecial();
  
  void endspecial();
  
  void beginraw();
  
  void endraw();
  
  void writepair(pair z) {
    *out << z;
  }
  
  void writeshifted(path p, bool newPath=true);
  
  // Draws label transformed by T at position z.
  void put(const mem::string& label, const transform& T, const pair& z,
	   const pair& Align);

  void beginlayer(const mem::string& psname);
  void endlayer();
  
};

} //namespace camp

#endif


