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

using std::string;

namespace camp {

extern std::list<string> TeXpipepreamble, TeXpreamble;

const double tex2ps=72.0/72.27;
const double ps2tex=1.0/tex2ps;
  
template<class T>
void texdocumentclass(T& out, bool pipe=false)
{
  if(pipe || !settings::getSetting<bool>("inlinetex"))
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
  texpreamble(out,preamble);
  out << "\\newbox\\ASYbox" << newl
      << "\\newdimen\\ASYdimen" << newl
      << "\\def\\ASYbase#1#2{\\setbox\\ASYbox=\\hbox{#1}"
      << "\\ASYdimen=\\ht\\ASYbox%" << newl
      << "\\setbox\\ASYbox=\\hbox{#2}\\lower\\ASYdimen\\box\\ASYbox}" << newl
//      << "\\usepackage{rotating}" << newl
      << "\\def\\ASYalign(#1,#2)(#3,#4)#5#6{\\leavevmode%" << newl
      << "\\setbox\\ASYbox=\\hbox{#6}%" << newl
//      << "\\put(#1,#2){\\begin{rotate}{#5}%" << newl
      << "\\put(#1,#2){\\special{ps: gsave currentpoint currentpoint translate"
      << newl
      << "[#5 0 0] concat neg exch neg exch translate}"
      << "\\ASYdimen=\\ht\\ASYbox%" << newl
      << "\\advance\\ASYdimen by\\dp\\ASYbox\\kern#3\\wd\\ASYbox"
      << "\\raise#4\\ASYdimen\\box\\ASYbox%" << newl
//      << "\\end{rotate}}}" << newl
      << "\\special{ps: currentpoint grestore moveto}}}" << newl
      << "\\def\\ASYgsave{\\special{ps: gsave}}" << newl
      << "\\def\\ASYgrestore{\\special{ps: grestore}}" << newl
      << "\\def\\ASYclip(#1,#2)#3{\\leavevmode%" << newl
      << "\\put(#1,#2){\\special{ps: currentpoint currentpoint translate" 
      << newl
      << "matrix currentmatrix" << newl
      << "[matrix defaultmatrix 0 get 0 0 matrix defaultmatrix 3 get" << newl
      << "matrix currentmatrix 4 get matrix currentmatrix 5 get] setmatrix"
      << newl
      << "#3" << newl
      << "setmatrix" << newl
      << "neg exch neg exch translate}}}" << newl;
  
  if(pipe || !settings::getSetting<bool>("inlinetex"))
    out << "\\usepackage{graphicx}" << newl;
  if(pipe) out << "\\begin{document}" << newl;
}
  
class texfile : public psfile {
  bbox box;
  pen lastpen;

public:
  texfile(const string& texname, const bbox& box);
  ~texfile();

  void prologue();

  void epilogue();

  void setpen(pen p);
  
  void gsave();
  
  void grestore();
  
  void openclip();
  
  void closeclip();
  
  // Draws label transformed by T at position z.
  void put(const string& label, const transform& T, const pair& z,
	   const pair& Align, const bbox& Box);

  void beginlayer(const string& psname);
  void endlayer();
  
  void verbatim(const string& s) {
    *out << s << newl;
  }
  
};

} //namespace camp

#endif


