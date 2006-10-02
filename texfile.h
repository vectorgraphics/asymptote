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

using std::string;

namespace camp {

class clipper : public gc {
public: 
  vm::array *p;
  pen rule;
  clipper(vm::array *p, const pen& rule) : p(p), rule(rule) {}
  
  // Return true if z is within clipping bounds.
  bool inside(const pair& z) {
    checkArray(p);
    size_t size=p->size();
    int count=0;
    for(size_t i=0; i < size; i++)
      count += vm::read<path *>(p,i)->inside(z);
    return rule.inside(count);
  }

};
  
typedef mem::list<clipper *> cliplist;
  
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
      << "\\put(#1,#2){\\special{ps: gsave currentpoint currentpoint" << newl
      << "translate [#5 0 0] concat neg exch neg exch translate}"
      << "\\ASYdimen=\\ht\\ASYbox%" << newl
      << "\\advance\\ASYdimen by\\dp\\ASYbox\\kern#3\\wd\\ASYbox"
      << "\\raise#4\\ASYdimen\\box\\ASYbox%" << newl
//      << "\\end{rotate}}}" << newl
      << "\\special{ps: currentpoint grestore moveto}}}" << newl;
  
  if(pipe || !settings::getSetting<bool>("inlinetex"))
    out << "\\usepackage{graphicx}" << newl;
  if(pipe) out << "\\begin{document}" << newl;
}
  
class texfile : public gc {
  ostream *out;
  pair offset;
  bbox box;
  pen lastpen;
  cliplist clipstack;

public:
  texfile(const string& texname, const bbox& box);
  ~texfile();

  void prologue();

  void epilogue();

  void setpen(pen p);
  
  void beginclip(clipper *c) {
    clipstack.push_back(c);
  }
  
  void endclip() {
    if(clipstack.size() < 1)
      reportError("endclip without matching beginclip");
    clipstack.pop_back();
  }
  
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


