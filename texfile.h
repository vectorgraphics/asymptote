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

namespace camp {

extern bool TeXcontaminated;
extern std::list<std::string> TeXpreamble;

const double tex2ps=72.0/72.27;
const double ps2tex=1.0/tex2ps;
  
template<class T>
void texdocumentclass(T& out) {
  out << "\\documentclass[12pt]{article}" << newl;
}
  
template<class T>
void texpreamble(T& out) {
  std::list<std::string>::iterator p;
  for (p = TeXpreamble.begin(); p != TeXpreamble.end(); ++p) {
    out << stripblanklines(*p);
    TeXcontaminated=true;
  }
  out << "\\newbox\\ASYbox" << newl
      << "\\newdimen\\ASYdimen" << newl;
}
  
class texfile {
  ostream *out;
  pair offset;
  bbox box;
  pen lastpen;

public:
  texfile(const std::string& texname, const bbox& box);
  ~texfile();

  void prologue();

  void epilogue();

  void setpen(pen p);
  
  // Draws label rotated by angle (relative to the horizontal) at position z.
  void put(const std::string& label, double angle, pair z);

  void beginlayer(const std::string& psname);
  void endlayer();
  
  void verbatim(const std::string& s) {
    *out << s << newl;
  }
  
};

} //namespace camp

#endif


