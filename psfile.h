/*****
 * psfile.h
 * Andy Hammerlindl 2002/06/10
 *
 * Encapsulates the writing of commands to a PostScript file.
 * Allows identification and removal of redundant commands.
 *****/

#ifndef PSFILE_H
#define PSFILE_H

#include <fstream>
#include <stack>
#include <iomanip>

#include "pair.h"
#include "path.h"
#include "bbox.h"
#include "pen.h"

namespace camp {

inline void BoundingBox(std::ostream& s, const bbox& box) 
{
  s << "%%BoundingBox: " << box.LowRes() << newl;
  s << "%%HiResBoundingBox: " << std::setprecision(9) << box << newl;
}

class psfile {
  std::string filename;
  bbox box;
  pair shift;
  bool rawmode;
  pen lastpen;
  ostream *out;
  std::stack<pen> pens;

  void writeUnshifted(pair z) {
    *out << " " << z.getx() << " " << z.gety();
  }

  void write(transform t) {
    *out << "[" << " " << t.getxx() << " " << t.getyx()
                << " " << t.getxy() << " " << t.getyy()
	        << " " << t.getx()  << " " << t.gety() << "]";
  }

public: 
  psfile(const std::string& filename, const bbox& box, const pair& shift);
  ~psfile();
  
  void prologue();
  void epilogue();

  void raw(bool mode) {rawmode=mode;}
  bool raw() {return rawmode;}
  
  void write(pair z) {
    if(rawmode) writeUnshifted(z);
    else {
      *out << " " << z.getx()+shift.getx() 
	   << " " << z.gety()+shift.gety();
    }
  }

  void write(double x) {
    *out << " " << x;
  }

  void resetpen() {
    lastpen=pen(initialpen);
    lastpen.convert();
  }
  
  void setpen(pen p);

  void write(pen p);
  
  void write(path p, bool newPath=true);
  
  void newpath() {
      *out << "newpath";
  }

  void moveto(pair z) {
      write(z);
      *out << " moveto" << newl;
  }

  void lineto(pair z) {
      write(z);
      *out << " lineto" << newl;
  }

  void curveto(pair zp, pair zm, pair z1) {
      write(zp); write(zm); write(z1);
      *out << " curveto" << newl;
  }

  void closepath() {
      *out << " closepath" << newl;
  }

  void rlineto(pair z) {
      writeUnshifted(z);
      *out << " rlineto" << newl;
  }

  void stroke() {
    *out << " stroke" << newl;
  }
  
  void fill(FillRule fillrule) {
    *out << (fillrule == EVENODD ? " eofill" : " fill") << newl;
  }
  
  void clip(FillRule fillrule) {
    *out << (fillrule == EVENODD ? " eoclip" : " clip") << newl;
  }
  
  void shade(bool axial, const std::string& colorspace,
	     const pen& pena, const pair& a, double ra,
	     const pen& penb, const pair& b, double rb);
  
  void gsave() {
    *out << " gsave" << newl;
    pens.push(lastpen);
  }
  
  void grestore() {
    if(pens.size() < 1) {
      reportError("grestore without matching gsave");
      return;
    }
    lastpen = pens.top();
    pens.pop();
    *out << " grestore" << newl;
  }

  void translate(pair z) {
    writeUnshifted(z);
    *out << " translate" << newl;
  }

  // Multiply on a transform to the transformation matrix.
  void concat(transform t) {
    write(t);
    *out << " concat" << newl;
  }
  
  void verbatimline(const std::string& s) {
    *out << s << newl;
  }
  
  void verbatim(const std::string& s) {
    *out << s;
  }

};

} //namespace camp

#endif
