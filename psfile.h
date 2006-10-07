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
#include <sstream>

#include "pair.h"
#include "path.h"
#include "bbox.h"
#include "pen.h"
#include "array.h"

namespace camp {

inline void BoundingBox(std::ostream& s, const bbox& box) 
{
  s << "%%BoundingBox: " << std::setprecision(0) << std::fixed 
       << box.LowRes() << newl;
  s.unsetf(std::ios::fixed);
  s << "%%HiResBoundingBox: " << std::setprecision(9) << box << newl;
}

class psfile {
  string filename;
  bool pdfformat;
  pen lastpen;
  std::stack<pen> pens;

  void write(transform t) {
    *out << "[" << " " << t.getxx() << " " << t.getyx()
	 << " " << t.getxy() << " " << t.getyy()
	 << " " << t.getx() << " " << t.gety() << "]";
  }

  void writeHex(unsigned int n) {
    *out << std::hex << std::setw(2) << std::setfill('0') << n << std::dec;
  }
  
protected:
  std::ostream *out;
  
public: 
  psfile(const string& filename, bool pdformat);
  psfile() {};
  ~psfile();
  
  void BoundingBox(const bbox& box) {
    camp::BoundingBox(*out,box);
  }
  
  void prologue(const bbox& box);
  void epilogue();

  void write(double x) {
    *out << " " << x;
  }

  void write(pair z) {
    *out << " " << z.getx() << " " << z.gety();
  }

  void writeHex(pen *p, int ncomponents);
  
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
      write(z);
      *out << " rlineto" << newl;
  }

  void stroke() {
    *out << " stroke" << newl;
  }
  
  void fill(const pen &p) {
    *out << (p.evenodd() ? " eofill" : " fill") << newl;
  }
  
  void clip(const pen &p) {
    *out << (p.evenodd() ? " eoclip" : " clip") << newl;
  }
  
  void shade(vm::array *a, const bbox& b);
  
  void shade(bool axial, const ColorSpace& colorspace,
	     const pen& pena, const pair& a, double ra,
	     const pen& penb, const pair& b, double rb);
  
  void shade(vm::array *pens, vm::array *vertices, vm::array *edges);
  
  void image(vm::array *a, vm::array *p);

  void gsave() {
    *out << "gsave" << newl;
    pens.push(lastpen);
  }
  
  void grestore() {
    if(pens.size() < 1)
      reportError("grestore without matching gsave");
    lastpen = pens.top();
    pens.pop();
    *out << "grestore" << newl;
  }

  void translate(pair z) {
    write(z);
    *out << " translate" << newl;
  }

  // Multiply on a transform to the transformation matrix.
  void concat(transform t) {
    write(t);
    *out << " concat" << newl;
  }
  
  void verbatimline(const string& s) {
    *out << s << newl;
  }
  
  void verbatim(const string& s) {
    *out << s;
  }

};

} //namespace camp

#endif
