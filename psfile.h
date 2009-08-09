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

// An ASCII85Encode filter.
class encode85 {
  ostream *out;
  int tuple;
  int pos;
  int count;
  static const int width=72;
public:
  encode85(ostream *out) : out(out), tuple(0), pos(0), count(0) {}
  
  ~encode85() {
    if(count > 0)
      encode(tuple, count);
    if(pos+2 > width)
      *out << '\n';
    *out << "~>\n";
  }
private:  
  void encode(unsigned int tuple, int count) {
    unsigned char buf[5], *s=buf;
    int i=5;
    do {
      *s++=tuple % 85;
      tuple /= 85;
    } while(--i > 0);
    i=count;
    do {
      *out << (unsigned char) (*--s + '!');
      if(pos++ >= width) {
        pos=0;
        *out << '\n';
      }
    } while(i-- > 0);
  }

public:  
  void put(unsigned char c) {
    switch (count++) {
      case 0:
        tuple |= (c << 24);
        break;
      case 1:
        tuple |= (c << 16);
        break;
      case 2:
        tuple |= (c <<  8);
        break;
      case 3:
        tuple |= c;
        if(tuple == 0) {
          *out << 'z';
          if(pos++ >= width) {
            pos=0;
            *out << '\n';
          }
        } else
          encode(tuple, count);
        tuple=0;
        count=0;
        break;
    }
  }
};

class psfile {
  string filename;
  bool pdfformat;    // Is final output format PDF?
  bool pdf;          // Output direct PDF?
  bool transparency; // Is transparency used?
  unsigned char *buffer;
  size_t count;
  mem::stack<pen> pens;
  
  void writeHex(pen *p, size_t ncomponents);
  void write(pen *p, size_t ncomponents);
  void writefromRGB(unsigned char r, unsigned char g, unsigned char b, 
                    ColorSpace colorspace, size_t ncomponents);
  
  void writeCompressed(const unsigned char *a, size_t size);
  void dealias(unsigned char *a, size_t width, size_t height, size_t n,
               bool convertrgb=false, ColorSpace colorspace=DEFCOLOR);
  
  void beginHex() {
    out->setf(std::ios::hex,std::ios::basefield);
    out->fill('0');
  }
  
  void endHex() {
    out->setf(std::ios::dec,std::ios::basefield);
    out->fill();
    *out << ">" << endl;
  }
  
  void beginImage(size_t n) {
    buffer=new unsigned char[n];
    count=0;
  }
  
  void outImage(bool antialias, size_t width, size_t height,
                size_t ncomponents);
  
  void endImage(bool antialias, size_t width, size_t height,
                size_t ncomponents) {
    outImage(antialias,width,height,ncomponents);
    delete[] buffer;
  }
  
  void writeByte(unsigned char n) {
    buffer[count++]=n;
  }
  
  void write2(unsigned n) {
    *out << std::setw(2) << n;
  }
  
protected:
  pen lastpen;
  std::ostream *out;
  
public: 
  psfile(const string& filename, bool pdfformat);
  
  psfile() {pdf=settings::pdf(settings::getSetting<string>("tex"));}

  ~psfile();
  
  void BoundingBox(const bbox& box) {
    camp::BoundingBox(*out,box);
  }
  
  void prologue(const bbox& box);
  void epilogue();
  void header();

  void close();
  
  void write(double x) {
    *out << " " << x;
  }

  void writenewl() {
    *out << newl;
  }
  
  bool Transparency() {
    return transparency;
  }
  
  void write(pair z) {
    *out << " " << z.getx() << " " << z.gety();
  }

  void write(transform t) {
    if(!pdf) *out << "[";
    *out << " " << t.getxx() << " " << t.getyx()
         << " " << t.getxy() << " " << t.getyy()
         << " " << t.getx() << " " << t.gety();
    if(!pdf) *out << "]";
  }

  void resetpen() {
    lastpen=pen(initialpen);
    lastpen.convert();
  }
  
  void setcolor(const pen& p, const string& begin, const string& end);
  void setopacity(const pen& p);

  void setpen(pen p);
  
  void write(const pen& p);
  
  void write(path p, bool newPath=true);
  
  void newpath() {
    if(!pdf) *out << "newpath";
  }

  void moveto(pair z) {
    write(z);
    if(pdf) *out << " m" << newl;
    else *out << " moveto" << newl;
  }

  void lineto(pair z) {
    write(z);
    if(pdf) *out << " l" << newl;
    else *out << " lineto" << newl;
  }

  void curveto(pair zp, pair zm, pair z1) {
    write(zp); write(zm); write(z1);
    if(pdf) *out << " c" << newl;
    else *out << " curveto" << newl;
  }

  void closepath() {
    if(pdf) *out << "h" << newl;
    else *out << "closepath" << newl;
  }

  void stroke() {
    if(pdf) *out << "S" << newl;
    else *out << "stroke" << newl;
  }
  
  void strokepath() {
    if(pdf) reportError("PDF does not support strokepath");
    else *out << "strokepath" << newl;
  }
  
  void fill(const pen &p) {
    if(p.evenodd()) {
      if(pdf) *out << "f*" << newl;
      else *out << "eofill" << newl;
    } else {
      if(pdf) *out << "f" << newl;
      else *out << "fill" << newl;
    }
  }
  
  void clip(const pen &p) {
    if(p.evenodd()) {
      if(pdf) *out << "W* n" << newl;
      else *out << "eoclip" << newl;
    } else {
      if(pdf) *out << "W n" << newl;
      else *out << "clip" << newl;
    }
  }
  
  void checkLevel() {
    int n=settings::getSetting<Int>("level");
    if(n < 3)
      reportError("PostScript shading requires -level 3");
  }
  
  void latticeshade(const vm::array& a, const bbox& b);
  
  void gradientshade(bool axial, ColorSpace colorspace,
                     const pen& pena, const pair& a, double ra,
                     const pen& penb, const pair& b, double rb);
  
  void gouraudshade(const vm::array& pens, const vm::array& vertices,
                    const vm::array& edges);
  void tensorshade(const vm::array& pens, const vm::array& boundaries,
                   const vm::array& z);
  
  void vertexpen(vm::array *pi, int j, ColorSpace colorspace);
  
  void imageheader(size_t width, size_t height, ColorSpace colorspace);
  
  void image(const vm::array& a, const vm::array& p, bool antialias);
  void image(const vm::array& a, bool antialias);
  void rawimage(unsigned char *a, size_t width, size_t height, bool antialias);

  void gsave(bool tex=false) {
    if(pdf) *out << "q";
    else *out << "gsave";
    if(!tex) *out << newl;
    pens.push(lastpen);
  }
  
  void grestore(bool tex=false) {
    if(pens.size() < 1)
      reportError("grestore without matching gsave");
    lastpen=pens.top();
    pens.pop();
    if(pdf) *out << "Q";
    else *out << "grestore";
    if(!tex) *out << newl;
  }

  void translate(pair z) {
    if(pdf) *out << " 1 0 0 1 " << newl;
    write(z);
    if(pdf) *out << " cm" << newl;
    *out << " translate" << newl;
  }

  // Multiply on a transform to the transformation matrix.
  void concat(transform t) {
    if(t.isIdentity()) return;
    write(t);
    if(pdf) *out << " cm" << newl;
    else *out << " concat" << newl;
  }
  
  void verbatimline(const string& s) {
    *out << s << newl;
  }
  
  void verbatim(const string& s) {
    *out << s;
  }

  // Determine shading and image transparency based on first pen.
  void setfirstopacity(const vm::array& pens) {
    if(pens.size() > 0) {
      pen *p=vm::read<pen *>(pens,0);
      setopacity(*p);
    }
  }
  
  ColorSpace maxcolorspace(const vm::array& pens) {
    ColorSpace colorspace=DEFCOLOR;
    size_t size=pens.size();
    for(size_t i=0; i < size; i++) {
      pen *p=vm::read<pen *>(pens,i);
      p->convert();
      colorspace=max(colorspace,p->colorspace());
    }
    return colorspace;
  }
  
  ColorSpace maxcolorspace2(const vm::array& penarray) {
    ColorSpace colorspace=DEFCOLOR;
    size_t size=penarray.size();
    for(size_t i=0; i < size; i++)
      colorspace=max(colorspace,
                     maxcolorspace(vm::read<vm::array>(penarray,i)));
    return colorspace;
  }

};

} //namespace camp

#endif
