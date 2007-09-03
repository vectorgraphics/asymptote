/*****
 * picture.h
 * Andy Hamerlindl 2002/06/06
 *
 * Stores a picture as a list of drawElements and handles its output to
 * PostScript.
 *****/

#ifndef PICTURE_H
#define PICTURE_H

#include <sstream>
#include <iostream>

#include "drawelement.h"

namespace camp {

class picture : public gc {
private:
  bool labels;
  size_t lastnumber;
  transform T; // Keep track of accumulative picture transform
  bbox b;
  boxvector labelbounds;
  bboxlist bboxstack;
  
  static bool epsformat,pdfformat,xobject,pdf,Labels;
  static double paperWidth,paperHeight;

public:
  typedef mem::list<drawElement*> nodelist;
  nodelist nodes;
  static std::ofstream bboxout;
  
  picture() : labels(false), lastnumber(0), T(identity) {}
  
  // Destroy all of the owned picture objects.
  ~picture();

  // Prepend an object to the picture.
  void prepend(drawElement *p);
  
  // Append an object to the picture.
  void append(drawElement *p);

  // Enclose each layer with begin and end.
  void enclose(drawElement *begin, drawElement *end);
  
  // Add the content of another picture.
  void add(picture &pic);
  void prepend(picture &pic);
  
  bool havelabels();
  bbox bounds();

  void texinit();

  int epstopdf(const string& epsname, const string& pdfname);
  
  bool texprocess(const string& texname, const string& tempname,
		  const string& prefix, const pair& bboxshift); 
    
  bool postprocess(const string& prename, const string& outname, 
		   const string& outputformat, double magnification,
		   bool wait, bool view);
    
  // Ship the picture out to PostScript & TeX files.
  bool shipout(picture* preamble, const string& prefix,
	       const string& format, double magnification=0, 
	       bool wait=false, bool view=true,
	       bool Delete=false);
 
  picture *transformed(const transform& t);
  
  bool null() {
    return nodes.empty();
  }
  
};

inline picture *transformed(const transform& t, picture *p)
{
  return p->transformed(t);
}

} //namespace camp

#endif
