/*****
 * picture.h
 * Andy Hamerlindl 2002/06/06
 *
 * Stores a picture as a list of drawElements and handles its output to
 * PostScript.
 *****/

#ifndef PICTURE_H
#define PICTURE_H

#include <list>
#include <sstream>
#include <iostream>


#include "drawelement.h"

namespace camp {

extern iopipestream tex; // Bi-directional pipe to latex (to find label bbox)

class picture : public gc {
private:
  bool labels;
  size_t lastnumber;
  bbox b;
  pair bboxshift;
  bool epsformat,pdfformat,tgifformat;
  boxvector labelbounds;
  bboxlist bboxstack;

public:
  std::list<drawElement*> nodes;
  
  picture() : labels(false), lastnumber(0) {}
  
  // Destroy all of the owned picture objects.
  ~picture();

  // Find beginning of current layer.
  std::list<drawElement*>::iterator layerstart();
  
  // Prepend an object to the picture.
  void prepend(drawElement *p);
  
  // Append an object to the picture.
  void append(drawElement *p);

  // Add the content of another picture.
  void add(picture &pic);
  void prepend(picture &pic);
  
  bbox bounds();

  void texinit();

  bool texprocess(const string& texname, const string& tempname,
		  const string& prefix, const bbox& bpos); 
    
  bool postprocess(const string& epsname, const string& outname, 
		   const string& outputformat, bool wait,
		   const bbox& bpos);
    
  // Ship the picture out to PostScript & TeX files.
  bool shipout(const picture& preamble, const string& prefix,
	       const string& format, bool wait, bool Delete=false);
 
  picture *transformed(const transform& t);
  
  bool null() {
    return nodes.empty();
  }
  
  bool empty() {
    bounds();
    return (b.right <= b.left && b.top <= b.bottom);
  }
};

inline picture *transformed(const transform& t, picture *p)
{
  return p->transformed(t);
}

} //namespace camp

#endif
