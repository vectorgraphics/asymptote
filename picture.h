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
  size_t lastnumber3;
  transform T; // Keep track of accumulative picture transform
  bbox b;
  bbox b_cached;   // Cached bounding box
  boxvector labelbounds;
  bboxlist bboxstack;
  bool transparency;
  
  static bool epsformat,pdfformat,xobject,pdf,Labels;
  static double paperWidth,paperHeight;

public:
  bbox3 b3; // 3D bounding box
  
  typedef mem::list<drawElement*> nodelist;
  nodelist nodes;
  
  picture() : labels(false), lastnumber(0), lastnumber3(0), T(identity),
	      transparency(false) {}
  
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
  bool have3D();

  bbox bounds();
  bbox3 bounds3();

  // Projected bounds of 3d picture given transform3 t (not cached).
  pair bounds(double (*m)(double, double),
	       double (*x)(const triple&, double*),
	       double (*y)(const triple&, double*),
	       double *t=NULL);
  
  void texinit();

  bool Transparency() {
    return transparency;
  }
  
  int epstopdf(const string& epsname, const string& pdfname);
  
  bool texprocess(const string& texname, const string& tempname,
		  const string& prefix, const pair& bboxshift); 
    
  bool postprocess(const string& prename, const string& outname, 
		   const string& outputformat, double magnification,
		   bool wait, bool view);
    
  // Ship the picture out to PostScript & TeX files.
  bool shipout(picture* preamble, const string& prefix,
	       const string& format, double magnification=0.0,
	       bool wait=false, bool view=true);
 
  void render(GLUnurbs *nurb, double size2,
	      const triple &Min, const triple& Max, double perspective,
	      bool transparent, bool twosided) const;
  bool shipout3(const string& prefix, const string& format,
		double width, double height, double angle, const triple& m,
		const triple& M, size_t nlights, triple *lights,
		double *diffuse, double *ambient, double *specular,
		bool viewportlighting, bool wait=false, bool view=true);
  
  bool shipout3(const string& prefix); // Embedded PRC
  
  bool reloadPDF(const string& Viewer, const string& outname) const;
  
  picture *transformed(const transform& t);
  picture *transformed(const vm::array& t);
  
  bool null() {
    return nodes.empty();
  }
  
};

inline picture *transformed(const transform& t, picture *p)
{
  return p->transformed(t);
}

inline picture *transformed(const vm::array& t, picture *p)
{
  return p->transformed(t);
}

const char *texpathmessage();
  
} //namespace camp

#endif
