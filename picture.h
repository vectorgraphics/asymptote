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

class picture : public gc, public IAsyPicture {
private:
  bool labels;
  size_t lastnumber;
  size_t lastnumber3;
  transform T; // Keep track of accumulative picture transform
  bbox b;
  bbox b_cached;   // Cached bounding box
  boxvector labelbounds;
  bboxlist bboxstack;
  groupsmap groups;
  unsigned billboard;
  bool deconstruct;
public:
  bbox3 b3; // 3D bounding box

  typedef mem::list<drawElement*> nodelist;
  nodelist nodes;

  picture(bool deconstruct=false) :
    labels(false), lastnumber(0), lastnumber3(0), T(identity), billboard(0),
    deconstruct(deconstruct) {}

  // Destroy all of the owned picture objects.
  ~picture() override;

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
  
  [[nodiscard]]
  bool have3D() const;
  
  [[nodiscard]]
  bool havepng() const;

  [[nodiscard]]
  unsigned int pagecount() const;

  bbox bounds();
  bbox3 bounds3();

  // Compute bounds on ratio (x,y)/z for 3d picture (not cached).
  pair ratio(double (*m)(double, double));

  int epstosvg(const string& epsname, const string& outname,
               unsigned int pages);
  int epstopdf(const string& epsname, const string& pdfname);
  int pdftoeps(const string& pdfname, const string& epsname, bool eps=true);

  bool texprocess(const string& texname, const string& tempname,
                  const string& prefix, const pair& bboxshift, bool svgformat);

  bool postprocess(const string& prename, const string& outname,
                   const string& outputformat, bool wait, bool view,
                   bool pdftex, bool epsformat, bool svg);

  bool display(const string& outname, const string& outputformat,
               bool wait, bool view, bool epsformat);

  // Ship the picture out to PostScript & TeX files.
  bool shipout(picture* preamble, const string& prefix,
               const string& format, bool wait=false, bool view=true);

  void render(double size2, const triple &Min, const triple& Max,
              double perspective, bool remesh) const;
  bool shipout3(const string& prefix, const string& format,
                double width, double height, double angle, double zoom,
                const triple& m, const triple& M, const pair& shift,
                const pair& margin, double *t, double *tup,
                double *background, size_t nlights, triple *lights,
                double *diffuse, bool view);

  // 3D output
  bool shipout3(const string& prefix, const string format);

  [[nodiscard]]
  bool reloadPDF(const string& Viewer, const string& outname) const;

  [[nodiscard]]
  picture *transformed(const transform& t) const;
  
  [[nodiscard]]
  picture *transformed(const vm::array& t) const;

  [[nodiscard]]
  bool null() const;
  
  // FFI functions

  void addToPicture(IAsyPicture* otherPic) override;
  void prependToPicture(IAsyPicture* otherPic) override;
  bool hasLabels() override;
  [[nodiscard]]
  bool has3D() const override;
  [[nodiscard]]
  bool hasPng() const override;
  IAsyBbox* getBounds() override;
  IAsyBbox3* getBounds3() override;
  [[nodiscard]]
  bool isNull() const override;
  [[nodiscard]]
  IAsyPicture* createTransformed(const IAsyTransform* transformPtr) const override;
  [[nodiscard]]
  IAsyPicture* createTransformedArray(const IAsyArray* arrayPtr) const override;
};

inline picture *transformed(const transform& t, picture *p)
{
  return p->transformed(t);
}

inline picture *transformed(const vm::array& t, picture *p)
{
  return p->transformed(t);
}

void texinit();
int opentex(const string& texname, const string& prefix, bool dvi=false);

const char *texpathmessage();

} //namespace camp

#endif
