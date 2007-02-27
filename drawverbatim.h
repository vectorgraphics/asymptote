/*****
 * drawverbatim.h
 * John Bowman 2003/03/18
 *
 * Add verbatim postscript to picture.
 *****/

#ifndef DRAWVERBATIM_H
#define DRAWVERBATIM_H

#include "drawelement.h"

namespace camp {

enum Language {PostScript,TeX};
  
class drawVerbatim : public drawElement {
private:
  Language language;
  mem::string text;
  bool havebounds;
  pair min,max;
public:
  drawVerbatim(Language language, const mem::string& text) : 
    language(language), text(text), havebounds(false) {}
  
  drawVerbatim(Language language, const mem::string& text, pair min,
	       pair max) : 
    language(language), text(text), havebounds(true), min(min), max(max) {}
  
  virtual ~drawVerbatim() {}

  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    if(havebounds) {
      b += min;
      b += max;
    }
  }
  
  bool islabel() {
    return language == TeX;
  }
  
  bool draw(psfile *out) {
    if(language == PostScript) out->verbatimline(text);
    return true;
  }

  bool write(texfile *out) {
    if(language == TeX) out->verbatimline(stripblanklines(text));
    return true;
  }
};

}

#endif
