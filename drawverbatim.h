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
  std::string text;
public:
  drawVerbatim(Language language, const std::string& text) : 
    language(language), text(text) {}
  
  virtual ~drawVerbatim() {}

  bool islabel() {
    return language == TeX;
  }
  
  bool draw(psfile *out) {
    if(language == PostScript) out->verbatimline(text);
    return true;
  }

  bool write(texfile *out) {
    if(language == TeX) out->verbatim(stripblanklines(text));
    return true;
  }

};

}

#endif
