/*****
 * drawverbatim.h
 * John Bowman 2003/03/18
 *
 * Add verbatim postscript to picture.
 *****/

#ifndef DRAWVERBATIM_H
#define DRAWVERBATIM_H

#include "drawelement.h"
#include "path.h"

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

  bool draw(psfile *out) {
    if(language == PostScript) out->verbatim(text);
    return true;
  }

  bool write(texfile *out) {
    if(language == TeX) out->verbatim(text);
    return true;
  }

};

}

#endif
