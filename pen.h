/*****
 * pen.h
 * John Bowman 2003/03/23
 *
 *****/

#ifndef PEN_H
#define PEN_H

#include <string>
#include "pool.h"
#include "transform.h"
#include "settings.h"
#include "bbox.h"

namespace camp {
  class pen;
}

namespace settings {
  extern camp::pen startupdefaultpen;
  extern camp::pen defaultpen;
}

namespace camp {

static const std::string DEFLINE="default";
static const std::string DEFPAT="<default>";
static const double DEFWIDTH=-1;
static const int DEFCAP=-1;
static const int DEFJOIN=-1;
  
static const struct transparentpen_t {} transparentpen={};
static const struct setlinewidth_t {} setlinewidth={};
static const struct setfontsize_t {} setfontsize={};
static const struct setpattern_t {} setpattern={};
static const struct setlinecap_t {} setlinecap={};
static const struct setlinejoin_t {} setlinejoin={};
static const struct setoverwrite_t {} setoverwrite={};
static const struct initialpen_t {} initialpen={};
  
static const std::string Cap[]={"square","round","extended"};
static const std::string Join[]={"miter","round","bevel"};
const int nCap=sizeof(Cap)/sizeof(std::string);
const int nJoin=sizeof(Join)/sizeof(std::string);
  
enum overwrite_t {DEFWRITE=-1,ALLOW,SUPPRESS,SUPPRESSQUIET,MOVE,MOVEQUIET};
static const std::string OverwriteTag[]={"Allow","Suppress","SupressQuiet",
					 "Move","MoveQuiet"};
const int nOverwrite=sizeof(OverwriteTag)/sizeof(std::string);
  
enum ColorSpace {TRANSPARENT,DEFCOLOR,GRAYSCALE,RGB,CMYK,PATTERN};
static const int ColorComponents[]={0,1,1,3,4,0};
static const std::string ColorDeviceSuffix[]={"","Gray","Gray","RGB","CMYK",
					      ""};
using settings::defaultpen;
  
class pen : public mempool::pooled<pen>
{ 
  // The string for the PostScript style line pattern.
  std::string line;

  // Width of line, in PS units.
  double linewidth;
  double fontsize;  
  
  ColorSpace color;
  double r,g,b; // RGB or CMY value
  double grey; // grayscale or K value

  // The name of the user-defined fill/draw pattern.
  std::string pattern;
  int linecap;
  int linejoin;
  overwrite_t overwrite;
  
  // The transformation applied to the pen nib for calligraphic effects.
  // Null means the identity transformation.
  const transform *t;
  
public:
  static double pos0(double x) {return x >= 0 ? x : 0;}
  
  void greyrange() {if(grey > 1.0) grey=1.0;}
  
  void rgbrange() {
    double sat=rgbsaturation();
    if(sat > 1.0) {
      double scale=1.0/sat;
      r *= scale;
      g *= scale;
      b *= scale;
    }
  }
  
  void cmykrange() {
    double sat=cmyksaturation();
    if(sat > 1.0) {
      double scale=1.0/sat;
      r *= scale;
      g *= scale;
      b *= scale;
      grey *= scale;
    }
  }
  
  pen() :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(DEFCOLOR), r(defaultpen.r), g(defaultpen.g), b(defaultpen.b),
    grey(defaultpen.grey), pattern(DEFPAT),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}

  pen(const std::string& line, double linewidth, double fontsize,
      ColorSpace color, double r, double g, double b,  double grey,
      const std::string& pattern, int linecap, int linejoin,
      overwrite_t overwrite, const transform *t) :
    line(line), linewidth(linewidth), fontsize(fontsize),
    color(color), r(r), g(g), b(b), grey(grey), pattern(pattern),
    linecap(linecap), linejoin(linejoin), overwrite(overwrite), t(t) {}
      
  pen(transparentpen_t) : 
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(TRANSPARENT), r(defaultpen.r), g(defaultpen.g), b(defaultpen.b),
    grey(defaultpen.grey), pattern(DEFPAT),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinewidth_t, double linewidth) : 
    line(DEFLINE), linewidth(linewidth), fontsize(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {}
  
  pen(const std::string& line) :
    line(line), linewidth(DEFWIDTH), fontsize(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {}
  
  pen(setfontsize_t, double fontsize) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(fontsize), color(DEFCOLOR), 
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {}
  
  pen(initialpen_t) : 
    line(DEFLINE), linewidth(-2), fontsize(-1), color(GRAYSCALE),
    r(0.0), g(0.0), b(0.0), grey(0.0),
    pattern(DEFPAT), linecap(-2), linejoin(-2), overwrite(DEFWRITE),
    t(0) {}
  
  pen(setpattern_t, const std::string& pattern) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(PATTERN),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(pattern), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {}
  
  pen(setlinecap_t, int linecap) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), linecap(linecap), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {}
  
  pen(setlinejoin_t, int linejoin) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(linejoin), overwrite(DEFWRITE),
    t(0) {}
  
  pen(setoverwrite_t, overwrite_t overwrite) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(overwrite),
    t(0) {}
  
  pen(double grey) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(GRAYSCALE), r(0.0), g(0.0), b(0.0), grey(pos0(grey)),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {greyrange();}
  
  pen(double r, double g, double b) : 
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(RGB), r(pos0(r)), g(pos0(g)), b(pos0(b)),  grey(0.0), 
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {rgbrange();}
  
  pen(double c, double m, double y, double k) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(CMYK), r(pos0(c)), g(pos0(m)), b(pos0(y)), grey(pos0(k)),
    pattern(DEFPAT), linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE),
    t(0) {cmykrange();}
  
  double width() const {
    return linewidth == DEFWIDTH ? defaultpen.linewidth : linewidth;
  }
  
  double size() const {
    return fontsize == 0.0 ? defaultpen.fontsize : fontsize;
  }
  
  std::string stroke() const {
    return line == DEFLINE ? defaultpen.line : line;
  }
  
  void setstroke(const std::string& s) {line=s;}
  
  std::string fillpattern() const {
    return pattern == DEFPAT ? "" : pattern;
  }
  
  int cap() const {
    return linecap == DEFCAP ? defaultpen.linecap : linecap;
  }
  
  int join() const {
    return linejoin == DEFJOIN ? defaultpen.linejoin : linejoin;
  }
  
  overwrite_t Overwrite() const {
    return overwrite == DEFWRITE ? defaultpen.overwrite : overwrite;
  }
  
  ColorSpace colorspace() const {
    return color == DEFCOLOR ? defaultpen.color : color;
  }
  
  bool transparent() const {return colorspace() == TRANSPARENT;}
  
  bool grayscale() const {return colorspace() == GRAYSCALE;}
  
  bool rgb() const {return colorspace() == RGB;}
  
  bool cmyk() const {return colorspace() == CMYK;}
  
  double gray() const {return color == DEFCOLOR ? defaultpen.grey : grey;}
  
  double red() const {return color == DEFCOLOR ? defaultpen.r : r;}
  
  double green() const {return color == DEFCOLOR ? defaultpen.g : g;}
  
  double blue() const {return color == DEFCOLOR ? defaultpen.b : b;}
  
  double cyan() const {return red();}
  
  double magenta() const {return green();}
  
  double yellow() const {return blue();}
  
  double black() const {return gray();}
  
  double rgbsaturation() const {return max(max(r,g),b);}
  
  double cmyksaturation() const {return max(rgbsaturation(),black());}
  
  void greytorgb() {
    r=g=b=grey; grey=0.0;
    color=RGB;
  }
  
  void greytocmyk() {
    grey=1.0-grey;
    color=CMYK;
  }
  
  void rgbtocmyk() {
    double sat=rgbsaturation();
    r=1.0-r;
    g=1.0-g;
    b=1.0-b;
    grey=1.0-sat;
    if(sat) {
      double scale=1.0/sat;
      r=(r-grey)*scale;
      g=(g-grey)*scale;
      b=(b-grey)*scale;
    }
    color=CMYK;
  }

  void cmyktorgb() {
    double sat=1.0-grey;
    r=(1.0-r)*sat;
    g=(1.0-g)*sat;
    b=(1.0-b)*sat;
    grey=0.0;
    color=RGB;
  }

  void convert() {
    if(settings::rgbonly && cmyk()) cmyktorgb();
    else if(settings::cmykonly && rgb()) rgbtocmyk();
  }    
  
  friend pen operator * (double x, const pen& q) {
    pen p=q;
    if(x < 0.0) x = 0.0;
    switch(p.color) {
    case PATTERN:
    case TRANSPARENT:
    case DEFCOLOR:
      break;
    case GRAYSCALE:
      {
	p.grey *= x;
	p.greyrange();
	break;
      }
      break;
    case RGB:
      {
	p.r *= x;
	p.g *= x;
	p.b *= x;
	p.rgbrange();
	break;
      }
    case CMYK:
      {
	p.r *= x;
	p.g *= x;
	p.b *= x;
	p.grey *= x;
	p.cmykrange();
	break;
      }
    }
    return p;
  }
  
  friend pen operator + (const pen& p, const pen& q) {
    double R=0.0,G=0.0,B=0.0,greyval=0.0;
    pen P=p;
    pen Q=q;
    
    if(P.color == PATTERN && P.pattern == "") P.color=DEFCOLOR;
    ColorSpace colorspace=(ColorSpace) max(P.color,Q.color);
    
  switch(colorspace) {
    case PATTERN:
    case TRANSPARENT:
    case DEFCOLOR:
      break;
    case GRAYSCALE:
      {
	if(p.color != GRAYSCALE) greyval=q.grey;
	else {
	  if(q.color != GRAYSCALE) greyval=p.grey;
	  else greyval=0.5*(p.grey+q.grey);
	}
	break;
      }
      
    case RGB:
      {
	double sat;
	if (P.color == GRAYSCALE) {
	  P.greytorgb();
	  sat=0.5*(P.rgbsaturation()+Q.rgbsaturation());
	} else {
	  if (Q.color == GRAYSCALE) {
	    Q.greytorgb();
	    sat=0.5*(P.rgbsaturation()+Q.rgbsaturation());
	  } else sat=max(P.rgbsaturation(),Q.rgbsaturation());
	}
	// Mix colors
	P.r += Q.r;
	P.g += Q.g;
	P.b += Q.b;
	double newsat=P.rgbsaturation();
	double scale=newsat ? sat/newsat: 1.0;
	R=P.r*scale;
	G=P.g*scale;
	B=P.b*scale;
	break;
      }
      
    case CMYK:
      {
	if (P.color == GRAYSCALE) P.greytocmyk();
	else if (Q.color == GRAYSCALE) Q.greytocmyk();
	
	if (P.color == RGB) P.rgbtocmyk();
	else if (Q.color == RGB) Q.rgbtocmyk();
	
	double sat=max(P.cmyksaturation(),Q.cmyksaturation());      
	// Mix colors
	P.r += Q.r;
	P.g += Q.g;
	P.b += Q.b;
	P.grey += Q.grey;
	double newsat=P.cmyksaturation();
	double scale=newsat ? sat/newsat: 1.0;
	R=P.r*scale;
	G=P.g*scale;
	B=P.b*scale;
	greyval=P.grey*scale;
	break;
      }
    }
    
    return pen(q.line == DEFLINE ? p.line : q.line,
	       q.linewidth == DEFWIDTH ? p.linewidth : q.linewidth,
	       q.fontsize == 0.0 ? p.fontsize : q.fontsize,
	       colorspace,R,G,B,greyval,
	       q.pattern == DEFPAT ? p.pattern : q.pattern,
	       q.linecap == DEFCAP ? p.linecap : q.linecap,
	       q.linejoin == DEFJOIN ? p.linejoin : q.linejoin,
	       q.overwrite == DEFWRITE ? p.overwrite : q.overwrite,
	       q.t == NULL ? p.t : q.t);
  }

  friend bool operator == (const pen& p, const pen& q) {
    return p.stroke() == q.stroke() 
      && p.width() == q.width() 
      && p.colorspace() == q.colorspace()
      && p.size() == q.size()
      && (!(p.grayscale() || p.cmyk()) || p.gray() == q.gray())
      && (!(p.rgb() || p.cmyk()) || (p.red() == q.red() &&
				     p.green() == q.green() &&
				     p.blue() == q.blue()))
      && p.pattern == q.pattern
      && p.cap() == q.cap()
      && p.join() == q.join()
      && p.Overwrite() == q.Overwrite()
      && (p.t ? *p.t : identity()) == (q.t ? *q.t : identity());
  }
  
  friend ostream& operator << (ostream& out, const pen& p) {
    out << "([" << p.line << "]";
    if(p.linewidth != DEFWIDTH)
      out << ", linewidth=" << p.linewidth;
    if(p.linecap != DEFCAP)
      out << ", linecap=" << Cap[p.linecap];
    if(p.linejoin != DEFJOIN)
      out << ", linejoin=" << Join[p.linejoin];
    if(p.fontsize)
      out << ", fontsize=" << p.fontsize;
    if(p.color == GRAYSCALE)
      out << ", gray=" << p.grey;
    if(p.color == RGB)
      out << ", red=" << p.red() << ", green=" << p.green() 
	  << ", blue=" << p.blue();
    if(p.color == CMYK)
      out << ", cyan=" << p.cyan() << ", magenta=" << p.magenta() 
	  << ", yellow=" << p.yellow() << ", black=" << p.black();
    if(p.pattern != DEFPAT)
      out << ", " << "\"" << p.pattern << "\"";
    if(p.overwrite != DEFWRITE)
      out << ", overwrite=" << OverwriteTag[p.overwrite];
    if(p.t)
      out << ", transform=" << *(p.t);
    out << ")";
    
    return out;
  }

  // The bounds of the circle or ellipse the pen describes.
  bbox bounds() const
  {
    double maxx, maxy;
    pair shift;

    if (t != 0) {
      double xx = t->getxx(), xy = t->getxy(),
             yx = t->getyx(), yy = t->getyy();

      // These are the maximum x and y values that a linear transform can map
      // a point in the unit circle.  This can be proven by the Lagrange
      // Multiplier theorem or by properties of the dot product.
      maxx = length(pair(xx,xy));
      maxy = length(pair(yx,yy));

      shift = *t*pair(0,0);
    }
    else {
      maxx = 1;
      maxy = 1;
      shift = pair(0,0);
    }

    bbox b;
    pair z = width() * pair(maxx,maxy);
    b += z + shift;
    b += -z + shift;

    return b;
  }

  friend pen transformed(const transform* t, const pen& p) {
    pen ret = p;
    ret.t = p.t ? new transform((*t)*(*p.t)) : t;
    return ret;
  }

  const transform *getTransform() const {
    return t;
  }
};
  
}

#endif
